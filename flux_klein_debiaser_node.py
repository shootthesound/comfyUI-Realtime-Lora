"""
DIT Deep Debiaser — FLUX.2 Klein (Architecture-Verified)
=========================================================
Built from empirical forward-pass hook tracing of the actual model.

VERIFIED ARCHITECTURE (from hook tracing, NOT from tensor name guesses):
┌─────────────────────────────────────────────────────────────────┐
│  Reference Latent [1,128,H,W] → patchify → [1, 4070, 128]     │
│  Noisy Latent [1,128,H,W]    → patchify → [1, 4070, 128]      │
│                                      │                          │
│                                 CONCATENATE                     │
│                                      ↓                          │
│                               [1, 8140, 128]                    │
│                                      │                          │
│                                   img_in  [4096, 128]           │
│                                      ↓                          │
│                               [1, 8140, 4096]                   │
│                                                                 │
│  Text [1,512,12288]  →  txt_in [4096, 12288]  → [1, 512, 4096] │
│                                                                 │
│  Timestep  →  time_in  → [4096]  (conditions via adaLN)        │
├─────────────────────────────────────────────────────────────────┤
│  DOUBLE BLOCKS (×8): SEPARATE STREAMS                           │
│    img_stream: [1, 8140, 4096]  (ref+noisy ONLY)               │
│    txt_stream: [1, 512, 4096]   (text ONLY)                    │
│    Each: img_attn, img_mlp, txt_attn, txt_mlp                  │
│    Modulated by: double_stream_modulation_img/txt               │
│    → NO cross-modal interaction in these blocks                 │
├─────────────────────────────────────────────────────────────────┤
│  SINGLE BLOCKS (×24): CONCATENATED JOINT PROCESSING             │
│    combined: [1, 8652, 4096]  (8140 img + 512 txt)              │
│    Each: linear1 [36864,4096], linear2 [4096,16384], norm       │
│    Modulated by: single_stream_modulation                       │
│    → THIS IS WHERE TEXT INFLUENCES IMAGE GENERATION             │
│    → Cross-attention over full joint sequence happens HERE       │
├─────────────────────────────────────────────────────────────────┤
│  final_layer → unpatchify → output                              │
└─────────────────────────────────────────────────────────────────┘

KEY INSIGHTS:
  - Double blocks CANNOT cause text-driven image corruption (streams isolated)
  - Single blocks ARE where text overrides your reference image
  - Reference latent = HALF of img_stream (4070 of 8140 tokens)
  - Weakening img_* in double blocks hits ref AND noisy equally
  - linear1 in single blocks packs QKV+MLP gate — can't split at weight level

201 tensors mapped to 63 sub-component controls:
  7 global + 32 double block (8×4) + 24 single blocks

LoRA-safe via ComfyUI's add_patches system.
"""

import re
import json
import os
import time
import torch
from pathlib import Path
from collections import defaultdict


# ============================================================================
# SAVE PATH CONFIG (matches .v2_save_paths.json pattern)
# ============================================================================

_SAVE_CONFIG_NAME = ".v2_save_paths.json"


def _find_comfyui_root():
    """Walk up from this file to find ComfyUI root."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "main.py").exists() or (parent / "comfy").is_dir():
            return parent
    return Path(__file__).resolve().parent.parent.parent


def _get_save_config_path():
    return (Path(__file__).resolve().parent / _SAVE_CONFIG_NAME)


def _load_save_config():
    cfg_path = _get_save_config_path()
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_save_config(config):
    cfg_path = _get_save_config_path()
    try:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"[FLUX Klein Debiaser] WARNING: Could not save config: {e}")


def _get_default_save_dir():
    root = _find_comfyui_root()
    cfg = _load_save_config()
    if "FluxKleinDeepDebiaser" in cfg:
        p = Path(cfg["FluxKleinDeepDebiaser"])
        if p.is_absolute():
            return p
        return root / p
    return root / "models" / "diffusion_models" / "debiased"


def _resolve_save_path(filename, save_dir_override=""):
    if save_dir_override.strip():
        base = Path(save_dir_override.strip())
        if not base.is_absolute():
            base = _find_comfyui_root() / base
    else:
        base = _get_default_save_dir()
    base.mkdir(parents=True, exist_ok=True)
    if not filename.endswith(".safetensors"):
        filename = filename + ".safetensors"
    return base / filename


# ============================================================================
# SAVE IMPLEMENTATION
# ============================================================================

def _build_modified_state_dict(state_dict, key_map, block_strengths):
    """Build a new state dict with strengths applied directly to weights.
    Returns (modified_sd, n_modified, n_total).
    """
    modified_sd = {}
    n_modified = 0

    # Build key→strength lookup
    key_strength = {}
    for bid, strength in block_strengths.items():
        for key in key_map.get(bid, []):
            key_strength[key] = strength

    for key, tensor in state_dict.items():
        s = key_strength.get(key, 1.0)
        if s != 1.0:
            modified_sd[key] = (tensor.float() * s).to(tensor.dtype)
            n_modified += 1
        else:
            modified_sd[key] = tensor

    return modified_sd, n_modified, len(state_dict)


def _build_diff_state_dict(state_dict, key_map, block_strengths):
    """Build a diff-only state dict: only tensors that changed, storing (w*s - w).
    Much smaller than full model. Can be applied later as patches.
    Returns (diff_sd, n_modified).
    """
    diff_sd = {}

    key_strength = {}
    for bid, strength in block_strengths.items():
        for key in key_map.get(bid, []):
            key_strength[key] = strength

    for key, tensor in state_dict.items():
        s = key_strength.get(key, 1.0)
        if s != 1.0:
            diff = (tensor.float() * (s - 1.0)).to(tensor.dtype)
            diff_sd[key] = diff

    return diff_sd, len(diff_sd)


def _generate_filename(block_strengths, preset_name="Custom"):
    """Auto-generate descriptive filename from settings."""
    modified = {k: v for k, v in block_strengths.items() if v != 1.0}
    if not modified:
        return "flux_klein_debiased_default"

    # Summarize what was changed
    parts = ["flux_klein"]

    # Check for common patterns
    sb_vals = [block_strengths.get(f"sb{i}", 1.0) for i in range(24)]
    sb_modified = [v for v in sb_vals if v != 1.0]

    if sb_modified and len(set(sb_modified)) == 1:
        parts.append(f"sb_all_{int(sb_modified[0]*100)}")
    elif sb_modified:
        parts.append(f"sb_{len(sb_modified)}mod")

    db_modified = sum(1 for b, v in block_strengths.items()
                      if b.startswith("db") and v != 1.0)
    if db_modified:
        parts.append(f"db_{db_modified}mod")

    global_modified = [b for b in GLOBAL_BLOCKS
                       if block_strengths.get(b, 1.0) != 1.0]
    if global_modified:
        parts.append("g_" + "_".join(global_modified[:3]))

    ts = time.strftime("%m%d_%H%M")
    parts.append(ts)

    return "_".join(parts)


# ============================================================================
# ARCHITECTURE CONSTANTS (verified by forward-pass hook tracing)
# ============================================================================

N_DOUBLE = 8    # double_blocks: separate img/txt streams
N_SINGLE = 24   # single_blocks: concatenated cross-modal
DB_SUBS = ["img_attn", "img_mlp", "txt_attn", "txt_mlp"]


# ============================================================================
# COMPLETE TENSOR INVENTORY (201 tensors, verified)
#
# Per double block (12 tensors × 8 = 96):
#   img_attn: qkv.weight[12288,4096], proj.weight[4096,4096],
#             norm.key_norm.scale[128], norm.query_norm.scale[128]
#   img_mlp:  0.weight[24576,4096], 2.weight[4096,12288]
#   txt_attn: (same as img_attn)
#   txt_mlp:  (same as img_mlp)
#
# Per single block (4 tensors × 24 = 96):
#   linear1.weight[36864,4096]  (packed QKV + MLP gate)
#   linear2.weight[4096,16384]  (output projection)
#   norm.key_norm.scale[128]
#   norm.query_norm.scale[128]
#
# Global (9 tensors):
#   img_in.weight[4096,128]
#   txt_in.weight[4096,12288]
#   time_in.in_layer.weight[4096,256]
#   time_in.out_layer.weight[4096,4096]
#   double_stream_modulation_img.lin.weight[24576,4096]
#   double_stream_modulation_txt.lin.weight[24576,4096]
#   single_stream_modulation.lin.weight[12288,4096]
#   final_layer.adaLN_modulation.1.weight[8192,4096]
#   final_layer.linear.weight[128,4096]
# ============================================================================


# ============================================================================
# SUB-COMPONENT DEFINITIONS (63 controls)
# ============================================================================

# --- 7 Global controls ---
GLOBAL_BLOCKS = [
    "img_in", "txt_in", "time_in",
    "db_mod_img", "db_mod_txt", "sb_mod",
    "final_layer",
]

GLOBAL_INFO = {
    "img_in": {
        "label": "img_in (Patch→4096d)",
        "tip": "Projects patchified image tokens (ref+noisy combined) from 128 to 4096 dims. "
               "Affects ALL image processing. Reference latent enters through here.",
    },
    "txt_in": {
        "label": "txt_in (Qwen3→4096d)",
        "tip": "Projects Qwen3 8B text embeddings from 12288 to 4096 dims. "
               "Affects ALL text influence. Weakening this globally reduces prompt strength.",
    },
    "time_in": {
        "label": "time_in (Timestep)",
        "tip": "Timestep embedding [256]→[4096]. Controls how model responds to denoising stage. "
               "Modifying this changes the entire denoising dynamics.",
    },
    "db_mod_img": {
        "label": "DB Modulation IMG",
        "tip": "adaLN modulation for image stream in double blocks [24576,4096]. "
               "Controls how timestep signal scales/shifts img processing.",
    },
    "db_mod_txt": {
        "label": "DB Modulation TXT",
        "tip": "adaLN modulation for text stream in double blocks [24576,4096]. "
               "Controls how timestep signal scales/shifts txt processing.",
    },
    "sb_mod": {
        "label": "SB Modulation (Joint)",
        "tip": "adaLN modulation for ALL single blocks [12288,4096]. This modulates "
               "the JOINT cross-modal processing. Very powerful — affects all 24 single blocks.",
    },
    "final_layer": {
        "label": "Final Layer (Output)",
        "tip": "Final output projection [128,4096] + adaLN modulation [8192,4096]. "
               "Last thing before unpatchify. Weakening this dampens ALL output.",
    },
}

# --- 32 Double block controls (8 blocks × 4 subs) ---
DB_BLOCKS = []
DB_INFO = {}

for i in range(N_DOUBLE):
    for sub in DB_SUBS:
        bid = f"db{i}_{sub}"
        DB_BLOCKS.append(bid)
        stream = "IMG" if sub.startswith("img") else "TXT"
        comp = "Attention" if "attn" in sub else "MLP"
        if stream == "IMG":
            tip = (
                f"Double block {i}, image {comp.lower()}. SEPARATE stream — processes "
                f"ref+noisy tokens [8140] independently from text. "
            )
            if "attn" in sub:
                tip += "QKV [12288,4096] computes self-attention over image tokens only."
            else:
                tip += "Up [24576,4096] + down [4096,12288] feed-forward on image features."
        else:
            tip = (
                f"Double block {i}, text {comp.lower()}. SEPARATE stream — processes "
                f"text tokens [512] independently from image. "
            )
            if "attn" in sub:
                tip += "QKV [12288,4096] computes self-attention over text tokens only."
            else:
                tip += "Up [24576,4096] + down [4096,12288] feed-forward on text features."
        DB_INFO[bid] = {"label": f"DB{i} {stream} {comp}", "tip": tip}

# --- 24 Single block controls ---
SB_BLOCKS = []
SB_INFO = {}

for i in range(N_SINGLE):
    bid = f"sb{i}"
    SB_BLOCKS.append(bid)
    phase = "early" if i < 8 else ("mid" if i < 16 else "late")
    tip = (
        f"Single block {i} ({phase}): Processes the CONCATENATED sequence "
        f"[8140 img + 512 txt = 8652 tokens]. THIS is where text tokens attend to "
        f"image tokens and vice versa. linear1 [36864,4096] packs QKV+MLP gate "
        f"(cannot split attention from MLP at weight level). "
        f"linear2 [4096,16384] is the output projection."
    )
    SB_INFO[bid] = {"label": f"SB{i} Joint ({phase})", "tip": tip}

# Complete ordered list
ALL_BLOCKS = GLOBAL_BLOCKS + DB_BLOCKS + SB_BLOCKS
ALL_INFO = {**GLOBAL_INFO, **DB_INFO, **SB_INFO}


# ============================================================================
# STATE_DICT KEY → BLOCK MAPPING (exhaustive, verified against 201 tensors)
# ============================================================================

def _key_to_block(key):
    """Map a single state_dict key to its sub-component block ID.

    Returns block_id string or None if unrecognized.
    Every key in FLUX.2 Klein should map to exactly one block.
    """
    # --- Global embedders ---
    if key.startswith("img_in."):
        return "img_in"
    if key.startswith("txt_in."):
        return "txt_in"
    if key.startswith("time_in."):
        return "time_in"

    # --- Modulations ---
    if key.startswith("double_stream_modulation_img."):
        return "db_mod_img"
    if key.startswith("double_stream_modulation_txt."):
        return "db_mod_txt"
    if key.startswith("single_stream_modulation."):
        return "sb_mod"

    # --- Final layer ---
    if key.startswith("final_layer."):
        return "final_layer"

    # --- Double blocks ---
    m = re.match(r"double_blocks\.(\d+)\.(img_attn|img_mlp|txt_attn|txt_mlp)\.", key)
    if m:
        return f"db{m.group(1)}_{m.group(2)}"

    # --- Single blocks ---
    m = re.match(r"single_blocks\.(\d+)\.", key)
    if m:
        return f"sb{m.group(1)}"

    return None


def _build_key_map(state_dict):
    """Build block_id → [keys] and find any unmapped keys."""
    mapping = defaultdict(list)
    unmapped = []
    for key in state_dict.keys():
        bid = _key_to_block(key)
        if bid is not None:
            mapping[bid].append(key)
        else:
            unmapped.append(key)
    return dict(mapping), unmapped


# ============================================================================
# PATCHING (LoRA-safe via add_patches)
# ============================================================================

def _apply_patches(model_patcher, key_map, block_strengths):
    """Apply strength modifications as additive patches. LoRA-safe."""

    inner = model_patcher.model
    diff = inner.diffusion_model if hasattr(inner, "diffusion_model") else inner
    pfx = "diffusion_model." if hasattr(inner, "diffusion_model") else ""
    sd = diff.state_dict()

    cloned = model_patcher.clone()
    patches = {}
    count = 0

    for bid, strength in block_strengths.items():
        if strength == 1.0:
            continue
        for key in key_map.get(bid, []):
            if key not in sd:
                continue
            w = sd[key]
            w_cpu = w.detach().cpu() if w.device.type != "cpu" else w.detach()
            patches[pfx + key] = (w_cpu * (strength - 1.0),)
            count += 1

    if patches:
        cloned.add_patches(patches, strength_patch=1.0)

    return cloned, count


# ============================================================================
# INFO / ANALYSIS
# ============================================================================

def _compute_stats(sd, key_map, block_strengths):
    """Compute per-block statistics from state dict."""
    stats = {}
    max_norm = 0.0

    for bid in ALL_BLOCKS:
        keys = key_map.get(bid, [])
        params = 0
        norm = 0.0
        for k in keys:
            if k in sd:
                t = sd[k]
                params += t.numel()
                norm += t.float().norm().item()
        s = block_strengths.get(bid, 1.0)
        stats[bid] = {
            "params": params, "norm": norm,
            "tensors": len(keys), "strength": s,
        }
        max_norm = max(max_norm, norm)

    for bid in stats:
        stats[bid]["score"] = stats[bid]["norm"] / max(max_norm, 1e-8) * 100.0

    return stats


def _format_info(stats, block_strengths, n_patched):
    """Human-readable summary."""
    total_p = sum(s["params"] for s in stats.values())
    lines = [
        "DIT Deep Debiaser — FLUX.2 Klein (Verified Architecture)",
        "=" * 60,
        f"Model: {total_p / 1e9:.2f}B params | "
        f"8 double blocks (SEPARATE) + 24 single blocks (JOINT)",
        "",
    ]

    sections = [
        ("GLOBAL", GLOBAL_BLOCKS),
        ("DOUBLE BLOCKS (separate img/txt streams)", DB_BLOCKS),
        ("SINGLE BLOCKS (joint cross-modal — where text→image happens)", SB_BLOCKS),
    ]

    modified = []
    untouched = 0

    for label, blocks in sections:
        sec_mods = []
        for bid in blocks:
            s = stats.get(bid)
            if not s:
                continue
            st = s["strength"]
            if st == 1.0:
                untouched += 1
                continue
            tag = "OFF" if st == 0.0 else f"{st:.2f}"
            info = ALL_INFO.get(bid, {})
            sec_mods.append(f"  {info.get('label', bid):<38} → {tag}")
        if sec_mods:
            modified.append(f"\n{label}:")
            modified.extend(sec_mods)

    if modified:
        lines.append("MODIFIED:")
        lines.extend(modified)
    else:
        lines.append("(All at 1.0 — no modifications)")

    lines.append(f"\n{untouched} sub-components unchanged at 1.00")
    lines.append(f"Patched {n_patched} tensors (LoRA-safe)")
    lines.append("=" * 60)
    return "\n".join(lines)


def _make_json(stats):
    return json.dumps({
        "architecture": "FLUX2_KLEIN_VERIFIED",
        "blocks": {
            bid: {
                "params": s["params"],
                "score": round(s["score"], 1),
                "strength": s["strength"],
                "tensors": s["tensors"],
            }
            for bid, s in stats.items()
        }
    })


# ============================================================================
# PRESETS
# ============================================================================

def _preset_strengths(name):
    """Return block_id → strength dict for a preset, or None for Custom."""

    if name == "Custom":
        return None

    base = {bid: 1.0 for bid in ALL_BLOCKS}

    if name == "Default":
        return base

    # --- Single-block (cross-modal) presets ---
    if name == "Weaken All Singles 95%":
        for bid in SB_BLOCKS:
            base[bid] = 0.95
    elif name == "Weaken All Singles 90%":
        for bid in SB_BLOCKS:
            base[bid] = 0.90
    elif name == "Weaken All Singles 85%":
        for bid in SB_BLOCKS:
            base[bid] = 0.85
    elif name == "Weaken Late Singles 90% (SB12-23)":
        for i in range(12, 24):
            base[f"sb{i}"] = 0.90
    elif name == "Weaken Late Singles 85% (SB12-23)":
        for i in range(12, 24):
            base[f"sb{i}"] = 0.85
    elif name == "Weaken Early Singles 90% (SB0-7)":
        for i in range(8):
            base[f"sb{i}"] = 0.90

    # --- Double-block stream presets ---
    elif name == "Weaken DB img_mlp 90%":
        for i in range(N_DOUBLE):
            base[f"db{i}_img_mlp"] = 0.90
    elif name == "Weaken DB txt_mlp 90%":
        for i in range(N_DOUBLE):
            base[f"db{i}_txt_mlp"] = 0.90
    elif name == "Weaken DB img_attn 90%":
        for i in range(N_DOUBLE):
            base[f"db{i}_img_attn"] = 0.90
    elif name == "Weaken DB txt_attn 90%":
        for i in range(N_DOUBLE):
            base[f"db{i}_txt_attn"] = 0.90
    elif name == "Weaken ALL img stream 90%":
        base["img_in"] = 0.90
        base["db_mod_img"] = 0.90
        for i in range(N_DOUBLE):
            base[f"db{i}_img_attn"] = 0.90
            base[f"db{i}_img_mlp"] = 0.90
    elif name == "Weaken ALL txt stream 90%":
        base["txt_in"] = 0.90
        base["db_mod_txt"] = 0.90
        for i in range(N_DOUBLE):
            base[f"db{i}_txt_attn"] = 0.90
            base[f"db{i}_txt_mlp"] = 0.90

    # --- Global presets ---
    elif name == "Boost img_in 115%":
        base["img_in"] = 1.15
    elif name == "Boost txt_in 115%":
        base["txt_in"] = 1.15
    elif name == "Weaken sb_mod 90%":
        base["sb_mod"] = 0.90
    elif name == "Global 95%":
        base = {bid: 0.95 for bid in ALL_BLOCKS}
    elif name == "Global 90%":
        base = {bid: 0.90 for bid in ALL_BLOCKS}

    # --- Combo presets ---
    elif name == "Protect Reference (edit mode)":
        # Weaken late singles where text overrides reference structure
        for i in range(12, 24):
            base[f"sb{i}"] = 0.88
        # Slightly boost img_in to strengthen reference signal
        base["img_in"] = 1.10
    elif name == "Stronger Prompt (edit mode)":
        # Boost text stream processing
        base["txt_in"] = 1.15
        base["db_mod_txt"] = 1.10
        for i in range(N_DOUBLE):
            base[f"db{i}_txt_attn"] = 1.10
            base[f"db{i}_txt_mlp"] = 1.10

    return base


PRESET_NAMES = [
    "Custom", "Default",
    # Cross-modal (single blocks) — the main lever
    "Weaken All Singles 95%",
    "Weaken All Singles 90%",
    "Weaken All Singles 85%",
    "Weaken Late Singles 90% (SB12-23)",
    "Weaken Late Singles 85% (SB12-23)",
    "Weaken Early Singles 90% (SB0-7)",
    # Stream-specific (double blocks)
    "Weaken DB img_mlp 90%",
    "Weaken DB txt_mlp 90%",
    "Weaken DB img_attn 90%",
    "Weaken DB txt_attn 90%",
    "Weaken ALL img stream 90%",
    "Weaken ALL txt stream 90%",
    # Global / embedders
    "Boost img_in 115%",
    "Boost txt_in 115%",
    "Weaken sb_mod 90%",
    "Global 95%", "Global 90%",
    # Combo
    "Protect Reference (edit mode)",
    "Stronger Prompt (edit mode)",
]


# ============================================================================
# NODE
# ============================================================================

class FluxKleinDeepDebiaser:
    """
    Architecture-verified debiaser for FLUX.2 Klein 9B.

    63 controls mapped to empirically verified forward-pass behavior.
    LoRA-safe via ComfyUI's add_patches system.
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {"required": {
            "model": ("MODEL", {"tooltip": "FLUX.2 Klein model to debias"}),
            "preset": (PRESET_NAMES, {"default": "Default"}),
            "save_model": ("BOOLEAN", {
                "default": False,
                "tooltip": "Save the modified model to disk as .safetensors",
            }),
            "save_mode": (["full_model", "diff_only"], {
                "default": "full_model",
                "tooltip": "full_model: complete ~18GB safetensors (load directly). "
                           "diff_only: small file with only changed tensors (apply as patch later).",
            }),
            "save_filename": ("STRING", {
                "default": "auto",
                "multiline": False,
                "tooltip": "'auto' generates a descriptive name. Otherwise enter a custom name "
                           "(extension added automatically).",
            }),
            "save_directory": ("STRING", {
                "default": "",
                "multiline": False,
                "tooltip": "Override save directory. Empty = use default from config "
                           "(models/diffusion_models/debiased). Can be absolute or relative to ComfyUI root.",
            }),
        }}

        for bid in GLOBAL_BLOCKS:
            info = GLOBAL_INFO[bid]
            inputs["required"][bid] = ("BOOLEAN", {
                "default": True, "tooltip": info["tip"],
            })
            inputs["required"][f"{bid}_str"] = ("FLOAT", {
                "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
                "tooltip": f"Strength: {info['label']}",
            })

        for bid in DB_BLOCKS:
            info = DB_INFO[bid]
            inputs["required"][bid] = ("BOOLEAN", {
                "default": True, "tooltip": info["tip"],
            })
            inputs["required"][f"{bid}_str"] = ("FLOAT", {
                "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
                "tooltip": f"Strength: {info['label']}",
            })

        for bid in SB_BLOCKS:
            info = SB_INFO[bid]
            inputs["required"][bid] = ("BOOLEAN", {
                "default": True, "tooltip": info["tip"],
            })
            inputs["required"][f"{bid}_str"] = ("FLOAT", {
                "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
                "tooltip": f"Strength: {info['label']}",
            })

        return inputs

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("model", "info", "save_path")
    OUTPUT_TOOLTIPS = (
        "Model with architecture-verified modifications (LoRA patches preserved)",
        "Summary of all modifications applied",
        "Path where model was saved (empty if save_model=False)",
    )
    FUNCTION = "debias"
    CATEGORY = "model_patches"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "FLUX.2 Klein debiaser built from verified forward-pass hook analysis.\n\n"
        "VERIFIED ARCHITECTURE:\n"
        "  Double blocks (0-7): SEPARATE img/txt streams — no cross-modal interaction\n"
        "    img_attn/img_mlp: processes ref+noisy image tokens [8140] ONLY\n"
        "    txt_attn/txt_mlp: processes text tokens [512] ONLY\n\n"
        "  Single blocks (0-23): CONCATENATED cross-modal [8652 tokens]\n"
        "    Text and image tokens interact HERE\n"
        "    This is where 'model overwrites your image' happens\n"
        "    linear1 packs QKV+MLP gate (can't split at weight level)\n\n"
        "  Global: img_in, txt_in, time_in, modulations, final_layer\n\n"
        "63 controls. Strength 1.0 = unchanged. LoRA-safe.\n\n"
        "SAVE: Export modified model as .safetensors:\n"
        "  full_model: Complete model (~18GB), load directly as replacement\n"
        "  diff_only: Just the changes (small), apply as additive patch"
    )

    def debias(self, model, preset, save_model, save_mode, save_filename,
               save_directory, **kwargs):
        print("[FLUX Klein Debiaser v2] Starting (verified architecture)...")

        # Build strength map from widgets
        block_strengths = {}
        for bid in ALL_BLOCKS:
            enabled = kwargs.get(bid, True)
            if enabled:
                block_strengths[bid] = kwargs.get(f"{bid}_str", 1.0)
            else:
                block_strengths[bid] = 0.0

        # Access model
        inner = model.model
        diff = inner.diffusion_model if hasattr(inner, "diffusion_model") else inner
        sd = diff.state_dict()

        # Build key mapping
        key_map, unmapped = _build_key_map(sd)
        n_mapped = sum(len(v) for v in key_map.values())
        print(f"[FLUX Klein Debiaser v2] Mapped {n_mapped}/201 tensors to "
              f"{len(key_map)} sub-components")
        if unmapped:
            print(f"[FLUX Klein Debiaser v2] WARNING: {len(unmapped)} unmapped: "
                  f"{unmapped[:5]}...")

        # Check if anything needs modification
        needs_mod = any(s != 1.0 for s in block_strengths.values())

        if needs_mod:
            model_out, n_patched = _apply_patches(model, key_map, block_strengths)
            print(f"[FLUX Klein Debiaser v2] Patched {n_patched} tensors (LoRA-safe)")
        else:
            model_out = model.clone()
            n_patched = 0
            print("[FLUX Klein Debiaser v2] No modifications (all at 1.0)")

        # Stats and info
        stats = _compute_stats(sd, key_map, block_strengths)
        info = _format_info(stats, block_strengths, n_patched)
        analysis_json = _make_json(stats)

        # ---- SAVE ----
        saved_path = ""
        if save_model and needs_mod:
            try:
                import safetensors.torch as sf_torch

                # Resolve filename
                if save_filename.strip().lower() in ("auto", ""):
                    fname = _generate_filename(block_strengths, preset)
                else:
                    fname = save_filename.strip()

                # Add mode suffix for diff
                if save_mode == "diff_only" and not fname.endswith("_diff"):
                    fname = fname.replace(".safetensors", "") + "_diff"

                full_path = _resolve_save_path(fname, save_directory)

                print(f"[FLUX Klein Debiaser v2] Saving ({save_mode}) to: {full_path}")
                t0 = time.time()

                if save_mode == "full_model":
                    modified_sd, n_mod, n_total = _build_modified_state_dict(
                        sd, key_map, block_strengths
                    )
                    # Add metadata
                    metadata = {
                        "debiaser": "FluxKleinDeepDebiaser_v2",
                        "architecture": "FLUX2_KLEIN_VERIFIED",
                        "save_mode": "full_model",
                        "modified_tensors": str(n_mod),
                        "total_tensors": str(n_total),
                        "preset": preset,
                        "strengths": json.dumps(
                            {k: v for k, v in block_strengths.items() if v != 1.0}
                        ),
                    }
                    sf_torch.save_file(modified_sd, str(full_path), metadata=metadata)
                    size_gb = full_path.stat().st_size / (1024**3)
                    elapsed = time.time() - t0
                    print(f"[FLUX Klein Debiaser v2] Saved full model: "
                          f"{size_gb:.2f} GB, {n_mod}/{n_total} modified, "
                          f"{elapsed:.1f}s")

                elif save_mode == "diff_only":
                    diff_sd, n_diff = _build_diff_state_dict(
                        sd, key_map, block_strengths
                    )
                    metadata = {
                        "debiaser": "FluxKleinDeepDebiaser_v2",
                        "architecture": "FLUX2_KLEIN_VERIFIED",
                        "save_mode": "diff_only",
                        "diff_tensors": str(n_diff),
                        "preset": preset,
                        "strengths": json.dumps(
                            {k: v for k, v in block_strengths.items() if v != 1.0}
                        ),
                        "usage": "Apply as additive patch: weight = base_weight + diff_weight",
                    }
                    sf_torch.save_file(diff_sd, str(full_path), metadata=metadata)
                    size_mb = full_path.stat().st_size / (1024**2)
                    elapsed = time.time() - t0
                    print(f"[FLUX Klein Debiaser v2] Saved diff: "
                          f"{size_mb:.1f} MB, {n_diff} tensors, {elapsed:.1f}s")

                saved_path = str(full_path)
                info += f"\n\nSaved to: {saved_path}"

                # Update save config with directory for next time
                if save_directory.strip():
                    cfg = _load_save_config()
                    cfg["FluxKleinDeepDebiaser"] = save_directory.strip()
                    _save_save_config(cfg)

            except ImportError:
                print("[FLUX Klein Debiaser v2] ERROR: safetensors not installed. "
                      "Run: pip install safetensors")
                info += "\n\nSAVE FAILED: safetensors package not installed"
            except Exception as e:
                print(f"[FLUX Klein Debiaser v2] SAVE ERROR: {e}")
                info += f"\n\nSAVE FAILED: {e}"

        elif save_model and not needs_mod:
            info += "\n\nSkipped save — no modifications to save (all at 1.0)"
            print("[FLUX Klein Debiaser v2] Skipped save — nothing modified")

        print("[FLUX Klein Debiaser v2] Done.")
        return {
            "ui": {"analysis_json": [analysis_json]},
            "result": (model_out, info, saved_path),
        }


# ============================================================================
# REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "FluxKleinDeepDebiaser": FluxKleinDeepDebiaser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxKleinDeepDebiaser": "DIT Deep Debiaser (FLUX.2 Klein — Verified)",
}
