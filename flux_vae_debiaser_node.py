"""
ComfyUI Flux VAE Deep Debiaser — Individual Tensor Control

For Flux 2 Klein 9B's VAE (Flux.1 Autoencoder variant, 32ch latent).

125 individually controllable tensor units:
  - 1 bn (batch norm)
  - 70 decoder tensor units (every conv, norm, shortcut, q/k/v/proj)
  - 54 encoder tensor units (every conv, norm, shortcut, q/k/v/proj)

Each knob controls weight+bias of a single functional unit.
All tensors F32 — fully modifiable.
"""

import re
import os
import json
import torch
import datetime
from collections import OrderedDict, defaultdict
from pathlib import Path

try:
    from safetensors.torch import save_file as safetensors_save
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False


# ============================================================================
# SAVE PATHS CONFIG
# ============================================================================

_SAVE_PATHS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_v2_save_paths.json")


def _load_save_paths():
    if os.path.exists(_SAVE_PATHS_FILE):
        try:
            with open(_SAVE_PATHS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _get_default_save_dir():
    if HAS_FOLDER_PATHS:
        try:
            vae_dirs = folder_paths.get_folder_paths("vae")
            if vae_dirs:
                save_dir = os.path.join(vae_dirs[0], "debiased")
                os.makedirs(save_dir, exist_ok=True)
                return save_dir
        except Exception:
            pass
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _get_save_dir():
    paths = _load_save_paths()
    custom = paths.get("FluxVAEDebiaser", "")
    if custom and os.path.isdir(custom):
        return custom
    return _get_default_save_dir()


# ============================================================================
# BLOCK DEFINITIONS — Every individual tensor unit
# ============================================================================
# Each block = one functional unit (weight+bias pair = one knob).
# block_id -> (display_label, [list of state_dict keys])

BLOCKS = OrderedDict()

# --- Top-level batch norm ---
BLOCKS["bn"] = ("BN (top-level)", [
    "bn.num_batches_tracked", "bn.running_mean", "bn.running_var"
])

# --- DECODER ---

BLOCKS["d_pqconv"] = ("Dec PostQuantConv", [
    "decoder.post_quant_conv.bias", "decoder.post_quant_conv.weight"
])
BLOCKS["d_conv_in"] = ("Dec ConvIn", [
    "decoder.conv_in.bias", "decoder.conv_in.weight"
])

# Mid attention
for sub, label in [("q", "Q"), ("k", "K"), ("v", "V"), ("proj_out", "Proj"), ("norm", "Norm")]:
    bid = f"d_mid_attn_{sub}"
    BLOCKS[bid] = (f"Dec Mid Attn {label}", [
        f"decoder.mid.attn_1.{sub}.bias", f"decoder.mid.attn_1.{sub}.weight"
    ])

# Mid res blocks 1 & 2
for blk in [1, 2]:
    for sub in ["conv1", "conv2", "norm1", "norm2"]:
        bid = f"d_mid_b{blk}_{sub}"
        BLOCKS[bid] = (f"Dec Mid B{blk} {sub}", [
            f"decoder.mid.block_{blk}.{sub}.bias", f"decoder.mid.block_{blk}.{sub}.weight"
        ])

# Decoder up stages 0-3
# nin_shortcut exists for: up.0.block.0 and up.1.block.0
_DEC_NIN = {(0, 0), (1, 0)}
for stage in range(4):
    for b in range(3):
        for sub in ["conv1", "conv2", "norm1", "norm2"]:
            bid = f"d_u{stage}b{b}_{sub}"
            BLOCKS[bid] = (f"Dec Up{stage}.B{b} {sub}", [
                f"decoder.up.{stage}.block.{b}.{sub}.bias",
                f"decoder.up.{stage}.block.{b}.{sub}.weight"
            ])
        if (stage, b) in _DEC_NIN:
            bid = f"d_u{stage}b{b}_nin_shortcut"
            BLOCKS[bid] = (f"Dec Up{stage}.B{b} NinShort", [
                f"decoder.up.{stage}.block.{b}.nin_shortcut.bias",
                f"decoder.up.{stage}.block.{b}.nin_shortcut.weight"
            ])
    # upsample for stages 1, 2, 3
    if stage in [1, 2, 3]:
        bid = f"d_u{stage}_up"
        BLOCKS[bid] = (f"Dec Up{stage} Upsample", [
            f"decoder.up.{stage}.upsample.conv.bias",
            f"decoder.up.{stage}.upsample.conv.weight"
        ])

BLOCKS["d_norm_out"] = ("Dec NormOut", [
    "decoder.norm_out.bias", "decoder.norm_out.weight"
])
BLOCKS["d_conv_out"] = ("Dec ConvOut", [
    "decoder.conv_out.bias", "decoder.conv_out.weight"
])

# --- ENCODER ---

BLOCKS["e_conv_in"] = ("Enc ConvIn", [
    "encoder.conv_in.bias", "encoder.conv_in.weight"
])

# Encoder down stages 0-3
# nin_shortcut exists for: down.1.block.0 and down.2.block.0
_ENC_NIN = {(1, 0), (2, 0)}
for stage in range(4):
    for b in range(2):
        for sub in ["conv1", "conv2", "norm1", "norm2"]:
            bid = f"e_d{stage}b{b}_{sub}"
            BLOCKS[bid] = (f"Enc Down{stage}.B{b} {sub}", [
                f"encoder.down.{stage}.block.{b}.{sub}.bias",
                f"encoder.down.{stage}.block.{b}.{sub}.weight"
            ])
        if (stage, b) in _ENC_NIN:
            bid = f"e_d{stage}b{b}_nin_shortcut"
            BLOCKS[bid] = (f"Enc Down{stage}.B{b} NinShort", [
                f"encoder.down.{stage}.block.{b}.nin_shortcut.bias",
                f"encoder.down.{stage}.block.{b}.nin_shortcut.weight"
            ])
    # downsample for stages 0, 1, 2
    if stage in [0, 1, 2]:
        bid = f"e_d{stage}_down"
        BLOCKS[bid] = (f"Enc Down{stage} Downsamp", [
            f"encoder.down.{stage}.downsample.conv.bias",
            f"encoder.down.{stage}.downsample.conv.weight"
        ])

# Encoder mid attention
for sub, label in [("q", "Q"), ("k", "K"), ("v", "V"), ("proj_out", "Proj"), ("norm", "Norm")]:
    bid = f"e_mid_attn_{sub}"
    BLOCKS[bid] = (f"Enc Mid Attn {label}", [
        f"encoder.mid.attn_1.{sub}.bias", f"encoder.mid.attn_1.{sub}.weight"
    ])

# Encoder mid res blocks 1 & 2
for blk in [1, 2]:
    for sub in ["conv1", "conv2", "norm1", "norm2"]:
        bid = f"e_mid_b{blk}_{sub}"
        BLOCKS[bid] = (f"Enc Mid B{blk} {sub}", [
            f"encoder.mid.block_{blk}.{sub}.bias", f"encoder.mid.block_{blk}.{sub}.weight"
        ])

BLOCKS["e_norm_out"] = ("Enc NormOut", [
    "encoder.norm_out.bias", "encoder.norm_out.weight"
])
BLOCKS["e_conv_out"] = ("Enc ConvOut", [
    "encoder.conv_out.bias", "encoder.conv_out.weight"
])
BLOCKS["e_qconv"] = ("Enc QuantConv", [
    "encoder.quant_conv.bias", "encoder.quant_conv.weight"
])


# Ordered lists for iteration
ALL_BLOCK_IDS = list(BLOCKS.keys())
DEC_BLOCK_IDS = [b for b in ALL_BLOCK_IDS if b.startswith("d_")]
ENC_BLOCK_IDS = [b for b in ALL_BLOCK_IDS if b.startswith("e_")]


# ============================================================================
# KEY MAPPING — reverse lookup from state_dict key → block_id
# ============================================================================

def _build_key_to_block():
    """Pre-build reverse map: state_dict key -> block_id."""
    k2b = {}
    for bid, (label, keys) in BLOCKS.items():
        for key in keys:
            k2b[key] = bid
    return k2b

_KEY_TO_BLOCK = _build_key_to_block()


def _build_block_key_map(state_dict):
    """Build block_id -> [matched keys] from actual state dict."""
    block_key_map = defaultdict(list)
    key_to_block = {}
    unmatched = []

    for key in state_dict.keys():
        bid = _KEY_TO_BLOCK.get(key)
        if bid is not None:
            block_key_map[bid].append(key)
            key_to_block[key] = bid
        else:
            unmatched.append(key)

    return dict(block_key_map), unmatched, key_to_block


# ============================================================================
# ANALYSIS
# ============================================================================

def _analyze_blocks(state_dict, block_key_map, enabled_blocks, block_strengths):
    block_stats = {}
    max_norm = 0.0
    for block_id, keys in block_key_map.items():
        total_params = 0
        total_bytes = 0
        total_norm = 0.0
        for key in keys:
            tensor = state_dict[key]
            total_params += tensor.numel()
            total_bytes += tensor.numel() * tensor.element_size()
            if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                total_norm += tensor.float().norm().item()
        block_stats[block_id] = {
            "param_count": total_params,
            "memory_mb": total_bytes / (1024 * 1024),
            "weight_norm": total_norm,
            "tensor_count": len(keys),
            "enabled": block_id in enabled_blocks,
            "strength": block_strengths.get(block_id, 1.0) if block_id in enabled_blocks else 0.0,
        }
        if total_norm > max_norm:
            max_norm = total_norm
    if max_norm > 0:
        for block_id in block_stats:
            block_stats[block_id]["score"] = (block_stats[block_id]["weight_norm"] / max_norm) * 100.0
    return block_stats


def _format_info(block_stats, enabled_blocks, block_strengths, modified_count, save_msg=""):
    total_params = sum(b["param_count"] for b in block_stats.values())
    total_memory = sum(b["memory_mb"] for b in block_stats.values())
    dec_params = sum(b["param_count"] for bid, b in block_stats.items() if bid.startswith("d_"))
    enc_params = sum(b["param_count"] for bid, b in block_stats.items() if bid.startswith("e_"))

    lines = [
        "Flux VAE Deep Debiaser — Individual Tensor Control",
        "=" * 65,
        f"Total: {total_params / 1e6:.1f}M params, {total_memory:.1f} MB",
        f"  Decoder: {dec_params / 1e6:.1f}M | Encoder: {enc_params / 1e6:.1f}M",
        f"  {len(ALL_BLOCK_IDS)} controllable tensor units",
        "",
        f"{'Unit':<28} {'Tens':>5} {'Strength':>10}",
        "-" * 65,
    ]

    modified_lines = []
    unmodified_count = 0
    for block_id in ALL_BLOCK_IDS:
        if block_id not in block_stats:
            continue
        stats = block_stats[block_id]
        strength = stats["strength"]
        if strength == 1.0:
            unmodified_count += 1
            continue
        if strength == 0.0:
            strength_str = "OFF ⚠"
        else:
            strength_str = f"{strength:.2f} ←"
        label = BLOCKS[block_id][0]
        modified_lines.append(f"{label:<28} {stats['tensor_count']:>5} {strength_str:>10}")

    if modified_lines:
        lines.append("MODIFIED:")
        lines.extend(modified_lines)
    else:
        lines.append("(no modifications)")
    if unmodified_count > 0:
        lines.append(f"... + {unmodified_count} tensor units at 1.00")
    lines.append("-" * 65)
    lines.append(f"Modified: {len(modified_lines)}/{len(ALL_BLOCK_IDS)} units ({modified_count} tensors patched)")
    if save_msg:
        lines.append("")
        lines.append(save_msg)
    return "\n".join(lines)


def _create_analysis_json(block_stats):
    result = {"architecture": "FLUX_VAE_32CH", "blocks": {}}
    for block_id, stats in block_stats.items():
        result["blocks"][block_id] = {
            "param_count": stats["param_count"],
            "memory_mb": round(stats["memory_mb"], 2),
            "score": round(stats.get("score", 50.0), 1),
            "enabled": stats["enabled"],
            "strength": stats["strength"],
            "tensor_count": stats["tensor_count"],
        }
    return json.dumps(result)


# ============================================================================
# PATCHING
# ============================================================================

def _apply_modifications(vae, block_key_map, enabled_blocks, block_strengths, original_sd):
    """
    Apply modifications from ORIGINAL weights every time.
    Never compounds — always: new_weight = original_weight * strength.
    """
    modified_count = 0
    new_sd = {}

    for key, orig_tensor in original_sd.items():
        bid = None
        for block_id, keys in block_key_map.items():
            if key in keys:
                bid = block_id
                break

        if bid is not None:
            if bid not in enabled_blocks:
                strength = 0.0
            else:
                strength = block_strengths.get(bid, 1.0)

            if strength != 1.0 and orig_tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                new_sd[key] = (orig_tensor.detach().float() * strength).to(orig_tensor.dtype)
                modified_count += 1
            else:
                new_sd[key] = orig_tensor.detach().clone()
        else:
            new_sd[key] = orig_tensor.detach().clone()

    # Load fresh computed weights into VAE
    try:
        vae.first_stage_model.load_state_dict(new_sd, strict=False)
    except Exception as e:
        print(f"[Flux VAE Debiaser] load_state_dict warning: {e}")

    return vae, modified_count


# ============================================================================
# SAVE / EXPORT
# ============================================================================

def _build_config_dict(enabled_blocks, block_strengths):
    config = {
        "format": "flux_vae_debiaser_v2",
        "architecture": "Flux VAE (Flux 2 Klein — 32ch latent)",
        "timestamp": datetime.datetime.now().isoformat(),
        "total_units": len(ALL_BLOCK_IDS),
        "modified_units": {},
        "unmodified_count": 0,
    }
    for block_id in ALL_BLOCK_IDS:
        if block_id not in enabled_blocks:
            config["modified_units"][block_id] = {"enabled": False, "strength": 0.0}
        else:
            strength = block_strengths.get(block_id, 1.0)
            if strength != 1.0:
                config["modified_units"][block_id] = {"enabled": True, "strength": strength}
            else:
                config["unmodified_count"] += 1
    config["modified_count"] = len(config["modified_units"])
    return config


def _auto_filename(enabled_blocks, block_strengths):
    modified = []
    for bid in ALL_BLOCK_IDS:
        if bid not in enabled_blocks:
            modified.append(bid)
        else:
            s = block_strengths.get(bid, 1.0)
            if s != 1.0:
                modified.append(bid)

    if len(modified) == 0:
        tag = "unmodified"
    elif len(modified) <= 4:
        tag = "_".join(modified)
    else:
        # Count by category
        counts = defaultdict(int)
        for b in modified:
            if "conv" in b:
                counts["conv"] += 1
            elif "norm" in b:
                counts["norm"] += 1
            elif "attn" in b:
                counts["attn"] += 1
            elif "nin" in b or "shortcut" in b:
                counts["shortcut"] += 1
            elif "up" in b or "down" in b:
                counts["sample"] += 1
            else:
                counts["other"] += 1
        parts = [f"{c}{t}" for t, c in sorted(counts.items(), key=lambda x: -x[1])]
        tag = "_".join(parts[:4])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"flux_vae_debiased_{tag}_{timestamp}"


def _save_full_model(state_dict, enabled_blocks, block_strengths,
                     save_dir, filename, key_to_block_map):
    if not HAS_SAFETENSORS:
        return "ERROR: safetensors not installed"

    save_dict = {}
    modified_count = 0
    for key, tensor in state_dict.items():
        save_dict[key] = tensor.detach().cpu().contiguous()
        bid = key_to_block_map.get(key)
        if bid is not None:
            s = block_strengths.get(bid, 1.0) if bid in enabled_blocks else 0.0
            if s != 1.0:
                modified_count += 1

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename + ".safetensors")
    safetensors_save(save_dict, filepath)
    size_mb = os.path.getsize(filepath) / (1024 ** 2)
    return f"SAVED full: {filepath}\n  {len(save_dict)} tensors, {modified_count} modified, {size_mb:.1f} MB"


def _save_diff_only(orig_state_dict, modified_state_dict, enabled_blocks, block_strengths,
                    save_dir, filename, key_to_block_map):
    if not HAS_SAFETENSORS:
        return "ERROR: safetensors not installed"

    diff_dict = {}
    for key in modified_state_dict.keys():
        bid = key_to_block_map.get(key)
        if bid is None:
            continue
        s = block_strengths.get(bid, 1.0) if bid in enabled_blocks else 0.0
        if s == 1.0:
            continue
        orig = orig_state_dict[key]
        modified = modified_state_dict[key]
        if orig.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            diff = (modified.detach().float() - orig.detach().float()).to(orig.dtype)
            diff_dict[key] = diff.cpu().contiguous()

    if not diff_dict:
        return "No diffs to save"

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename + "_diff.safetensors")
    safetensors_save(diff_dict, filepath)
    size_mb = os.path.getsize(filepath) / (1024 ** 2)
    return f"SAVED diff: {filepath}\n  {len(diff_dict)} delta tensors, {size_mb:.1f} MB"


def _save_config(enabled_blocks, block_strengths, save_dir, filename):
    config = _build_config_dict(enabled_blocks, block_strengths)
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename + "_config.json")
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)
    return f"SAVED config: {filepath}"


# ============================================================================
# PRESETS
# ============================================================================

PRESET_NAMES = [
    "Custom",
    "Default",
    # Decoder up-stage presets
    "Dec All Up0 convs 90%",
    "Dec All Up0 convs 85%",
    "Dec All Up0+Up1 convs 90%",
    "Dec All Up convs 90%",
    "Dec All Up convs 85%",
    "Dec All Up norms 90%",
    # Mid attn
    "Dec Mid Attn all 90%",
    "Dec Mid Attn all 80%",
    "Dec Mid Attn Q+K 90%",
    # Norms
    "Dec All norms 90%",
    "Dec All norms 85%",
    # Boosts
    "Dec Up0 convs boost 110%",
    "Dec Up0 convs boost 120%",
    "Dec NormOut boost 110%",
    "Dec ConvOut boost 110%",
    # Global
    "All Decoder 95%",
    "All Decoder 90%",
    "All Decoder 85%",
    "Global 95%",
    "Global 90%",
]


# ============================================================================
# NODE CLASS
# ============================================================================

class FluxVAEDebiaser:
    """
    Individual tensor-level debiaser for Flux VAE (Flux 2 Klein — 32ch latent).

    125 individually controllable tensor units — every conv, norm, shortcut,
    q/k/v/proj_out gets its own toggle + strength slider.

    All tensors F32 — fully modifiable.
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "vae": ("VAE", {
                    "tooltip": "VAE model (Flux.1 Autoencoder for Flux 2 Klein)"
                }),
                "preset": (PRESET_NAMES, {
                    "default": "Default",
                }),
                "save_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Save the modified VAE to disk",
                }),
                "save_mode": (["full_model", "diff_only", "both"], {
                    "default": "full_model",
                }),
                "filename": ("STRING", {
                    "default": "auto",
                    "tooltip": "Filename (no ext). 'auto' = generate from settings.",
                }),
            },
        }

        # Register every block as a toggle + strength float
        for bid in ALL_BLOCK_IDS:
            inputs["required"][bid] = ("BOOLEAN", {"default": True})
            inputs["required"][f"{bid}_str"] = ("FLOAT", {
                "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
            })

        return inputs

    RETURN_TYPES = ("VAE", "STRING")
    RETURN_NAMES = ("vae", "info")
    OUTPUT_TOOLTIPS = (
        "VAE with per-tensor modifications",
        "Text summary",
    )
    FUNCTION = "debias"
    CATEGORY = "model_patches"
    OUTPUT_NODE = True
    DESCRIPTION = """Individual tensor-level debiaser for Flux VAE (Flux 2 Klein — 32ch latent).

125 tensor units — every conv1, conv2, norm1, norm2, nin_shortcut,
Q, K, V, proj_out, upsample, downsample gets its own knob.

All tensors F32 — fully modifiable (no quantization issues).

Strength 1.0 = unchanged. < 1.0 = weaken. > 1.0 = boost.

🎨 COLOR TIPS:
  • d_u0b*_conv* (Dec Up0, 128ch final stage) — fine color precision
  • d_u1b*_conv* (Dec Up1, 256ch) — color gradients & transitions
  • d_u2b*/d_u3b* (512ch) — coarse structure & base tone
  • d_mid_attn_* — global spatial color relationships
  • d_norm_out — overall brightness/contrast
  • d_conv_out — final RGB mapping (VERY sensitive!)

⚠️ Encoder units only matter for img2img/inpainting.
For txt2img, only decoder units affect output.

SAVE: Enable save_model to export modified VAE."""

    def debias(self, vae, preset, save_model=False, save_mode="full_model",
               filename="auto", **kwargs):
        print(f"[Flux VAE Debiaser] Starting... ({len(ALL_BLOCK_IDS)} tensor units)")

        enabled_blocks = set()
        block_strengths = {}
        for bid in ALL_BLOCK_IDS:
            if kwargs.get(bid, True):
                enabled_blocks.add(bid)
                block_strengths[bid] = kwargs.get(f"{bid}_str", 1.0)
            else:
                block_strengths[bid] = 0.0

        # Snapshot original weights on first run so we never compound modifications.
        # Stored on the vae object itself to survive across queue executions.
        if not hasattr(vae, '_flux_vae_debiaser_original_sd'):
            print(f"[Flux VAE Debiaser] Snapshotting original weights (first run)")
            vae._flux_vae_debiaser_original_sd = {
                k: v.detach().clone() for k, v in vae.first_stage_model.state_dict().items()
            }

        original_sd = vae._flux_vae_debiaser_original_sd

        block_key_map, unmatched, key_to_block_map = _build_block_key_map(original_sd)
        matched_keys = sum(len(v) for v in block_key_map.values())
        print(f"[Flux VAE Debiaser] Mapped {matched_keys} tensors across {len(block_key_map)} units")
        if unmatched:
            print(f"[Flux VAE Debiaser] {len(unmatched)} unmatched: {unmatched[:5]}...")

        needs_modification = any(
            bid not in enabled_blocks or block_strengths.get(bid, 1.0) != 1.0
            for bid in ALL_BLOCK_IDS
        )

        modified_count = 0
        if needs_modification:
            vae, modified_count = _apply_modifications(
                vae, block_key_map, enabled_blocks, block_strengths, original_sd
            )
            print(f"[Flux VAE Debiaser] Applied {modified_count} modifications (from originals, no compounding)")
        else:
            # Restore originals if user set everything back to 1.0
            vae.first_stage_model.load_state_dict(
                {k: v.detach().clone() for k, v in original_sd.items()}, strict=False
            )
            print(f"[Flux VAE Debiaser] Restored originals (all at 1.0)")

        # ---- SAVE ----
        save_msg = ""
        if save_model and needs_modification:
            save_dir = _get_save_dir()
            if not filename or filename.strip().lower() == "auto":
                fname = _auto_filename(enabled_blocks, block_strengths)
            else:
                fname = re.sub(r'[<>:"/\\|?*]', '_', filename.strip())
                if not fname:
                    fname = _auto_filename(enabled_blocks, block_strengths)

            save_lines = []
            current_sd = vae.first_stage_model.state_dict()

            if save_mode in ("full_model", "both"):
                msg = _save_full_model(current_sd, enabled_blocks, block_strengths,
                                       save_dir, fname, key_to_block_map)
                save_lines.append(msg)

            if save_mode in ("diff_only", "both"):
                msg = _save_diff_only(original_sd, current_sd, enabled_blocks,
                                      block_strengths, save_dir, fname, key_to_block_map)
                save_lines.append(msg)

            cfg_msg = _save_config(enabled_blocks, block_strengths, save_dir, fname)
            save_lines.append(cfg_msg)
            save_msg = "\n".join(save_lines)
            for line in save_lines:
                print(f"[Flux VAE Debiaser] {line}")

        elif save_model and not needs_modification:
            save_msg = "SAVE skipped: no modifications"

        block_stats = _analyze_blocks(original_sd, block_key_map, enabled_blocks, block_strengths)
        info = _format_info(block_stats, enabled_blocks, block_strengths, modified_count, save_msg)
        analysis_json = _create_analysis_json(block_stats)

        print(f"[Flux VAE Debiaser] Done.")
        return {"ui": {"analysis_json": [analysis_json]}, "result": (vae, info)}


# ============================================================================
# REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "FluxVAEDebiaser": FluxVAEDebiaser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxVAEDebiaser": "VAE Deep Debiaser (Flux 2 Klein — 125 Tensors)",
}
