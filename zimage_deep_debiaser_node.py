"""
ComfyUI DIT Deep Debiaser — Sub-Component Control for Z-Image Turbo

Splits every block into its functional sub-components:

  context_refiner N:  attn | attn_norm | ffn | ffn_norm         (4 subs)
  layer N:            adaLN | attn | attn_norm | ffn | ffn_norm (5 subs)
  noise_refiner N:    adaLN | attn | attn_norm | ffn | ffn_norm (5 subs)

174 controllable sub-components:
  - 5 embedder/token blocks
  - 2 context_refiner × 4 subs = 8
  - 30 layers × 5 subs = 150
  - 2 noise_refiner × 5 subs = 10
  - 1 final_layer
  = 174 total

WARNING: Setting any sub-component to 0.0 will produce static noise.
LoRA-safe via ComfyUI's add_patches system.
"""

import re
import os
import json
import torch
import datetime
from collections import defaultdict
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
    """Load save paths from config JSON."""
    if os.path.exists(_SAVE_PATHS_FILE):
        try:
            with open(_SAVE_PATHS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_save_paths(paths_dict):
    """Write save paths config JSON."""
    try:
        with open(_SAVE_PATHS_FILE, "w") as f:
            json.dump(paths_dict, f, indent=2)
    except Exception as e:
        print(f"[Z-Image Deep Debiaser] Warning: could not write save paths: {e}")

def _get_default_save_dir():
    """Get default save directory, creating it if needed."""
    # Try ComfyUI checkpoints folder first
    if HAS_FOLDER_PATHS:
        try:
            ckpt_dirs = folder_paths.get_folder_paths("checkpoints")
            if ckpt_dirs:
                save_dir = os.path.join(ckpt_dirs[0], "debiased")
                os.makedirs(save_dir, exist_ok=True)
                return save_dir
        except Exception:
            pass

    # Fallback: next to this file
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def _get_save_dir():
    """Get save directory from config or default."""
    paths = _load_save_paths()
    custom = paths.get("ZImageDeepDebiaser", "")
    if custom and os.path.isdir(custom):
        return custom
    return _get_default_save_dir()


# ============================================================================
# BLOCK DEFINITIONS
# ============================================================================

# Embedders & tokens
EMBED_BLOCKS = [
    "cap_embedder",
    "t_embedder",
    "x_embedder",
    "cap_pad_token",
    "x_pad_token",
]

EMBED_LABELS = {
    "cap_embedder": "Caption Embedder",
    "t_embedder": "Timestep Embedder",
    "x_embedder": "Image Embedder",
    "cap_pad_token": "Caption Pad Token",
    "x_pad_token": "Image Pad Token",
}

# Context refiner sub-components (no adaLN in these)
CR_SUBS = ["attn", "attn_norm", "ffn", "ffn_norm"]
CR_BLOCKS = []
CR_LABELS = {}
for i in range(2):
    for sub in CR_SUBS:
        block_id = f"cr{i}_{sub}"
        CR_BLOCKS.append(block_id)
        CR_LABELS[block_id] = f"CR{i} {sub}"

# Main layer sub-components (has adaLN)
LAYER_SUBS = ["adaLN", "attn", "attn_norm", "ffn", "ffn_norm"]
LAYER_BLOCKS = []
LAYER_LABELS = {}
for i in range(30):
    for sub in LAYER_SUBS:
        block_id = f"l{i}_{sub}"
        LAYER_BLOCKS.append(block_id)
        LAYER_LABELS[block_id] = f"L{i} {sub}"

# Noise refiner sub-components (has adaLN)
NR_SUBS = ["adaLN", "attn", "attn_norm", "ffn", "ffn_norm"]
NR_BLOCKS = []
NR_LABELS = {}
for i in range(2):
    for sub in NR_SUBS:
        block_id = f"nr{i}_{sub}"
        NR_BLOCKS.append(block_id)
        NR_LABELS[block_id] = f"NR{i} {sub}"

FINAL_BLOCKS = ["final_layer"]
FINAL_LABELS = {"final_layer": "Final Layer"}

# Complete ordered list
BLOCKS = EMBED_BLOCKS + CR_BLOCKS + LAYER_BLOCKS + NR_BLOCKS + FINAL_BLOCKS
BLOCK_LABELS = {**EMBED_LABELS, **CR_LABELS, **LAYER_LABELS, **NR_LABELS, **FINAL_LABELS}


# ============================================================================
# KEY MAPPING
# ============================================================================

def _classify_sub(rest):
    """Classify a sub-key within a layer/refiner block."""
    if rest.startswith("adaLN_modulation"):
        return "adaLN"
    if rest.startswith("attention."):
        return "attn"
    if rest.startswith("attention_norm"):
        return "attn_norm"
    if rest.startswith("feed_forward."):
        return "ffn"
    if rest.startswith("ffn_norm"):
        return "ffn_norm"
    return None


def _get_block_for_key(key: str):
    """Map a state_dict key to its sub-component block name."""

    # Embedders & tokens
    if key.startswith("cap_embedder."):
        return "cap_embedder"
    if key.startswith("t_embedder."):
        return "t_embedder"
    if key.startswith("x_embedder."):
        return "x_embedder"
    if key == "cap_pad_token":
        return "cap_pad_token"
    if key == "x_pad_token":
        return "x_pad_token"

    # Context refiner: context_refiner.N.sub...
    m = re.match(r'context_refiner\.(\d+)\.(.+)', key)
    if m:
        idx, rest = m.group(1), m.group(2)
        sub = _classify_sub(rest)
        if sub and sub != "adaLN":  # context_refiner has no adaLN
            return f"cr{idx}_{sub}"
        # fallback
        return f"cr{idx}_attn"

    # Main layers: layers.N.sub...
    m = re.match(r'layers\.(\d+)\.(.+)', key)
    if m:
        idx, rest = m.group(1), m.group(2)
        sub = _classify_sub(rest)
        if sub:
            return f"l{idx}_{sub}"
        return f"l{idx}_attn"

    # Noise refiner: noise_refiner.N.sub...
    m = re.match(r'noise_refiner\.(\d+)\.(.+)', key)
    if m:
        idx, rest = m.group(1), m.group(2)
        sub = _classify_sub(rest)
        if sub:
            return f"nr{idx}_{sub}"
        return f"nr{idx}_attn"

    # Final layer
    if key.startswith("final_layer."):
        return "final_layer"

    return None


def _build_block_key_map(state_dict):
    """Build mapping of block_name -> list of state_dict keys."""
    block_key_map = defaultdict(list)
    unmatched = []
    for key in state_dict.keys():
        block = _get_block_for_key(key)
        if block is not None:
            block_key_map[block].append(key)
        else:
            unmatched.append(key)
    return dict(block_key_map), unmatched


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
    lines = [
        "DIT Deep Debiaser — Z-Image Turbo",
        "=" * 60,
        f"Total: {total_params / 1e9:.2f}B params, {total_memory / 1024:.2f} GB",
        "",
        f"{'Sub-Component':<20} {'Tens':>5} {'Strength':>10}",
        "-" * 60,
    ]
    modified_lines = []
    unmodified_count = 0
    for block_id in BLOCKS:
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
        label = BLOCK_LABELS.get(block_id, block_id)
        modified_lines.append(f"{label:<20} {stats['tensor_count']:>5} {strength_str:>10}")
    if modified_lines:
        lines.append("MODIFIED:")
        lines.extend(modified_lines)
    else:
        lines.append("(no modifications)")
    if unmodified_count > 0:
        lines.append(f"... + {unmodified_count} sub-components at 1.00")
    lines.append("-" * 60)
    lines.append(f"Modified: {len(modified_lines)}/{len(BLOCKS)} sub-components ({modified_count} tensors patched)")
    lines.append("LoRA patches: preserved ✓")
    if save_msg:
        lines.append("")
        lines.append(save_msg)
    return "\n".join(lines)


def _create_analysis_json(block_stats):
    result = {"architecture": "ZIMAGE_TURBO_DEEP", "blocks": {}}
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
# PATCHING (LoRA-safe)
# ============================================================================

def _apply_modifications(model_patcher, block_key_map, enabled_blocks, block_strengths):
    model = model_patcher.model
    if hasattr(model, 'diffusion_model'):
        diff_model = model.diffusion_model
        key_prefix = "diffusion_model."
    else:
        diff_model = model
        key_prefix = ""

    state_dict = diff_model.state_dict()
    cloned = model_patcher.clone()
    patches = {}
    modified_count = 0

    for block_id in BLOCKS:
        if block_id not in enabled_blocks:
            strength = 0.0
        else:
            strength = block_strengths.get(block_id, 1.0)
        if strength == 1.0:
            continue
        keys = block_key_map.get(block_id, [])
        for key in keys:
            if key not in state_dict:
                continue
            weight = state_dict[key]
            weight_cpu = weight.detach().cpu() if weight.device.type != "cpu" else weight.detach()
            diff = weight_cpu * (strength - 1.0)
            patch_key = key_prefix + key
            patches[patch_key] = (diff,)
            modified_count += 1

    if patches:
        cloned.add_patches(patches, strength_patch=1.0)
    return cloned, modified_count


# ============================================================================
# SAVE / EXPORT
# ============================================================================

def _build_config_dict(enabled_blocks, block_strengths):
    """Build a JSON-serializable config of all debiaser settings."""
    config = {
        "format": "zimage_deep_debiaser_v1",
        "architecture": "Z-Image Turbo (NextDiT)",
        "timestamp": datetime.datetime.now().isoformat(),
        "total_blocks": len(BLOCKS),
        "modified_blocks": {},
        "unmodified_count": 0,
    }
    for block_id in BLOCKS:
        if block_id not in enabled_blocks:
            config["modified_blocks"][block_id] = {"enabled": False, "strength": 0.0}
        else:
            strength = block_strengths.get(block_id, 1.0)
            if strength != 1.0:
                config["modified_blocks"][block_id] = {"enabled": True, "strength": strength}
            else:
                config["unmodified_count"] += 1
    config["modified_count"] = len(config["modified_blocks"])
    return config


def _auto_filename(enabled_blocks, block_strengths):
    """Generate descriptive filename from current settings."""
    modified = []
    for block_id in BLOCKS:
        if block_id not in enabled_blocks:
            modified.append(block_id)
        else:
            s = block_strengths.get(block_id, 1.0)
            if s != 1.0:
                modified.append(block_id)

    if len(modified) == 0:
        tag = "unmodified"
    elif len(modified) <= 5:
        tag = "_".join(modified)
    else:
        # Summarize: count by type
        counts = defaultdict(int)
        for b in modified:
            for sub in ["adaLN", "attn_norm", "attn", "ffn_norm", "ffn"]:
                if b.endswith("_" + sub):
                    counts[sub] += 1
                    break
            else:
                counts["other"] += 1
        parts = [f"{c}{t}" for t, c in sorted(counts.items(), key=lambda x: -x[1])]
        tag = "_".join(parts[:4])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"zimage_debiased_{tag}_{timestamp}"


def _save_full_model(state_dict, block_key_map, enabled_blocks, block_strengths,
                     save_dir, filename, original_dtype):
    """Save full model with modifications baked in."""
    if not HAS_SAFETENSORS:
        return "ERROR: safetensors not installed. Run: pip install safetensors"

    # Build modified state dict
    save_dict = {}
    modified_count = 0
    for key, tensor in state_dict.items():
        block_id = _get_block_for_key(key)
        if block_id is not None:
            if block_id not in enabled_blocks:
                strength = 0.0
            else:
                strength = block_strengths.get(block_id, 1.0)

            if strength != 1.0:
                t = tensor.detach().float() * strength
                save_dict[key] = t.to(original_dtype)
                modified_count += 1
            else:
                save_dict[key] = tensor.detach().cpu()
        else:
            # Unmatched key — keep as-is
            save_dict[key] = tensor.detach().cpu()

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename + ".safetensors")

    # Ensure all tensors are contiguous and on CPU
    for k in save_dict:
        if not save_dict[k].is_contiguous():
            save_dict[k] = save_dict[k].contiguous()
        if save_dict[k].device.type != "cpu":
            save_dict[k] = save_dict[k].cpu()

    safetensors_save(save_dict, filepath)

    size_gb = os.path.getsize(filepath) / (1024 ** 3)
    return (f"SAVED full model: {filepath}\n"
            f"  {len(save_dict)} tensors, {modified_count} modified, {size_gb:.2f} GB")


def _save_diff_only(state_dict, block_key_map, enabled_blocks, block_strengths,
                    save_dir, filename, original_dtype):
    """Save only the delta tensors (much smaller file)."""
    if not HAS_SAFETENSORS:
        return "ERROR: safetensors not installed. Run: pip install safetensors"

    diff_dict = {}
    for block_id in BLOCKS:
        if block_id not in enabled_blocks:
            strength = 0.0
        else:
            strength = block_strengths.get(block_id, 1.0)
        if strength == 1.0:
            continue
        keys = block_key_map.get(block_id, [])
        for key in keys:
            if key not in state_dict:
                continue
            tensor = state_dict[key]
            diff = (tensor.detach().float() * (strength - 1.0)).to(original_dtype)
            diff_dict[key] = diff.cpu().contiguous()

    if not diff_dict:
        return "No modifications to save (all at 1.0)"

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename + "_diff.safetensors")

    safetensors_save(diff_dict, filepath)

    size_mb = os.path.getsize(filepath) / (1024 ** 2)
    return (f"SAVED diff: {filepath}\n"
            f"  {len(diff_dict)} delta tensors, {size_mb:.1f} MB")


def _save_config(enabled_blocks, block_strengths, save_dir, filename):
    """Save JSON config file with all settings."""
    config = _build_config_dict(enabled_blocks, block_strengths)
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename + "_config.json")
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)
    return f"SAVED config: {filepath}"


# ============================================================================
# NODE CLASS
# ============================================================================

class ZImageDeepDebiaser:
    """
    Sub-component debiaser for Z-Image Turbo.

    174 individually controllable sub-components:
      context_refiner: attn | attn_norm | ffn | ffn_norm
      layers:          adaLN | attn | attn_norm | ffn | ffn_norm
      noise_refiner:   adaLN | attn | attn_norm | ffn | ffn_norm
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Z-Image Turbo model"
                }),
                "preset": (["Custom", "Default",
                            "Weaken ALL attn 90%", "Weaken ALL attn 85%",
                            "Weaken ALL attn 80%",
                            "Weaken ALL ffn 90%", "Weaken ALL ffn 85%",
                            "Weaken ALL adaLN 90%", "Weaken ALL adaLN 85%",
                            "Weaken ALL attn+ffn 90%",
                            "Weaken ALL attn+ffn 85%",
                            "Weaken ALL attn_norm+ffn_norm 90%",
                            "Global 95%", "Global 90%", "Global 85%"], {
                    "default": "Default",
                }),
                "save_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Save the modified model to disk when enabled",
                }),
                "save_mode": (["full_model", "diff_only", "both"], {
                    "default": "full_model",
                    "tooltip": "full_model = complete safetensors (~12GB). "
                               "diff_only = only changed weights (small). "
                               "both = save both files.",
                }),
                "filename": ("STRING", {
                    "default": "auto",
                    "tooltip": "Filename (no extension). 'auto' = generate from settings.",
                }),
            },
        }

        # Embedders & tokens
        for block in EMBED_BLOCKS:
            inputs["required"][block] = ("BOOLEAN", {"default": True})
            inputs["required"][f"{block}_str"] = ("FLOAT", {
                "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
            })

        # Context refiners — 4 subs each
        for i in range(2):
            for sub in CR_SUBS:
                block_id = f"cr{i}_{sub}"
                inputs["required"][block_id] = ("BOOLEAN", {"default": True})
                inputs["required"][f"{block_id}_str"] = ("FLOAT", {
                    "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
                })

        # Main layers — 5 subs each
        for i in range(30):
            for sub in LAYER_SUBS:
                block_id = f"l{i}_{sub}"
                inputs["required"][block_id] = ("BOOLEAN", {"default": True})
                inputs["required"][f"{block_id}_str"] = ("FLOAT", {
                    "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
                })

        # Noise refiners — 5 subs each
        for i in range(2):
            for sub in NR_SUBS:
                block_id = f"nr{i}_{sub}"
                inputs["required"][block_id] = ("BOOLEAN", {"default": True})
                inputs["required"][f"{block_id}_str"] = ("FLOAT", {
                    "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
                })

        # Final layer
        inputs["required"]["final_layer"] = ("BOOLEAN", {"default": True})
        inputs["required"]["final_layer_str"] = ("FLOAT", {
            "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
        })

        return inputs

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    OUTPUT_TOOLTIPS = (
        "Model with per-sub-component modifications (LoRA-safe)",
        "Text summary of modifications",
    )
    FUNCTION = "debias"
    CATEGORY = "model_patches"
    OUTPUT_NODE = True
    DESCRIPTION = """Deep sub-component debiaser for Z-Image Turbo.

174 individual controls splitting every block into functional parts:
  context_refiner: attn | attn_norm | ffn | ffn_norm
  layers:          adaLN | attn | attn_norm | ffn | ffn_norm
  noise_refiner:   adaLN | attn | attn_norm | ffn | ffn_norm

adaLN = timestep/conditioning modulation
attn = self-attention (qkv, output, norms)
attn_norm = attention layer normalization
ffn = feed-forward network (SwiGLU w1/w2/w3)
ffn_norm = FFN layer normalization

Strength 1.0 = unchanged. Below 1.0 = weakened. Never use 0.0.
LoRA-safe: patches stack independently with any applied LoRAs.

SAVE: Enable save_model to export the modified weights.
  full_model = complete safetensors file (loadable as checkpoint)
  diff_only  = only modified weights (small delta file)
  both       = saves both files
Config JSON is always saved alongside with all your settings."""

    def debias(self, model, preset, save_model=False, save_mode="full_model",
               filename="auto", **kwargs):
        print(f"[Z-Image Deep Debiaser] Starting...")

        enabled_blocks = set()
        block_strengths = {}
        for block in BLOCKS:
            if kwargs.get(block, True):
                enabled_blocks.add(block)
                block_strengths[block] = kwargs.get(f"{block}_str", 1.0)
            else:
                block_strengths[block] = 0.0

        inner_model = model.model
        if hasattr(inner_model, 'diffusion_model'):
            diff_model = inner_model.diffusion_model
        else:
            diff_model = inner_model

        state_dict = diff_model.state_dict()
        block_key_map, unmatched = _build_block_key_map(state_dict)

        matched_blocks = len(block_key_map)
        matched_keys = sum(len(v) for v in block_key_map.values())
        print(f"[Z-Image Deep Debiaser] Mapped {matched_keys} tensors across {matched_blocks} sub-components")
        if unmatched:
            print(f"[Z-Image Deep Debiaser] {len(unmatched)} unmatched: {unmatched[:5]}")

        needs_modification = False
        for block in BLOCKS:
            if block not in enabled_blocks or block_strengths.get(block, 1.0) != 1.0:
                needs_modification = True
                break

        modified_count = 0
        if needs_modification:
            model_out, modified_count = _apply_modifications(
                model, block_key_map, enabled_blocks, block_strengths
            )
            print(f"[Z-Image Deep Debiaser] Patched {modified_count} tensors (LoRA-safe)")
        else:
            model_out = model.clone()
            print(f"[Z-Image Deep Debiaser] No modifications (all at 1.0)")

        # ---- SAVE ----
        save_msg = ""
        if save_model and needs_modification:
            save_dir = _get_save_dir()

            # Resolve filename
            if not filename or filename.strip().lower() == "auto":
                fname = _auto_filename(enabled_blocks, block_strengths)
            else:
                # Sanitize
                fname = re.sub(r'[<>:"/\\|?*]', '_', filename.strip())
                if not fname:
                    fname = _auto_filename(enabled_blocks, block_strengths)

            # Detect original dtype for saving
            sample_key = next(iter(state_dict))
            original_dtype = state_dict[sample_key].dtype

            save_lines = []
            print(f"[Z-Image Deep Debiaser] Saving to: {save_dir}")

            if save_mode in ("full_model", "both"):
                msg = _save_full_model(state_dict, block_key_map, enabled_blocks,
                                       block_strengths, save_dir, fname, original_dtype)
                save_lines.append(msg)
                print(f"[Z-Image Deep Debiaser] {msg}")

            if save_mode in ("diff_only", "both"):
                msg = _save_diff_only(state_dict, block_key_map, enabled_blocks,
                                      block_strengths, save_dir, fname, original_dtype)
                save_lines.append(msg)
                print(f"[Z-Image Deep Debiaser] {msg}")

            # Always save config JSON
            cfg_msg = _save_config(enabled_blocks, block_strengths, save_dir, fname)
            save_lines.append(cfg_msg)
            print(f"[Z-Image Deep Debiaser] {cfg_msg}")

            save_msg = "\n".join(save_lines)

        elif save_model and not needs_modification:
            save_msg = "SAVE skipped: no modifications to save (all at 1.0)"
            print(f"[Z-Image Deep Debiaser] {save_msg}")

        block_stats = _analyze_blocks(state_dict, block_key_map, enabled_blocks, block_strengths)
        info = _format_info(block_stats, enabled_blocks, block_strengths, modified_count, save_msg)
        analysis_json = _create_analysis_json(block_stats)

        print(f"[Z-Image Deep Debiaser] Done.")
        return {"ui": {"analysis_json": [analysis_json]}, "result": (model_out, info)}


# ============================================================================
# REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ZImageDeepDebiaser": ZImageDeepDebiaser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageDeepDebiaser": "DIT Deep Debiaser (Z-Image Sub-Component)",
}
