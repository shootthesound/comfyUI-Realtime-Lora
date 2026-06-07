"""
ComfyUI Qwen3-4B Text Encoder Deep Debiaser — Sub-Component Control

For Qwen3-4B text encoder (e.g., used in Flux workflows).

Splits every layer into its functional sub-components:
  layer N: input_norm | attn | attn_norm | mlp | post_norm (5 subs)

182 controllable sub-components:
  - 1 embed_tokens
  - 36 layers × 5 subs = 180
  - 1 final_norm
  = 182 total

Qwen3-4B Architecture:
  - 36 layers
  - Hidden size: 2560
  - MLP intermediate: 9728
  - Attention heads with QK norms

Handles quantized weights (comfy_quant) - only modifies .weight tensors.
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
        print(f"[Qwen3-4B TE Debiaser] Warning: could not write save paths: {e}")


def _get_default_save_dir():
    """Get default save directory, creating it if needed."""
    if HAS_FOLDER_PATHS:
        try:
            te_dirs = folder_paths.get_folder_paths("text_encoders")
            if te_dirs:
                save_dir = os.path.join(te_dirs[0], "debiased")
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
    custom = paths.get("Qwen3_4BTextEncoderDebiaser", "")
    if custom and os.path.isdir(custom):
        return custom
    return _get_default_save_dir()


# ============================================================================
# BLOCK DEFINITIONS
# ============================================================================

NUM_LAYERS = 36  # Qwen3-4B has 36 layers (0-35)

# Special blocks
EMBED_BLOCKS = ["embed_tokens"]
EMBED_LABELS = {"embed_tokens": "Embedding Tokens"}

# Layer sub-components
LAYER_SUBS = ["input_norm", "attn", "attn_norm", "mlp", "post_norm"]
LAYER_BLOCKS = []
LAYER_LABELS = {}
for i in range(NUM_LAYERS):
    for sub in LAYER_SUBS:
        block_id = f"l{i}_{sub}"
        LAYER_BLOCKS.append(block_id)
        LAYER_LABELS[block_id] = f"L{i} {sub}"

# Final norm
FINAL_BLOCKS = ["final_norm"]
FINAL_LABELS = {"final_norm": "Final Norm"}

# Complete ordered list
BLOCKS = EMBED_BLOCKS + LAYER_BLOCKS + FINAL_BLOCKS
BLOCK_LABELS = {**EMBED_LABELS, **LAYER_LABELS, **FINAL_LABELS}


# ============================================================================
# KEY MAPPING
# ============================================================================

def _is_weight_tensor(key):
    """
    Check if this is a weight tensor we should modify.
    Skip quantization metadata (comfy_quant, weight_scale, weight_scale_2).
    """
    if key.endswith('.weight'):
        return True
    if 'layernorm' in key.lower() or '_norm' in key.lower():
        return True
    return False


def _is_quant_metadata(key):
    """Check if this is quantization metadata that should NOT be modified."""
    return ('comfy_quant' in key or 
            'weight_scale' in key or 
            key.endswith('_scale') or
            key.endswith('_scale_2'))


def _classify_sub(rest):
    """Classify a sub-key within a layer block."""
    if rest.startswith("input_layernorm"):
        return "input_norm"
    if rest.startswith("self_attn.q_proj") or rest.startswith("self_attn.k_proj") or \
       rest.startswith("self_attn.v_proj") or rest.startswith("self_attn.o_proj"):
        return "attn"
    if rest.startswith("self_attn.q_norm") or rest.startswith("self_attn.k_norm"):
        return "attn_norm"
    if rest.startswith("mlp."):
        return "mlp"
    if rest.startswith("post_attention_layernorm"):
        return "post_norm"
    return None


def _get_block_for_key(key: str, prefix: str = ""):
    """Map a state_dict key to its sub-component block name."""
    if _is_quant_metadata(key):
        return None
    
    if prefix and key.startswith(prefix):
        key = key[len(prefix):]
    
    work_key = key
    
    for p in ["transformer.", "text_model.", "encoder."]:
        if work_key.startswith(p):
            work_key = work_key[len(p):]
    
    if "embed_tokens" in work_key:
        return "embed_tokens"
    
    if work_key == "model.norm.weight" or (work_key.endswith(".norm.weight") and "layers" not in work_key):
        return "final_norm"
    
    m = re.match(r'(?:model\.)?layers\.(\d+)\.(.+)', work_key)
    if m:
        idx, rest = m.group(1), m.group(2)
        idx = int(idx)
        if idx >= NUM_LAYERS:
            return None
        sub = _classify_sub(rest)
        if sub:
            return f"l{idx}_{sub}"
        return f"l{idx}_attn"
    
    return None


def _build_block_key_map(state_dict, prefix=""):
    """
    Build mapping of block_name -> list of state_dict keys.
    Also returns reverse mapping: key -> block_name (for save functions).
    """
    block_key_map = defaultdict(list)
    key_to_block = {}
    unmatched = []
    quant_metadata = []
    
    for key in state_dict.keys():
        if _is_quant_metadata(key):
            quant_metadata.append(key)
            continue
            
        block = _get_block_for_key(key, prefix)
        if block is not None:
            block_key_map[block].append(key)
            key_to_block[key] = block
        else:
            unmatched.append(key)
    
    return dict(block_key_map), unmatched, key_to_block, quant_metadata


def _detect_key_prefix(state_dict):
    """Detect the key prefix used in the CLIP model's state_dict."""
    for key in state_dict.keys():
        if key.startswith("qwen3_4b."):
            if ".model.layers." in key:
                idx = key.index(".model.layers.")
                return key[:idx + 1]
            if ".model.embed_tokens" in key:
                idx = key.index(".model.embed_tokens")
                return key[:idx + 1]
            if ".transformer.model.layers." in key:
                idx = key.index(".transformer.model.layers.")
                return key[:idx + len(".transformer.") + 1]
        
        if key.startswith("qwen3_8b."):
            if ".model.layers." in key:
                idx = key.index(".model.layers.")
                return key[:idx + 1]
        
        if key.startswith("t5xxl."):
            if ".encoder.block." in key:
                idx = key.index(".encoder.block.")
                return key[:idx + 1]
    
    return ""


def _get_comfyui_wrapper_prefix(state_dict):
    """Get the full ComfyUI wrapper prefix that needs to be stripped for saving."""
    for key in state_dict.keys():
        if "qwen3_4b." in key:
            if ".transformer.model." in key:
                idx = key.index(".transformer.model.")
                return key[:idx + len(".transformer.")], "qwen3_4b_transformer"
            elif ".model." in key:
                idx = key.index(".model.")
                return key[:idx + 1], "qwen3_4b_direct"
        
        if "qwen3_8b." in key:
            if ".transformer.model." in key:
                idx = key.index(".transformer.model.")
                return key[:idx + len(".transformer.")], "qwen3_8b_transformer"
            elif ".model." in key:
                idx = key.index(".model.")
                return key[:idx + 1], "qwen3_8b_direct"
    
    return "", "unknown"


def _strip_wrapper_prefix(key, wrapper_prefix, structure_type):
    """Strip ComfyUI wrapper prefix to get original model key format."""
    if not wrapper_prefix:
        return key
    if key.startswith(wrapper_prefix):
        return key[len(wrapper_prefix):]
    return key


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
            else:
                total_norm += tensor.numel() * 0.01
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
        "Qwen3-4B Text Encoder Deep Debiaser",
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
    result = {"architecture": "QWEN3_4B_TEXT_ENCODER", "blocks": {}}
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

def _apply_modifications(clip_patcher, block_key_map, enabled_blocks, block_strengths, state_dict):
    """Apply modifications via ComfyUI's patching system (LoRA-safe)."""
    cloned = clip_patcher.clone()
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
            
            if weight.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                print(f"[Qwen3-4B TE Debiaser] Skipping quantized tensor: {key} ({weight.dtype})")
                continue
            
            weight_cpu = weight.detach().cpu() if weight.device.type != "cpu" else weight.detach()
            diff = weight_cpu * (strength - 1.0)
            patches[key] = (diff,)
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
        "format": "qwen3_4b_te_debiaser_v1",
        "architecture": "Qwen3-4B Text Encoder",
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
        counts = defaultdict(int)
        for b in modified:
            for sub in ["input_norm", "attn_norm", "attn", "post_norm", "mlp"]:
                if b.endswith("_" + sub):
                    counts[sub] += 1
                    break
            else:
                counts["other"] += 1
        parts = [f"{c}{t}" for t, c in sorted(counts.items(), key=lambda x: -x[1])]
        tag = "_".join(parts[:4])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"qwen3_4b_te_debiased_{tag}_{timestamp}"


def _save_full_model(state_dict, block_key_map, enabled_blocks, block_strengths,
                     save_dir, filename, original_dtype, key_to_block_map, wrapper_prefix, structure_type):
    """Save full model with modifications baked in."""
    if not HAS_SAFETENSORS:
        return "ERROR: safetensors not installed. Run: pip install safetensors"

    print(f"[Qwen3-4B TE Save] Building modified state dict on GPU...")
    print(f"[Qwen3-4B TE Save] Stripping wrapper prefix: '{wrapper_prefix}' (type: {structure_type})")
    
    save_dict = {}
    modified_count = 0
    
    for key, tensor in state_dict.items():
        save_key = _strip_wrapper_prefix(key, wrapper_prefix, structure_type)
        block_id = key_to_block_map.get(key)
        
        is_float = tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]
        
        if block_id is not None and is_float:
            if block_id not in enabled_blocks:
                strength = 0.0
            else:
                strength = block_strengths.get(block_id, 1.0)

            if strength != 1.0:
                t = tensor.detach().float() * strength
                save_dict[save_key] = t.to(original_dtype if is_float else tensor.dtype)
                modified_count += 1
            else:
                save_dict[save_key] = tensor.detach().clone()
        else:
            save_dict[save_key] = tensor.detach().clone()

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename + ".safetensors")

    print(f"[Qwen3-4B TE Save] Moving {len(save_dict)} tensors to CPU for saving...")
    
    for k in save_dict:
        if not save_dict[k].is_contiguous():
            save_dict[k] = save_dict[k].contiguous()
        if save_dict[k].device.type != "cpu":
            save_dict[k] = save_dict[k].cpu()

    print(f"[Qwen3-4B TE Save] Writing to {filepath}...")
    print(f"[Qwen3-4B TE Save] Sample keys being saved: {list(save_dict.keys())[:3]}")
    safetensors_save(save_dict, filepath)

    size_gb = os.path.getsize(filepath) / (1024 ** 3)
    return (f"SAVED full model: {filepath}\n"
            f"  {len(save_dict)} tensors, {modified_count} modified, {size_gb:.2f} GB")


def _save_diff_only(state_dict, block_key_map, enabled_blocks, block_strengths,
                    save_dir, filename, original_dtype, key_to_block_map, wrapper_prefix, structure_type):
    """Save only the delta tensors (much smaller file)."""
    if not HAS_SAFETENSORS:
        return "ERROR: safetensors not installed. Run: pip install safetensors"

    print(f"[Qwen3-4B TE Save] Building diff tensors on GPU...")
    
    diff_dict = {}
    for key, tensor in state_dict.items():
        block_id = key_to_block_map.get(key)
        if block_id is None:
            continue
        
        if tensor.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            continue
            
        if block_id not in enabled_blocks:
            strength = 0.0
        else:
            strength = block_strengths.get(block_id, 1.0)
        
        if strength == 1.0:
            continue
        
        save_key = _strip_wrapper_prefix(key, wrapper_prefix, structure_type)
        diff = (tensor.detach().float() * (strength - 1.0)).to(original_dtype)
        diff_dict[save_key] = diff

    if not diff_dict:
        return "No modifications to save (all at 1.0 or only quantized tensors)"

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename + "_diff.safetensors")

    print(f"[Qwen3-4B TE Save] Moving {len(diff_dict)} diff tensors to CPU...")
    
    for k in diff_dict:
        if not diff_dict[k].is_contiguous():
            diff_dict[k] = diff_dict[k].contiguous()
        if diff_dict[k].device.type != "cpu":
            diff_dict[k] = diff_dict[k].cpu()

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

class Qwen3_4BTextEncoderDebiaser:
    """
    Sub-component debiaser for Qwen3-4B Text Encoder.

    182 individually controllable sub-components:
      layers: input_norm | attn | attn_norm | mlp | post_norm
    
    Handles quantized models - only modifies float weight tensors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "CLIP/Text Encoder model (Qwen3-4B)"
                }),
                "preset": (["Custom", "Default",
                            "Weaken ALL attn 90%", "Weaken ALL attn 85%",
                            "Weaken ALL attn 80%",
                            "Weaken ALL mlp 90%", "Weaken ALL mlp 85%",
                            "Weaken ALL attn+mlp 90%",
                            "Weaken ALL attn+mlp 85%",
                            "Weaken ALL norms 90%",
                            "Global 95%", "Global 90%", "Global 85%"], {
                    "default": "Default",
                }),
                "save_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Save the modified model to disk when enabled",
                }),
                "save_mode": (["full_model", "diff_only", "both"], {
                    "default": "full_model",
                    "tooltip": "full_model = complete safetensors. "
                               "diff_only = only changed weights (small). "
                               "both = save both files.",
                }),
                "filename": ("STRING", {
                    "default": "auto",
                    "tooltip": "Filename (no extension). 'auto' = generate from settings.",
                }),
            },
        }

        # Embedding
        inputs["required"]["embed_tokens"] = ("BOOLEAN", {"default": True})
        inputs["required"]["embed_tokens_str"] = ("FLOAT", {
            "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
        })

        # Main layers — 5 subs each
        for i in range(NUM_LAYERS):
            for sub in LAYER_SUBS:
                block_id = f"l{i}_{sub}"
                inputs["required"][block_id] = ("BOOLEAN", {"default": True})
                inputs["required"][f"{block_id}_str"] = ("FLOAT", {
                    "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
                })

        # Final norm
        inputs["required"]["final_norm"] = ("BOOLEAN", {"default": True})
        inputs["required"]["final_norm_str"] = ("FLOAT", {
            "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
        })

        return inputs

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "info")
    OUTPUT_TOOLTIPS = (
        "CLIP with per-sub-component modifications (LoRA-safe)",
        "Text summary of modifications",
    )
    FUNCTION = "debias"
    CATEGORY = "model_patches"
    OUTPUT_NODE = True
    DESCRIPTION = """Deep sub-component debiaser for Qwen3-4B Text Encoder.

182 individual controls splitting every layer into functional parts:
  layers: input_norm | attn | attn_norm | mlp | post_norm

⚠️ QUANTIZED MODEL: Only float tensors (norms, some weights) can be modified.
Quantized layers (U8, F8_E4M3) are preserved unchanged.

Strength 1.0 = unchanged. Below 1.0 = weakened. Above 1.0 = boosted.
LoRA-safe: patches stack independently with any applied LoRAs.

TIP: Boost embed_tokens (1.5-2.0) for better prompt adherence.

SAVE: Enable save_model to export the modified weights.
  full_model = complete safetensors file
  diff_only  = only modified weights (small delta file)
  both       = saves both files"""

    def debias(self, clip, preset, save_model=False, save_mode="full_model",
               filename="auto", **kwargs):
        print(f"[Qwen3-4B TE Debiaser] Starting...")

        enabled_blocks = set()
        block_strengths = {}
        for block in BLOCKS:
            if kwargs.get(block, True):
                enabled_blocks.add(block)
                block_strengths[block] = kwargs.get(f"{block}_str", 1.0)
            else:
                block_strengths[block] = 0.0

        cond_stage_model = clip.cond_stage_model
        state_dict = cond_stage_model.state_dict()
        
        prefix = _detect_key_prefix(state_dict)
        print(f"[Qwen3-4B TE Debiaser] Detected key prefix: '{prefix}'")

        block_key_map, unmatched, key_to_block_map, quant_metadata = _build_block_key_map(state_dict, prefix)

        matched_blocks = len(block_key_map)
        matched_keys = sum(len(v) for v in block_key_map.values())
        print(f"[Qwen3-4B TE Debiaser] Mapped {matched_keys} tensors across {matched_blocks} sub-components")
        print(f"[Qwen3-4B TE Debiaser] {len(quant_metadata)} quantization metadata tensors (preserved)")
        if unmatched:
            print(f"[Qwen3-4B TE Debiaser] {len(unmatched)} unmatched keys (will be preserved): {unmatched[:3]}...")

        needs_modification = False
        for block in BLOCKS:
            if block not in enabled_blocks or block_strengths.get(block, 1.0) != 1.0:
                needs_modification = True
                break

        modified_count = 0
        if needs_modification:
            clip_out, modified_count = _apply_modifications(
                clip, block_key_map, enabled_blocks, block_strengths, state_dict
            )
            print(f"[Qwen3-4B TE Debiaser] Patched {modified_count} tensors (LoRA-safe)")
        else:
            clip_out = clip.clone()
            print(f"[Qwen3-4B TE Debiaser] No modifications (all at 1.0)")

        save_msg = ""
        if save_model and needs_modification:
            save_dir = _get_save_dir()

            if not filename or filename.strip().lower() == "auto":
                fname = _auto_filename(enabled_blocks, block_strengths)
            else:
                fname = re.sub(r'[<>:"/\\|?*]', '_', filename.strip())
                if not fname:
                    fname = _auto_filename(enabled_blocks, block_strengths)

            original_dtype = torch.bfloat16
            for key, tensor in state_dict.items():
                if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    original_dtype = tensor.dtype
                    break
            
            wrapper_prefix, structure_type = _get_comfyui_wrapper_prefix(state_dict)
            print(f"[Qwen3-4B TE Debiaser] Detected wrapper prefix: '{wrapper_prefix}' ({structure_type})")

            save_lines = []
            print(f"[Qwen3-4B TE Debiaser] Saving to: {save_dir}")

            if save_mode in ("full_model", "both"):
                msg = _save_full_model(state_dict, block_key_map, enabled_blocks,
                                       block_strengths, save_dir, fname, original_dtype,
                                       key_to_block_map, wrapper_prefix, structure_type)
                save_lines.append(msg)
                print(f"[Qwen3-4B TE Debiaser] {msg}")

            if save_mode in ("diff_only", "both"):
                msg = _save_diff_only(state_dict, block_key_map, enabled_blocks,
                                      block_strengths, save_dir, fname, original_dtype,
                                      key_to_block_map, wrapper_prefix, structure_type)
                save_lines.append(msg)
                print(f"[Qwen3-4B TE Debiaser] {msg}")

            cfg_msg = _save_config(enabled_blocks, block_strengths, save_dir, fname)
            save_lines.append(cfg_msg)
            print(f"[Qwen3-4B TE Debiaser] {cfg_msg}")

            save_msg = "\n".join(save_lines)

        elif save_model and not needs_modification:
            save_msg = "SAVE skipped: no modifications to save (all at 1.0)"
            print(f"[Qwen3-4B TE Debiaser] {save_msg}")

        block_stats = _analyze_blocks(state_dict, block_key_map, enabled_blocks, block_strengths)
        info = _format_info(block_stats, enabled_blocks, block_strengths, modified_count, save_msg)
        analysis_json = _create_analysis_json(block_stats)

        print(f"[Qwen3-4B TE Debiaser] Done.")
        return {"ui": {"analysis_json": [analysis_json]}, "result": (clip_out, info)}


# ============================================================================
# REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "Qwen3_4BTextEncoderDebiaser": Qwen3_4BTextEncoderDebiaser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3_4BTextEncoderDebiaser": "Text Encoder Deep Debiaser (Qwen3-4B)",
}
