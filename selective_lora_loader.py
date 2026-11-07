"""
Selective LoRA Loaders for ComfyUI
User-friendly loaders with architecture-specific controls.
"""

import os
import re
from typing import Dict, List, Optional

import torch
import folder_paths
import comfy.sd
from safetensors.torch import load_file

from .moss_lora import MOSS_BLOCKS, is_moss_lora, parse_moss_profile


def _detect_architecture(keys):
    """Identify LoRA architecture from key patterns."""
    keys_lower = [k.lower() for k in keys]
    keys_str = ' '.join(keys_lower)
    num_keys = len(keys)

    if is_moss_lora(keys):
        return 'MOSS_IMAGE'

    if any(
        re.search(r'(?:diffusion_model|transformer)\.blocks[._]\d+\..*(?:qkv_proj|out_proj|mlp\.fc[12])', k)
        for k in keys_lower
    ) or any(
        re.search(r'lora_unet_blocks_\d+_(?:attn_qkv_proj|attn_out_proj|mlp_fc[12])', k)
        for k in keys_lower
    ):
        return 'MINIMAX_H3'
    if any('transformer_blocks' in k and any(x in k for x in ['img_mlp', 'txt_mlp', 'img_mod', 'txt_mod']) for k in keys_lower):
        return 'QWEN_IMAGE'
    if 'lora_krea2' in keys_str or 'krea2' in keys_str or 'krea_2' in keys_str:
        return 'KREA2'
    if any(x in keys_str for x in ['txtfusion', 'txtmlp', 'tmlp', 'tproj']) and any(re.search(r'blocks[._]\d+', k) for k in keys_lower):
        return 'KREA2'
    if any(re.search(r'blocks[._]\d+[._].*(?:attn[._])?(?:wq|wk|wv|wo|gate)', k) for k in keys_lower):
        return 'KREA2'
    if any('diffusion_model.layers.' in k and ('attention' in k or 'adaln' in k.lower()) for k in keys_lower):
        return 'ZIMAGE'
    # Musubi Tuner Z-Image format (lora_unet_layers_N_attention_...)
    if any('lora_unet_layers_' in k and 'attention' in k for k in keys_lower):
        return 'ZIMAGE'
    # FLUX detection - check AI-Toolkit format BEFORE Z-Image single_transformer_blocks check
    # AI-Toolkit uses: transformer.transformer_blocks.N (double) and transformer.single_transformer_blocks.N (single)
    if any('transformer.single_transformer_blocks' in k or 'transformer.transformer_blocks' in k for k in keys_lower):
        return 'FLUX'
    # Kohya/other format: lora_transformer_single_transformer_blocks / lora_transformer_double_blocks (underscores)
    if any('transformer_single_transformer_blocks' in k or 'transformer_double_blocks' in k for k in keys_lower):
        return 'FLUX'
    if any('double_blocks' in k or 'single_blocks' in k for k in keys_lower):
        return 'FLUX'
    # Z-Image old format (single_transformer_blocks WITHOUT transformer. prefix)
    if any('single_transformer_blocks' in k and 'transformer.single_transformer_blocks' not in k for k in keys_lower):
        return 'ZIMAGE'
    if any(('blocks.' in k or 'blocks_' in k) and any(x in k for x in ['self_attn', 'cross_attn', 'ffn'])
           for k in keys_lower):
        return 'WAN'
    has_te1 = 'lora_te1_' in keys_str or 'text_encoder_1' in keys_str
    has_te2 = 'lora_te2_' in keys_str or 'text_encoder_2' in keys_str
    if has_te1 and has_te2:
        return 'SDXL'
    if num_keys > 1500:
        return 'SDXL'
    if any('input_blocks_7' in k or 'input_blocks_8' in k or
           'input_blocks.7' in k or 'input_blocks.8' in k for k in keys_lower):
        return 'SDXL'
    if any('lora_unet_' in k or 'lora_te_' in k for k in keys_lower):
        return 'SD15'
    if num_keys > 1000:
        return 'SDXL'
    if any('input_blocks' in k for k in keys_lower):
        return 'SD15'
    return 'UNKNOWN'


def _get_architecture_blocks(architecture: str) -> List[str]:
    """Get the ordered list of block names for string chaining."""
    if architecture == 'MOSS_IMAGE':
        return MOSS_BLOCKS[:-1]
    if architecture == 'SDXL':
        return [
            'text_encoder_1', 'text_encoder_2', 'input_4', 'input_5', 'input_7',
            'input_8', 'unet_mid', 'output_0', 'output_1', 'output_2', 'output_3',
            'output_4', 'output_5',
        ]
    if architecture == 'ZIMAGE':
        return [f'layer_{i}' for i in range(30)]
    if architecture == 'FLUX':
        return [f'double_{i}' for i in range(19)] + [f'single_{i}' for i in range(38)]
    if architecture == 'WAN':
        return [f'block_{i}' for i in range(40)]
    if architecture == 'QWEN':
        return [f'block_{i}' for i in range(60)]
    if architecture == 'KREA2':
        return [f'block_{i}' for i in range(28)]
    if architecture == 'MINIMAX_H3':
        return [f'block_{i}' for i in range(50)]
    if architecture == 'FLUX_KLEIN':
        return [f'double_{i}' for i in range(8)] + [f'single_{i}' for i in range(24)]
    return []


def _parse_block_weights_string(weights_str: str, architecture: str) -> Optional[Dict[str, tuple[bool, float]]]:
    """
    Parse block weights from either:
    - positional format: "1.0, 0.0, 0.5, ..."
    - named format: "%default=1.0, te1=0.5, in7-8=1.2"
    """
    if architecture == 'MOSS_IMAGE':
        return parse_moss_profile(weights_str)
    if not weights_str or not weights_str.strip():
        return None

    weights_str = weights_str.strip()

    if not weights_str.startswith('%'):
        try:
            values = [float(v.strip()) for v in weights_str.split(',') if v.strip()]
        except (TypeError, ValueError):
            return None

        block_names = _get_architecture_blocks(architecture)
        if not values or len(values) not in (len(block_names), len(block_names) + 1):
            return None

        block_values = values[:len(block_names)]
        parsed = {
            block_name: (value != 0.0, value)
            for block_name, value in zip(block_names, block_values)
        }
        other_value = values[-1] if len(values) == len(block_names) + 1 else 1.0
        parsed["other_weights"] = (other_value != 0.0, other_value)
        return parsed

    pairs = [part.strip() for part in weights_str[1:].split(',') if part.strip()]
    if not pairs:
        return None

    block_names = _get_architecture_blocks(architecture)
    parsed: Dict[str, Optional[tuple[bool, float]]] = {block_name: None for block_name in block_names}
    parsed["other_weights"] = None
    default_val = 1.0

    def set_named_targets(prefix: str, value: float):
        for block_name in block_names:
            if block_name.startswith(prefix):
                parsed[block_name] = (value != 0.0, value)

    def set_numeric_range(prefix: str, spec: str, value: float):
        for chunk in spec.split(','):
            chunk = chunk.strip()
            if not chunk:
                continue
            if '-' in chunk:
                start_s, end_s = chunk.split('-', 1)
                start_i = int(start_s)
                end_i = int(end_s)
                for idx in range(start_i, end_i + 1):
                    parsed[f"{prefix}_{idx}"] = (value != 0.0, value)
            else:
                parsed[f"{prefix}_{int(chunk)}"] = (value != 0.0, value)

    alias_map = {
        "te1": ["text_encoder_1"],
        "te2": ["text_encoder_2"],
        "mid": ["unet_mid"],
        "other": ["other_weights"],
    }

    for pair in pairs:
        if '=' not in pair:
            return None
        key, value_str = [part.strip() for part in pair.split('=', 1)]
        try:
            value = float(value_str)
        except ValueError:
            return None

        key_lower = key.lower()
        if key_lower == "default":
            default_val = value
            continue

        if key_lower in alias_map:
            for block_name in alias_map[key_lower]:
                parsed[block_name] = (value != 0.0, value)
            continue

        if architecture == 'SDXL':
            if key_lower == "in":
                for name in ["input_4", "input_5", "input_7", "input_8"]:
                    parsed[name] = (value != 0.0, value)
                continue
            if key_lower.startswith("in"):
                set_numeric_range("input", key_lower[2:], value)
                continue
            if key_lower == "out":
                for name in [f"output_{i}" for i in range(6)]:
                    parsed[name] = (value != 0.0, value)
                continue
            if key_lower.startswith("out"):
                set_numeric_range("output", key_lower[3:], value)
                continue
        elif architecture == 'FLUX':
            if key_lower == "double":
                set_named_targets("double_", value)
                continue
            if key_lower.startswith("double"):
                set_numeric_range("double", key_lower[6:], value)
                continue
            if key_lower == "single":
                set_named_targets("single_", value)
                continue
            if key_lower.startswith("single"):
                set_numeric_range("single", key_lower[6:], value)
                continue
        elif architecture in ('ZIMAGE', 'WAN', 'QWEN', 'KREA2', 'MINIMAX_H3'):
            prefix = "layer" if architecture == 'ZIMAGE' else "block"
            if key_lower == prefix:
                set_named_targets(f"{prefix}_", value)
                continue
            if key_lower.startswith(prefix):
                set_numeric_range(prefix, key_lower[len(prefix):], value)
                continue

        if key_lower in parsed:
            parsed[key_lower] = (value != 0.0, value)
            continue

        return None

    final_parsed: Dict[str, tuple[bool, float]] = {}
    for block_name, current in parsed.items():
        final_parsed[block_name] = current if current is not None else (default_val != 0.0, default_val)

    return final_parsed


def _extract_block_id_sdxl(key: str) -> str:
    """Extract block ID for SDXL/SD15 architecture."""
    key_lower = key.lower()

    te = re.search(r'lora_te(\d?)_', key_lower)
    if te:
        return f"text_encoder_{te.group(1) or '1'}"

    down = re.search(r'down_blocks?[._]?(\d+)', key_lower)
    if down:
        return f"unet_down_{down.group(1)}"
    if 'mid_block' in key_lower or 'middle_block' in key_lower:
        return "unet_mid"
    up = re.search(r'up_blocks?[._]?(\d+)', key_lower)
    if up:
        return f"unet_up_{up.group(1)}"

    inp = re.search(r'input_blocks?[._]?(\d+)', key_lower)
    if inp:
        return f"input_{inp.group(1)}"
    out = re.search(r'output_blocks?[._]?(\d+)', key_lower)
    if out:
        return f"output_{out.group(1)}"

    return 'other'


def _extract_layer_num_zimage(key: str) -> Optional[int]:
    """Extract layer number for Z-Image architecture."""
    match = re.search(r'diffusion_model\.layers\.(\d+)', key)
    if match:
        return int(match.group(1))
    match = re.search(r'single_transformer_blocks\.(\d+)', key)
    if match:
        return int(match.group(1))
    return None


def _extract_block_id_flux(key: str) -> str:
    """Extract block ID for FLUX architecture."""
    key_lower = key.lower()

    # FLUX has double blocks (19) and single blocks (38)
    # Different trainers use different naming:
    #   - Standard: double_blocks.N, single_blocks.N
    #   - AI-Toolkit: transformer.transformer_blocks.N (double), transformer.single_transformer_blocks.N (single)
    #   - Kohya/other: lora_transformer_single_transformer_blocks_N, lora_transformer_double_blocks_N

    # Check single blocks FIRST (because "single_transformer_blocks" contains "transformer_blocks")
    # Handles: single_transformer_blocks.N, single_transformer_blocks_N, transformer_single_transformer_blocks_N
    single = re.search(r'single_transformer_blocks[._]?(\d+)', key_lower)
    if single:
        return f"single_{single.group(1)}"
    single = re.search(r'single_blocks[._]?(\d+)', key_lower)
    if single:
        return f"single_{single.group(1)}"

    # Double blocks - standard format
    double = re.search(r'(?:transformer\.)?double_blocks?[._]?(\d+)', key_lower)
    if double:
        return f"double_{double.group(1)}"
    # AI-Toolkit format: transformer.transformer_blocks.N (these are double blocks)
    double = re.search(r'transformer\.transformer_blocks[._]?(\d+)', key_lower)
    if double:
        return f"double_{double.group(1)}"
    # Kohya/other format: transformer_double_blocks_N (underscores, these are double blocks)
    double = re.search(r'transformer_double_blocks[._]?(\d+)', key_lower)
    if double:
        return f"double_{double.group(1)}"

    return 'other'


def _extract_block_id_wan(key: str) -> Optional[int]:
    """Extract block number for Wan architecture."""
    # Handle both blocks.N and blocks_N patterns
    match = re.search(r'blocks[._](\d+)', key)
    if match:
        return int(match.group(1))
    return None


def _extract_block_id_qwen(key: str) -> Optional[int]:
    """Extract block number for Qwen-Image architecture."""
    match = re.search(r'transformer_blocks[._](\d+)', key)
    if match:
        return int(match.group(1))
    return None


def _extract_block_id_krea2(key: str) -> Optional[int]:
    """Extract main SingleStreamBlock number for Krea 2 architecture."""
    key_lower = key.lower()
    if any(part in key_lower for part in ['txtfusion', 'txtmlp', 'tmlp', 'tproj', 'first', 'last']):
        return None
    match = re.search(r'blocks[._](\d+)', key_lower)
    if match:
        return int(match.group(1))
    return None


def _extract_block_id_minimax_h3(key: str) -> Optional[int]:
    """Extract a top-level MiniMax H3 DiT block, excluding token-refiner blocks."""
    key_lower = key.lower()
    match = re.search(r'(?:^|_)lora_unet_blocks_(\d+)_', key_lower)
    if match:
        return int(match.group(1))
    match = re.search(r'(?:^|\.)(?:diffusion_model|transformer)\.blocks[._](\d+)', key_lower)
    if match:
        return int(match.group(1))
    return None


def _coerce_scalar_strength(value) -> float:
    """Normalize a ComfyUI strength input before it reaches the LoRA patcher."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Overall LoRA strength must be a single number")
        value = value.item()
    elif isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Overall LoRA strength must be a single number")
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Overall LoRA strength must be a single number") from error


def _scale_minimax_h3_tensor(key: str, value, strength: float):
    """Scale one LoRA factor so a block strength stays linear and signed."""
    strength = _coerce_scalar_strength(strength)
    if strength == 1.0:
        return value

    key_lower = key.lower()
    if key_lower.endswith((".alpha", ".dora_scale", ".reshape_weight")):
        return value

    # ComfyUI multiplies the two LoRA factors together at load time. Scaling
    # only the output/up factor avoids turning a 0.5 weight into 0.25 and
    # preserves negative block strengths.
    is_output_factor = (
        re.search(r"(?:^|[._])lora_(?:up|b)(?:\.default)?(?:\.weight)?$", key_lower)
        or re.search(r"(?:^|[._])lora\.up(?:\.weight)?$", key_lower)
        or key_lower.endswith(".lora_linear_layer.up.weight")
        or "lokr_w1" in key_lower
        or "hada_w1" in key_lower
    )
    return value * strength if is_output_factor else value


def _save_minimax_h3_filtered_lora(
    filtered_lora: dict,
    source_path: str,
    save_path: str,
    save_filename: str,
) -> Optional[str]:
    """Save the exact tensor dictionary applied by the MiniMax H3 loader."""
    if not save_path or not save_path.strip():
        return None

    output_dir = os.path.expanduser(save_path.strip())
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as error:
        print(f"[MiniMax H3 Selective Loader] Could not create save directory: {error}")
        return None

    base_name = save_filename.strip() if save_filename else "minimax_h3_selective"
    if base_name.lower().endswith(".safetensors"):
        base_name = base_name[:-12]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{base_name}_{timestamp}.safetensors")

    metadata = {}
    if source_path.lower().endswith(".safetensors"):
        try:
            with safe_open(source_path, framework="pt", device="cpu") as source:
                metadata = dict(source.metadata() or {})
        except Exception as error:
            print(f"[MiniMax H3 Selective Loader] Could not read source metadata: {error}")

    metadata.update({
        "refined_by": "Selective LoRA Loader (MiniMax H3)",
        "refined_date": datetime.now().isoformat(),
        "refined_source": os.path.basename(source_path),
    })

    try:
        save_file(filtered_lora, output_path, metadata=metadata)
        print(f"[MiniMax H3 Selective Loader] Saved filtered LoRA: {output_path}")
        return output_path
    except Exception as error:
        print(f"[MiniMax H3 Selective Loader] Could not save filtered LoRA: {error}")
        return None


# SDXL block presets - only blocks with attention layers that LoRA trains
# SDXL has: text_encoder_1, text_encoder_2, input_4/5/7/8, unet_mid, output_0-5
# Other input/output blocks are ResNet-only (no attention) and not trained by standard LoRA
SDXL_VALID_BLOCKS = {"text_encoder_1", "text_encoder_2", "input_4", "input_5", "input_7", "input_8", "unet_mid", "output_0", "output_1", "output_2", "output_3", "output_4", "output_5"}

SDXL_PRESETS = {
    "All Blocks": SDXL_VALID_BLOCKS.copy(),
    "All Off": set(),  # All blocks disabled including other_weights
    "UNet Only": {"input_4", "input_5", "input_7", "input_8", "unet_mid", "output_0", "output_1", "output_2", "output_3", "output_4", "output_5"},
    "High Impact": {"input_7", "input_8", "unet_mid", "output_0", "output_1", "output_2"},
    "Text Encoders Only": {"text_encoder_1", "text_encoder_2"},
    "Decoders Only": {"output_0", "output_1", "output_2", "output_3", "output_4", "output_5"},
    "Encoders Only": {"input_4", "input_5", "input_7", "input_8"},
    "Style Focus": {"output_1", "output_2"},  # output_1 is strongest for style/color
    "Composition Focus": {"input_8", "unet_mid", "output_0"},  # composition and structure
    "Face Focus": {"input_7", "input_8", "unet_mid", "output_0", "output_1", "output_2", "output_3"},  # OUT3 best for faces, upper blocks for identity
    "Custom": None,  # Use individual toggles
}

# Z-Image layer presets
ZIMAGE_PRESETS = {
    "All Layers": set(range(30)),
    "All Off": set(),  # All layers disabled including other_weights
    "Late Only (20-29)": set(range(20, 30)),
    "Mid-Late (15-29)": set(range(15, 30)),
    "Skip Early (10-29)": set(range(10, 30)),
    "Mid Only (10-19)": set(range(10, 20)),
    "Early Only (0-9)": set(range(10)),
    "Peak Impact (18-25)": set(range(18, 26)),
    "Face Priority (16-24)": set(range(16, 25)),
    "Face Priority Aggressive (14-25)": set(range(14, 26)),
    "Evens Only": set(range(0, 30, 2)),  # 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28
    "Odds Only": set(range(1, 30, 2)),   # 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29
    "Custom": None,  # Use individual toggles
}

# FLUX block presets - 19 double blocks (0-18) + 38 single blocks (0-37) = 57 total
FLUX_ALL_BLOCKS = (
    {f"double_{i}" for i in range(19)} |
    {f"single_{i}" for i in range(38)}
)

# Facial layers from lora-the-explorer (github.com/shootthesound/lora-the-explorer)
# Double blocks: 7, 12, 16 | Single blocks: 7, 12, 16, 20
FLUX_FACE_DOUBLE = {"double_7", "double_12", "double_16"}
FLUX_FACE_SINGLE = {"single_7", "single_12", "single_16", "single_20"}
FLUX_FACE_BLOCKS = FLUX_FACE_DOUBLE | FLUX_FACE_SINGLE

# Aggressive facial (for overtrained LoRAs) - excludes out-of-range double_19
FLUX_FACE_AGGRESSIVE_DOUBLE = {"double_4", "double_7", "double_8", "double_12", "double_15", "double_16"}
FLUX_FACE_AGGRESSIVE_SINGLE = {"single_4", "single_7", "single_8", "single_12", "single_15", "single_16", "single_19", "single_20"}
FLUX_FACE_AGGRESSIVE = FLUX_FACE_AGGRESSIVE_DOUBLE | FLUX_FACE_AGGRESSIVE_SINGLE

# Style = all blocks except facial
FLUX_STYLE_BLOCKS = FLUX_ALL_BLOCKS - FLUX_FACE_BLOCKS

FLUX_PRESETS = {
    "All Blocks": FLUX_ALL_BLOCKS.copy(),
    "All Off": set(),  # All blocks disabled including other_weights
    "Double Blocks Only": {f"double_{i}" for i in range(19)},
    "Single Blocks Only": {f"single_{i}" for i in range(38)},
    "High Impact Double": {f"double_{i}" for i in range(6, 19)},  # double_6-18 tend to be highest
    "Core Double": {f"double_{i}" for i in range(8, 18)},  # Peak impact range
    "Face Focus": FLUX_FACE_BLOCKS.copy(),  # double 7,12,16 + single 7,12,16,20
    "Face Aggressive": FLUX_FACE_AGGRESSIVE.copy(),  # Extended for overtrained LoRAs
    "Style Only (No Face)": FLUX_STYLE_BLOCKS.copy(),  # All except facial layers
    "Evens Only": {f"double_{i}" for i in range(0, 19, 2)} | {f"single_{i}" for i in range(0, 38, 2)},
    "Odds Only": {f"double_{i}" for i in range(1, 19, 2)} | {f"single_{i}" for i in range(1, 38, 2)},
    "Custom": None,  # Use individual toggles
}

# Wan 2.2 block presets - 40 transformer blocks
WAN_PRESETS = {
    "All Blocks": set(range(40)),
    "All Off": set(),  # All blocks disabled including other_weights
    "Late Only (30-39)": set(range(30, 40)),
    "Mid-Late (20-39)": set(range(20, 40)),
    "Skip Early (10-39)": set(range(10, 40)),
    "Mid Only (15-25)": set(range(15, 26)),
    "Early Only (0-19)": set(range(20)),
    "Evens Only": set(range(0, 40, 2)),
    "Odds Only": set(range(1, 40, 2)),
    "Custom": None,  # Use individual toggles
}

# Qwen-Image block presets - 60 transformer blocks
QWEN_PRESETS = {
    "All Blocks": set(range(60)),
    "All Off": set(),  # All blocks disabled including other_weights
    "Late Only (45-59)": set(range(45, 60)),
    "Mid-Late (30-59)": set(range(30, 60)),
    "Skip Early (15-59)": set(range(15, 60)),
    "Mid Only (20-40)": set(range(20, 41)),
    "Early Only (0-29)": set(range(30)),
    "Evens Only": set(range(0, 60, 2)),
    "Odds Only": set(range(1, 60, 2)),
    "Custom": None,  # Use individual toggles
}

# Krea 2 block presets - 28 main SingleStreamBlocks
KREA2_PRESETS = {
    "All Blocks": set(range(28)),
    "All Off": set(),  # All blocks disabled including other_weights
    "Late Only (21-27)": set(range(21, 28)),
    "Mid-Late (14-27)": set(range(14, 28)),
    "Skip Early (7-27)": set(range(7, 28)),
    "Mid Only (9-18)": set(range(9, 19)),
    "Early Only (0-8)": set(range(9)),
    "Evens Only": set(range(0, 28, 2)),
    "Odds Only": set(range(1, 28, 2)),
    "Custom": None,  # Use individual toggles
}

# MiniMax H3 block presets - 50 top-level packed DiT blocks
MINIMAX_H3_PRESETS = {
    "All Blocks": set(range(50)),
    "All Off": set(),
    "Late Only (38-49)": set(range(38, 50)),
    "Mid-Late (25-49)": set(range(25, 50)),
    "Skip Early (13-49)": set(range(13, 50)),
    "Mid Only (17-32)": set(range(17, 33)),
    "Early Only (0-16)": set(range(17)),
    "Evens Only": set(range(0, 50, 2)),
    "Odds Only": set(range(1, 50, 2)),
    "Custom": None,
}


class SDXLSelectiveLoRALoader:
    """
    Selective LoRA Loader for SDXL models.

    Toggle individual blocks on/off to control which parts of the LoRA are applied.
    Use the LoRA Analyzer first to see which blocks have the most impact.

    Block Guide (13 blocks with attention layers):
    - text_encoder_1/2: CLIP text encoders (CLIP-L and CLIP-G)
    - input_4, input_5: Mid encoder blocks with attention
    - input_7, input_8: Deep encoder blocks (high impact, composition)
    - unet_mid: Bottleneck (moderate-high impact)
    - output_0: Primary decoder (composition, high impact)
    - output_1: Style block (strongest for style/color)
    - output_2-5: Decoder blocks (decreasing impact)

    Note: Other input/output blocks (0-3, 6, 9-11) are ResNet-only without
    attention layers and are not trained by standard LoRA.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "SDXL LoRA file to load"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Overall LoRA strength"
                }),
                "preset": (list(SDXL_PRESETS.keys()), {
                    "default": "All Blocks",
                    "tooltip": "Quick preset selection. Choose 'Custom' to use individual toggles below."
                }),
                # Text encoders
                "text_encoder_1": ("BOOLEAN", {"default": True}),
                "text_encoder_1_str": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "text_encoder_2": ("BOOLEAN", {"default": True}),
                "text_encoder_2_str": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                # Input blocks with attention (only 4, 5, 7, 8 have attention in SDXL)
                "input_4": ("BOOLEAN", {"default": True}),
                "input_4_str": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "input_5": ("BOOLEAN", {"default": True}),
                "input_5_str": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "input_7": ("BOOLEAN", {"default": True}),
                "input_7_str": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "input_8": ("BOOLEAN", {"default": True}),
                "input_8_str": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                # Mid block
                "unet_mid": ("BOOLEAN", {"default": True}),
                "unet_mid_str": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                # Output blocks with attention (only 0-5 have attention in SDXL)
                "output_0": ("BOOLEAN", {"default": True}),
                "output_0_str": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "output_1": ("BOOLEAN", {"default": True}),
                "output_1_str": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "output_2": ("BOOLEAN", {"default": True}),
                "output_2_str": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "output_3": ("BOOLEAN", {"default": True}),
                "output_3_str": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "output_4": ("BOOLEAN", {"default": True}),
                "output_4_str": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "output_5": ("BOOLEAN", {"default": True}),
                "output_5_str": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                # Other weights (keys not matching known blocks)
                "other_weights": ("BOOLEAN", {"default": True}),
                "other_weights_str": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
            },
            "optional": {
                "lora_path_opt": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer to use its selected LoRA"}),
                "analysis_json": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer for impact-colored checkboxes"}),
                "block_weights_string": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Input/Output block profile string. Positional text syncs with the UI. String input overrides UI values."
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "info", "weights_output")
    OUTPUT_NODE = True
    FUNCTION = "load_lora"
    CATEGORY = "loaders/lora"
    DESCRIPTION = """Selective LoRA loader for SDXL. Toggle blocks on/off.

TIP: Use 'LoRA Loader + Analyzer' first to see which blocks matter for your LoRA.
Then disable low-scoring blocks here to reduce unwanted effects.

SDXL has 13 blocks with attention layers that LoRA trains.
output_1 is strongest for style/color, input_8/output_0 for composition."""

    def load_lora(self, model, clip, lora_name, strength, preset,
                  text_encoder_1, text_encoder_1_str, text_encoder_2, text_encoder_2_str,
                  input_4, input_4_str, input_5, input_5_str, input_7, input_7_str, input_8, input_8_str,
                  unet_mid, unet_mid_str,
                  output_0, output_0_str, output_1, output_1_str, output_2, output_2_str,
                  output_3, output_3_str, output_4, output_4_str, output_5, output_5_str,
                  other_weights, other_weights_str,
                  lora_path_opt=None, analysis_json=None, block_weights_string=""):
        # Store analysis_json for UI callback
        self._analysis_json = analysis_json

        # Valid SDXL blocks (only those with attention layers)
        all_valid_blocks = ["text_encoder_1", "text_encoder_2", "input_4", "input_5", "input_7", "input_8", "unet_mid", "output_0", "output_1", "output_2", "output_3", "output_4", "output_5"]

        parsed_weights = _parse_block_weights_string(block_weights_string, 'SDXL')
        if parsed_weights:
            enabled_blocks = set()
            block_strengths = {}
            for block_name, (enabled, blk_str) in parsed_weights.items():
                if block_name == "other_weights":
                    other_enabled = enabled
                    other_str = blk_str
                elif enabled:
                    enabled_blocks.add(block_name)
                    block_strengths[block_name] = blk_str
            using_preset = "String Input"
        else:
            block_settings = {
                "text_encoder_1": (text_encoder_1, text_encoder_1_str),
                "text_encoder_2": (text_encoder_2, text_encoder_2_str),
                "input_4": (input_4, input_4_str),
                "input_5": (input_5, input_5_str),
                "input_7": (input_7, input_7_str),
                "input_8": (input_8, input_8_str),
                "unet_mid": (unet_mid, unet_mid_str),
                "output_0": (output_0, output_0_str),
                "output_1": (output_1, output_1_str),
                "output_2": (output_2, output_2_str),
                "output_3": (output_3, output_3_str),
                "output_4": (output_4, output_4_str),
                "output_5": (output_5, output_5_str),
            }

            if preset != "Custom":
                enabled_blocks = SDXL_PRESETS[preset].copy()
                block_strengths = {b: 1.0 for b in enabled_blocks}
                other_enabled = preset != "All Off"
                other_str = 1.0
                using_preset = preset
            else:
                enabled_blocks = set()
                block_strengths = {}
                for block_id, (enabled, blk_str) in block_settings.items():
                    if enabled:
                        enabled_blocks.add(block_id)
                        block_strengths[block_id] = blk_str
                other_enabled = other_weights
                other_str = other_weights_str
                using_preset = None

        # Load LoRA - use optional path if provided, otherwise use dropdown selection
        if lora_path_opt and os.path.exists(lora_path_opt):
            lora_path = lora_path_opt
        else:
            lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.exists(lora_path):
            return (model, clip, "Error: LoRA not found")

        if lora_path.endswith('.safetensors'):
            lora_state_dict = load_file(lora_path)
        else:
            lora_state_dict = torch.load(lora_path, map_location='cpu')

        # Filter and scale tensors by block strength
        filtered_dict = {}
        for key, value in lora_state_dict.items():
            block_id = _extract_block_id_sdxl(key)
            if block_id in enabled_blocks:
                blk_str = block_strengths.get(block_id, 1.0)
                filtered_dict[key] = value * blk_str if blk_str != 1.0 else value
            elif block_id == 'other' and other_enabled:
                filtered_dict[key] = value * other_str if other_str != 1.0 else value

        original_count = len(lora_state_dict)
        filtered_count = len(filtered_dict)

        if filtered_count == 0:
            return {"ui": {"analysis_json": [analysis_json or ""]}, "result": (model, clip, "Warning: All blocks disabled, no LoRA applied", "")}

        # Apply filtered LoRA
        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, filtered_dict, strength, strength
        )

        disabled = [b for b in all_valid_blocks if b not in enabled_blocks]
        scaled = [f"{b}={block_strengths[b]:.2f}" for b in enabled_blocks if block_strengths.get(b, 1.0) != 1.0]

        info = f"Loaded {filtered_count}/{original_count} tensors\n"
        if using_preset:
            info += f"Preset: {using_preset}\n"
        else:
            info += "Preset: Custom\n"
        info += f"Enabled: {len(enabled_blocks)}/13 blocks\n"
        if scaled:
            info += f"Scaled: {', '.join(scaled)}\n"
        if disabled:
            info += f"Disabled: {', '.join(disabled)}"
        else:
            info += "All blocks enabled"

        weights_output = ", ".join(
            f"{(block_strengths.get(block, 0.0) if block in enabled_blocks else 0.0):.2f}"
            for block in all_valid_blocks
        )
        return {"ui": {"analysis_json": [analysis_json or ""]}, "result": (model_lora, clip_lora, info, weights_output)}


class ZImageSelectiveLoRALoader:
    """
    Selective LoRA Loader for Z-Image Turbo models.

    Toggle individual layers (0-29) on/off to control which parts of the LoRA are applied.
    Use the LoRA Analyzer first to see which layers have the most impact.

    Layer Guide:
    - Layers 0-9: Early processing (usually low impact, ~7-25%)
    - Layers 10-19: Mid processing (moderate impact, ~25-70%)
    - Layers 20-29: Late processing (usually highest impact, ~70-100%)

    Most LoRAs have their main effect in layers 18-29.
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "Z-Image LoRA file to load"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Overall LoRA strength"
                }),
                "preset": (list(ZIMAGE_PRESETS.keys()), {
                    "default": "All Layers",
                    "tooltip": "Quick preset selection. Choose 'Custom' to use individual toggles below."
                }),
            },
        }

        # Add layer toggles and strengths (0-29)
        for i in range(30):
            inputs["required"][f"layer_{i}"] = ("BOOLEAN", {"default": True})
            inputs["required"][f"layer_{i}_str"] = ("FLOAT", {
                "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05
            })

        # Other weights (keys not matching known layers)
        inputs["required"]["other_weights"] = ("BOOLEAN", {"default": True})
        inputs["required"]["other_weights_str"] = ("FLOAT", {
            "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05
        })

        inputs["optional"] = {
            "lora_path_opt": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer to use its selected LoRA"}),
            "analysis_json": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer for impact-colored checkboxes"}),
            "block_weights_string": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "Input/Output block profile string. Positional text syncs with the UI. String input overrides UI values."
            }),
        }

        return inputs

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "info", "weights_output")
    OUTPUT_NODE = True
    FUNCTION = "load_lora"
    CATEGORY = "loaders/lora"
    DESCRIPTION = """Selective LoRA loader for Z-Image Turbo. Toggle each layer on/off.

TIP: Use 'LoRA Loader + Analyzer' first to see which layers matter for YOUR LoRA.
Late layers (20-29) usually have the most effect.
Try disabling early layers (0-9) to reduce style bleed while keeping identity."""

    def load_lora(self, model, clip, lora_name, strength, preset, **kwargs):
        # Get optional inputs from kwargs
        lora_path_opt = kwargs.get("lora_path_opt")
        analysis_json = kwargs.get("analysis_json")
        block_weights_string = kwargs.get("block_weights_string", "")

        # Store analysis_json for UI callback
        self._analysis_json = analysis_json
        parsed_weights = _parse_block_weights_string(block_weights_string, 'ZIMAGE')
        if parsed_weights:
            enabled_layers = set()
            layer_strengths = {}
            for block_name, (enabled, blk_str) in parsed_weights.items():
                if block_name == "other_weights":
                    other_enabled = enabled
                    other_str = blk_str
                elif block_name.startswith("layer_") and enabled:
                    layer_num = int(block_name.split("_")[1])
                    enabled_layers.add(layer_num)
                    layer_strengths[layer_num] = blk_str
            using_preset = "String Input"
        elif preset != "Custom":
            enabled_layers = ZIMAGE_PRESETS[preset].copy()
            layer_strengths = {i: 1.0 for i in enabled_layers}
            # All Off preset disables other_weights too
            other_enabled = preset != "All Off"
            other_str = 1.0
            using_preset = preset
        else:
            # Build from individual toggles and strengths
            enabled_layers = set()
            layer_strengths = {}
            for i in range(30):
                if kwargs.get(f"layer_{i}", True):
                    enabled_layers.add(i)
                    layer_strengths[i] = kwargs.get(f"layer_{i}_str", 1.0)
            other_enabled = kwargs.get("other_weights", True)
            other_str = kwargs.get("other_weights_str", 1.0)
            using_preset = None

        # Load LoRA - use optional path if provided, otherwise use dropdown selection
        if lora_path_opt and os.path.exists(lora_path_opt):
            lora_path = lora_path_opt
        else:
            lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.exists(lora_path):
            return (model, clip, "Error: LoRA not found")

        if lora_path.endswith('.safetensors'):
            lora_state_dict = load_file(lora_path)
        else:
            lora_state_dict = torch.load(lora_path, map_location='cpu')

        # Filter and scale tensors by layer strength
        filtered_dict = {}
        for key, value in lora_state_dict.items():
            layer_num = _extract_layer_num_zimage(key)
            if layer_num is not None:
                if layer_num in enabled_layers:
                    lyr_str = layer_strengths.get(layer_num, 1.0)
                    filtered_dict[key] = value * lyr_str if lyr_str != 1.0 else value
            elif other_enabled:
                # Include non-layer keys (text encoder, etc.) based on other_weights setting
                filtered_dict[key] = value * other_str if other_str != 1.0 else value

        original_count = len(lora_state_dict)
        filtered_count = len(filtered_dict)

        if filtered_count == 0:
            return {"ui": {"analysis_json": [analysis_json or ""]}, "result": (model, clip, "Warning: All layers disabled, no LoRA applied", "")}

        # Apply filtered LoRA
        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, filtered_dict, strength, strength
        )

        disabled_layers = [i for i in range(30) if i not in enabled_layers]
        scaled = [f"{i}={layer_strengths[i]:.2f}" for i in enabled_layers if layer_strengths.get(i, 1.0) != 1.0]

        info = f"Loaded {filtered_count}/{original_count} tensors\n"
        if using_preset:
            info += f"Preset: {using_preset}\n"
        else:
            info += "Preset: Custom\n"
        info += f"Enabled: {len(enabled_layers)}/30 layers\n"
        if scaled:
            info += f"Scaled: {', '.join(scaled)}\n"
        if disabled_layers:
            info += f"Disabled: {', '.join(str(l) for l in disabled_layers)}"
        else:
            info += "All layers enabled"

        weights_output = ", ".join(
            f"{(layer_strengths.get(i, 0.0) if i in enabled_layers else 0.0):.2f}"
            for i in range(30)
        )
        return {"ui": {"analysis_json": [analysis_json or ""]}, "result": (model_lora, clip_lora, info, weights_output)}


class FLUXSelectiveLoRALoader:
    """
    Selective LoRA Loader for FLUX models.

    Toggle individual blocks on/off to control which parts of the LoRA are applied.
    Use the LoRA Analyzer first to see which blocks have the most impact.

    Block Guide (57 total):
    - double_0-18: Double transformer blocks (19 blocks, higher impact)
    - single_0-37: Single transformer blocks (38 blocks, lower impact)

    Double blocks typically have higher impact than single blocks.
    Peak impact is usually in double_8-17 range.
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "FLUX LoRA file to load"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Overall LoRA strength"
                }),
                "preset": (list(FLUX_PRESETS.keys()), {
                    "default": "All Blocks",
                    "tooltip": "Quick preset selection. Choose 'Custom' to use individual toggles below."
                }),
            },
        }

        # Add double block toggles and strengths (0-18)
        for i in range(19):
            inputs["required"][f"double_{i}"] = ("BOOLEAN", {"default": True})
            inputs["required"][f"double_{i}_str"] = ("FLOAT", {
                "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05
            })

        # Add single block toggles and strengths (0-37)
        for i in range(38):
            inputs["required"][f"single_{i}"] = ("BOOLEAN", {"default": True})
            inputs["required"][f"single_{i}_str"] = ("FLOAT", {
                "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05
            })

        # Other weights (keys not matching known blocks)
        inputs["required"]["other_weights"] = ("BOOLEAN", {"default": True})
        inputs["required"]["other_weights_str"] = ("FLOAT", {
            "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05
        })

        inputs["optional"] = {
            "lora_path_opt": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer to use its selected LoRA"}),
            "analysis_json": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer for impact-colored checkboxes"}),
            "block_weights_string": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "Input/Output block profile string. Positional text syncs with the UI. String input overrides UI values."
            }),
        }

        return inputs

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "info", "weights_output")
    OUTPUT_NODE = True
    FUNCTION = "load_lora"
    CATEGORY = "loaders/lora"
    DESCRIPTION = """Selective LoRA loader for FLUX. Toggle blocks on/off.

TIP: Use 'LoRA Loader + Analyzer' first to see which blocks matter for your LoRA.
Double blocks (0-18) typically have more impact than single blocks (0-37)."""

    def load_lora(self, model, clip, lora_name, strength, preset, **kwargs):
        # Get optional inputs from kwargs
        lora_path_opt = kwargs.get("lora_path_opt")
        analysis_json = kwargs.get("analysis_json")
        block_weights_string = kwargs.get("block_weights_string", "")

        # Store analysis_json for UI callback
        self._analysis_json = analysis_json
        parsed_weights = _parse_block_weights_string(block_weights_string, 'FLUX')
        if parsed_weights:
            enabled_blocks = set()
            block_strengths = {}
            for block_name, (enabled, blk_str) in parsed_weights.items():
                if block_name == "other_weights":
                    other_enabled = enabled
                    other_str = blk_str
                elif enabled:
                    enabled_blocks.add(block_name)
                    block_strengths[block_name] = blk_str
            using_preset = "String Input"
        elif preset != "Custom":
            enabled_blocks = FLUX_PRESETS[preset].copy()
            block_strengths = {b: 1.0 for b in enabled_blocks}
            # All Off preset disables other_weights too
            other_enabled = preset != "All Off"
            other_str = 1.0
            using_preset = preset
        else:
            # Build from individual toggles and strengths
            enabled_blocks = set()
            block_strengths = {}
            for i in range(19):
                block_id = f"double_{i}"
                if kwargs.get(block_id, True):
                    enabled_blocks.add(block_id)
                    block_strengths[block_id] = kwargs.get(f"{block_id}_str", 1.0)
            for i in range(38):
                block_id = f"single_{i}"
                if kwargs.get(block_id, True):
                    enabled_blocks.add(block_id)
                    block_strengths[block_id] = kwargs.get(f"{block_id}_str", 1.0)
            other_enabled = kwargs.get("other_weights", True)
            other_str = kwargs.get("other_weights_str", 1.0)
            using_preset = None

        # Load LoRA - use optional path if provided, otherwise use dropdown selection
        if lora_path_opt and os.path.exists(lora_path_opt):
            lora_path = lora_path_opt
        else:
            lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.exists(lora_path):
            return (model, clip, "Error: LoRA not found")

        if lora_path.endswith('.safetensors'):
            lora_state_dict = load_file(lora_path)
        else:
            lora_state_dict = torch.load(lora_path, map_location='cpu')

        # Filter and scale tensors by block strength
        filtered_dict = {}
        for key, value in lora_state_dict.items():
            block_id = _extract_block_id_flux(key)
            if block_id in enabled_blocks:
                blk_str = block_strengths.get(block_id, 1.0)
                filtered_dict[key] = value * blk_str if blk_str != 1.0 else value
            elif block_id == 'other' and other_enabled:
                filtered_dict[key] = value * other_str if other_str != 1.0 else value

        original_count = len(lora_state_dict)
        filtered_count = len(filtered_dict)

        if filtered_count == 0:
            return {"ui": {"analysis_json": [analysis_json or ""]}, "result": (model, clip, "Warning: All blocks disabled, no LoRA applied", "")}

        # Apply filtered LoRA
        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, filtered_dict, strength, strength
        )

        all_blocks = [f"double_{i}" for i in range(19)] + [f"single_{i}" for i in range(38)]
        disabled = [b for b in all_blocks if b not in enabled_blocks]
        scaled = [f"{b}={block_strengths[b]:.2f}" for b in enabled_blocks if block_strengths.get(b, 1.0) != 1.0]

        info = f"Loaded {filtered_count}/{original_count} tensors\n"
        if using_preset:
            info += f"Preset: {using_preset}\n"
        else:
            info += "Preset: Custom\n"
        info += f"Enabled: {len(enabled_blocks)}/57 blocks\n"
        if scaled:
            info += f"Scaled: {', '.join(scaled[:10])}"  # Limit to first 10 for readability
            if len(scaled) > 10:
                info += f" (+{len(scaled)-10} more)\n"
            else:
                info += "\n"
        if disabled:
            # Summarize disabled blocks
            disabled_double = [b for b in disabled if b.startswith("double_")]
            disabled_single = [b for b in disabled if b.startswith("single_")]
            if disabled_double:
                info += f"Disabled double: {', '.join(b.replace('double_', '') for b in disabled_double)}\n"
            if disabled_single:
                info += f"Disabled single: {', '.join(b.replace('single_', '') for b in disabled_single)}"
        else:
            info += "All blocks enabled"

        weights_output = ", ".join(
            [f"{(block_strengths.get(f'double_{i}', 0.0) if f'double_{i}' in enabled_blocks else 0.0):.2f}" for i in range(19)] +
            [f"{(block_strengths.get(f'single_{i}', 0.0) if f'single_{i}' in enabled_blocks else 0.0):.2f}" for i in range(38)]
        )
        return {"ui": {"analysis_json": [analysis_json or ""]}, "result": (model_lora, clip_lora, info, weights_output)}


class WanSelectiveLoRALoader:
    """
    Selective LoRA Loader for Wan 2.2 models.

    Toggle individual transformer blocks on/off to control which parts of the LoRA are applied.
    Use the LoRA Analyzer first to see which blocks have the most impact.

    Block Guide (40 total):
    - block_0-9: Early transformer blocks
    - block_10-19: Early-mid blocks
    - block_20-29: Mid-late blocks
    - block_30-39: Late blocks
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "Wan 2.2 LoRA file to load"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Overall LoRA strength"
                }),
                "preset": (list(WAN_PRESETS.keys()), {
                    "default": "All Blocks",
                    "tooltip": "Quick preset selection. Choose 'Custom' to use individual toggles below."
                }),
            },
        }

        # Add block toggles and strengths (0-39)
        for i in range(40):
            inputs["required"][f"block_{i}"] = ("BOOLEAN", {"default": True})
            inputs["required"][f"block_{i}_str"] = ("FLOAT", {
                "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05
            })

        # Other weights (keys not matching known blocks)
        inputs["required"]["other_weights"] = ("BOOLEAN", {"default": True})
        inputs["required"]["other_weights_str"] = ("FLOAT", {
            "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05
        })

        inputs["optional"] = {
            "lora_path_opt": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer to use its selected LoRA"}),
            "analysis_json": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer for impact-colored checkboxes"}),
            "block_weights_string": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "Input/Output block profile string. Positional text syncs with the UI. String input overrides UI values."
            }),
        }

        return inputs

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "info", "weights_output")
    OUTPUT_NODE = True
    FUNCTION = "load_lora"
    CATEGORY = "loaders/lora"
    DESCRIPTION = """Selective LoRA loader for Wan 2.2. Toggle blocks on/off.

TIP: Use 'LoRA Loader + Analyzer' first to see which blocks matter for your LoRA."""

    def load_lora(self, model, clip, lora_name, strength, preset, **kwargs):
        # Get optional inputs from kwargs
        lora_path_opt = kwargs.get("lora_path_opt")
        analysis_json = kwargs.get("analysis_json")
        block_weights_string = kwargs.get("block_weights_string", "")

        # Store analysis_json for UI callback
        self._analysis_json = analysis_json
        parsed_weights = _parse_block_weights_string(block_weights_string, 'WAN')
        if parsed_weights:
            enabled_blocks = set()
            block_strengths = {}
            for block_name, (enabled, blk_str) in parsed_weights.items():
                if block_name == "other_weights":
                    other_enabled = enabled
                    other_str = blk_str
                elif block_name.startswith("block_") and enabled:
                    block_num = int(block_name.split("_")[1])
                    enabled_blocks.add(block_num)
                    block_strengths[block_num] = blk_str
            using_preset = "String Input"
        elif preset != "Custom":
            enabled_blocks = WAN_PRESETS[preset].copy()
            block_strengths = {i: 1.0 for i in enabled_blocks}
            # All Off preset disables other_weights too
            other_enabled = preset != "All Off"
            other_str = 1.0
            using_preset = preset
        else:
            # Build from individual toggles and strengths
            enabled_blocks = set()
            block_strengths = {}
            for i in range(40):
                if kwargs.get(f"block_{i}", True):
                    enabled_blocks.add(i)
                    block_strengths[i] = kwargs.get(f"block_{i}_str", 1.0)
            other_enabled = kwargs.get("other_weights", True)
            other_str = kwargs.get("other_weights_str", 1.0)
            using_preset = None

        # Load LoRA - use optional path if provided, otherwise use dropdown selection
        if lora_path_opt and os.path.exists(lora_path_opt):
            lora_path = lora_path_opt
        else:
            lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.exists(lora_path):
            return (model, clip, "Error: LoRA not found")

        if lora_path.endswith('.safetensors'):
            lora_state_dict = load_file(lora_path)
        else:
            lora_state_dict = torch.load(lora_path, map_location='cpu')

        # Filter and scale tensors by block strength
        filtered_dict = {}
        for key, value in lora_state_dict.items():
            block_num = _extract_block_id_wan(key)
            if block_num is not None:
                if block_num in enabled_blocks:
                    blk_str = block_strengths.get(block_num, 1.0)
                    filtered_dict[key] = value * blk_str if blk_str != 1.0 else value
            elif other_enabled:
                # Include non-block keys based on other_weights setting
                filtered_dict[key] = value * other_str if other_str != 1.0 else value

        original_count = len(lora_state_dict)
        filtered_count = len(filtered_dict)

        if filtered_count == 0:
            return {"ui": {"analysis_json": [analysis_json or ""]}, "result": (model, clip, "Warning: All blocks disabled, no LoRA applied", "")}

        # Apply filtered LoRA
        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, filtered_dict, strength, strength
        )

        disabled_blocks = [i for i in range(40) if i not in enabled_blocks]
        scaled = [f"{i}={block_strengths[i]:.2f}" for i in enabled_blocks if block_strengths.get(i, 1.0) != 1.0]

        info = f"Loaded {filtered_count}/{original_count} tensors\n"
        if using_preset:
            info += f"Preset: {using_preset}\n"
        else:
            info += "Preset: Custom\n"
        info += f"Enabled: {len(enabled_blocks)}/40 blocks\n"
        if scaled:
            info += f"Scaled: {', '.join(scaled[:10])}"
            if len(scaled) > 10:
                info += f" (+{len(scaled)-10} more)\n"
            else:
                info += "\n"
        if disabled_blocks:
            info += f"Disabled: {', '.join(str(b) for b in disabled_blocks)}"
        else:
            info += "All blocks enabled"

        weights_output = ", ".join(
            f"{(block_strengths.get(i, 0.0) if i in enabled_blocks else 0.0):.2f}"
            for i in range(40)
        )
        return {"ui": {"analysis_json": [analysis_json or ""]}, "result": (model_lora, clip_lora, info, weights_output)}


class QwenSelectiveLoRALoader:
    """
    Selective LoRA Loader for Qwen-Image models.

    Toggle individual transformer blocks on/off to control which parts of the LoRA are applied.
    Use the LoRA Analyzer first to see which blocks have the most impact.

    Block Guide (60 total):
    - block_0-14: Early transformer blocks
    - block_15-29: Early-mid blocks
    - block_30-44: Mid-late blocks
    - block_45-59: Late blocks
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "Qwen-Image LoRA file to load"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Overall LoRA strength"
                }),
                "preset": (list(QWEN_PRESETS.keys()), {
                    "default": "All Blocks",
                    "tooltip": "Quick preset selection. Choose 'Custom' to use individual toggles below."
                }),
            },
        }

        # Add block toggles and strengths (0-59)
        for i in range(60):
            inputs["required"][f"block_{i}"] = ("BOOLEAN", {"default": True})
            inputs["required"][f"block_{i}_str"] = ("FLOAT", {
                "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05
            })

        # Other weights (keys not matching known blocks)
        inputs["required"]["other_weights"] = ("BOOLEAN", {"default": True})
        inputs["required"]["other_weights_str"] = ("FLOAT", {
            "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05
        })

        inputs["optional"] = {
            "lora_path_opt": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer to use its selected LoRA"}),
            "analysis_json": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer for impact-colored checkboxes"}),
            "block_weights_string": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "Input/Output block profile string. Positional text syncs with the UI. String input overrides UI values."
            }),
        }

        return inputs

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "info", "weights_output")
    OUTPUT_NODE = True
    FUNCTION = "load_lora"
    CATEGORY = "loaders/lora"
    DESCRIPTION = """Selective LoRA loader for Qwen-Image. Toggle blocks on/off.

TIP: Use 'LoRA Loader + Analyzer' first to see which blocks matter for your LoRA."""

    def load_lora(self, model, clip, lora_name, strength, preset, **kwargs):
        # Get optional inputs from kwargs
        lora_path_opt = kwargs.get("lora_path_opt")
        analysis_json = kwargs.get("analysis_json")
        block_weights_string = kwargs.get("block_weights_string", "")

        # Store analysis_json for UI callback
        self._analysis_json = analysis_json
        parsed_weights = _parse_block_weights_string(block_weights_string, 'QWEN')
        if parsed_weights:
            enabled_blocks = set()
            block_strengths = {}
            for block_name, (enabled, blk_str) in parsed_weights.items():
                if block_name == "other_weights":
                    other_enabled = enabled
                    other_str = blk_str
                elif block_name.startswith("block_") and enabled:
                    block_num = int(block_name.split("_")[1])
                    enabled_blocks.add(block_num)
                    block_strengths[block_num] = blk_str
            using_preset = "String Input"
        elif preset != "Custom":
            enabled_blocks = QWEN_PRESETS[preset].copy()
            block_strengths = {i: 1.0 for i in enabled_blocks}
            # All Off preset disables other_weights too
            other_enabled = preset != "All Off"
            other_str = 1.0
            using_preset = preset
        else:
            # Build from individual toggles and strengths
            enabled_blocks = set()
            block_strengths = {}
            for i in range(60):
                if kwargs.get(f"block_{i}", True):
                    enabled_blocks.add(i)
                    block_strengths[i] = kwargs.get(f"block_{i}_str", 1.0)
            other_enabled = kwargs.get("other_weights", True)
            other_str = kwargs.get("other_weights_str", 1.0)
            using_preset = None

        # Load LoRA - use optional path if provided, otherwise use dropdown selection
        if lora_path_opt and os.path.exists(lora_path_opt):
            lora_path = lora_path_opt
        else:
            lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.exists(lora_path):
            return (model, clip, "Error: LoRA not found")

        if lora_path.endswith('.safetensors'):
            lora_state_dict = load_file(lora_path)
        else:
            lora_state_dict = torch.load(lora_path, map_location='cpu')

        # Filter and scale tensors by block strength
        filtered_dict = {}
        for key, value in lora_state_dict.items():
            block_num = _extract_block_id_qwen(key)
            if block_num is not None:
                if block_num in enabled_blocks:
                    blk_str = block_strengths.get(block_num, 1.0)
                    filtered_dict[key] = value * blk_str if blk_str != 1.0 else value
            elif other_enabled:
                # Include non-block keys based on other_weights setting
                filtered_dict[key] = value * other_str if other_str != 1.0 else value

        original_count = len(lora_state_dict)
        filtered_count = len(filtered_dict)

        if filtered_count == 0:
            return {"ui": {"analysis_json": [analysis_json or ""]}, "result": (model, clip, "Warning: All blocks disabled, no LoRA applied", "")}

        # Apply filtered LoRA
        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, filtered_dict, strength, strength
        )

        disabled_blocks = [i for i in range(60) if i not in enabled_blocks]
        scaled = [f"{i}={block_strengths[i]:.2f}" for i in enabled_blocks if block_strengths.get(i, 1.0) != 1.0]

        info = f"Loaded {filtered_count}/{original_count} tensors\n"
        if using_preset:
            info += f"Preset: {using_preset}\n"
        else:
            info += "Preset: Custom\n"
        info += f"Enabled: {len(enabled_blocks)}/60 blocks\n"
        if scaled:
            info += f"Scaled: {', '.join(scaled[:10])}"
            if len(scaled) > 10:
                info += f" (+{len(scaled)-10} more)\n"
            else:
                info += "\n"
        if disabled_blocks:
            info += f"Disabled: {', '.join(str(b) for b in disabled_blocks)}"
        else:
            info += "All blocks enabled"

        weights_output = ", ".join(
            f"{(block_strengths.get(i, 0.0) if i in enabled_blocks else 0.0):.2f}"
            for i in range(60)
        )
        return {"ui": {"analysis_json": [analysis_json or ""]}, "result": (model_lora, clip_lora, info, weights_output)}


class Krea2SelectiveLoRALoader:
    """
    Selective LoRA Loader for Krea 2 models.

    Toggle individual main SingleStreamBlocks on/off to control which parts of the LoRA are applied.
    Non-main-block Linear layers such as first, last.linear, tmlp, txtmlp, tproj, and txtfusion are
    controlled by other_weights.

    Block Guide (28 total):
    - block_0-8: Early main blocks
    - block_9-18: Mid main blocks
    - block_19-27: Late main blocks
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "Krea 2 LoRA file to load"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Overall LoRA strength"
                }),
                "preset": (list(KREA2_PRESETS.keys()), {
                    "default": "All Blocks",
                    "tooltip": "Quick preset selection. Choose 'Custom' to use individual toggles below."
                }),
            },
        }

        # Add main block toggles and strengths (0-27)
        for i in range(28):
            inputs["required"][f"block_{i}"] = ("BOOLEAN", {"default": True})
            inputs["required"][f"block_{i}_str"] = ("FLOAT", {
                "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05
            })

        # Other Krea 2 Linear layers: first, last.linear, tmlp, txtmlp, tproj, txtfusion, etc.
        inputs["required"]["other_weights"] = ("BOOLEAN", {"default": True})
        inputs["required"]["other_weights_str"] = ("FLOAT", {
            "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05
        })

        inputs["optional"] = {
            "lora_path_opt": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer to use its selected LoRA"}),
            "analysis_json": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer for impact-colored checkboxes"}),
            "block_weights_string": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "Input/Output block profile string. Positional text syncs with the UI. String input overrides UI values."
            }),
        }

        return inputs

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "info", "weights_output")
    OUTPUT_NODE = True
    FUNCTION = "load_lora"
    CATEGORY = "loaders/lora"
    DESCRIPTION = """Selective LoRA loader for Krea 2. Toggle the 28 main SingleStreamBlocks on/off.

TIP: Use 'LoRA Loader + Analyzer' first to see which blocks matter for your LoRA.
Use other_weights for non-main-block Krea 2 modules like txtfusion, tmlp, txtmlp, tproj, first, and last.linear."""

    def load_lora(self, model, clip, lora_name, strength, preset, **kwargs):
        # Get optional inputs from kwargs
        lora_path_opt = kwargs.get("lora_path_opt")
        analysis_json = kwargs.get("analysis_json")
        block_weights_string = kwargs.get("block_weights_string", "")

        # Store analysis_json for UI callback
        self._analysis_json = analysis_json
        parsed_weights = _parse_block_weights_string(block_weights_string, 'KREA2')
        if parsed_weights:
            enabled_blocks = set()
            block_strengths = {}
            for block_name, (enabled, block_strength) in parsed_weights.items():
                if block_name == "other_weights":
                    other_enabled = enabled
                    other_str = block_strength
                elif block_name.startswith("block_") and enabled:
                    block_num = int(block_name.split("_")[1])
                    enabled_blocks.add(block_num)
                    block_strengths[block_num] = block_strength
            using_preset = "String Input"
        elif preset != "Custom":
            enabled_blocks = KREA2_PRESETS[preset].copy()
            block_strengths = {i: 1.0 for i in enabled_blocks}
            # All Off preset disables other_weights too
            other_enabled = preset != "All Off"
            other_str = 1.0
            using_preset = preset
        else:
            # Build from individual toggles and strengths
            enabled_blocks = set()
            block_strengths = {}
            for i in range(28):
                if kwargs.get(f"block_{i}", True):
                    enabled_blocks.add(i)
                    block_strengths[i] = kwargs.get(f"block_{i}_str", 1.0)
            other_enabled = kwargs.get("other_weights", True)
            other_str = kwargs.get("other_weights_str", 1.0)
            using_preset = None

        # Load LoRA - use optional path if provided, otherwise use dropdown selection
        if lora_path_opt and os.path.exists(lora_path_opt):
            lora_path = lora_path_opt
        else:
            lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.exists(lora_path):
            return (model, clip, "Error: LoRA not found", "")

        if lora_path.endswith('.safetensors'):
            lora_state_dict = load_file(lora_path)
        else:
            lora_state_dict = torch.load(lora_path, map_location='cpu')

        # Filter and scale tensors by block strength
        filtered_dict = {}
        for key, value in lora_state_dict.items():
            block_num = _extract_block_id_krea2(key)
            if block_num is not None:
                if block_num in enabled_blocks:
                    blk_str = block_strengths.get(block_num, 1.0)
                    filtered_dict[key] = value * blk_str if blk_str != 1.0 else value
            elif other_enabled:
                # Include non-main-block keys based on other_weights setting
                filtered_dict[key] = value * other_str if other_str != 1.0 else value

        original_count = len(lora_state_dict)
        filtered_count = len(filtered_dict)

        if filtered_count == 0:
            return {"ui": {"analysis_json": [analysis_json or ""]}, "result": (model, clip, "Warning: All blocks disabled, no LoRA applied", "")}

        # Apply filtered LoRA
        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, filtered_dict, strength, strength
        )

        disabled_blocks = [i for i in range(28) if i not in enabled_blocks]
        scaled = [f"{i}={block_strengths[i]:.2f}" for i in enabled_blocks if block_strengths.get(i, 1.0) != 1.0]

        info = f"Loaded {filtered_count}/{original_count} tensors\n"
        if using_preset:
            info += f"Preset: {using_preset}\n"
        else:
            info += "Preset: Custom\n"
        info += f"Enabled: {len(enabled_blocks)}/28 blocks\n"
        info += f"Other weights: {'enabled' if other_enabled else 'disabled'}"
        if other_enabled and other_str != 1.0:
            info += f" ({other_str:.2f})"
        info += "\n"
        if scaled:
            info += f"Scaled: {', '.join(scaled[:10])}"
            if len(scaled) > 10:
                info += f" (+{len(scaled)-10} more)\n"
            else:
                info += "\n"
        if disabled_blocks:
            info += f"Disabled: {', '.join(str(b) for b in disabled_blocks)}"
        else:
            info += "All blocks enabled"

        output_values = [
            block_strengths.get(i, 0.0) if i in enabled_blocks else 0.0
            for i in range(28)
        ]
        output_values.append(other_str if other_enabled else 0.0)
        weights_output = ", ".join(f"{value:.2f}" for value in output_values)
        return {"ui": {"analysis_json": [analysis_json or ""]}, "result": (model_lora, clip_lora, info, weights_output)}


class MiniMaxH3SelectiveLoRALoader:
    """Selective LoRA loader for MiniMax H3's 50 packed DiT blocks."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "MiniMax H3 LoRA file to load"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Overall LoRA strength"
                }),
                "preset": (list(MINIMAX_H3_PRESETS.keys()), {
                    "default": "All Blocks",
                    "tooltip": "Quick preset selection. Choose 'Custom' to use individual toggles below."
                }),
            },
        }

        for i in range(50):
            inputs["required"][f"block_{i}"] = ("BOOLEAN", {"default": True})
            inputs["required"][f"block_{i}_str"] = ("FLOAT", {
                "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05
            })

        inputs["required"]["other_weights"] = ("BOOLEAN", {"default": True})
        inputs["required"]["other_weights_str"] = ("FLOAT", {
            "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05
        })

        inputs["optional"] = {
            "lora_path_opt": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer to use its selected LoRA"}),
            "analysis_json": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer for impact-colored checkboxes"}),
            "block_weights_string": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "Optional positional or named block profile. A connected string overrides the individual controls."
            }),
            "save_refined_lora": ("BOOLEAN", {
                "default": False,
                "tooltip": "Save the exact filtered/scaled LoRA applied by this node."
            }),
            "save_path": ("STRING", {
                "default": "",
                "tooltip": "Directory where the filtered LoRA will be saved."
            }),
            "save_filename": ("STRING", {
                "default": "",
                "tooltip": "Base filename. A timestamp and .safetensors are added automatically."
            }),
        }
        return inputs

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "info", "weights_output")
    OUTPUT_NODE = True
    FUNCTION = "load_lora"
    CATEGORY = "loaders/lora"
    DESCRIPTION = """Selective LoRA loader for MiniMax H3. Toggle the 50 main packed DiT blocks on/off.

Use other_weights for token-refiner and any non-main H3 tensors. Supports both native
diffusion_model.blocks.* keys and lora_unet_blocks_* training keys."""

    def load_lora(self, model, clip, lora_name, strength, preset, **kwargs):
        strength = _coerce_scalar_strength(strength)
        lora_path_opt = kwargs.get("lora_path_opt")
        analysis_json = kwargs.get("analysis_json")
        block_weights_string = kwargs.get("block_weights_string", "")
        save_refined_lora = kwargs.get("save_refined_lora", False)
        save_path = kwargs.get("save_path", "")
        save_filename = kwargs.get("save_filename", "")

        self._analysis_json = analysis_json
        parsed_weights = _parse_block_weights_string(block_weights_string, "MINIMAX_H3")
        if parsed_weights:
            enabled_blocks = set()
            block_strengths = {}
            other_enabled = True
            other_str = 1.0
            for block_name, (enabled, blk_str) in parsed_weights.items():
                if block_name == "other_weights":
                    other_enabled = enabled
                    other_str = blk_str
                elif block_name.startswith("block_") and enabled:
                    block_num = int(block_name.split("_")[1])
                    enabled_blocks.add(block_num)
                    block_strengths[block_num] = blk_str
            using_preset = "String Input"
        elif preset != "Custom":
            enabled_blocks = MINIMAX_H3_PRESETS[preset].copy()
            block_strengths = {i: 1.0 for i in enabled_blocks}
            other_enabled = preset != "All Off"
            other_str = 1.0
            using_preset = preset
        else:
            enabled_blocks = set()
            block_strengths = {}
            for i in range(50):
                if kwargs.get(f"block_{i}", True):
                    enabled_blocks.add(i)
                    block_strengths[i] = kwargs.get(f"block_{i}_str", 1.0)
            other_enabled = kwargs.get("other_weights", True)
            other_str = kwargs.get("other_weights_str", 1.0)
            using_preset = None

        if lora_path_opt and os.path.exists(lora_path_opt):
            lora_path = lora_path_opt
        else:
            lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.exists(lora_path):
            return (model, clip, "Error: LoRA not found", "")

        if lora_path.endswith(".safetensors"):
            lora_state_dict = load_file(lora_path)
        else:
            lora_state_dict = torch.load(lora_path, map_location="cpu")

        filtered_dict = {}
        for key, value in lora_state_dict.items():
            block_num = _extract_block_id_minimax_h3(key)
            if block_num is not None:
                if block_num in enabled_blocks:
                    blk_str = block_strengths.get(block_num, 1.0)
                    filtered_dict[key] = _scale_minimax_h3_tensor(key, value, blk_str)
            elif other_enabled:
                filtered_dict[key] = _scale_minimax_h3_tensor(key, value, other_str)

        original_count = len(lora_state_dict)
        filtered_count = len(filtered_dict)
        if filtered_count == 0:
            output_values = [
                block_strengths.get(i, 0.0) if i in enabled_blocks else 0.0
                for i in range(50)
            ]
            output_values.append(other_str if other_enabled else 0.0)
            weights_output = ", ".join(f"{value:.2f}" for value in output_values)
            return {
                "ui": {"analysis_json": [analysis_json or ""]},
                "result": (model, clip, "Warning: All blocks disabled, no LoRA applied", weights_output),
            }

        saved_path = None
        if save_refined_lora and save_path.strip():
            saved_path = _save_minimax_h3_filtered_lora(
                filtered_dict, lora_path, save_path, save_filename
            )

        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, filtered_dict, strength, strength
        )

        disabled_blocks = [i for i in range(50) if i not in enabled_blocks]
        scaled = [
            f"{i}={block_strengths[i]:.2f}"
            for i in sorted(enabled_blocks)
            if block_strengths.get(i, 1.0) != 1.0
        ]

        info = f"Loaded {filtered_count}/{original_count} tensors\n"
        info += f"Preset: {using_preset or 'Custom'}\n"
        info += f"Enabled: {len(enabled_blocks)}/50 blocks\n"
        info += f"Other weights: {'enabled' if other_enabled else 'disabled'}"
        if other_enabled and other_str != 1.0:
            info += f" ({other_str:.2f})"
        info += "\n"
        if scaled:
            info += f"Scaled: {', '.join(scaled[:10])}"
            if len(scaled) > 10:
                info += f" (+{len(scaled)-10} more)\n"
            else:
                info += "\n"
        if disabled_blocks:
            info += f"Disabled: {', '.join(str(b) for b in disabled_blocks)}"
        else:
            info += "All blocks enabled"
        if save_refined_lora:
            if saved_path:
                info += f"\nSaved: {saved_path}"
            elif not save_path.strip():
                info += "\nSave skipped: save_path is empty"
            else:
                info += "\nSave failed; check the ComfyUI console"

        output_values = [
            block_strengths.get(i, 0.0) if i in enabled_blocks else 0.0
            for i in range(50)
        ]
        output_values.append(other_str if other_enabled else 0.0)
        weights_output = ", ".join(f"{value:.2f}" for value in output_values)
        return {"ui": {"analysis_json": [analysis_json or ""]}, "result": (model_lora, clip_lora, info, weights_output)}


NODE_CLASS_MAPPINGS = {
    "SDXLSelectiveLoRALoader": SDXLSelectiveLoRALoader,
    "ZImageSelectiveLoRALoader": ZImageSelectiveLoRALoader,
    "FLUXSelectiveLoRALoader": FLUXSelectiveLoRALoader,
    "WanSelectiveLoRALoader": WanSelectiveLoRALoader,
    "QwenSelectiveLoRALoader": QwenSelectiveLoRALoader,
    "Krea2SelectiveLoRALoader": Krea2SelectiveLoRALoader,
    "MiniMaxH3SelectiveLoRALoader": MiniMaxH3SelectiveLoRALoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXLSelectiveLoRALoader": "Selective LoRA Loader (SDXL)",
    "ZImageSelectiveLoRALoader": "Selective LoRA Loader (Z-Image)",
    "FLUXSelectiveLoRALoader": "Selective LoRA Loader (FLUX)",
    "WanSelectiveLoRALoader": "Selective LoRA Loader (Wan)",
    "QwenSelectiveLoRALoader": "Selective LoRA Loader (Qwen)",
    "Krea2SelectiveLoRALoader": "Selective LoRA Loader (Krea 2)",
    "MiniMaxH3SelectiveLoRALoader": "Selective LoRA Loader (MiniMax H3)",
}
