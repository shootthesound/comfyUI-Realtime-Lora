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


def _detect_architecture(keys):
    """Identify LoRA architecture from key patterns."""
    keys_lower = [k.lower() for k in keys]
    keys_str = ' '.join(keys_lower)
    num_keys = len(keys)

    if any('transformer_blocks' in k and any(x in k for x in ['img_mlp', 'txt_mlp', 'img_mod', 'txt_mod']) for k in keys_lower):
        return 'QWEN_IMAGE'
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


def _parse_block_weights_string(weights_str: str, architecture: str) -> Optional[Dict]:
    """
    Parse Inspire Pack-style block weight syntax.

    Syntax: %default=1.0, te1=0.5, te2=0.5, in=0.8, in7-8=1.2, mid=1.0, out=0.9, out0-2=1.5

    For FLUX: %default=1.0, double=0.8, double7,12,16=1.5, single=0.5
    For Z-Image/Wan/Qwen: %default=1.0, layer=0.8, layer18-25=1.2 (or block for Wan/Qwen)

    Returns dict: {block_name: (enabled, strength)} or None if no string
    """
    if not weights_str or not weights_str.strip():
        return None

    weights_str = weights_str.strip()
    if not weights_str.startswith('%'):
        return None

    # Remove leading %
    weights_str = weights_str[1:].strip()

    # Parse into dict
    result = {}
    default_val = 1.0

    # Split by comma
    parts = [p.strip() for p in weights_str.split(',') if p.strip()]

    # Process default first
    for part in parts:
        if '=' not in part:
            continue
        key, val = part.split('=', 1)
        key = key.strip()
        try:
            val = float(val.strip())
        except ValueError:
            continue

        if key == 'default':
            default_val = val
            break

    # Initialize all blocks with default
    if architecture == 'SDXL':
        block_names = ['text_encoder_1', 'text_encoder_2', 'input_4', 'input_5',
                      'input_7', 'input_8', 'unet_mid',
                      'output_0', 'output_1', 'output_2', 'output_3', 'output_4', 'output_5',
                      'other_weights']
        aliases = {'te1': 'text_encoder_1', 'te2': 'text_encoder_2', 'in': 'input',
                  'mid': 'unet_mid', 'out': 'output', 'other': 'other_weights'}
    elif architecture == 'FLUX':
        block_names = [f'double_{i}' for i in range(19)] + [f'single_{i}' for i in range(38)] + ['other_weights']
        aliases = {'other': 'other_weights'}
    elif architecture == 'ZIMAGE':
        block_names = [f'layer_{i}' for i in range(30)] + ['other_weights']
        aliases = {'other': 'other_weights'}
    elif architecture == 'WAN':
        block_names = [f'block_{i}' for i in range(40)] + ['other_weights']
        aliases = {'other': 'other_weights'}
    elif architecture == 'QWEN':
        block_names = [f'block_{i}' for i in range(60)] + ['other_weights']
        aliases = {'other': 'other_weights'}
    else:
        return None

    # Initialize all with default
    for name in block_names:
        result[name] = (default_val != 0.0, default_val)

    # Process named assignments
    for part in parts:
        if '=' not in part:
            continue
        key, val = part.split('=', 1)
        key = key.strip()
        try:
            val = float(val.strip())
        except ValueError:
            continue

        if key == 'default':
            continue  # Already processed

        # Check for aliases
        orig_key = key
        for alias, full_name in aliases.items():
            if key.startswith(alias):
                key = key.replace(alias, full_name, 1)
                break

        # Check for range (e.g., "input7-8" or "double7-12")
        if '-' in key and any(c.isdigit() for c in key):
            # Extract base name and range
            base = ''.join(c for c in key if not c.isdigit() and c != '-')
            nums = ''.join(c if c.isdigit() or c == '-' else '' for c in key)

            if '-' in nums:
                try:
                    start, end = nums.split('-')
                    start, end = int(start), int(end)
                    for i in range(start, end + 1):
                        block_name = f"{base}_{i}" if base else f"{i}"
                        # Handle SDXL special case (input_7, output_0, etc)
                        if architecture == 'SDXL' and base in ['input', 'output']:
                            block_name = f"{base}_{i}"
                        if block_name in result:
                            result[block_name] = (val != 0.0, val)
                except (ValueError, IndexError):
                    pass
        # Check for comma-separated list (e.g., "double7,12,16=1.5")
        elif ',' in key:
            continue  # Will be handled by individual processing
        # Check for single numbered block (e.g., "input7", "double12")
        elif any(c.isdigit() for c in key):
            base = ''.join(c for c in key if not c.isdigit())
            num = ''.join(c for c in key if c.isdigit())
            if num:
                try:
                    num = int(num)
                    block_name = f"{base}_{num}" if base else f"{num}"
                    if architecture == 'SDXL' and base in ['input', 'output']:
                        block_name = f"{base}_{num}"
                    if block_name in result:
                        result[block_name] = (val != 0.0, val)
                except ValueError:
                    pass
        # Whole category (e.g., "in=0.8" means all input blocks)
        else:
            if architecture == 'SDXL':
                if key == 'input':
                    for name in block_names:
                        if name.startswith('input_'):
                            result[name] = (val != 0.0, val)
                elif key == 'output':
                    for name in block_names:
                        if name.startswith('output_'):
                            result[name] = (val != 0.0, val)
                elif key == 'text_encoder_1' or key == 'text_encoder_2' or key == 'unet_mid' or key == 'other_weights':
                    if key in result:
                        result[key] = (val != 0.0, val)
            elif architecture == 'FLUX':
                if key == 'double':
                    for i in range(19):
                        result[f'double_{i}'] = (val != 0.0, val)
                elif key == 'single':
                    for i in range(38):
                        result[f'single_{i}'] = (val != 0.0, val)
                elif key in result:
                    result[key] = (val != 0.0, val)
            elif architecture in ['ZIMAGE', 'WAN', 'QWEN']:
                prefix = 'layer' if architecture == 'ZIMAGE' else 'block'
                count = 30 if architecture == 'ZIMAGE' else (40 if architecture == 'WAN' else 60)
                if key == prefix:
                    for i in range(count):
                        result[f'{prefix}_{i}'] = (val != 0.0, val)
                elif key in result:
                    result[key] = (val != 0.0, val)

    return result


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
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Overall LoRA strength"
                }),
                "preset": (list(SDXL_PRESETS.keys()), {
                    "default": "All Blocks",
                    "tooltip": "Quick preset selection. Choose 'Custom' to use individual toggles below."
                }),
                # Text encoders
                "text_encoder_1": ("BOOLEAN", {"default": True}),
                "text_encoder_1_str": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "text_encoder_2": ("BOOLEAN", {"default": True}),
                "text_encoder_2_str": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                # Input blocks with attention (only 4, 5, 7, 8 have attention in SDXL)
                "input_4": ("BOOLEAN", {"default": True}),
                "input_4_str": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "input_5": ("BOOLEAN", {"default": True}),
                "input_5_str": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "input_7": ("BOOLEAN", {"default": True}),
                "input_7_str": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "input_8": ("BOOLEAN", {"default": True}),
                "input_8_str": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                # Mid block
                "unet_mid": ("BOOLEAN", {"default": True}),
                "unet_mid_str": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                # Output blocks with attention (only 0-5 have attention in SDXL)
                "output_0": ("BOOLEAN", {"default": True}),
                "output_0_str": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "output_1": ("BOOLEAN", {"default": True}),
                "output_1_str": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "output_2": ("BOOLEAN", {"default": True}),
                "output_2_str": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "output_3": ("BOOLEAN", {"default": True}),
                "output_3_str": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "output_4": ("BOOLEAN", {"default": True}),
                "output_4_str": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "output_5": ("BOOLEAN", {"default": True}),
                "output_5_str": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                # Other weights (keys not matching known blocks)
                "other_weights": ("BOOLEAN", {"default": True}),
                "other_weights_str": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "lora_path_opt": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer to use its selected LoRA"}),
                "analysis_json": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer for impact-colored checkboxes"}),
                "block_weights_string": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Input/Output: Block weights in positional (1.0, 0.5, 1.2...) or named format (%default=1.0, te1=0.5, in7-8=1.2). Syncs bidirectionally with UI sliders. String input overrides UI."
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

        # Check if string input is provided (overrides UI)
        parsed_weights = _parse_block_weights_string(block_weights_string, 'SDXL')

        if parsed_weights:
            # Use parsed weights from string
            enabled_blocks = set()
            block_strengths = {}
            for block_name, (enabled, blk_str) in parsed_weights.items():
                if block_name == 'other_weights':
                    other_enabled = enabled
                    other_str = blk_str
                elif enabled:
                    enabled_blocks.add(block_name)
                    block_strengths[block_name] = blk_str
            using_preset = "String Input"
        else:
            # Build block settings: {block_id: (enabled, strength)}
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

            # Use preset or custom toggles
            if preset != "Custom":
                enabled_blocks = SDXL_PRESETS[preset].copy()
                # Presets use strength 1.0 for all enabled blocks
                block_strengths = {b: 1.0 for b in enabled_blocks}
                # All Off preset disables other_weights too
                other_enabled = preset != "All Off"
                other_str = 1.0
                using_preset = preset
            else:
                # Build from individual toggles and strengths
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
            weights_out = "%default=0.0"
            return {"ui": {"analysis_json": [analysis_json or ""]}, "result": (model, clip, "Warning: All blocks disabled, no LoRA applied", weights_out)}

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

        # Generate weights_output string - simple positional format (13 blocks)
        weights_list = []
        for block in all_valid_blocks:
            blk_str = block_strengths.get(block, 0.0) if block in enabled_blocks else 0.0
            weights_list.append(f"{blk_str:.2f}")
        weights_output = ", ".join(weights_list)

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
                    "min": -2.0,
                    "max": 2.0,
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
                "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05
            })

        # Other weights (keys not matching known layers)
        inputs["required"]["other_weights"] = ("BOOLEAN", {"default": True})
        inputs["required"]["other_weights_str"] = ("FLOAT", {
            "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05
        })

        inputs["optional"] = {
            "lora_path_opt": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer to use its selected LoRA"}),
            "analysis_json": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer for impact-colored checkboxes"}),
            "block_weights_string": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "Optional: Named weight syntax (overrides UI). Format: %default=1.0, layer=0.8, layer18-25=1.2, other=1.0"
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

        # Check if string input is provided (overrides UI)
        parsed_weights = _parse_block_weights_string(block_weights_string, 'ZIMAGE')

        if parsed_weights:
            # Use parsed weights from string
            enabled_layers = set()
            layer_strengths = {}
            for block_name, (enabled, blk_str) in parsed_weights.items():
                if block_name == 'other_weights':
                    other_enabled = enabled
                    other_str = blk_str
                elif block_name.startswith('layer_'):
                    layer_num = int(block_name.split('_')[1])
                    if enabled:
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
            weights_out = "%default=0.0"
            return {"ui": {"analysis_json": [analysis_json or ""]}, "result": (model, clip, "Warning: All layers disabled, no LoRA applied", weights_out)}

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

        # Generate weights_output string - simple positional format
        weights_list = []
        for i in range(30):
            lyr_str = layer_strengths.get(i, 0.0) if i in enabled_layers else 0.0
            weights_list.append(f"{lyr_str:.2f}")
        weights_output = ", ".join(weights_list)

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
                    "min": -2.0,
                    "max": 2.0,
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
                "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05
            })

        # Add single block toggles and strengths (0-37)
        for i in range(38):
            inputs["required"][f"single_{i}"] = ("BOOLEAN", {"default": True})
            inputs["required"][f"single_{i}_str"] = ("FLOAT", {
                "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05
            })

        # Other weights (keys not matching known blocks)
        inputs["required"]["other_weights"] = ("BOOLEAN", {"default": True})
        inputs["required"]["other_weights_str"] = ("FLOAT", {
            "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05
        })

        inputs["optional"] = {
            "lora_path_opt": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer to use its selected LoRA"}),
            "analysis_json": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer for impact-colored checkboxes"}),
                "block_weights_string": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional: Named weight syntax (overrides UI). Format: %default=1.0, double=0.8, double7-12=1.2, single=0.5, other=1.0"
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

        # Check if string input is provided (overrides UI)
        parsed_weights = _parse_block_weights_string(block_weights_string, 'FLUX')

        if parsed_weights:
            # Use parsed weights from string
            enabled_blocks = set()
            block_strengths = {}
            for block_name, (enabled, blk_str) in parsed_weights.items():
                if block_name == 'other_weights':
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
            weights_out = "%default=0.0"
            return {"ui": {"analysis_json": [analysis_json or ""]}, "result": (model, clip, "Warning: All blocks disabled, no LoRA applied", weights_out)}

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

        # Generate weights_output string - simple positional format (57 blocks: 19 double + 38 single)
        weights_list = []
        for i in range(19):
            block_id = f"double_{i}"
            blk_str = block_strengths.get(block_id, 0.0) if block_id in enabled_blocks else 0.0
            weights_list.append(f"{blk_str:.2f}")
        for i in range(38):
            block_id = f"single_{i}"
            blk_str = block_strengths.get(block_id, 0.0) if block_id in enabled_blocks else 0.0
            weights_list.append(f"{blk_str:.2f}")
        weights_output = ", ".join(weights_list)

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
                    "min": -2.0,
                    "max": 2.0,
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
                "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05
            })

        # Other weights (keys not matching known blocks)
        inputs["required"]["other_weights"] = ("BOOLEAN", {"default": True})
        inputs["required"]["other_weights_str"] = ("FLOAT", {
            "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05
        })

        inputs["optional"] = {
            "lora_path_opt": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer to use its selected LoRA"}),
            "analysis_json": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer for impact-colored checkboxes"}),
                "block_weights_string": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional: Named weight syntax (overrides UI). Format: %default=1.0, block=0.8, block20-39=1.2, other=1.0"
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

        # Check if string input is provided (overrides UI)
        parsed_weights = _parse_block_weights_string(block_weights_string, 'WAN')

        if parsed_weights:
            # Use parsed weights from string
            enabled_blocks = set()
            block_strengths = {}
            for block_name, (enabled, blk_str) in parsed_weights.items():
                if block_name == 'other_weights':
                    other_enabled = enabled
                    other_str = blk_str
                elif block_name.startswith('block_'):
                    block_num = int(block_name.split('_')[1])
                    if enabled:
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
            weights_out = "%default=0.0"
            return {"ui": {"analysis_json": [analysis_json or ""]}, "result": (model, clip, "Warning: All blocks disabled, no LoRA applied", weights_out)}

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

        # Generate weights_output string - simple positional format
        weights_list = []
        for i in range(40):
            blk_str = block_strengths.get(i, 0.0) if i in enabled_blocks else 0.0
            weights_list.append(f"{blk_str:.2f}")
        weights_output = ", ".join(weights_list)

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
                    "min": -2.0,
                    "max": 2.0,
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
                "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05
            })

        # Other weights (keys not matching known blocks)
        inputs["required"]["other_weights"] = ("BOOLEAN", {"default": True})
        inputs["required"]["other_weights_str"] = ("FLOAT", {
            "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05
        })

        inputs["optional"] = {
            "lora_path_opt": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer to use its selected LoRA"}),
            "analysis_json": ("STRING", {"forceInput": True, "tooltip": "Optional: Connect from LoRA Analyzer for impact-colored checkboxes"}),
                "block_weights_string": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional: Named weight syntax (overrides UI). Format: %default=1.0, block=0.8, block45-59=1.2, other=1.0"
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

        # Check if string input is provided (overrides UI)
        parsed_weights = _parse_block_weights_string(block_weights_string, 'QWEN')

        if parsed_weights:
            # Use parsed weights from string
            enabled_blocks = set()
            block_strengths = {}
            for block_name, (enabled, blk_str) in parsed_weights.items():
                if block_name == 'other_weights':
                    other_enabled = enabled
                    other_str = blk_str
                elif block_name.startswith('block_'):
                    block_num = int(block_name.split('_')[1])
                    if enabled:
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
            weights_out = "%default=0.0"
            return {"ui": {"analysis_json": [analysis_json or ""]}, "result": (model, clip, "Warning: All blocks disabled, no LoRA applied", weights_out)}

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

        # Generate weights_output string - simple positional format
        weights_list = []
        for i in range(60):
            blk_str = block_strengths.get(i, 0.0) if i in enabled_blocks else 0.0
            weights_list.append(f"{blk_str:.2f}")
        weights_output = ", ".join(weights_list)

        return {"ui": {"analysis_json": [analysis_json or ""]}, "result": (model_lora, clip_lora, info, weights_output)}


NODE_CLASS_MAPPINGS = {
    "SDXLSelectiveLoRALoader": SDXLSelectiveLoRALoader,
    "ZImageSelectiveLoRALoader": ZImageSelectiveLoRALoader,
    "FLUXSelectiveLoRALoader": FLUXSelectiveLoRALoader,
    "WanSelectiveLoRALoader": WanSelectiveLoRALoader,
    "QwenSelectiveLoRALoader": QwenSelectiveLoRALoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXLSelectiveLoRALoader": "Selective LoRA Loader (SDXL)",
    "ZImageSelectiveLoRALoader": "Selective LoRA Loader (Z-Image)",
    "FLUXSelectiveLoRALoader": "Selective LoRA Loader (FLUX)",
    "WanSelectiveLoRALoader": "Selective LoRA Loader (Wan)",
    "QwenSelectiveLoRALoader": "Selective LoRA Loader (Qwen)",
}
