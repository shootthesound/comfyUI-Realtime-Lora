"""
LoRA Loader with Analysis V2 for ComfyUI

Features:
- Improved architecture detection using metadata, scoring, and block counting
- Combined analyzer + selective loader in one node
- Strength shaping (scheduling) via ComfyUI hooks
- Per-block toggles and strength sliders with impact coloring

Strength Schedule Format: "0:.2,.2:.3,.5:.6,1:.9"
- Each pair is "step_percent:strength"
- Values interpolate linearly between keyframes
"""

import os
import re
import json
from collections import defaultdict
from datetime import datetime
import threading

import torch
import folder_paths
import comfy.sd
import comfy.model_patcher
import comfy.hooks
import comfy.utils
from safetensors.torch import load_file, save_file
from safetensors import safe_open

# Path to store per-node config (last used save paths)
_NODE_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_SAVE_PATHS_CONFIG = os.path.join(_NODE_CONFIG_DIR, ".v2_save_paths.json")

# ============================================================================
# STRENGTH SCHEDULE PRESETS
# ============================================================================
# Format: "step:strength" pairs, comma-separated. Steps are 0-1 (percent of generation).
# These presets are shown in dropdown, value populates the editable text field.

SCHEDULE_PRESETS = {
    # === No Schedule ===
    "Custom": "",
    "Constant 1.0 (No Change)": "0:1, 1:1",

    # === Basic Fades ===
    "Linear In (0→1)": "0:0, 1:1",
    "Linear Out (1→0)": "0:1, 1:0",

    # === Ease Curves ===
    "Ease In (slow start)": "0:0, 0.3:0.1, 0.7:0.5, 1:1",
    "Ease Out (slow end)": "0:1, 0.3:0.5, 0.7:0.9, 1:0",
    "Ease In-Out": "0:0, 0.3:0.1, 0.7:0.9, 1:1",

    # === Bell Curves ===
    "Bell Curve (peak middle)": "0:0, 0.5:1, 1:0",
    "Wide Bell": "0:0, 0.3:0.8, 0.7:0.8, 1:0",
    "Sharp Bell": "0:0, 0.4:0.2, 0.5:1, 0.6:0.2, 1:0",

    # === Structure LoRA Favorites ===
    "High Start → Cut Low": "0:1, 0.3:1, 0.35:0.2, 1:0.2",
    "High Start → Cut Off": "0:1, 0.3:1, 0.35:0, 1:0",
    "High Early → Fade": "0:1, 0.2:0.8, 0.5:0.3, 1:0",

    # === Detail LoRA Favorites ===
    "Low Start → Ramp Late": "0:0, 0.6:0.1, 0.8:0.7, 1:1",
    "Off → Kick In Late": "0:0, 0.7:0, 0.75:0.8, 1:1",
    "Low → Boost End": "0:0.2, 0.5:0.2, 0.7:0.6, 1:1",

    # === Step Functions ===
    "Step Up Mid": "0:0.2, 0.45:0.2, 0.55:0.8, 1:0.8",
    "Step Down Mid": "0:0.8, 0.45:0.8, 0.55:0.2, 1:0.2",
    "Two Steps Up": "0:0, 0.3:0, 0.35:0.5, 0.65:0.5, 0.7:1, 1:1",

    # === Pulses ===
    "Pulse Early": "0:1, 0.25:1, 0.35:0.2, 1:0.2",
    "Pulse Mid": "0:0.2, 0.4:0.2, 0.45:1, 0.55:1, 0.6:0.2, 1:0.2",
    "Pulse Late": "0:0.2, 0.65:0.2, 0.75:1, 1:1",

    # === Constant (for testing) ===
    "Constant Half": "0:0.5, 1:0.5",
    "Constant Low": "0:0.3, 1:0.3",

    # =========================================================================
    # INVERTED VERSIONS (same order as above)
    # =========================================================================

    # === Basic Fades (Inverted) ===
    "INV: Linear In (1→0)": "0:1, 1:0",
    "INV: Linear Out (0→1)": "0:0, 1:1",

    # === Ease Curves (Inverted) ===
    "INV: Ease In": "0:1, 0.3:0.9, 0.7:0.5, 1:0",
    "INV: Ease Out": "0:0, 0.3:0.5, 0.7:0.1, 1:1",
    "INV: Ease In-Out": "0:1, 0.3:0.9, 0.7:0.1, 1:0",

    # === Bell Curves (Inverted) ===
    "INV: Bell (dip middle)": "0:1, 0.5:0, 1:1",
    "INV: Wide Bell": "0:1, 0.3:0.2, 0.7:0.2, 1:1",
    "INV: Sharp Bell": "0:1, 0.4:0.8, 0.5:0, 0.6:0.8, 1:1",

    # === Structure Inverted ===
    "INV: Low Start → Boost High": "0:0, 0.3:0, 0.35:0.8, 1:0.8",
    "INV: Low Start → Full On": "0:0, 0.3:0, 0.35:1, 1:1",
    "INV: Low Early → Build": "0:0, 0.2:0.2, 0.5:0.7, 1:1",

    # === Detail Inverted ===
    "INV: High Start → Drop Late": "0:1, 0.6:0.9, 0.8:0.3, 1:0",
    "INV: Full → Cut Off Late": "0:1, 0.7:1, 0.75:0.2, 1:0",
    "INV: High → Drop End": "0:0.8, 0.5:0.8, 0.7:0.4, 1:0",

    # === Step Functions (Inverted) ===
    "INV: Step Down Mid": "0:0.8, 0.45:0.8, 0.55:0.2, 1:0.2",
    "INV: Step Up Mid": "0:0.2, 0.45:0.2, 0.55:0.8, 1:0.8",
    "INV: Two Steps Down": "0:1, 0.3:1, 0.35:0.5, 0.65:0.5, 0.7:0, 1:0",

    # === Pulses (Inverted) ===
    "INV: Dip Early": "0:0, 0.25:0, 0.35:0.8, 1:0.8",
    "INV: Dip Mid": "0:0.8, 0.4:0.8, 0.45:0, 0.55:0, 0.6:0.8, 1:0.8",
    "INV: Dip Late": "0:0.8, 0.65:0.8, 0.75:0, 1:0",
}

# List for dropdown (maintains order)
SCHEDULE_PRESET_LIST = list(SCHEDULE_PRESETS.keys())


def _load_save_paths_config() -> dict:
    """Load saved paths config from JSON file."""
    if os.path.exists(_SAVE_PATHS_CONFIG):
        try:
            with open(_SAVE_PATHS_CONFIG, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_save_paths_config(config: dict):
    """Save paths config to JSON file."""
    try:
        with open(_SAVE_PATHS_CONFIG, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"[LoRA V2] Warning: Could not save config: {e}")


def _save_refined_lora(filtered_lora: dict, save_path: str, save_filename: str,
                       node_id: str, architecture: str = None, original_metadata: dict = None) -> str:
    """
    Save a refined LoRA to disk.

    Args:
        filtered_lora: The filtered/scaled LoRA state dict
        save_path: Directory to save to
        save_filename: Base filename (timestamp will be appended)
        node_id: Node type ID for default filename
        architecture: Model architecture (ZIMAGE, SDXL, FLUX, etc.)
        original_metadata: Metadata from original LoRA to preserve

    Returns:
        Full path to saved file, or None if not saved
    """
    if not save_path or not save_path.strip():
        return None

    save_path = os.path.expanduser(save_path.strip())  # Support ~/paths on Linux/Mac

    # Ensure directory exists
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path, exist_ok=True)
        except Exception as e:
            print(f"[LoRA V2] Error creating directory {save_path}: {e}")
            return None

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_filename and save_filename.strip():
        base_name = save_filename.strip()
        # Remove .safetensors extension if user added it
        if base_name.lower().endswith('.safetensors'):
            base_name = base_name[:-12]
        filename = f"{base_name}_{timestamp}.safetensors"
    else:
        # Use node ID as default name
        filename = f"{node_id}_{timestamp}.safetensors"

    full_path = os.path.join(save_path, filename)

    # Prepare metadata
    metadata = {}
    if original_metadata:
        # Copy relevant metadata from original
        for key in ['ss_base_model_version', 'ss_network_module', 'ss_network_dim',
                    'ss_network_alpha', 'modelspec.architecture', 'modelspec.title']:
            if key in original_metadata:
                metadata[key] = original_metadata[key]

    # Add our own metadata
    metadata['refined_by'] = 'comfyui-zimage-realtime-lora V2'
    metadata['refined_url'] = 'https://github.com/ShootTheSound/comfyUI-Realtime-Lora'
    metadata['refined_date'] = datetime.now().isoformat()
    metadata['refined_node'] = node_id
    if architecture:
        metadata['refined_architecture'] = architecture

    try:
        save_file(filtered_lora, full_path, metadata=metadata)
        print(f"[LoRA V2] Saved refined LoRA: {full_path}")
        return full_path
    except Exception as e:
        print(f"[LoRA V2] Error saving LoRA to {full_path}: {e}")
        return None


# ============================================================================
# STRENGTH SCHEDULE PARSING
# ============================================================================

def _parse_strength_schedule(schedule_str: str) -> list:
    """
    Parse a strength schedule string into keyframes.

    Format: "0:.2,.2:.3,.5:.6,1:.9"
    Each pair is "percent:strength" where:
    - percent is 0.0 to 1.0 (proportion of steps)
    - strength is the LoRA strength at that point

    Values are interpolated linearly between keyframes.

    Returns list of (percent, strength) tuples, sorted by percent.
    """
    if not schedule_str or not schedule_str.strip():
        return None

    schedule_str = schedule_str.strip()

    # Check if it's just a float (no schedule)
    try:
        val = float(schedule_str)
        return None  # Just a regular float, no schedule
    except ValueError:
        pass

    keyframes = []
    pairs = schedule_str.split(',')

    for pair in pairs:
        pair = pair.strip()
        if ':' not in pair:
            continue

        parts = pair.split(':')
        if len(parts) != 2:
            continue

        try:
            percent = float(parts[0].strip())
            strength = float(parts[1].strip())

            # Clamp percent to 0-1
            percent = max(0.0, min(1.0, percent))

            keyframes.append((percent, strength))
        except ValueError:
            continue

    if not keyframes:
        return None

    # Sort by percent
    keyframes.sort(key=lambda x: x[0])

    return keyframes


def _create_hook_keyframes_interpolated(keyframes: list, num_keyframes: int = 20) -> comfy.hooks.HookKeyframeGroup:
    """
    Create hook keyframes with linear interpolation between user-defined points.

    keyframes: list of (percent, strength) tuples
    num_keyframes: number of keyframes to generate for smooth interpolation
    """
    if not keyframes:
        return None

    kf_group = comfy.hooks.HookKeyframeGroup()

    # Generate interpolated keyframes
    for i in range(num_keyframes + 1):
        percent = i / num_keyframes

        # Find surrounding keyframes for interpolation
        strength = _interpolate_strength(keyframes, percent)

        guarantee_steps = 1 if i == 0 else 0
        kf = comfy.hooks.HookKeyframe(strength=strength, start_percent=percent, guarantee_steps=guarantee_steps)
        kf_group.add(kf)

    return kf_group


def _interpolate_strength(keyframes: list, percent: float) -> float:
    """Linearly interpolate strength at a given percent."""
    if not keyframes:
        return 1.0

    # Before first keyframe
    if percent <= keyframes[0][0]:
        return keyframes[0][1]

    # After last keyframe
    if percent >= keyframes[-1][0]:
        return keyframes[-1][1]

    # Find surrounding keyframes
    for i in range(len(keyframes) - 1):
        p1, s1 = keyframes[i]
        p2, s2 = keyframes[i + 1]

        if p1 <= percent <= p2:
            # Linear interpolation
            if p2 == p1:
                return s1
            t = (percent - p1) / (p2 - p1)
            return s1 + t * (s2 - s1)

    return keyframes[-1][1]


def _load_lora_with_schedule(model, clip, lora_path: str, strength_model: float, strength_clip: float, schedule_str: str = None):
    """
    Load LoRA with optional strength scheduling.

    If schedule_str is provided and valid, returns (model, clip, hooks, is_scheduled).
    If no schedule, returns (model_lora, clip_lora, None, False).
    """
    # Load LoRA file
    if lora_path.endswith('.safetensors'):
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    else:
        lora = torch.load(lora_path, map_location='cpu')

    # Check for schedule
    schedule = _parse_strength_schedule(schedule_str) if schedule_str else None

    if schedule:
        # Use hook system for scheduling
        print(f"[LoRA V2] Using strength schedule: {schedule}")

        # Create hooks
        hooks = comfy.hooks.create_hook_lora(lora=lora, strength_model=strength_model, strength_clip=strength_clip)

        # Create keyframes
        kf_group = _create_hook_keyframes_interpolated(schedule)

        # Apply keyframes to all hooks in group
        if kf_group and hooks:
            for hook in hooks.hooks:
                hook.hook_keyframe = kf_group

        return (model, clip, hooks, True)
    else:
        # Standard LoRA loading
        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )
        return (model_lora, clip_lora, None, False)


# ============================================================================
# V2 IMPROVED ARCHITECTURE DETECTION
# ============================================================================

def _get_metadata(lora_path: str) -> dict:
    """Extract metadata from safetensors file."""
    if not lora_path.endswith('.safetensors'):
        return {}
    try:
        with safe_open(lora_path, framework="pt") as f:
            metadata = f.metadata()
            return metadata if metadata else {}
    except Exception as e:
        print(f"[LoRA Analyzer V2] Could not read metadata: {e}")
        return {}


def _detect_from_metadata(metadata: dict) -> str:
    """Try to detect architecture from safetensors metadata."""
    if not metadata:
        return None

    # Check modelspec.architecture (newer LoRAs)
    arch = metadata.get('modelspec.architecture', '').lower()
    if 'flux' in arch:
        return 'FLUX'
    if 'sdxl' in arch:
        return 'SDXL'
    if 'sd1' in arch or 'sd15' in arch:
        return 'SD15'

    # Check ss_base_model_version (Kohya format)
    base_model = metadata.get('ss_base_model_version', '').lower()
    if 'sdxl' in base_model:
        return 'SDXL'
    if 'sd_v1' in base_model or 'sd1' in base_model:
        return 'SD15'
    if 'flux' in base_model:
        return 'FLUX'

    # Check ss_network_module (Kohya format)
    network_module = metadata.get('ss_network_module', '').lower()
    if 'flux' in network_module:
        return 'FLUX'
    if 'zimage' in network_module or 'z_image' in network_module:
        return 'ZIMAGE'
    if 'wan' in network_module:
        return 'WAN'
    if 'qwen' in network_module:
        return 'QWEN_IMAGE'

    # Check ss_sd_model_name for hints
    model_name = metadata.get('ss_sd_model_name', '').lower()
    if 'flux' in model_name:
        return 'FLUX'
    if 'sdxl' in model_name or 'xl' in model_name:
        return 'SDXL'
    if 'z-image' in model_name or 'zimage' in model_name:
        return 'ZIMAGE'

    return None


def _count_unique_blocks(keys: list) -> dict:
    """Count unique block numbers for each potential architecture."""
    counts = {
        'zimage_layers': set(),
        'flux_double': set(),
        'flux_single': set(),
        'wan_blocks': set(),
        'qwen_blocks': set(),
        'sdxl_blocks': set(),
        'sd15_blocks': set(),
    }

    for key in keys:
        key_lower = key.lower()

        # Z-Image layers (0-29)
        match = re.search(r'diffusion_model\.layers\.(\d+)', key)
        if match:
            counts['zimage_layers'].add(int(match.group(1)))
        match = re.search(r'lora_unet_layers_(\d+)_', key)
        if match:
            counts['zimage_layers'].add(int(match.group(1)))
        # LyCORIS/LoKR format: lycoris_layers_N_...
        match = re.search(r'lycoris_layers_(\d+)_', key)
        if match:
            counts['zimage_layers'].add(int(match.group(1)))

        # FLUX double blocks (0-18) and single blocks (0-37)
        # Standard format
        match = re.search(r'double_blocks?[._]?(\d+)', key_lower)
        if match:
            counts['flux_double'].add(int(match.group(1)))
        # AI-Toolkit format: transformer.transformer_blocks.N (double blocks)
        if 'single_transformer_blocks' not in key_lower:
            match = re.search(r'transformer\.transformer_blocks[._]?(\d+)', key_lower)
            if match:
                counts['flux_double'].add(int(match.group(1)))
        # Kohya format: transformer_double_blocks_N (double blocks)
        match = re.search(r'transformer_double_blocks[._]?(\d+)', key_lower)
        if match:
            counts['flux_double'].add(int(match.group(1)))
        # Single blocks - handles all formats including transformer_single_transformer_blocks_N
        match = re.search(r'single_transformer_blocks[._]?(\d+)', key_lower)
        if match:
            counts['flux_single'].add(int(match.group(1)))
        match = re.search(r'(?<!transformer_)single_blocks[._]?(\d+)', key_lower)
        if match:
            counts['flux_single'].add(int(match.group(1)))

        # Wan blocks (0-39)
        if any(x in key_lower for x in ['self_attn', 'cross_attn', 'ffn']):
            match = re.search(r'blocks[._](\d+)', key_lower)
            if match:
                counts['wan_blocks'].add(int(match.group(1)))

        # Qwen blocks (0-59)
        if any(x in key_lower for x in ['img_mlp', 'txt_mlp', 'img_mod', 'txt_mod']):
            match = re.search(r'transformer_blocks[._](\d+)', key)
            if match:
                counts['qwen_blocks'].add(int(match.group(1)))

        # SDXL/SD15 blocks
        match = re.search(r'input_blocks?[._]?(\d+)', key_lower)
        if match:
            block_num = int(match.group(1))
            if block_num >= 7:
                counts['sdxl_blocks'].add(block_num)
            counts['sd15_blocks'].add(block_num)

    return {k: len(v) for k, v in counts.items()}


def _score_architecture(keys: list, num_keys: int, block_counts: dict) -> dict:
    """Score each architecture based on multiple signals."""
    scores = {
        'QWEN_IMAGE': 0,
        'FLUX': 0,
        'ZIMAGE': 0,
        'WAN': 0,
        'SDXL': 0,
        'SD15': 0,
    }

    keys_lower = [k.lower() for k in keys]
    keys_str = ' '.join(keys_lower)

    # === QWEN_IMAGE scoring ===
    if any('transformer_blocks' in k and any(x in k for x in ['img_mlp', 'txt_mlp', 'img_mod', 'txt_mod']) for k in keys_lower):
        scores['QWEN_IMAGE'] += 50
    if block_counts['qwen_blocks'] >= 50:  # Expect ~60 blocks
        scores['QWEN_IMAGE'] += 30
    elif block_counts['qwen_blocks'] >= 30:
        scores['QWEN_IMAGE'] += 15

    # === FLUX scoring ===
    # AI-Toolkit format with transformer. prefix
    if any('transformer.single_transformer_blocks' in k or 'transformer.double_blocks' in k for k in keys_lower):
        scores['FLUX'] += 50
    # AI-Toolkit alternate format: transformer.transformer_blocks (double blocks)
    if any('transformer.transformer_blocks' in k and 'single_transformer_blocks' not in k for k in keys_lower):
        scores['FLUX'] += 40
    # Kohya/other format: lora_transformer_single_transformer_blocks / lora_transformer_double_blocks (underscores)
    if any('transformer_single_transformer_blocks' in k or 'transformer_double_blocks' in k for k in keys_lower):
        scores['FLUX'] += 45
    # Standard double_blocks/single_blocks
    if any('double_blocks' in k for k in keys_lower):
        scores['FLUX'] += 25
    if any('single_blocks' in k and 'transformer_blocks' not in k for k in keys_lower):
        scores['FLUX'] += 20
    # Block count check (19 double + up to 38 single = 57 total possible)
    if block_counts['flux_double'] >= 15:
        scores['FLUX'] += 20
    if block_counts['flux_single'] >= 30:
        scores['FLUX'] += 15

    # === Z-IMAGE scoring ===
    # ComfyUI/AI-Toolkit format
    if any('diffusion_model.layers.' in k and ('attention' in k or 'adaln' in k.lower()) for k in keys_lower):
        scores['ZIMAGE'] += 50
    # Musubi Tuner format
    if any('lora_unet_layers_' in k and 'attention' in k for k in keys_lower):
        scores['ZIMAGE'] += 50
    # Old format (but NOT with transformer. prefix - that's FLUX)
    if any('single_transformer_blocks' in k and 'transformer.single_transformer_blocks' not in k for k in keys_lower):
        scores['ZIMAGE'] += 30
    # Block count check (exactly 30 layers)
    if block_counts['zimage_layers'] >= 25:
        scores['ZIMAGE'] += 25
    if block_counts['zimage_layers'] == 30:
        scores['ZIMAGE'] += 15  # Bonus for exact match

    # === WAN scoring ===
    if any(('blocks.' in k or 'blocks_' in k) and any(x in k for x in ['self_attn', 'cross_attn', 'ffn']) for k in keys_lower):
        scores['WAN'] += 50
    if block_counts['wan_blocks'] >= 35:  # Expect ~40 blocks
        scores['WAN'] += 25
    elif block_counts['wan_blocks'] >= 20:
        scores['WAN'] += 10

    # === SDXL scoring ===
    has_te1 = 'lora_te1_' in keys_str or 'text_encoder_1' in keys_str
    has_te2 = 'lora_te2_' in keys_str or 'text_encoder_2' in keys_str
    if has_te1 and has_te2:
        scores['SDXL'] += 50
    if any('input_blocks_7' in k or 'input_blocks_8' in k or 'input_blocks.7' in k or 'input_blocks.8' in k for k in keys_lower):
        scores['SDXL'] += 30
    if num_keys > 1500:
        scores['SDXL'] += 25
    elif num_keys > 1000:
        scores['SDXL'] += 10

    # === SD15 scoring ===
    if any('lora_unet_' in k for k in keys_lower) and not any('lora_unet_layers_' in k for k in keys_lower):
        scores['SD15'] += 30
    if any('lora_te_' in k and 'lora_te1_' not in k and 'lora_te2_' not in k for k in keys_lower):
        scores['SD15'] += 20
    if any('input_blocks' in k for k in keys_lower) and num_keys < 1000:
        scores['SD15'] += 20
    if 600 <= num_keys <= 900:
        scores['SD15'] += 15

    # Penalize SD15 if we have strong signals for other architectures
    if scores['ZIMAGE'] > 40:
        scores['SD15'] -= 30
    if scores['FLUX'] > 40:
        scores['SD15'] -= 30

    return scores


def _detect_lora_type(keys: list) -> str:
    """
    Detect the LoRA training method (standard LoRA, LoKR, LoHa, etc.)
    This is SEPARATE from architecture detection - a LoKR can be trained for any architecture.

    Returns: 'LoRA', 'LoKR', 'LoHa', or 'GLoRA'
    """
    for key in keys:
        key_lower = key.lower()
        # LoKR detection: lokr_w1, lokr_w2, or decomposed forms (lokr_w1_a, lokr_w1_b)
        if '.lokr_w1' in key_lower or '.lokr_w1_a' in key_lower:
            return 'LoKR'
        # LoHa detection: hada_w1_a, hada_w1_b, hada_w2_a, hada_w2_b
        if '.hada_w1_a' in key_lower or '.hada_w2_a' in key_lower:
            return 'LoHa'
        # GLoRA detection: glora_a, glora_b
        if '.glora_a' in key_lower or '.glora_b' in key_lower:
            return 'GLoRA'
    # Default: standard LoRA (uses lora_up/lora_down or lora_A/lora_B)
    return 'LoRA'


def _detect_architecture_v2(keys: list, metadata: dict = None) -> tuple:
    """
    V2 architecture detection using metadata, scoring, and block counting.
    Returns (architecture, confidence, method).
    """
    num_keys = len(keys)

    # Method 1: Try metadata first (most reliable)
    if metadata:
        arch = _detect_from_metadata(metadata)
        if arch:
            return (arch, 'high', 'metadata')

    # Method 2: Count unique blocks
    block_counts = _count_unique_blocks(keys)

    # Method 3: Score each architecture
    scores = _score_architecture(keys, num_keys, block_counts)

    # Find highest scoring architecture
    best_arch = max(scores, key=scores.get)
    best_score = scores[best_arch]

    # Determine confidence
    if best_score >= 50:
        confidence = 'high'
    elif best_score >= 30:
        confidence = 'medium'
    else:
        confidence = 'low'

    # If confidence is low, fall back to UNKNOWN
    if best_score < 20:
        return ('UNKNOWN', 'low', 'fallback')

    return (best_arch, confidence, 'scoring')


# ============================================================================
# BLOCK EXTRACTION (same as V1 but with improvements)
# ============================================================================

def _extract_block_id_v2(key: str, architecture: str) -> str:
    """Extract block identifier from a LoRA/model weight key."""
    key_lower = key.lower()

    if architecture == 'QWEN_IMAGE':
        match = re.search(r'transformer_blocks[._](\d+)', key)
        return f"block_{match.group(1)}" if match else 'other'

    elif architecture == 'ZIMAGE':
        # AI-Toolkit format: diffusion_model.layers.N.attention/adaLN_modulation
        match = re.search(r'diffusion_model\.layers\.(\d+)', key)
        if match:
            return f"layer_{match.group(1)}"
        # Musubi Tuner format: lora_unet_layers_N_attention_...
        match = re.search(r'lora_unet_layers_(\d+)_', key)
        if match:
            return f"layer_{match.group(1)}"
        # LyCORIS/LoKR format: lycoris_layers_N_...
        match = re.search(r'lycoris_layers_(\d+)_', key)
        if match:
            return f"layer_{match.group(1)}"
        # Old format: single_transformer_blocks.N
        match = re.search(r'single_transformer_blocks\.(\d+)', key)
        if match:
            return f"block_{match.group(1)}"
        # Context refiner (modal alignment blocks)
        if 'context_refiner' in key_lower:
            return 'context_refiner'
        # Noise refiner
        if 'noise_refiner' in key_lower:
            return 'noise_refiner'
        # Final layer
        if 'all_final_layer' in key_lower or 'final_layer' in key_lower:
            return 'final_layer'
        # X embedder
        if 'all_x_embedder' in key_lower or 'x_embedder' in key_lower:
            return 'x_embedder'
        return 'other'

    elif architecture == 'WAN':
        match = re.search(r'blocks[._](\d+)', key)
        return f"block_{match.group(1)}" if match else 'other'

    elif architecture == 'FLUX':
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

    elif architecture in ['SDXL', 'SD15']:
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

    return 'other'


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def _analyze_patches_v2(model_patcher, architecture: str) -> dict:
    """Analyze the patches stored in the ModelPatcher."""
    block_analysis = defaultdict(lambda: {
        'patch_count': 0,
        'total_strength': 0.0,
        'total_norm': 0.0,
        'keys': []
    })

    if not hasattr(model_patcher, 'patches'):
        return dict(block_analysis)

    for weight_key, patch_list in model_patcher.patches.items():
        block_id = _extract_block_id_v2(weight_key, architecture)

        for patch_tuple in patch_list:
            if len(patch_tuple) < 3:
                continue

            strength = patch_tuple[0]
            patch_data = patch_tuple[1]
            strength_model = patch_tuple[2]

            effective_strength = abs(strength * strength_model)
            block_analysis[block_id]['patch_count'] += 1
            block_analysis[block_id]['total_strength'] += effective_strength
            block_analysis[block_id]['keys'].append(weight_key)

            norm_value = 0.0

            try:
                if hasattr(patch_data, 'weights') and isinstance(patch_data.weights, tuple):
                    weights = patch_data.weights
                    if len(weights) >= 2:
                        lora_up = weights[0]
                        lora_down = weights[1]
                        if hasattr(lora_up, 'norm') and hasattr(lora_down, 'norm'):
                            up_norm = lora_up.float().norm().item()
                            down_norm = lora_down.float().norm().item()
                            norm_value = up_norm * down_norm

                elif hasattr(patch_data, 'lora_up') and hasattr(patch_data, 'lora_down'):
                    lora_up = patch_data.lora_up
                    lora_down = patch_data.lora_down
                    if lora_up is not None and lora_down is not None:
                        up_norm = lora_up.float().norm().item()
                        down_norm = lora_down.float().norm().item()
                        norm_value = up_norm * down_norm

                elif isinstance(patch_data, tuple) and len(patch_data) >= 2:
                    patch_type = patch_data[0]
                    patch_content = patch_data[1]

                    if patch_type == "lora" and isinstance(patch_content, tuple) and len(patch_content) >= 2:
                        lora_up = patch_content[0]
                        lora_down = patch_content[1]
                        if hasattr(lora_up, 'norm') and hasattr(lora_down, 'norm'):
                            up_norm = lora_up.float().norm().item()
                            down_norm = lora_down.float().norm().item()
                            norm_value = up_norm * down_norm

                elif hasattr(patch_data, 'norm'):
                    norm_value = patch_data.float().norm().item()

            except Exception:
                pass

            block_analysis[block_id]['total_norm'] += norm_value * effective_strength

    return dict(block_analysis)


def _format_patch_analysis_v2(block_analysis: dict, architecture: str, confidence: str, method: str, lora_type: str = 'LoRA') -> str:
    """Format patch analysis as readable text."""
    if not block_analysis:
        return "No patches found."

    max_norm = max((d['total_norm'] for d in block_analysis.values()), default=1.0)
    if max_norm == 0:
        max_norm = 1.0

    # Show LoRA type if not standard LoRA
    arch_display = f"{architecture} ({lora_type})" if lora_type != 'LoRA' else architecture

    lines = [
        f"LoRA Patch Analysis V2 ({arch_display})",
        f"Detection: {confidence} confidence via {method}",
        "=" * 60,
        f"{'Block':<25} {'Score':>8} {'Patches':>10} {'Strength':>10}",
        "-" * 60
    ]

    sorted_blocks = sorted(
        block_analysis.items(),
        key=lambda x: x[1]['total_norm'],
        reverse=True
    )

    for block_id, data in sorted_blocks:
        score = (data['total_norm'] / max_norm) * 100
        bar_len = int(score / 5)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        lines.append(f"{block_id:<25} [{bar}] {score:5.1f}  ({data['patch_count']:>3})  {data['total_strength']:>8.3f}")

    lines.append("-" * 60)
    lines.append(f"Total patched layers: {sum(d['patch_count'] for d in block_analysis.values())}")

    return '\n'.join(lines)


def _create_analysis_json_v2(block_analysis: dict, architecture: str, lora_name: str, confidence: str, method: str, metadata: dict, lora_type: str = 'LoRA') -> str:
    """Create JSON analysis output for use by selective loaders."""
    if not block_analysis:
        return json.dumps({"architecture": architecture, "lora_name": lora_name, "lora_type": lora_type, "blocks": {}})

    max_norm = max((d['total_norm'] for d in block_analysis.values()), default=1.0)
    if max_norm == 0:
        max_norm = 1.0

    blocks = {}
    for block_id, data in block_analysis.items():
        score = (data['total_norm'] / max_norm) * 100
        blocks[block_id] = {
            "score": round(score, 1),
            "patch_count": data['patch_count'],
            "strength": round(data['total_strength'], 4)
        }

    result = {
        "architecture": architecture,
        "lora_type": lora_type,
        "lora_name": lora_name,
        "detection_confidence": confidence,
        "detection_method": method,
        "blocks": blocks
    }

    # Include useful metadata if present
    if metadata:
        meta_info = {}
        for key in ['ss_base_model_version', 'ss_network_module', 'ss_output_name', 'modelspec.architecture', 'modelspec.title']:
            if key in metadata:
                meta_info[key] = metadata[key]
        if meta_info:
            result['metadata'] = meta_info

    return json.dumps(result)


# ============================================================================
# NODE CLASS
# ============================================================================

class LoRALoaderWithAnalysisV2:
    """
    V2 LoRA Loader with improved architecture detection.

    Uses multiple detection methods:
    1. Safetensors metadata (most reliable)
    2. Block counting (validates expected structure)
    3. Scoring system (weighs multiple signals)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "LoRA file to load and analyze"
                }),
                "strength_model": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "LoRA strength for model (UNet/DiT)"
                }),
                "strength_clip": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "LoRA strength for CLIP text encoder"
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "analysis", "analysis_json", "lora_path")
    OUTPUT_TOOLTIPS = (
        "Model with LoRA applied.",
        "CLIP with LoRA applied.",
        "Per-block patch analysis with detection confidence.",
        "JSON analysis data. Connect to Selective LoRA Loader for impact-colored UI.",
        "Full path to the loaded LoRA file. Connect to Selective LoRA Loader."
    )
    FUNCTION = "load_lora_with_analysis"
    CATEGORY = "loaders/lora"
    OUTPUT_NODE = True
    DESCRIPTION = "V2 analyzer with improved detection using metadata, scoring, and block counting."

    def load_lora_with_analysis(self, model, clip, lora_name, strength_model, strength_clip):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.exists(lora_path):
            return (model, clip, "Error: LoRA file not found", "{}", "")

        print(f"[LoRA Analyzer V2] Loading: {lora_name}")

        # Get metadata first
        metadata = _get_metadata(lora_path)
        if metadata:
            print(f"[LoRA Analyzer V2] Metadata found: {list(metadata.keys())}")

        # Load LoRA state dict
        if lora_path.endswith('.safetensors'):
            lora_state_dict = load_file(lora_path)
        else:
            lora_state_dict = torch.load(lora_path, map_location='cpu')

        lora_keys = list(lora_state_dict.keys())

        # V2 detection
        architecture, confidence, method = _detect_architecture_v2(lora_keys, metadata)
        lora_type = _detect_lora_type(lora_keys)
        arch_display = f"{architecture} ({lora_type})" if lora_type != 'LoRA' else architecture
        print(f"[LoRA Analyzer V2] Architecture: {arch_display} ({confidence} confidence via {method})")
        print(f"[LoRA Analyzer V2] Tensors: {len(lora_state_dict)}")
        print(f"[LoRA Analyzer V2] Sample keys: {lora_keys[:5]}")

        # Load the LoRA using ComfyUI's standard method
        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model,
            clip,
            lora_state_dict,
            strength_model,
            strength_clip
        )

        # Analyze the patches
        patch_analysis = _analyze_patches_v2(model_lora, architecture)
        analysis_text = _format_patch_analysis_v2(patch_analysis, architecture, confidence, method, lora_type)
        analysis_json = _create_analysis_json_v2(patch_analysis, architecture, lora_name, confidence, method, metadata, lora_type)

        print(f"[LoRA Analyzer V2] Found {len(patch_analysis)} blocks with patches")
        print(analysis_text)

        return (model_lora, clip_lora, analysis_text, analysis_json, lora_path)


# ============================================================================
# ARCHITECTURE CONFIGS FOR COMBINED ANALYZER + SELECTIVE LOADER
# ============================================================================
# Each config defines the blocks, presets, and filtering logic for an architecture.
# Adding a new architecture just requires adding a new config dict here.

ARCH_CONFIGS = {
    "ZIMAGE": {
        "node_id": "ZImageAnalyzerSelectiveLoaderV2",
        "display_name": "Z-Image Analyzer + Selective Loader V2",
        "description": """Combined analyzer and selective loader for Z-Image LoRAs.
Analyzes block impact and allows per-layer control with strength shaping.

Layer Guide:
- Layers 0-9: Early processing (usually low impact, ~7-25%)
- Layers 10-19: Mid processing (moderate impact, ~25-70%)
- Layers 20-29: Late processing (usually highest impact, ~70-100%)

Supports strength scheduling format: 0:.2,.5:.8,1:1.0""",
        "architecture": "ZIMAGE",
        "blocks": [f"layer_{i}" for i in range(30)] + ["context_refiner", "noise_refiner", "final_layer", "x_embedder", "other_weights"],
        "block_labels": {f"layer_{i}": f"Layer {i}" for i in range(30)} | {"context_refiner": "Context Refiner", "noise_refiner": "Noise Refiner", "final_layer": "Final Layer", "x_embedder": "X Embedder", "other_weights": "Other Weights"},
        "presets": {
            "Default": {"enabled": "ALL", "strength": 1.0},
            "All Off": {"enabled": [], "strength": 0.0},
            "Half Strength": {"enabled": "ALL", "strength": 0.5},
            "Late Only (20-29)": {"enabled": [f"layer_{i}" for i in range(20, 30)] + ["other_weights"], "strength": 1.0},
            "Mid-Late (15-29)": {"enabled": [f"layer_{i}" for i in range(15, 30)] + ["other_weights"], "strength": 1.0},
            "Skip Early (10-29)": {"enabled": [f"layer_{i}" for i in range(10, 30)] + ["other_weights"], "strength": 1.0},
            "Mid Only (10-19)": {"enabled": [f"layer_{i}" for i in range(10, 20)], "strength": 1.0},
            "Early Only (0-9)": {"enabled": [f"layer_{i}" for i in range(10)], "strength": 1.0},
            "Peak Impact (18-25)": {"enabled": [f"layer_{i}" for i in range(18, 26)], "strength": 1.0},
            "Face Priority (16-24)": {"enabled": [f"layer_{i}" for i in range(16, 25)], "strength": 1.0},
            "Face Priority Aggressive (14-25)": {"enabled": [f"layer_{i}" for i in range(14, 26)], "strength": 1.0},
            "Evens Only": {"enabled": [f"layer_{i}" for i in range(0, 30, 2)], "strength": 1.0},
            "Odds Only": {"enabled": [f"layer_{i}" for i in range(1, 30, 2)], "strength": 1.0},
            "Custom": None,  # Use individual toggles (JS always sends this)
        },
        "filter_key_pattern": r'diffusion_model\.layers\.(\d+)|lora_unet_layers_(\d+)',
        "filter_key_to_block": lambda match: f"layer_{match.group(1) or match.group(2)}",
    },
    "SDXL": {
        "node_id": "SDXLAnalyzerSelectiveLoaderV2",
        "display_name": "SDXL Analyzer + Selective Loader V2",
        "description": """Combined analyzer and selective loader for SDXL LoRAs.
Analyzes block impact and allows per-block control with strength shaping.

Block Guide (13 blocks with attention layers):
- text_encoder_1/2: CLIP text encoders (CLIP-L and CLIP-G)
- input_4, input_5: Mid encoder blocks with attention
- input_7, input_8: Deep encoder blocks (high impact, composition)
- unet_mid: Bottleneck (moderate-high impact)
- output_0: Primary decoder (composition, high impact)
- output_1: Style block (strongest for style/color)
- output_2-5: Decoder blocks (decreasing impact)

Supports strength scheduling format: 0:.2,.5:.8,1:1.0""",
        "architecture": "SDXL",
        "blocks": ["text_encoder_1", "text_encoder_2", "input_4", "input_5", "input_7", "input_8",
                   "unet_mid", "output_0", "output_1", "output_2", "output_3", "output_4", "output_5", "other_weights"],
        "block_labels": {
            "text_encoder_1": "TE1 (CLIP-L)", "text_encoder_2": "TE2 (CLIP-G)",
            "input_4": "Input 4", "input_5": "Input 5", "input_7": "Input 7", "input_8": "Input 8",
            "unet_mid": "Mid Block",
            "output_0": "Output 0", "output_1": "Output 1", "output_2": "Output 2",
            "output_3": "Output 3", "output_4": "Output 4", "output_5": "Output 5",
            "other_weights": "Other Weights"
        },
        "presets": {
            "Default": {"enabled": "ALL", "strength": 1.0},
            "All Off": {"enabled": [], "strength": 0.0},
            "Half Strength": {"enabled": "ALL", "strength": 0.5},
            "UNet Only": {"enabled": ["input_4", "input_5", "input_7", "input_8", "unet_mid",
                                      "output_0", "output_1", "output_2", "output_3", "output_4", "output_5", "other_weights"], "strength": 1.0},
            "High Impact": {"enabled": ["input_7", "input_8", "unet_mid", "output_0", "output_1", "output_2"], "strength": 1.0},
            "Text Encoders Only": {"enabled": ["text_encoder_1", "text_encoder_2"], "strength": 1.0},
            "Decoders Only": {"enabled": ["output_0", "output_1", "output_2", "output_3", "output_4", "output_5"], "strength": 1.0},
            "Encoders Only": {"enabled": ["input_4", "input_5", "input_7", "input_8"], "strength": 1.0},
            "Style Focus": {"enabled": ["output_1", "output_2"], "strength": 1.0},
            "Composition Focus": {"enabled": ["input_8", "unet_mid", "output_0"], "strength": 1.0},
            "Face Focus": {"enabled": ["input_7", "input_8", "unet_mid", "output_0", "output_1", "output_2", "output_3"], "strength": 1.0},
            "Custom": None,
        },
    },
    "FLUX": {
        "node_id": "FLUXAnalyzerSelectiveLoaderV2",
        "display_name": "FLUX Analyzer + Selective Loader V2",
        "description": """Combined analyzer and selective loader for FLUX LoRAs.
Analyzes block impact and allows per-block control with strength shaping.

Block Guide (57 total):
- double_0-18: Double transformer blocks (19 blocks, higher impact)
- single_0-37: Single transformer blocks (38 blocks, lower impact)

Double blocks typically have higher impact than single blocks.
Peak impact is usually in double_8-17 range.

Face blocks (from lora-the-explorer): double 7,12,16 | single 7,12,16,20

Supports strength scheduling format: 0:.2,.5:.8,1:1.0""",
        "architecture": "FLUX",
        "blocks": [f"double_{i}" for i in range(19)] + [f"single_{i}" for i in range(38)] + ["other_weights"],
        "block_labels": ({f"double_{i}": f"Double {i}" for i in range(19)} |
                        {f"single_{i}": f"Single {i}" for i in range(38)} |
                        {"other_weights": "Other Weights"}),
        "presets": {
            "Default": {"enabled": "ALL", "strength": 1.0},
            "All Off": {"enabled": [], "strength": 0.0},
            "Half Strength": {"enabled": "ALL", "strength": 0.5},
            "Double Blocks Only": {"enabled": [f"double_{i}" for i in range(19)] + ["other_weights"], "strength": 1.0},
            "Single Blocks Only": {"enabled": [f"single_{i}" for i in range(38)] + ["other_weights"], "strength": 1.0},
            "High Impact Double": {"enabled": [f"double_{i}" for i in range(6, 19)], "strength": 1.0},
            "Core Double": {"enabled": [f"double_{i}" for i in range(8, 18)], "strength": 1.0},
            "Face Focus": {"enabled": ["double_7", "double_12", "double_16", "single_7", "single_12", "single_16", "single_20"], "strength": 1.0},
            "Face Aggressive": {"enabled": ["double_4", "double_7", "double_8", "double_12", "double_15", "double_16",
                                            "single_4", "single_7", "single_8", "single_12", "single_15", "single_16", "single_19", "single_20"], "strength": 1.0},
            "Style Only (No Face)": {"enabled": [b for b in [f"double_{i}" for i in range(19)] + [f"single_{i}" for i in range(38)]
                                                 if b not in ["double_7", "double_12", "double_16", "single_7", "single_12", "single_16", "single_20"]], "strength": 1.0},
            "Evens Only": {"enabled": [f"double_{i}" for i in range(0, 19, 2)] + [f"single_{i}" for i in range(0, 38, 2)], "strength": 1.0},
            "Odds Only": {"enabled": [f"double_{i}" for i in range(1, 19, 2)] + [f"single_{i}" for i in range(1, 38, 2)], "strength": 1.0},
            "Custom": None,
        },
    },
    "WAN": {
        "node_id": "WanAnalyzerSelectiveLoaderV2",
        "display_name": "Wan Analyzer + Selective Loader V2",
        "description": """Combined analyzer and selective loader for Wan 2.2 LoRAs.
Analyzes block impact and allows per-block control with strength shaping.

Block Guide (40 total):
- block_0-9: Early transformer blocks
- block_10-19: Early-mid blocks
- block_20-29: Mid-late blocks
- block_30-39: Late blocks

Supports strength scheduling format: 0:.2,.5:.8,1:1.0""",
        "architecture": "WAN",
        "blocks": [f"block_{i}" for i in range(40)] + ["other_weights"],
        "block_labels": {f"block_{i}": f"Block {i}" for i in range(40)} | {"other_weights": "Other Weights"},
        "presets": {
            "Default": {"enabled": "ALL", "strength": 1.0},
            "All Off": {"enabled": [], "strength": 0.0},
            "Half Strength": {"enabled": "ALL", "strength": 0.5},
            "Late Only (30-39)": {"enabled": [f"block_{i}" for i in range(30, 40)] + ["other_weights"], "strength": 1.0},
            "Mid-Late (20-39)": {"enabled": [f"block_{i}" for i in range(20, 40)] + ["other_weights"], "strength": 1.0},
            "Skip Early (10-39)": {"enabled": [f"block_{i}" for i in range(10, 40)] + ["other_weights"], "strength": 1.0},
            "Mid Only (15-25)": {"enabled": [f"block_{i}" for i in range(15, 26)], "strength": 1.0},
            "Early Only (0-19)": {"enabled": [f"block_{i}" for i in range(20)], "strength": 1.0},
            "Evens Only": {"enabled": [f"block_{i}" for i in range(0, 40, 2)], "strength": 1.0},
            "Odds Only": {"enabled": [f"block_{i}" for i in range(1, 40, 2)], "strength": 1.0},
            "Custom": None,
        },
    },
    "QWEN_IMAGE": {
        "node_id": "QwenAnalyzerSelectiveLoaderV2",
        "display_name": "Qwen Analyzer + Selective Loader V2",
        "description": """Combined analyzer and selective loader for Qwen-Image LoRAs.
Analyzes block impact and allows per-block control with strength shaping.

Block Guide (60 total):
- block_0-14: Early transformer blocks
- block_15-29: Early-mid blocks
- block_30-44: Mid-late blocks
- block_45-59: Late blocks

Supports strength scheduling format: 0:.2,.5:.8,1:1.0""",
        "architecture": "QWEN_IMAGE",
        "blocks": [f"block_{i}" for i in range(60)] + ["other_weights"],
        "block_labels": {f"block_{i}": f"Block {i}" for i in range(60)} | {"other_weights": "Other Weights"},
        "presets": {
            "Default": {"enabled": "ALL", "strength": 1.0},
            "All Off": {"enabled": [], "strength": 0.0},
            "Half Strength": {"enabled": "ALL", "strength": 0.5},
            "Late Only (45-59)": {"enabled": [f"block_{i}" for i in range(45, 60)] + ["other_weights"], "strength": 1.0},
            "Mid-Late (30-59)": {"enabled": [f"block_{i}" for i in range(30, 60)] + ["other_weights"], "strength": 1.0},
            "Skip Early (15-59)": {"enabled": [f"block_{i}" for i in range(15, 60)] + ["other_weights"], "strength": 1.0},
            "Mid Only (20-40)": {"enabled": [f"block_{i}" for i in range(20, 41)], "strength": 1.0},
            "Early Only (0-29)": {"enabled": [f"block_{i}" for i in range(30)], "strength": 1.0},
            "Evens Only": {"enabled": [f"block_{i}" for i in range(0, 60, 2)], "strength": 1.0},
            "Odds Only": {"enabled": [f"block_{i}" for i in range(1, 60, 2)], "strength": 1.0},
            "Custom": None,
        },
    },
}


def _filter_lora_by_blocks(lora_state_dict: dict, enabled_blocks: set, block_strengths: dict,
                           architecture: str, other_enabled: bool = True, other_strength: float = 1.0) -> dict:
    """
    Filter LoRA state dict to only include enabled blocks with their strengths applied.
    Returns a new dict with filtered and scaled weights.

    For LoKR/LoHa, only scales the first weight matrix (lokr_w1/hada_w1) to avoid
    squaring the strength effect (since these use product decomposition).
    """
    filtered_dict = {}

    # Detect LoRA type once for the whole dict
    keys = list(lora_state_dict.keys())
    lora_type = _detect_lora_type(keys)

    for key, value in lora_state_dict.items():
        block_id = _extract_block_id_v2(key, architecture)
        key_lower = key.lower()

        # Determine if this key should be scaled
        # For LoKR: only scale lokr_w1 (not lokr_w2) to avoid squaring the strength
        # For LoHa: only scale hada_w1 (not hada_w2) for the same reason
        should_scale = True
        if lora_type == 'LoKR':
            # Don't scale lokr_w2 variants - only lokr_w1
            if '.lokr_w2' in key_lower:
                should_scale = False
        elif lora_type == 'LoHa':
            # Don't scale hada_w2 variants - only hada_w1
            if '.hada_w2' in key_lower:
                should_scale = False

        if block_id == 'other':
            if other_enabled:
                if other_strength != 1.0 and should_scale:
                    filtered_dict[key] = value * other_strength
                else:
                    filtered_dict[key] = value
        elif block_id in enabled_blocks:
            strength = block_strengths.get(block_id, 1.0)
            if strength != 1.0 and should_scale:
                filtered_dict[key] = value * strength
            else:
                filtered_dict[key] = value

    return filtered_dict


def _create_combined_node_class(config: dict):
    """
    Factory function to create a combined Analyzer + Selective Loader node class
    from an architecture config.
    """

    class CombinedAnalyzerSelectiveLoader:
        """Combined Analyzer + Selective Loader node (generated from config)."""

        _config = config

        @classmethod
        def INPUT_TYPES(cls):
            cfg = cls._config
            blocks = cfg["blocks"]
            presets = cfg["presets"]

            inputs = {
                "required": {
                    "model": ("MODEL",),
                    "positive": ("CONDITIONING", {"tooltip": "Positive conditioning from CLIP encode"}),
                    "negative": ("CONDITIONING", {"tooltip": "Negative conditioning from CLIP encode"}),
                    "lora_name": (folder_paths.get_filename_list("loras"), {
                        "tooltip": "LoRA file to load and analyze"
                    }),
                    "strength": ("FLOAT", {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.05,
                        "tooltip": "Overall LoRA strength (ignored when using schedule)"
                    }),
                    "preset": (list(presets.keys()), {
                        "default": "Default",
                        "tooltip": "Quick preset selection. Individual toggles below override when changed."
                    }),
                },
            }

            # Add per-block toggle and strength inputs in required section
            # This ensures they always show and the JS extension can combine them
            for block in blocks:
                inputs["required"][block] = ("BOOLEAN", {"default": True})
                inputs["required"][f"{block}_str"] = ("FLOAT", {
                    "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05
                })

            # Load last used save path for this node type
            save_paths_config = _load_save_paths_config()
            last_save_path = save_paths_config.get(cfg["node_id"], "")

            # Optional inputs for path override, scheduling, and saving
            inputs["optional"] = {
                "lora_path_opt": ("STRING", {"forceInput": True, "tooltip": "Optional: Override LoRA selection with a path"}),
                "schedule_preset": (SCHEDULE_PRESET_LIST, {
                    "default": "Custom",
                    "tooltip": "Select a preset schedule (populates the text field below for editing)"
                }),
                "strength_schedule": ("STRING", {
                    "default": "",
                    "tooltip": "Strength schedule: 0:.2,.5:.8,1:1.0 (step:strength pairs). Edit freely after selecting preset."
                }),
                "save_refined_lora": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable to save the refined LoRA to disk"
                }),
                "save_path": ("STRING", {
                    "default": last_save_path,
                    "tooltip": "Directory to save refined LoRA"
                }),
                "save_filename": ("STRING", {
                    "default": "",
                    "tooltip": "Filename for saved LoRA (timestamp auto-appended). Leave empty for auto-name."
                }),
                "block_weights_string": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Input/Output: Block weights in positional (1.0, 0.5, 1.2...) or named format (%default=1.0, te1=0.5, in7-8=1.2). Syncs bidirectionally with UI sliders. String input overrides UI."
                }),
            }

            return inputs

        RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "STRING", "STRING", "STRING")
        RETURN_NAMES = ("model", "positive", "negative", "analysis", "analysis_json", "weights_output")
        OUTPUT_TOOLTIPS = (
            "Model with LoRA applied (filtered by enabled blocks).",
            "Positive conditioning (with hooks if using schedule).",
            "Negative conditioning (with hooks if using schedule).",
            "Per-block analysis showing impact scores.",
            "JSON analysis data for UI coloring.",
            "Block weights string output (syncs with UI sliders).",
        )
        FUNCTION = "load_analyze_and_filter"
        CATEGORY = "loaders/lora"
        OUTPUT_NODE = True
        DESCRIPTION = config["description"]

        def load_analyze_and_filter(self, model, positive, negative, lora_name, strength, preset,
                                    schedule_preset="Custom", strength_schedule="", lora_path_opt=None, save_refined_lora=False, save_path="", save_filename="", block_weights_string="", **kwargs):
            cfg = self._config
            architecture = cfg["architecture"]
            blocks = cfg["blocks"]
            presets = cfg["presets"]

            # Get LoRA path - use optional override if provided
            if lora_path_opt and os.path.exists(lora_path_opt):
                lora_path = lora_path_opt
            else:
                lora_path = folder_paths.get_full_path("loras", lora_name)
            if not lora_path or not os.path.exists(lora_path):
                return {"ui": {"analysis_json": ["{}"]}, "result": (model, positive, negative, "Error: LoRA file not found", "{}", "")}

            print(f"[{cfg['display_name']}] Loading: {lora_name}")

            # Get metadata and load LoRA
            metadata = _get_metadata(lora_path)
            if lora_path.endswith('.safetensors'):
                lora_state_dict = load_file(lora_path)
            else:
                lora_state_dict = torch.load(lora_path, map_location='cpu')

            lora_keys = list(lora_state_dict.keys())

            # Detect architecture and LoRA type (for analysis display)
            detected_arch, confidence, method = _detect_architecture_v2(lora_keys, metadata)
            lora_type = _detect_lora_type(lora_keys)
            arch_display = f"{detected_arch} ({lora_type})" if lora_type != 'LoRA' else detected_arch
            print(f"[{cfg['display_name']}] Detected: {arch_display} ({confidence} via {method})")

            # Check if string input is provided (overrides UI)
            from .selective_lora_loader import _parse_block_weights_string
            parsed_weights = _parse_block_weights_string(block_weights_string, architecture)

            # Determine enabled blocks and strengths
            # Note: JS always sends "Custom" so we read individual widget values
            if parsed_weights:
                # Use parsed weights from string
                enabled_blocks = set()
                block_strengths = {}
                for block_name, (enabled, blk_str) in parsed_weights.items():
                    if block_name == 'other_weights':
                        other_enabled = enabled
                        other_strength = blk_str
                    elif enabled:
                        enabled_blocks.add(block_name)
                        block_strengths[block_name] = blk_str
                using_preset = "String Input"
            else:
                preset_cfg = presets.get(preset)
                if preset_cfg is not None:
                    # Using a preset (only when JS is not present)
                    if preset_cfg["enabled"] == "ALL":
                        enabled_blocks = set(blocks)
                    else:
                        enabled_blocks = set(preset_cfg["enabled"])
                    block_strengths = {b: preset_cfg["strength"] for b in enabled_blocks}
                    other_enabled = "other_weights" in enabled_blocks or preset != "All Off"
                    other_strength = preset_cfg["strength"]
                    using_preset = preset
                else:
                    # Custom mode - read from kwargs
                    enabled_blocks = set()
                    block_strengths = {}
                    for block in blocks:
                        if block == "other_weights":
                            continue
                        if kwargs.get(block, True):
                            enabled_blocks.add(block)
                            block_strengths[block] = kwargs.get(f"{block}_str", 1.0)
                    other_enabled = kwargs.get("other_weights", True)
                    other_strength = kwargs.get("other_weights_str", 1.0)
                    using_preset = None

            # Filter LoRA by enabled blocks
            original_count = len(lora_state_dict)
            filtered_lora = _filter_lora_by_blocks(
                lora_state_dict, enabled_blocks, block_strengths,
                architecture, other_enabled, other_strength
            )
            filtered_count = len(filtered_lora)

            print(f"[{cfg['display_name']}] Filtered: {filtered_count}/{original_count} tensors")

            # Save refined LoRA if enabled and path provided
            saved_path = None
            if save_refined_lora and save_path and save_path.strip():
                saved_path = _save_refined_lora(
                    filtered_lora, save_path, save_filename,
                    cfg["node_id"], architecture, metadata
                )
                # Remember this save path for next time
                if saved_path:
                    save_paths_config = _load_save_paths_config()
                    save_paths_config[cfg["node_id"]] = save_path.strip()
                    _save_save_paths_config(save_paths_config)

            # Check for strength schedule
            # Priority: 1) strength_schedule text field, 2) schedule_preset lookup
            effective_schedule = strength_schedule.strip() if strength_schedule else ""
            if not effective_schedule and schedule_preset and schedule_preset != "Custom":
                # Fallback to preset value (for API usage without JS)
                effective_schedule = SCHEDULE_PRESETS.get(schedule_preset, "")
            schedule = _parse_strength_schedule(effective_schedule)
            using_schedule = False  # Track if we actually use scheduling

            # Initialize conditioning outputs
            positive_out = positive
            negative_out = negative

            if schedule:
                # Use hook system for scheduling
                # Base strength is 1.0 so keyframe values ARE the actual strengths
                # Schedule format: "0:.2,.8:.6,1:.4" means:
                #   - At 0% of steps: strength 0.2
                #   - At 80% of steps: strength 0.6
                #   - At 100% of steps: strength 0.4
                # Values interpolate linearly between keyframes
                using_schedule = True
                print(f"[{cfg['display_name']}] Using strength schedule: {schedule}")
                hooks = comfy.hooks.create_hook_lora(lora=filtered_lora, strength_model=1.0, strength_clip=0.0)
                kf_group = _create_hook_keyframes_interpolated(schedule)
                if kf_group and hooks:
                    hooks.set_keyframes_on_hooks(kf_group)

                # Clone model and register hook patches (needed for hook system)
                model_out = model.clone()
                target_dict = comfy.hooks.create_target_dict(comfy.hooks.EnumWeightTarget.Model)
                model_out.register_all_hook_patches(hooks, target_dict)

                # Attach hooks to conditioning - this is what makes scheduling work!
                positive_out = comfy.hooks.set_hooks_for_conditioning(positive, hooks)
                negative_out = comfy.hooks.set_hooks_for_conditioning(negative, hooks)
                print(f"[{cfg['display_name']}] Hooks attached to conditioning")
            else:
                # Standard loading (no schedule) - apply LoRA directly to model
                model_out, _ = comfy.sd.load_lora_for_models(
                    model, None, filtered_lora, strength, 0.0
                )

            # Analyze patches for display
            if not using_schedule:
                patch_analysis = _analyze_patches_v2(model_out, architecture)
            else:
                # For scheduled LoRAs, analyze the filtered dict directly
                # Create a temporary load to get patch info
                temp_model, _ = comfy.sd.load_lora_for_models(model, None, filtered_lora, 1.0, 0.0)
                patch_analysis = _analyze_patches_v2(temp_model, architecture)

            analysis_text = _format_patch_analysis_v2(patch_analysis, architecture, confidence, method, lora_type)
            analysis_json = _create_analysis_json_v2(patch_analysis, architecture, lora_name, confidence, method, metadata, lora_type)

            print(f"[{cfg['display_name']}] {len(patch_analysis)} blocks analyzed")

            # Build info string - count only layer blocks (not other_weights)
            layer_blocks = [b for b in blocks if b != "other_weights"]
            layer_blocks_enabled = len([b for b in enabled_blocks if b != "other_weights"])
            other_status = "on" if other_enabled else "off"
            info_lines = [analysis_text, "", f"Enabled: {layer_blocks_enabled}/{len(layer_blocks)} layers (other: {other_status})"]
            if using_schedule:
                info_lines.append(f"Schedule: {len(schedule)} keyframes (attached to conditioning)")
            if saved_path:
                info_lines.append(f"Saved: {os.path.basename(saved_path)}")

            # Generate weights_output string - simple positional format
            weights_list = []
            for block in blocks:
                if block == "other_weights":
                    continue  # Skip other_weights in positional format
                blk_str = block_strengths.get(block, 0.0) if block in enabled_blocks else 0.0
                weights_list.append(f"{blk_str:.2f}")
            weights_output = ", ".join(weights_list)

            # Return with UI format for analysis_json passthrough to JS
            return {"ui": {"analysis_json": [analysis_json]}, "result": (model_out, positive_out, negative_out, "\n".join(info_lines), analysis_json, weights_output)}

    # Set class attributes from config
    CombinedAnalyzerSelectiveLoader.__name__ = config["node_id"]
    CombinedAnalyzerSelectiveLoader.__doc__ = config["description"]

    return CombinedAnalyzerSelectiveLoader


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    # Keep the basic V2 analyzer
    "LoRALoaderWithAnalysisV2": LoRALoaderWithAnalysisV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRALoaderWithAnalysisV2": "LoRA Loader + Analyzer V2",
}

# Generate and register combined nodes from configs
for arch_name, config in ARCH_CONFIGS.items():
    node_class = _create_combined_node_class(config)
    NODE_CLASS_MAPPINGS[config["node_id"]] = node_class
    NODE_DISPLAY_NAME_MAPPINGS[config["node_id"]] = config["display_name"]
