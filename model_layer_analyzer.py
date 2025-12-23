"""
Model Layer Analyzer + Editor for ComfyUI

Analyzes and allows per-block control of diffusion model layers.
Similar to LoRA V2 analyzer but for base models.

Supports: SDXL, SD 1.5, FLUX, Z-Image, Wan, Qwen architectures.
"""

import os
import re
import json
from datetime import datetime
from collections import defaultdict

import torch
import folder_paths
import comfy.sd
import comfy.model_patcher
from safetensors.torch import save_file
from safetensors import safe_open


# ============================================================================
# ORIGINAL WEIGHTS CACHE
# ============================================================================
# Cache original weights on first access to prevent reading patched values
# Key: id(model), Value: state_dict with cloned tensors

_original_weights_cache = {}


def _get_original_weights(model_patcher):
    """
    Get the original, unmodified weights for a model.

    Caches on first access because:
    1. The underlying model is shared between all model_patcher clones
    2. state_dict() returns current values which may be patched
    3. weight_backup is only available AFTER patch_model() during inference

    Returns (state_dict, key_prefix) where state_dict contains original weights.
    """
    global _original_weights_cache

    model = model_patcher.model
    if hasattr(model, 'diffusion_model'):
        diff_model = model.diffusion_model
        key_prefix = "diffusion_model."
    else:
        diff_model = model
        key_prefix = ""

    model_id = id(model)

    if model_id not in _original_weights_cache:
        # First access - cache the current state (should be clean on first load)
        # Store on CPU to avoid GPU memory bloat
        _original_weights_cache[model_id] = {
            k: v.detach().clone().cpu() for k, v in diff_model.state_dict().items()
        }

    return _original_weights_cache[model_id], key_prefix


def clear_model_cache(model_id=None):
    """Clear cached weights. Call when model is reloaded."""
    global _original_weights_cache
    if model_id is None:
        _original_weights_cache.clear()
        print("[Model Layer Editor] Cleared all weight caches")
    elif model_id in _original_weights_cache:
        del _original_weights_cache[model_id]
        print(f"[Model Layer Editor] Cleared cache for model {model_id}")


# ============================================================================
# USER PRESETS
# ============================================================================
# Save/load user-defined presets per node type
# Preset system: localStorage for instant UI updates, Python config file for persistence across sessions

def _get_user_presets_path(node_id: str) -> str:
    """Get path to user presets file for a node type."""
    return os.path.join(os.path.dirname(__file__), f".model_layer_presets_{node_id}.json")


def _load_user_presets(node_id: str) -> dict:
    """Load user presets for a node type from config file."""
    path = _get_user_presets_path(node_id)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError, OSError) as e:
            print(f"[Model Layer Editor] Error loading presets: {e}")
    return {}


def _save_user_presets(node_id: str, presets: dict):
    """Save user presets for a node type to config file."""
    path = _get_user_presets_path(node_id)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(presets, f, indent=2)
        print(f"[Model Layer Editor] Saved presets to {path}")
    except (IOError, OSError) as e:
        print(f"[Model Layer Editor] Error saving presets: {e}")


def _save_current_as_preset(node_id: str, preset_name: str, blocks: list,
                             enabled_blocks: set, block_strengths: dict):
    """Save current block settings as a named preset."""
    if not preset_name or not preset_name.strip():
        return False

    preset_name = preset_name.strip()

    # Build preset data
    preset_data = {
        "enabled": list(enabled_blocks),
        "strength": 1.0,  # Base strength (individual values in overrides)
        "overrides": {}
    }

    # Check if all blocks are enabled
    if set(blocks) == enabled_blocks:
        preset_data["enabled"] = "ALL"

    # Add strength overrides for non-1.0 values
    for block in blocks:
        strength = block_strengths.get(block, 1.0)
        if strength != 1.0:
            preset_data["overrides"][block] = strength

    # Load existing presets, add/update, save
    presets = _load_user_presets(node_id)
    presets[preset_name] = preset_data
    _save_user_presets(node_id, presets)

    print(f"[Model Layer Editor] Saved preset '{preset_name}' for {node_id}")
    return True


def _delete_user_preset(node_id: str, preset_name: str) -> bool:
    """Delete a user preset by name."""
    if not preset_name or not preset_name.strip():
        return False

    preset_name = preset_name.strip()
    presets = _load_user_presets(node_id)

    if preset_name in presets:
        del presets[preset_name]
        _save_user_presets(node_id, presets)
        print(f"[Model Layer Editor] Deleted preset '{preset_name}' from {node_id}")
        return True
    else:
        print(f"[Model Layer Editor] Preset '{preset_name}' not found in {node_id}")
        return False


# Cache for loaded user presets (reloaded when saving)
_user_presets_cache = {}


def _get_all_presets(node_id: str, builtin_presets: dict) -> dict:
    """Get combined builtin + user presets for a node."""
    global _user_presets_cache

    # Load user presets (refresh from disk)
    user_presets = _load_user_presets(node_id)
    _user_presets_cache[node_id] = user_presets

    # Combine: builtin first, then user presets
    # User presets with same name as builtin will override
    combined = dict(builtin_presets)

    # Add user presets (they appear after builtins, can override)
    for name, data in user_presets.items():
        combined[name] = data

    return combined


# ============================================================================
# ARCHITECTURE DETECTION
# ============================================================================

def _detect_model_architecture(model_patcher) -> tuple:
    """
    Detect model architecture from the model structure.
    Returns (architecture, confidence, block_info).
    """
    try:
        # Get the underlying model
        model = model_patcher.model

        # Try to get state dict keys
        if hasattr(model, 'state_dict'):
            state_dict = model.state_dict()
            keys = list(state_dict.keys())
        elif hasattr(model, 'diffusion_model') and hasattr(model.diffusion_model, 'state_dict'):
            state_dict = model.diffusion_model.state_dict()
            keys = list(state_dict.keys())
        else:
            return ('UNKNOWN', 'low', {})

        num_keys = len(keys)
        keys_str = ' '.join(keys[:100]).lower()  # Sample for detection

        # Z-Image detection (30 layers)
        if any('layers.' in k and 'attention' in k.lower() for k in keys[:50]):
            layer_nums = set()
            for k in keys:
                match = re.search(r'layers\.(\d+)', k)
                if match:
                    layer_nums.add(int(match.group(1)))
            if len(layer_nums) >= 25:
                return ('ZIMAGE', 'high', {'layers': sorted(layer_nums)})

        # FLUX detection (double + single blocks)
        double_blocks = set()
        single_blocks = set()
        for k in keys:
            match = re.search(r'double_blocks?\.(\d+)', k)
            if match:
                double_blocks.add(int(match.group(1)))
            match = re.search(r'single_blocks?\.(\d+)', k)
            if match:
                single_blocks.add(int(match.group(1)))
        if len(double_blocks) >= 15 or len(single_blocks) >= 30:
            return ('FLUX', 'high', {'double_blocks': sorted(double_blocks), 'single_blocks': sorted(single_blocks)})

        # Wan detection (blocks with self_attn, ffn)
        if any('self_attn' in k and 'blocks.' in k for k in keys):
            wan_blocks = set()
            for k in keys:
                match = re.search(r'blocks\.(\d+)', k)
                if match:
                    wan_blocks.add(int(match.group(1)))
            if len(wan_blocks) >= 30:
                return ('WAN', 'high', {'blocks': sorted(wan_blocks)})

        # Qwen detection (transformer_blocks with img_mlp)
        if any('transformer_blocks' in k and 'img_mlp' in k for k in keys):
            qwen_blocks = set()
            for k in keys:
                match = re.search(r'transformer_blocks\.(\d+)', k)
                if match:
                    qwen_blocks.add(int(match.group(1)))
            if len(qwen_blocks) >= 50:
                return ('QWEN', 'high', {'blocks': sorted(qwen_blocks)})

        # SDXL/SD15 detection (input_blocks, output_blocks)
        input_blocks = set()
        output_blocks = set()
        has_mid = False
        for k in keys:
            if 'input_blocks.' in k or 'input_blocks_' in k:
                match = re.search(r'input_blocks[._](\d+)', k)
                if match:
                    input_blocks.add(int(match.group(1)))
            if 'output_blocks.' in k or 'output_blocks_' in k:
                match = re.search(r'output_blocks[._](\d+)', k)
                if match:
                    output_blocks.add(int(match.group(1)))
            if 'middle_block' in k or 'mid_block' in k:
                has_mid = True

        if len(input_blocks) >= 10 and len(output_blocks) >= 10:
            # Distinguish SDXL from SD15 by parameter count
            if num_keys > 1500:
                return ('SDXL', 'high', {'input_blocks': sorted(input_blocks), 'output_blocks': sorted(output_blocks), 'has_mid': has_mid})
            else:
                return ('SD15', 'high', {'input_blocks': sorted(input_blocks), 'output_blocks': sorted(output_blocks), 'has_mid': has_mid})

        return ('UNKNOWN', 'low', {})

    except Exception as e:
        print(f"[Model Analyzer] Detection error: {e}")
        return ('UNKNOWN', 'low', {})


# ============================================================================
# BLOCK ANALYSIS
# ============================================================================

def _analyze_model_blocks(model_patcher, architecture: str) -> dict:
    """
    Analyze model blocks and return stats per block.
    Returns dict: {block_id: {param_count, memory_mb, dtype, keys}}
    """
    try:
        model = model_patcher.model

        # Get state dict from the right place
        if hasattr(model, 'diffusion_model'):
            state_dict = model.diffusion_model.state_dict()
            prefix = 'diffusion_model.'
        else:
            state_dict = model.state_dict()
            prefix = ''

        blocks = defaultdict(lambda: {
            'param_count': 0,
            'memory_bytes': 0,
            'keys': [],
            'dtype': None
        })

        for key, tensor in state_dict.items():
            block_id = _extract_block_id(key, architecture)

            blocks[block_id]['param_count'] += tensor.numel()
            blocks[block_id]['memory_bytes'] += tensor.numel() * tensor.element_size()
            blocks[block_id]['keys'].append(prefix + key)
            if blocks[block_id]['dtype'] is None:
                blocks[block_id]['dtype'] = str(tensor.dtype)

        # Convert memory to MB
        for block_id in blocks:
            blocks[block_id]['memory_mb'] = blocks[block_id]['memory_bytes'] / (1024 * 1024)

        return dict(blocks)

    except Exception as e:
        print(f"[Model Analyzer] Analysis error: {e}")
        return {}


def _extract_block_id(key: str, architecture: str) -> str:
    """Extract block identifier from a model weight key."""
    key_lower = key.lower()

    if architecture == 'ZIMAGE':
        match = re.search(r'layers\.(\d+)', key)
        if match:
            return f"layer_{match.group(1)}"
        if 'final_layer' in key_lower:
            return 'final_layer'
        if 'x_embedder' in key_lower:
            return 'x_embedder'
        return 'other'

    elif architecture == 'FLUX':
        match = re.search(r'double_blocks?\.(\d+)', key)
        if match:
            return f"double_{match.group(1)}"
        match = re.search(r'single_blocks?\.(\d+)', key)
        if match:
            return f"single_{match.group(1)}"
        return 'other'

    elif architecture == 'WAN':
        match = re.search(r'blocks\.(\d+)', key)
        if match:
            return f"block_{match.group(1)}"
        return 'other'

    elif architecture == 'QWEN':
        match = re.search(r'transformer_blocks\.(\d+)', key)
        if match:
            return f"block_{match.group(1)}"
        return 'other'

    elif architecture in ['SDXL', 'SD15']:
        match = re.search(r'input_blocks?[._](\d+)', key)
        if match:
            return f"input_{match.group(1)}"
        match = re.search(r'output_blocks?[._](\d+)', key)
        if match:
            return f"output_{match.group(1)}"
        if 'middle_block' in key_lower or 'mid_block' in key_lower:
            return 'mid'
        if 'time_embed' in key_lower:
            return 'time_embed'
        if 'label_emb' in key_lower:
            return 'label_emb'
        return 'other'

    return 'other'


# ============================================================================
# BLOCK SCALING (using ComfyUI's patching system)
# ============================================================================

def _apply_block_modifications(model_patcher, block_analysis: dict, enabled_blocks: set,
                               block_strengths: dict, architecture: str):
    """
    Apply block modifications using ComfyUI's patching system.

    CRITICAL: We clone the model_patcher and add patches to scale weights.
    This is non-destructive - original model weights are never modified.

    For scaling weight W by factor s:
    - new_weight = W * s = W + W * (s - 1)
    - We add a patch of W * (s - 1) with strength 1.0

    Returns (cloned_model_patcher, modified_count).
    """
    try:
        # Get the ORIGINAL weights from cache (not current state_dict which may be patched)
        base_state_dict, key_prefix = _get_original_weights(model_patcher)

        # Clone the model patcher so we don't affect the original
        cloned = model_patcher.clone()

        # CRITICAL: Clear any inherited patches to prevent accumulation
        if hasattr(cloned, 'patches'):
            cloned.patches = {}
        if hasattr(cloned, 'object_patches'):
            cloned.object_patches = {}

        # Build patches dict for add_patches
        # Format: {key: (diff_tensor,)} where the diff is added to original
        patches = {}
        modified_count = 0

        for block_id, block_info in block_analysis.items():
            # Only modify blocks that are in our control list (block_strengths)
            # Skip blocks we don't have controls for (like extra blocks found in model)
            if block_id not in block_strengths:
                continue

            strength = block_strengths[block_id]

            # Skip if no modification needed
            if strength == 1.0:
                continue

            # Create patches for this block's weights
            for full_key in block_info['keys']:
                # Get the key without prefix for state_dict lookup
                key = full_key.replace('diffusion_model.', '')

                # Use base_state_dict for original unpatched weights
                if key not in base_state_dict:
                    continue

                weight = base_state_dict[key]

                # Calculate the diff: to get W*s, we add W*(s-1) to W
                # diff = W * (s - 1)
                # IMPORTANT: Do this on CPU to avoid GPU OOM errors
                # ComfyUI's patching system will handle device placement
                weight_cpu = weight.to("cpu") if weight.device.type != "cpu" else weight
                diff = weight_cpu * (strength - 1.0)

                # Patch key needs the prefix
                patch_key = key_prefix + key

                # Format for add_patches: tuple of (diff_tensor,)
                # This gets added to the original weight
                patches[patch_key] = (diff,)
                modified_count += 1

        # Apply patches if any
        if patches:
            # add_patches(patches, strength_patch, strength_model=None)
            # strength_patch=1.0 means apply the full diff
            cloned.add_patches(patches, strength_patch=1.0)

        return cloned, modified_count

    except Exception as e:
        import traceback
        print(f"[Model Analyzer] Patching error: {e}")
        traceback.print_exc()
        return model_patcher.clone(), 0


# ============================================================================
# MODEL SAVING
# ============================================================================

def _save_modified_model(model_patcher, save_path: str, save_filename: str,
                         architecture: str, block_info: dict, source_model_path: str = "") -> str:
    """
    Save modified model to safetensors.

    Since we use patching, we need to temporarily apply patches to get
    the full modified state_dict, then unpatch after saving.

    Returns full path to saved file, or None if not saved.
    """
    if not save_path or not save_path.strip():
        return None

    save_path = os.path.expanduser(save_path.strip())

    # Ensure directory exists
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path, exist_ok=True)
        except Exception as e:
            print(f"[Model Analyzer] Error creating directory: {e}")
            return None

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_filename and save_filename.strip():
        base_name = save_filename.strip()
        if base_name.lower().endswith('.safetensors'):
            base_name = base_name[:-12]
        filename = f"{base_name}_{timestamp}.safetensors"
    else:
        filename = f"modified_model_{timestamp}.safetensors"

    full_path = os.path.join(save_path, filename)

    try:
        # Try to get original model's metadata
        original_metadata = {}
        source_path = None

        # Use provided source path first
        if source_model_path and source_model_path.strip():
            source_path = os.path.expanduser(source_model_path.strip())

        # Fallback: Check if model_patcher has path info
        if not source_path:
            if hasattr(model_patcher, 'model_options'):
                opts = model_patcher.model_options
                if isinstance(opts, dict):
                    source_path = opts.get('model_path') or opts.get('path')

        # Fallback: Check for model.model_path attribute
        if not source_path and hasattr(model_patcher, 'model'):
            if hasattr(model_patcher.model, 'model_path'):
                source_path = model_patcher.model.model_path

        # Try to read metadata from source file
        if source_path and os.path.exists(source_path) and source_path.endswith('.safetensors'):
            try:
                with safe_open(source_path, framework="pt") as f:
                    original_metadata = dict(f.metadata()) if f.metadata() else {}
                print(f"[Model Analyzer] Copied metadata from source: {os.path.basename(source_path)}")
            except Exception as e:
                print(f"[Model Analyzer] Could not read source metadata: {e}")
        else:
            print(f"[Model Analyzer] Warning: No source model specified - saved model may not load correctly")

        # Load original file and apply modifications directly to preserve key format
        if source_path and os.path.exists(source_path) and source_path.endswith('.safetensors'):
            print(f"[Model Analyzer] Loading weights from source file...")
            from safetensors.torch import load_file
            state_dict = load_file(source_path)

            # Get the patches from model_patcher
            patches = model_patcher.patches.copy() if hasattr(model_patcher, 'patches') else {}

            if len(patches) > 0:
                import comfy.lora

                # Build prefix mapping: patches use diffusion_model. prefix but file doesn't
                # Try to detect the prefix used by patches
                patch_keys = list(patches.keys())
                file_keys = list(state_dict.keys())

                # Common prefixes in patched models
                possible_prefixes = ['diffusion_model.', 'model.diffusion_model.', 'model.', '']
                detected_prefix = ''

                for prefix in possible_prefixes:
                    # Check if removing this prefix from patch keys matches file keys
                    test_key = patch_keys[0]
                    if test_key.startswith(prefix):
                        stripped_key = test_key[len(prefix):]
                        if stripped_key in file_keys:
                            detected_prefix = prefix
                            print(f"[Model Analyzer] Detected patch key prefix: '{prefix}'")
                            break

                # Apply patches using ComfyUI's calculate_weight function
                modified_keys = 0
                for patch_key, patch_list in patches.items():
                    # Strip the prefix to get the original file key
                    file_key = patch_key[len(detected_prefix):] if patch_key.startswith(detected_prefix) else patch_key

                    if file_key in state_dict:
                        original_weight = state_dict[file_key]
                        # Use ComfyUI's proper patch calculation
                        # calculate_weight expects weight to be on the same device as patches
                        modified_weight = comfy.lora.calculate_weight(
                            patch_list,
                            original_weight.clone(),
                            file_key,
                            intermediate_dtype=torch.float32
                        )
                        state_dict[file_key] = modified_weight.to(original_weight.dtype)
                        modified_keys += 1

                print(f"[Model Analyzer] Applied patches to {modified_keys} keys")
            else:
                print(f"[Model Analyzer] No patches to apply")
        else:
            # Fallback: use model's state dict (may have different key format)
            print("[Model Analyzer] Warning: No source file - using internal model keys (may not load correctly)")
            model_patcher.patch_model()
            model = model_patcher.model

            if hasattr(model, 'state_dict'):
                state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                print("[Model Analyzer] Could not get model state_dict")
                model_patcher.unpatch_model()
                return None

            model_patcher.unpatch_model()

        # Start with original metadata, then add our modification info
        metadata = original_metadata.copy()
        metadata.update({
            'modified_by': 'comfyui-zimage-realtime-lora ModelLayerEditor',
            'modified_url': 'https://github.com/ShootTheSound/comfyUI-Realtime-Lora',
            'modified_date': datetime.now().isoformat(),
            'modified_architecture': architecture,
            'block_modifications': json.dumps(block_info)
        })

        # Estimate file size
        total_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
        size_gb = total_bytes / (1024**3)
        print(f"[Model Analyzer] Saving model ({size_gb:.2f} GB) to: {full_path}")

        # Save
        save_file(state_dict, full_path, metadata=metadata)
        print(f"[Model Analyzer] Saved successfully: {full_path}")

        return full_path

    except Exception as e:
        import traceback
        print(f"[Model Analyzer] Error saving model: {e}")
        traceback.print_exc()
        # Make sure to unpatch if there was an error
        try:
            model_patcher.unpatch_model()
        except:
            pass
        return None


# ============================================================================
# FORMATTING
# ============================================================================

def _format_analysis(block_analysis: dict, architecture: str,
                     enabled_blocks: set, block_strengths: dict) -> str:
    """Format block analysis as readable text."""
    if not block_analysis:
        return "No blocks found."

    total_params = sum(b['param_count'] for b in block_analysis.values())
    total_memory = sum(b['memory_mb'] for b in block_analysis.values())

    lines = [
        f"Model Layer Analysis ({architecture})",
        "=" * 60,
        f"Total Parameters: {total_params:,}",
        f"Total Memory: {total_memory:.2f} MB ({total_memory/1024:.2f} GB)",
        "",
        f"{'Block':<20} {'Params':>12} {'Memory':>10} {'Strength':>10}",
        "-" * 60
    ]

    # Sort blocks logically
    def block_sort_key(item):
        block_id = item[0]
        # Extract number from block_id
        match = re.search(r'(\d+)', block_id)
        num = int(match.group(1)) if match else 999
        # Prefix for ordering
        if block_id.startswith('input'):
            return (0, num)
        elif block_id.startswith('layer'):
            return (1, num)
        elif block_id.startswith('double'):
            return (1, num)
        elif block_id == 'mid':
            return (2, 0)
        elif block_id.startswith('output'):
            return (3, num)
        elif block_id.startswith('single'):
            return (4, num)
        elif block_id.startswith('block'):
            return (5, num)
        else:
            return (9, num)

    sorted_blocks = sorted(block_analysis.items(), key=block_sort_key)

    for block_id, data in sorted_blocks:
        if block_id == 'other':
            continue  # Skip 'other' in main display

        params_str = f"{data['param_count']/1e6:.1f}M" if data['param_count'] >= 1e6 else f"{data['param_count']/1e3:.0f}K"
        memory_str = f"{data['memory_mb']:.1f} MB"

        if block_id in enabled_blocks:
            strength = block_strengths.get(block_id, 1.0)
            strength_str = f"{strength*100:.0f}%"
        else:
            strength_str = "OFF"

        lines.append(f"{block_id:<20} {params_str:>12} {memory_str:>10} {strength_str:>10}")

    # Show 'other' at end
    if 'other' in block_analysis:
        data = block_analysis['other']
        params_str = f"{data['param_count']/1e6:.1f}M" if data['param_count'] >= 1e6 else f"{data['param_count']/1e3:.0f}K"
        memory_str = f"{data['memory_mb']:.1f} MB"
        lines.append(f"{'other':<20} {params_str:>12} {memory_str:>10} {'100%':>10}")

    lines.append("-" * 60)
    lines.append(f"Enabled: {len(enabled_blocks)}/{len(block_analysis)} blocks")

    scaled = [f"{b}={block_strengths[b]:.0%}" for b in enabled_blocks
              if block_strengths.get(b, 1.0) != 1.0]
    if scaled:
        lines.append(f"Scaled: {', '.join(scaled)}")

    return '\n'.join(lines)


def _create_analysis_json(block_analysis: dict, architecture: str,
                          enabled_blocks: set, block_strengths: dict) -> str:
    """Create JSON analysis output."""
    result = {
        'architecture': architecture,
        'total_params': sum(b['param_count'] for b in block_analysis.values()),
        'total_memory_mb': sum(b['memory_mb'] for b in block_analysis.values()),
        'blocks': {}
    }

    for block_id, data in block_analysis.items():
        result['blocks'][block_id] = {
            'param_count': data['param_count'],
            'memory_mb': round(data['memory_mb'], 2),
            'enabled': block_id in enabled_blocks,
            'strength': block_strengths.get(block_id, 1.0) if block_id in enabled_blocks else 0.0
        }

    return json.dumps(result)


# ============================================================================
# ARCHITECTURE CONFIGS
# ============================================================================

ARCH_CONFIGS = {
    "SDXL": {
        "node_id": "SDXLModelLayerEditor",
        "display_name": "SDXL Model Layer Editor",
        "description": """Per-block control of SDXL model layers.

Block Guide:
- Input 0-11: Encoder blocks (structure, composition)
- Mid: Bottleneck (high-level features)
- Output 0-5: Main decoder (style in 1-2)
- Output 6-11: Late decoder (details, textures)""",
        "architecture": "SDXL",
        "blocks": [f"input_{i}" for i in range(12)] + ["mid"] + [f"output_{i}" for i in range(12)] + ["other"],
        "block_labels": {f"input_{i}": f"Input {i}" for i in range(12)} |
                        {"mid": "Middle"} |
                        {f"output_{i}": f"Output {i}" for i in range(12)} |
                        {"other": "Other Weights"},
        "presets": {
            "Default": {"enabled": "ALL", "strength": 1.0},
            "All Off": {"enabled": [], "strength": 0.0},
            "Half Strength": {"enabled": "ALL", "strength": 0.5},
            "Outputs Only": {"enabled": [f"output_{i}" for i in range(12)] + ["mid", "other"], "strength": 1.0},
            "Inputs Only": {"enabled": [f"input_{i}" for i in range(12)] + ["mid", "other"], "strength": 1.0},
            "Custom": None,
        }
    },
    "SD15": {
        "node_id": "SD15ModelLayerEditor",
        "display_name": "SD 1.5 Model Layer Editor",
        "description": """Per-block control of SD 1.5 model layers.""",
        "architecture": "SD15",
        "blocks": [f"input_{i}" for i in range(12)] + ["mid"] + [f"output_{i}" for i in range(12)] + ["other"],
        "block_labels": {f"input_{i}": f"Input {i}" for i in range(12)} |
                        {"mid": "Middle"} |
                        {f"output_{i}": f"Output {i}" for i in range(12)} |
                        {"other": "Other Weights"},
        "presets": {
            "Default": {"enabled": "ALL", "strength": 1.0},
            "All Off": {"enabled": [], "strength": 0.0},
            "Half Strength": {"enabled": "ALL", "strength": 0.5},
            "Custom": None,
        }
    },
    "FLUX": {
        "node_id": "FLUXModelLayerEditor",
        "display_name": "FLUX Model Layer Editor",
        "description": """Per-block control of FLUX model layers.

Block Guide:
- Double 0-18: Main transformer blocks (19 total)
- Single 0-37: Secondary blocks (38 total)""",
        "architecture": "FLUX",
        "blocks": [f"double_{i}" for i in range(19)] + [f"single_{i}" for i in range(38)] + ["other"],
        "block_labels": {f"double_{i}": f"Double {i}" for i in range(19)} |
                        {f"single_{i}": f"Single {i}" for i in range(38)} |
                        {"other": "Other Weights"},
        "presets": {
            "Default": {"enabled": "ALL", "strength": 1.0},
            "All Off": {"enabled": [], "strength": 0.0},
            "Half Strength": {"enabled": "ALL", "strength": 0.5},
            "Double Only": {"enabled": [f"double_{i}" for i in range(19)] + ["other"], "strength": 1.0},
            "Single Only": {"enabled": [f"single_{i}" for i in range(38)] + ["other"], "strength": 1.0},
            "Custom": None,
        }
    },
    "ZIMAGE": {
        "node_id": "ZImageModelLayerEditor",
        "display_name": "Z-Image Model Layer Editor",
        "description": """Per-block control of Z-Image model layers.

Layer Guide:
- Layers 0-9: Early processing (structure)
- Layers 10-19: Mid processing (features)
- Layers 20-29: Late processing (details)""",
        "architecture": "ZIMAGE",
        "blocks": [f"layer_{i}" for i in range(30)] + ["other"],
        "block_labels": {f"layer_{i}": f"Layer {i}" for i in range(30)} | {"other": "Other Weights"},
        "presets": {
            "Default": {"enabled": "ALL", "strength": 1.0},
            "All Off": {"enabled": [], "strength": 0.0},
            "Half Strength": {"enabled": "ALL", "strength": 0.5},
            "Late Only (20-29)": {"enabled": [f"layer_{i}" for i in range(20, 30)] + ["other"], "strength": 1.0},
            "Mid-Late (15-29)": {"enabled": [f"layer_{i}" for i in range(15, 30)] + ["other"], "strength": 1.0},
            "Custom": None,
        }
    },
    "WAN": {
        "node_id": "WanModelLayerEditor",
        "display_name": "Wan Model Layer Editor",
        "description": """Per-block control of Wan 2.2 model layers (40 blocks).""",
        "architecture": "WAN",
        "blocks": [f"block_{i}" for i in range(40)] + ["other"],
        "block_labels": {f"block_{i}": f"Block {i}" for i in range(40)} | {"other": "Other Weights"},
        "presets": {
            "Default": {"enabled": "ALL", "strength": 1.0},
            "All Off": {"enabled": [], "strength": 0.0},
            "Half Strength": {"enabled": "ALL", "strength": 0.5},
            "Late Only (30-39)": {"enabled": [f"block_{i}" for i in range(30, 40)] + ["other"], "strength": 1.0},
            "Custom": None,
        }
    },
    "QWEN": {
        "node_id": "QwenModelLayerEditor",
        "display_name": "Qwen Model Layer Editor",
        "description": """Per-block control of Qwen-Image model layers (60 blocks).""",
        "architecture": "QWEN",
        "blocks": [f"block_{i}" for i in range(60)] + ["other"],
        "block_labels": {f"block_{i}": f"Block {i}" for i in range(60)} | {"other": "Other Weights"},
        "presets": {
            "Default": {"enabled": "ALL", "strength": 1.0},
            "All Off": {"enabled": [], "strength": 0.0},
            "Half Strength": {"enabled": "ALL", "strength": 0.5},
            "Late Only (45-59)": {"enabled": [f"block_{i}" for i in range(45, 60)] + ["other"], "strength": 1.0},
            "Custom": None,
        }
    },
}


# ============================================================================
# PER-ARCHITECTURE NODE FACTORY
# ============================================================================

def _create_model_layer_editor_class(config: dict):
    """
    Factory function to create a per-architecture Model Layer Editor node.
    Similar to LoRA V2 combined nodes but for base models.
    """

    class ModelLayerEditor:
        """Per-architecture Model Layer Editor (generated from config)."""

        _config = config

        @classmethod
        def INPUT_TYPES(cls):
            cfg = cls._config
            blocks = cfg["blocks"]
            node_id = cfg["node_id"]
            builtin_presets = cfg["presets"]

            # Get combined builtin + user presets
            all_presets = _get_all_presets(node_id, builtin_presets)

            inputs = {
                "required": {
                    "model": ("MODEL", {
                        "tooltip": "Model from LoadCheckpoint or LoadDiffusionModel"
                    }),
                    "preset": (list(all_presets.keys()), {
                        "default": "Default",
                        "tooltip": "Quick preset selection. Individual controls below override when changed."
                    }),
                },
                "optional": {
                    "save_preset_name": ("STRING", {
                        "default": "",
                        "tooltip": "Enter preset name, then click Save Preset button (or leave for file sync on execute)"
                    }),
                    "delete_preset_name": ("STRING", {
                        "default": "",
                        "tooltip": "Preset to delete (filled automatically by Delete button)"
                    }),
                    "browser_presets_json": ("STRING", {
                        "default": "{}",
                        "tooltip": "JSON of all browser presets (auto-filled by JS for sync)"
                    }),
                    "save_model": ("BOOLEAN", {
                        "default": False,
                        "tooltip": "Save modified model to disk (large files!)"
                    }),
                    "save_path": ("STRING", {
                        "default": "",
                        "tooltip": "Directory to save modified model"
                    }),
                    "save_filename": ("STRING", {
                        "default": "",
                        "tooltip": "Filename for saved model (timestamp added)"
                    }),
                    "source_model_path": ("STRING", {
                        "default": "",
                        "tooltip": "Path to original model file (for copying metadata). Required for saved model to load correctly."
                    }),
                }
            }

            # Add per-block toggle and strength inputs
            for block in blocks:
                inputs["required"][block] = ("BOOLEAN", {"default": True})
                inputs["required"][f"{block}_str"] = ("FLOAT", {
                    "default": 1.0,
                    "min": -2.0,  # Must match JS hardcoded value
                    "max": 2.0,
                    "step": 0.05
                })

            return inputs

        RETURN_TYPES = ("MODEL", "STRING", "STRING")
        RETURN_NAMES = ("model", "analysis", "analysis_json")
        OUTPUT_TOOLTIPS = (
            "Model with block modifications applied",
            "Text analysis of model structure",
            "JSON analysis data"
        )
        FUNCTION = "edit_model"
        CATEGORY = "model"
        OUTPUT_NODE = True

        def edit_model(self, model, preset, save_preset_name="", delete_preset_name="", browser_presets_json="{}",
                       save_model=False, save_path="", save_filename="", source_model_path="",
                       **kwargs):
            cfg = self._config
            blocks = cfg["blocks"]
            node_id = cfg["node_id"]
            architecture = cfg["architecture"]

            print(f"[{cfg['display_name']}] Starting...")

            # Analyze the model
            block_analysis = _analyze_model_blocks(model, architecture)
            print(f"[{cfg['display_name']}] Found {len(block_analysis)} blocks")

            # Determine enabled blocks and strengths from kwargs
            # NOTE: Always read from kwargs (actual widget values), not preset config
            # The preset dropdown only tells JavaScript which values to set on the widgets.
            # Python should always use the actual widget values so manual adjustments work.
            enabled_blocks = set()
            block_strengths = {}

            for block in blocks:
                if kwargs.get(block, True):
                    enabled_blocks.add(block)
                    block_strengths[block] = kwargs.get(f"{block}_str", 1.0)
                else:
                    # Block is disabled - add with strength 0.0 so we know it's in our config
                    block_strengths[block] = 0.0

            print(f"[{cfg['display_name']}] Enabled: {len(enabled_blocks)}/{len(blocks)} blocks")

            # Full sync between browser presets and file presets
            # This handles: adding new presets, deleting removed presets
            print(f"[Model Layer Editor] Browser presets JSON: {browser_presets_json[:100] if browser_presets_json else 'None'}...")

            if browser_presets_json and isinstance(browser_presets_json, str) and browser_presets_json.strip():
                try:
                    browser_presets = json.loads(browser_presets_json)
                    if isinstance(browser_presets, dict):
                        file_presets = _load_user_presets(node_id)
                        changed = False

                        print(f"[Model Layer Editor] Syncing: {len(browser_presets)} browser presets, {len(file_presets)} file presets")

                        # Add presets that are in browser but not in file
                        for name, data in browser_presets.items():
                            if not name or name.lower() in ('true', 'false', 'none', ''):
                                continue
                            if name not in file_presets:
                                file_presets[name] = data
                                changed = True
                                print(f"[Model Layer Editor] Added preset '{name}' to file")

                        # Remove presets that are in file but not in browser (user deleted them)
                        file_preset_names = list(file_presets.keys())
                        for name in file_preset_names:
                            if name not in browser_presets:
                                del file_presets[name]
                                changed = True
                                print(f"[Model Layer Editor] Removed preset '{name}' from file")

                        if changed:
                            _save_user_presets(node_id, file_presets)
                        else:
                            print(f"[Model Layer Editor] Presets already in sync")

                except (json.JSONDecodeError, TypeError) as e:
                    print(f"[Model Layer Editor] Error parsing browser presets: {e}")
            else:
                print(f"[Model Layer Editor] No browser presets to sync (empty or invalid)")

            # Save as preset if requested (syncs JS localStorage save to file)
            # Skip if it's a boolean or boolean-like string (widget corruption)
            saved_preset_data = None
            if save_preset_name and isinstance(save_preset_name, str) and save_preset_name.strip():
                save_name = save_preset_name.strip()
                if save_name.lower() not in ('true', 'false', 'none', ''):
                    _save_current_as_preset(node_id, save_name, blocks, enabled_blocks, block_strengths)
                    # Prepare preset data for JS sync (only if we actually saved)
                    saved_preset_data = {
                        "node_id": node_id,
                        "preset_name": save_name,
                        "preset": {
                            "enabled": "ALL" if set(blocks) == enabled_blocks else list(enabled_blocks),
                            "strength": 1.0,
                            "overrides": {b: s for b, s in block_strengths.items() if s != 1.0}
                        }
                    }

            # Check if any modifications needed
            needs_modification = False
            for block in blocks:
                if block not in enabled_blocks:
                    needs_modification = True
                    break
                if block_strengths.get(block, 1.0) != 1.0:
                    needs_modification = True
                    break

            # Apply modifications
            modified_count = 0
            if needs_modification:
                model, modified_count = _apply_block_modifications(
                    model, block_analysis, enabled_blocks, block_strengths, architecture
                )
                print(f"[{cfg['display_name']}] Modified {modified_count} tensors")

            # Format output
            analysis_text = _format_analysis(block_analysis, architecture, enabled_blocks, block_strengths)
            analysis_json = _create_analysis_json(block_analysis, architecture, enabled_blocks, block_strengths)

            # Include ALL user presets in JSON for JS sync (handles new computer/cleared storage)
            analysis_dict = json.loads(analysis_json)
            all_user_presets = _load_user_presets(node_id)
            if all_user_presets:
                analysis_dict["user_presets"] = {
                    "node_id": node_id,
                    "presets": all_user_presets
                }

            # Include just-saved preset info for confirmation
            if saved_preset_data:
                analysis_dict["saved_preset"] = saved_preset_data
                analysis_text += f"\n\n✓ Saved preset '{save_preset_name.strip()}' (restart ComfyUI to see in dropdown)"

            analysis_json = json.dumps(analysis_dict)

            # Save model if requested
            if save_model and needs_modification:
                block_info = {
                    'enabled': list(enabled_blocks),
                    'strengths': block_strengths,
                    'modified_count': modified_count
                }
                saved_path = _save_modified_model(
                    model, save_path, save_filename, architecture, block_info, source_model_path
                )
                if saved_path:
                    analysis_text += f"\n\nSaved model: {saved_path}"

            return (model, analysis_text, analysis_json)

    # Set class attributes from config
    ModelLayerEditor.__name__ = config["node_id"]
    ModelLayerEditor.__doc__ = config.get("description", "Model Layer Editor")
    ModelLayerEditor.DESCRIPTION = config.get("description", "Model Layer Editor")

    return ModelLayerEditor


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Generate per-architecture editor nodes
for arch_name, config in ARCH_CONFIGS.items():
    if "node_id" in config:  # Only configs with node_id
        node_class = _create_model_layer_editor_class(config)
        NODE_CLASS_MAPPINGS[config["node_id"]] = node_class
        NODE_DISPLAY_NAME_MAPPINGS[config["node_id"]] = config["display_name"]
