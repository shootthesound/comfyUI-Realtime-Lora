"""
LoRA Loader with Analysis for ComfyUI
Loads a LoRA and measures per-block contributions during inference.
Shows which blocks actually contribute to the generated image.
"""

import os
import re
import json
from collections import defaultdict
import threading

import torch
import folder_paths
import comfy.sd
import comfy.model_patcher
from safetensors.torch import load_file

from .moss_lora import extract_moss_layer, is_moss_lora


def _detect_architecture(keys):
    """Identify LoRA architecture from key patterns."""
    keys_lower = [k.lower() for k in keys]
    keys_str = ' '.join(keys_lower)
    num_keys = len(keys)

    if is_moss_lora(keys):
        return 'MOSS_IMAGE'

    # Check for Qwen-Image (transformer_blocks with img_mlp/txt_mlp/img_mod/txt_mod)
    if any('transformer_blocks' in k and any(x in k for x in ['img_mlp', 'txt_mlp', 'img_mod', 'txt_mod']) for k in keys_lower):
        return 'QWEN_IMAGE'

    # Check for Krea 2 (28 main SingleStreamBlocks plus txtfusion/projection modules)
    if 'lora_krea2' in keys_str or 'krea2' in keys_str or 'krea_2' in keys_str:
        return 'KREA2'
    if any(x in keys_str for x in ['txtfusion', 'txtmlp', 'tmlp', 'tproj']) and any(re.search(r'blocks[._]\d+', k) for k in keys_lower):
        return 'KREA2'
    if any(re.search(r'blocks[._]\d+[._].*(?:attn[._])?(?:wq|wk|wv|wo|gate)', k) for k in keys_lower):
        return 'KREA2'

    # Check for Flux formats - must check BEFORE Z-Image since both have single_transformer_blocks
    # AI-Toolkit format: transformer.single_transformer_blocks / transformer.double_blocks
    if any('transformer.single_transformer_blocks' in k or 'transformer.double_blocks' in k for k in keys_lower):
        return 'FLUX'
    # AI-Toolkit alternate format: transformer.transformer_blocks (double blocks)
    if any('transformer.transformer_blocks' in k and 'single_transformer_blocks' not in k for k in keys_lower):
        return 'FLUX'
    # Kohya/other format: lora_transformer_single_transformer_blocks / lora_transformer_double_blocks (underscores)
    if any('transformer_single_transformer_blocks' in k or 'transformer_double_blocks' in k for k in keys_lower):
        return 'FLUX'

    # Check for Z-Image patterns:
    # - diffusion_model.layers.N.attention/adaLN_modulation (ComfyUI/AI-Toolkit format)
    # - lora_unet_layers_N_attention (Musubi Tuner format)
    # - single_transformer_blocks WITHOUT transformer. prefix (older Z-Image format)
    if any('diffusion_model.layers.' in k and ('attention' in k or 'adaln' in k.lower()) for k in keys_lower):
        return 'ZIMAGE'
    if any('lora_unet_layers_' in k and 'attention' in k for k in keys_lower):
        return 'ZIMAGE'
    if any('single_transformer_blocks' in k and 'transformer.single_transformer_blocks' not in k for k in keys_lower):
        return 'ZIMAGE'

    # Check for Flux (double_blocks/single_blocks - ComfyUI/other formats)
    if any('double_blocks' in k or 'single_blocks' in k for k in keys_lower):
        return 'FLUX'

    # Check for Wan (blocks.N or blocks_N with self_attn/cross_attn/ffn)
    if any(('blocks.' in k or 'blocks_' in k) and any(x in k for x in ['self_attn', 'cross_attn', 'ffn'])
           for k in keys_lower):
        return 'WAN'

    # Check for SDXL - look for dual text encoders
    has_te1 = 'lora_te1_' in keys_str or 'text_encoder_1' in keys_str
    has_te2 = 'lora_te2_' in keys_str or 'text_encoder_2' in keys_str
    if has_te1 and has_te2:
        return 'SDXL'

    # SDXL has way more tensors than SD15 (2000+ vs 600-800)
    if num_keys > 1500:
        return 'SDXL'

    if any('input_blocks_7' in k or 'input_blocks_8' in k or
           'input_blocks.7' in k or 'input_blocks.8' in k for k in keys_lower):
        return 'SDXL'

    # Check for SD1.5 patterns
    if any('lora_unet_' in k or 'lora_te_' in k for k in keys_lower):
        return 'SD15'

    # Fallback based on tensor count
    if num_keys > 1000:
        return 'SDXL'

    if any('input_blocks' in k for k in keys_lower):
        return 'SD15'

    return 'UNKNOWN'


def _extract_block_id(key: str, architecture: str) -> str:
    """Extract block identifier from a LoRA/model weight key."""
    key_lower = key.lower()

    if architecture == 'MOSS_IMAGE':
        return extract_moss_layer(key)

    if architecture == 'QWEN_IMAGE':
        match = re.search(r'transformer_blocks[._](\d+)', key)
        return f"block_{match.group(1)}" if match else 'other'

    elif architecture == 'KREA2':
        if any(part in key_lower for part in ['txtfusion', 'txtmlp', 'tmlp', 'tproj', 'first', 'last']):
            return 'other'
        match = re.search(r'blocks[._](\d+)', key_lower)
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
        # Old format: single_transformer_blocks.N
        match = re.search(r'single_transformer_blocks\.(\d+)', key)
        if match:
            return f"block_{match.group(1)}"
        return 'other'

    elif architecture == 'WAN':
        # Handle both blocks.N and blocks_N patterns
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
        # Match diffusion_model patterns
        down = re.search(r'down_blocks?[._]?(\d+)', key_lower)
        if down:
            return f"unet_down_{down.group(1)}"
        if 'mid_block' in key_lower or 'middle_block' in key_lower:
            return "unet_mid"
        up = re.search(r'up_blocks?[._]?(\d+)', key_lower)
        if up:
            return f"unet_up_{up.group(1)}"
        # Input/output blocks for SD
        inp = re.search(r'input_blocks?[._]?(\d+)', key_lower)
        if inp:
            return f"input_{inp.group(1)}"
        out = re.search(r'output_blocks?[._]?(\d+)', key_lower)
        if out:
            return f"output_{out.group(1)}"
        return 'other'

    return 'other'


class LoRAContributionTracker:
    """
    Singleton that tracks LoRA contributions during inference.
    Uses weight_function wrappers to measure actual contributions.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.contributions = defaultdict(lambda: {'count': 0, 'total_norm': 0.0, 'total_delta': 0.0})
                    cls._instance.architecture = 'UNKNOWN'
                    cls._instance.enabled = False
                    cls._instance.lora_name = ""
        return cls._instance

    def reset(self):
        self.contributions = defaultdict(lambda: {'count': 0, 'total_norm': 0.0, 'total_delta': 0.0})

    def record(self, block_id: str, delta_norm: float):
        if self.enabled:
            self.contributions[block_id]['count'] += 1
            self.contributions[block_id]['total_delta'] += delta_norm

    def get_report(self) -> str:
        if not self.contributions:
            return "No LoRA contributions recorded yet.\nGenerate an image first, then check this output."

        total = sum(d['total_delta'] for d in self.contributions.values())
        if total == 0:
            return "No significant LoRA contributions detected."

        lines = [
            f"LoRA: {self.lora_name}",
            f"Architecture: {self.architecture}",
            "=" * 60,
            f"{'Block':<25} {'Contribution':>15} {'Calls':>10}",
            "-" * 60
        ]

        sorted_blocks = sorted(
            self.contributions.items(),
            key=lambda x: x[1]['total_delta'],
            reverse=True
        )

        for block_id, data in sorted_blocks:
            pct = (data['total_delta'] / total) * 100
            bar_len = int(pct / 5)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            lines.append(f"{block_id:<25} [{bar}] {pct:5.1f}%  ({data['count']:>5})")

        lines.append("-" * 60)
        lines.append(f"Total forward passes with LoRA: {sum(d['count'] for d in self.contributions.values())}")

        return '\n'.join(lines)


# Global tracker
_tracker = LoRAContributionTracker()


def _create_measuring_weight_function(original_weight_func, block_id: str, weight_key: str):
    """
    Wrap a weight function to measure LoRA contribution.
    The weight function is called with the base weight and returns the patched weight.
    """
    def measuring_wrapper(weight, *args, **kwargs):
        global _tracker

        # Get the original weight norm before patching
        original_norm = weight.norm().item()

        # Apply the original function (which applies LoRA patches)
        result = original_weight_func(weight, *args, **kwargs)

        # Measure the delta
        if _tracker.enabled and result is not None:
            try:
                result_norm = result.norm().item()
                delta = abs(result_norm - original_norm)
                _tracker.record(block_id, delta)
            except Exception:
                pass

        return result

    return measuring_wrapper


def _analyze_patches(model_patcher, architecture: str) -> dict:
    """
    Analyze the patches stored in the ModelPatcher.
    Returns per-block analysis of patch strengths and norms.

    Patch structure: (strength, patch_data, strength_model, offset, function)
    patch_data can be:
    - LoRAAdapter object (comfy.weight_adapter.lora.LoRAAdapter)
    - tuple like ("lora", (lora_up, lora_down, alpha, ...))
    - tensor directly
    """
    block_analysis = defaultdict(lambda: {
        'patch_count': 0,
        'total_strength': 0.0,
        'total_norm': 0.0,
        'keys': []
    })

    if not hasattr(model_patcher, 'patches'):
        return dict(block_analysis)

    for weight_key, patch_list in model_patcher.patches.items():
        block_id = _extract_block_id(weight_key, architecture)

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
                # Handle LoRAAdapter object (ComfyUI format)
                # weights tuple: (lora_up, lora_down, alpha, ...)
                if hasattr(patch_data, 'weights') and isinstance(patch_data.weights, tuple):
                    weights = patch_data.weights
                    if len(weights) >= 2:
                        lora_up = weights[0]  # shape: [out_features, rank]
                        lora_down = weights[1]  # shape: [rank, in_features]
                        if hasattr(lora_up, 'norm') and hasattr(lora_down, 'norm'):
                            up_norm = lora_up.float().norm().item()
                            down_norm = lora_down.float().norm().item()
                            norm_value = up_norm * down_norm

                # Handle older LoRAAdapter with direct attributes
                elif hasattr(patch_data, 'lora_up') and hasattr(patch_data, 'lora_down'):
                    lora_up = patch_data.lora_up
                    lora_down = patch_data.lora_down
                    if lora_up is not None and lora_down is not None:
                        up_norm = lora_up.float().norm().item()
                        down_norm = lora_down.float().norm().item()
                        norm_value = up_norm * down_norm

                # Handle tuple format ("lora", (up, down, alpha, ...))
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

                # Direct tensor
                elif hasattr(patch_data, 'norm'):
                    norm_value = patch_data.float().norm().item()

            except Exception as e:
                pass

            block_analysis[block_id]['total_norm'] += norm_value * effective_strength

    return dict(block_analysis)


def _format_patch_analysis(block_analysis: dict, architecture: str) -> str:
    """Format patch analysis as readable text."""
    if not block_analysis:
        return "No patches found."

    # Normalize scores to 0-100
    max_norm = max((d['total_norm'] for d in block_analysis.values()), default=1.0)
    if max_norm == 0:
        max_norm = 1.0

    lines = [
        f"LoRA Patch Analysis ({architecture})",
        "=" * 60,
        f"{'Block':<25} {'Score':>8} {'Patches':>10} {'Strength':>10}",
        "-" * 60
    ]

    # Sort by normalized score
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


def _create_analysis_json(block_analysis: dict, architecture: str, lora_name: str) -> str:
    """Create JSON analysis output for use by selective loaders."""
    if not block_analysis:
        return json.dumps({"architecture": architecture, "lora_name": lora_name, "blocks": {}})

    # Normalize scores to 0-100
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

    return json.dumps({
        "architecture": architecture,
        "lora_name": lora_name,
        "blocks": blocks
    })


class LoRALoaderWithAnalysis:
    """
    Loads a LoRA and provides per-block contribution analysis.

    Analysis output shows:
    1. Static analysis: Which blocks have patches and their relative strength
    2. Runtime analysis: After generation, shows actual contributions (use GetLoRAAnalysis node)
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
        "Per-block patch analysis. Shows which blocks are affected by this LoRA.",
        "JSON analysis data. Connect to Selective LoRA Loader for impact-colored UI.",
        "Full path to the loaded LoRA file. Connect to Selective LoRA Loader."
    )
    FUNCTION = "load_lora_with_analysis"
    CATEGORY = "loaders/lora"
    OUTPUT_NODE = True
    DESCRIPTION = "Loads a LoRA and analyzes which blocks it affects. Connect analysis_json to Selective Loaders for impact-colored checkboxes."

    def load_lora_with_analysis(self, model, clip, lora_name, strength_model, strength_clip):
        global _tracker

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.exists(lora_path):
            return {"ui": {"analysis_json": ["{}"]}, "result": (model, clip, "Error: LoRA file not found", "{}", "")}

        print(f"[LoRA Analyzer] Loading: {lora_name}")

        # Load LoRA state dict to detect architecture
        if lora_path.endswith('.safetensors'):
            lora_state_dict = load_file(lora_path)
        else:
            lora_state_dict = torch.load(lora_path, map_location='cpu')

        lora_keys = list(lora_state_dict.keys())
        architecture = _detect_architecture(lora_keys)
        print(f"[LoRA Analyzer] Architecture: {architecture}, {len(lora_state_dict)} tensors")

        # Debug: show sample keys to help identify architecture issues
        print(f"[LoRA Analyzer] Sample keys: {lora_keys[:10]}")

        # Reset tracker
        _tracker.reset()
        _tracker.architecture = architecture
        _tracker.lora_name = lora_name
        _tracker.enabled = True

        # Load the LoRA using ComfyUI's standard method
        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model,
            clip,
            lora_state_dict,
            strength_model,
            strength_clip
        )

        # Analyze the patches that were applied
        patch_analysis = _analyze_patches(model_lora, architecture)
        analysis_text = _format_patch_analysis(patch_analysis, architecture)
        analysis_json = _create_analysis_json(patch_analysis, architecture, lora_name)

        print(f"[LoRA Analyzer] Found {len(patch_analysis)} blocks with patches")
        print(analysis_text)

        return {"ui": {"analysis_json": [analysis_json]}, "result": (model_lora, clip_lora, analysis_text, analysis_json, lora_path)}


NODE_CLASS_MAPPINGS = {
    "LoRALoaderWithAnalysis": LoRALoaderWithAnalysis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRALoaderWithAnalysis": "LoRA Loader + Analyzer",
}
