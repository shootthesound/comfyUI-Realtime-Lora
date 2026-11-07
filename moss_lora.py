"""MOSS-Image layer mapping, filtering, and selective loader."""

import math
import os
import re

import torch
import comfy.sd
import folder_paths
from safetensors.torch import load_file


MOSS_LAYER_COUNT = 42
MOSS_BLOCKS = [f"layer_{i}" for i in range(MOSS_LAYER_COUNT)] + ["other_weights"]
MOSS_PRESETS = {
    "Default": {"enabled": "ALL", "strength": 1.0},
    "All Off": {"enabled": [], "strength": 0.0},
    "Half Strength": {"enabled": "ALL", "strength": 0.5},
    "Early Only (0-13)": {"enabled": MOSS_BLOCKS[:14] + ["other_weights"], "strength": 1.0},
    "Mid Only (14-27)": {"enabled": MOSS_BLOCKS[14:28] + ["other_weights"], "strength": 1.0},
    "Late Only (28-41)": {"enabled": MOSS_BLOCKS[28:], "strength": 1.0},
    "Evens Only": {"enabled": MOSS_BLOCKS[:42:2] + ["other_weights"], "strength": 1.0},
    "Odds Only": {"enabled": MOSS_BLOCKS[1:42:2] + ["other_weights"], "strength": 1.0},
    "Custom": None,
}


def extract_moss_layer(key):
    key = key.lower()
    match = re.match(r"^(?:base_model\.model\.)?(?:(?:diffusion_model|transformer)\.)?layers\.(\d+)\.", key)
    if not match:
        match = re.match(r"^(?:lora_unet_|lora_transformer_|lycoris_)layers_(\d+)_", key)
    return f"layer_{int(match.group(1))}" if match else "other"


def is_moss_lora(keys):
    return any(
        extract_moss_layer(key) != "other"
        and re.search(r"[._]proj[._](?:ffn_up_gate|ffn_down|qkv|q|k|v|o)(?:[._]|$)", key.lower())
        for key in keys
    )


def parse_moss_profile(text):
    if not text or not text.strip():
        return None
    text = text.strip()
    try:
        if not text.startswith("%"):
            values = [float(part.strip()) for part in text.split(",")]
            if len(values) not in (42, 43):
                raise ValueError("Expected 42 layer values or 43 including other_weights")
            if len(values) == 42:
                values.append(1.0)
            result = dict(zip(MOSS_BLOCKS, values))
        else:
            overrides = {}
            default = 1.0
            for part in text[1:].split(","):
                name, value = part.strip().split("=", 1)
                name, value = name.strip().lower(), float(value)
                if name == "default":
                    default = value
                elif name in ("other", "other_weights"):
                    overrides["other_weights"] = value
                elif name == "layer":
                    overrides.update({block: value for block in MOSS_BLOCKS[:-1]})
                else:
                    match = re.fullmatch(r"layer_?(\d+)(?:-(\d+))?", name)
                    if not match:
                        raise ValueError(f"Unknown layer name: {name}")
                    start = int(match.group(1))
                    end = int(match.group(2)) if match.group(2) else start
                    if not 0 <= start <= end < MOSS_LAYER_COUNT:
                        raise ValueError(f"Layer range must be within 0-41: {name}")
                    overrides.update({f"layer_{i}": value for i in range(start, end + 1)})
            result = {block: overrides.get(block, default) for block in MOSS_BLOCKS}
        if not all(math.isfinite(value) for value in result.values()):
            raise ValueError("Layer strengths must be finite numbers")
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid MOSS-Image block profile: {error}") from error
    return {block: (value != 0.0, value) for block, value in result.items()}


def filter_moss_lora(state_dict, enabled_blocks, block_strengths, other_enabled=True, other_strength=1.0):
    filtered = {}
    for key, tensor in state_dict.items():
        layer = extract_moss_layer(key)
        if layer != "other" and layer not in MOSS_BLOCKS:
            raise ValueError(f"MOSS-Image layer outside the supported 0-41 range: {key}")
        if layer == "other":
            if not other_enabled:
                continue
            strength = float(other_strength)
        else:
            if layer not in enabled_blocks:
                continue
            strength = float(block_strengths.get(layer, 1.0))
        if not math.isfinite(strength):
            raise ValueError("MOSS-Image layer strengths must be finite numbers")
        if strength == 0.0:
            continue

        # Scale one factor only: scaling both A and B squares the weight and loses its sign.
        scales_delta = re.search(
            r"(?:^|[.])(?:lora_(?:up|b)(?:\.default)?(?:\.weight|\.bias)?|"
            r"lora\.up(?:\.weight)?|lora_linear_layer\.up\.weight|"
            r"lokr_w1(?:_a)?|hada_w1_a|diff|diff_b)$", key.lower()
        )
        filtered[key] = tensor * strength if scales_delta and strength != 1.0 else tensor
    return filtered


class MOSSImageSelectiveLoRALoader:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "preset": (list(MOSS_PRESETS), {"default": "Default"}),
            },
            "optional": {
                "lora_path_opt": ("STRING", {"forceInput": True}),
                "analysis_json": ("STRING", {"forceInput": True}),
                "block_weights_string": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "42 layer values, optionally followed by other_weights. Connected strings override sliders. Named example: %default=0, layer14-27=1, other=0.2",
                }),
                "save_refined_lora": ("BOOLEAN", {"default": False}),
                "save_path": ("STRING", {"default": ""}),
                "save_filename": ("STRING", {"default": ""}),
            },
        }
        for block in MOSS_BLOCKS:
            inputs["required"][block] = ("BOOLEAN", {"default": True})
            inputs["required"][f"{block}_str"] = ("FLOAT", {
                "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
            })
        return inputs

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "info", "weights_output")
    OUTPUT_NODE = True
    FUNCTION = "load_lora"
    CATEGORY = "loaders/lora"
    DESCRIPTION = """Selective LoRA loader for MOSS-Image's 42 main layers.
Use other_weights for text-refiner, embedding, modulation, and output weights.
Saving bakes in layer weights only. Apply the saved LoRA at the same overall strength."""

    def load_lora(self, model, clip, lora_name, strength, preset, **kwargs):
        from .selective_lora_loader import _coerce_scalar_strength, _parse_block_weights_string
        from .lora_analyzer_v2 import _get_metadata, _save_refined_lora

        strength = _coerce_scalar_strength(strength)
        if not math.isfinite(strength):
            raise ValueError("Overall LoRA strength must be finite")
        parsed = _parse_block_weights_string(kwargs.get("block_weights_string", ""), "MOSS_IMAGE")
        if parsed is not None:
            values = {block: value if enabled else 0.0 for block, (enabled, value) in parsed.items()}
        else:
            preset_cfg = MOSS_PRESETS[preset]
            if preset_cfg is None:
                values = {
                    block: float(kwargs.get(f"{block}_str", 1.0)) if kwargs.get(block, True) else 0.0
                    for block in MOSS_BLOCKS
                }
            else:
                enabled = MOSS_BLOCKS if preset_cfg["enabled"] == "ALL" else preset_cfg["enabled"]
                values = {block: preset_cfg["strength"] if block in enabled else 0.0 for block in MOSS_BLOCKS}
        weights_output = ", ".join(f"{values[block]:.2f}" for block in MOSS_BLOCKS)
        analysis_json = kwargs.get("analysis_json") or ""
        lora_path = kwargs.get("lora_path_opt") or folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.isfile(lora_path):
            return {"ui": {"analysis_json": [analysis_json]}, "result": (model, clip, "Error: LoRA not found", weights_output)}
        if lora_path.lower().endswith(".safetensors"):
            state_dict = load_file(lora_path)
        else:
            state_dict = torch.load(lora_path, map_location="cpu", weights_only=True)
        if not is_moss_lora(state_dict):
            raise ValueError("No MOSS-Image layer projections found in this LoRA")
        enabled = {block for block in MOSS_BLOCKS[:-1] if values[block] != 0.0}
        filtered = filter_moss_lora(state_dict, enabled, values, values["other_weights"] != 0.0, values["other_weights"])
        if not filtered:
            return {"ui": {"analysis_json": [analysis_json]}, "result": (model, clip, "All layers disabled; no LoRA applied or saved", weights_output)}

        info = f"Loaded {len(filtered)}/{len(state_dict)} tensors\nEnabled: {len(enabled)}/42 layers"
        if kwargs.get("save_refined_lora"):
            save_path = kwargs.get("save_path", "")
            if save_path.strip():
                saved = _save_refined_lora(filtered, save_path, kwargs.get("save_filename", ""),
                                           "MOSSImageSelectiveLoRALoader", "MOSS_IMAGE", _get_metadata(lora_path))
                info += f"\nSaved: {saved}" if saved else "\nSave failed; check the ComfyUI console"
            else:
                info += "\nSave skipped: save_path is empty"
        model_out, clip_out = comfy.sd.load_lora_for_models(model, clip, filtered, strength, strength)
        return {"ui": {"analysis_json": [analysis_json]}, "result": (model_out, clip_out, info, weights_output)}
