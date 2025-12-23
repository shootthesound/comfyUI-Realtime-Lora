"""
Scheduled LoRA Loader for ComfyUI

A standalone LoRA loader with strength scheduling support.
No block selection - just simple load with optional scheduling.

Strength Schedule Format: "0:.2,.5:.8,1:1.0"
- Each pair is "step_percent:strength"
- Values interpolate linearly between keyframes
"""

import os

import torch
import folder_paths
import comfy.sd
import comfy.hooks
import comfy.utils


# ============================================================================
# STRENGTH SCHEDULE PRESETS
# ============================================================================

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
    # INVERTED VERSIONS
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

SCHEDULE_PRESET_LIST = list(SCHEDULE_PRESETS.keys())


# ============================================================================
# SCHEDULE PARSING AND INTERPOLATION
# ============================================================================

def _parse_strength_schedule(schedule_str: str) -> list:
    """
    Parse a strength schedule string into keyframes.

    Format: "0:.2,.2:.3,.5:.6,1:.9"
    Each pair is "percent:strength" where:
    - percent is 0.0 to 1.0 (proportion of steps)
    - strength is the LoRA strength at that point

    Returns list of (percent, strength) tuples, sorted by percent.
    """
    if not schedule_str or not schedule_str.strip():
        return None

    schedule_str = schedule_str.strip()

    # Check if it's just a float (no schedule)
    try:
        float(schedule_str)
        return None
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
            percent = max(0.0, min(1.0, percent))
            keyframes.append((percent, strength))
        except ValueError:
            continue

    if not keyframes:
        return None

    keyframes.sort(key=lambda x: x[0])
    return keyframes


def _interpolate_strength(keyframes: list, percent: float) -> float:
    """Linearly interpolate strength at a given percent."""
    if not keyframes:
        return 1.0

    if percent <= keyframes[0][0]:
        return keyframes[0][1]

    if percent >= keyframes[-1][0]:
        return keyframes[-1][1]

    for i in range(len(keyframes) - 1):
        p1, s1 = keyframes[i]
        p2, s2 = keyframes[i + 1]

        if p1 <= percent <= p2:
            if p2 == p1:
                return s1
            t = (percent - p1) / (p2 - p1)
            return s1 + t * (s2 - s1)

    return keyframes[-1][1]


def _create_hook_keyframes(keyframes: list, num_keyframes: int = 20) -> comfy.hooks.HookKeyframeGroup:
    """
    Create hook keyframes with linear interpolation between user-defined points.
    """
    if not keyframes:
        return None

    kf_group = comfy.hooks.HookKeyframeGroup()

    for i in range(num_keyframes + 1):
        percent = i / num_keyframes
        strength = _interpolate_strength(keyframes, percent)
        guarantee_steps = 1 if i == 0 else 0
        kf = comfy.hooks.HookKeyframe(strength=strength, start_percent=percent, guarantee_steps=guarantee_steps)
        kf_group.add(kf)

    return kf_group


def _invert_schedule(schedule_str: str) -> str:
    """
    Invert a schedule string by subtracting each strength from 1.0.

    "0:1, 0.5:0.3, 1:0" becomes "0:0, 0.5:0.7, 1:1"

    Useful for pairing two LoRAs that crossfade.
    """
    if not schedule_str or not schedule_str.strip():
        return ""

    keyframes = _parse_strength_schedule(schedule_str)
    if not keyframes:
        return ""

    inverted_pairs = []
    for percent, strength in keyframes:
        inv_strength = 1.0 - strength
        inverted_pairs.append(f"{percent}:{inv_strength}")

    return ", ".join(inverted_pairs)


# ============================================================================
# NODE CLASS
# ============================================================================

class ScheduledLoRALoader:
    """
    Simple LoRA loader with strength scheduling.

    No block selection - just straightforward loading with
    strength scheduling over the generation steps.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning from CLIP encode"}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning from CLIP encode"}),
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "LoRA file to load"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "LoRA strength (ignored when using schedule)"
                }),
                "schedule_preset": (SCHEDULE_PRESET_LIST, {
                    "default": "Custom",
                    "tooltip": "Select a preset schedule (populates the text field)"
                }),
                "strength_schedule": ("STRING", {
                    "default": "",
                    "tooltip": "Strength schedule: 0:.2,.5:.8,1:1.0 (step:strength pairs)"
                }),
            },
            "optional": {
                "schedule_in": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Schedule input from another node (overrides preset/text field)"
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "schedule_out", "schedule_inv")
    OUTPUT_TOOLTIPS = (
        "Model with LoRA applied.",
        "Positive conditioning (with hooks if using schedule).",
        "Negative conditioning (with hooks if using schedule).",
        "Active schedule string (for chaining to other loaders).",
        "Inverted schedule (1 - strength at each keyframe, for complementary LoRAs).",
    )
    FUNCTION = "load_lora"
    CATEGORY = "loaders/lora"
    DESCRIPTION = """Simple LoRA loader with strength scheduling.

Schedule format: "step:strength" pairs, comma-separated.
Example: "0:.2, .5:.8, 1:1.0"
  - 0% of steps: strength 0.2
  - 50% of steps: strength 0.8
  - 100% of steps: strength 1.0

Values interpolate linearly between keyframes.

Chain multiple loaders using schedule_out → schedule_in.
Use schedule_inv for complementary LoRA pairs (crossfade effect)."""

    def load_lora(self, model, positive, negative, lora_name, strength,
                  schedule_preset="Custom", strength_schedule="", schedule_in=None):

        # Get LoRA path
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.exists(lora_path):
            print(f"[ScheduledLoRALoader] Error: LoRA not found: {lora_name}")
            return (model, positive, negative, "", "")

        print(f"[ScheduledLoRALoader] Loading: {lora_name}")

        # Load LoRA file
        if lora_path.endswith('.safetensors'):
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        else:
            lora = torch.load(lora_path, map_location='cpu')

        # Determine active schedule (priority: schedule_in > text field > preset)
        if schedule_in and schedule_in.strip():
            effective_schedule = schedule_in.strip()
            print(f"[ScheduledLoRALoader] Using schedule from input")
        else:
            effective_schedule = strength_schedule.strip() if strength_schedule else ""
            if not effective_schedule and schedule_preset and schedule_preset != "Custom":
                effective_schedule = SCHEDULE_PRESETS.get(schedule_preset, "")

        # Generate inverted schedule for output
        schedule_inv = _invert_schedule(effective_schedule)

        schedule = _parse_strength_schedule(effective_schedule)

        if schedule:
            # Use hook system for scheduling
            print(f"[ScheduledLoRALoader] Using schedule: {effective_schedule}")

            # Create hooks - strength_model=1.0 so keyframe values ARE the strengths
            hooks = comfy.hooks.create_hook_lora(lora=lora, strength_model=1.0, strength_clip=0.0)
            kf_group = _create_hook_keyframes(schedule)

            if kf_group and hooks:
                hooks.set_keyframes_on_hooks(kf_group)

            # Clone model and register hooks
            model_out = model.clone()
            target_dict = comfy.hooks.create_target_dict(comfy.hooks.EnumWeightTarget.Model)
            model_out.register_all_hook_patches(hooks, target_dict)

            # Attach hooks to conditioning
            positive_out = comfy.hooks.set_hooks_for_conditioning(positive, hooks)
            negative_out = comfy.hooks.set_hooks_for_conditioning(negative, hooks)

            print(f"[ScheduledLoRALoader] Schedule applied with {len(schedule)} keyframes")
            return (model_out, positive_out, negative_out, effective_schedule, schedule_inv)

        else:
            # Standard loading without schedule
            model_out, _ = comfy.sd.load_lora_for_models(
                model, None, lora, strength, 0.0
            )

            print(f"[ScheduledLoRALoader] Loaded with strength={strength}")
            return (model_out, positive, negative, effective_schedule, schedule_inv)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ScheduledLoRALoader": ScheduledLoRALoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScheduledLoRALoader": "LoRA Loader (Scheduled)",
}
