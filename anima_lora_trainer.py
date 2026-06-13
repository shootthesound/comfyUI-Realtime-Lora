"""
Anima LoRA Trainer Node for ComfyUI

Trains Anima LoRAs using kohya-ss/sd-scripts (anima_train_network.py).
Anima is a model by Circlestone Labs using a Qwen3-0.6B text encoder
and the Qwen-Image VAE. Requires a recent sd-scripts checkout with
Anima support (anima_train_network.py present).
"""

import os
import sys
import json
import hashlib
import tempfile
import shutil
import subprocess
from datetime import datetime
import numpy as np
from PIL import Image

import folder_paths

from .anima_config_template import (
    generate_anima_training_config,
    save_config,
    ANIMA_VRAM_PRESETS,
)


# Global config for Anima trainer
_anima_config = {}
_anima_config_file = os.path.join(os.path.dirname(__file__), ".anima_config.json")

# Global cache for trained LoRAs
_anima_lora_cache = {}
_anima_cache_file = os.path.join(os.path.dirname(__file__), ".anima_lora_cache.json")


def _load_anima_config():
    """Load Anima config from disk."""
    global _anima_config
    if os.path.exists(_anima_config_file):
        try:
            with open(_anima_config_file, "r", encoding="utf-8") as f:
                _anima_config = json.load(f)
        except:
            _anima_config = {}


def _save_anima_config():
    """Save Anima config to disk."""
    try:
        with open(_anima_config_file, "w", encoding="utf-8") as f:
            json.dump(_anima_config, f, indent=2)
    except:
        pass


def _load_anima_cache():
    """Load Anima LoRA cache from disk."""
    global _anima_lora_cache
    if os.path.exists(_anima_cache_file):
        try:
            with open(_anima_cache_file, "r", encoding="utf-8") as f:
                _anima_lora_cache = json.load(f)
        except:
            _anima_lora_cache = {}


def _save_anima_cache():
    """Save Anima LoRA cache to disk."""
    try:
        with open(_anima_cache_file, "w", encoding="utf-8") as f:
            json.dump(_anima_lora_cache, f)
    except:
        pass


def _compute_image_hash(
    images,
    captions,
    training_steps,
    learning_rate,
    lora_rank,
    vram_mode,
    output_name,
    extra="",
    use_folder_path=False,
):
    """Compute a hash of all images, captions, and training parameters."""
    hasher = hashlib.sha256()

    if use_folder_path:
        # For folder paths, hash the file paths and modification times
        for img_path in images:
            hasher.update(img_path.encode("utf-8"))
            if os.path.exists(img_path):
                hasher.update(str(os.path.getmtime(img_path)).encode("utf-8"))
    else:
        # For tensor inputs, hash the image data
        for img_tensor in images:
            img_np = (img_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            img_bytes = img_np.tobytes()
            hasher.update(img_bytes)

    # Include all captions in hash
    captions_str = "|".join(captions)
    params_str = f"anima|{captions_str}|{training_steps}|{learning_rate}|{lora_rank}|{vram_mode}|{output_name}|{extra}|{len(images)}"
    hasher.update(params_str.encode("utf-8"))

    return hasher.hexdigest()[:16]


def _resolve_launcher(sd_scripts_path, custom_python_exe=""):
    """Resolve the command prefix used to launch accelerate.

    Tries, in order:
    1. custom_python_exe (accelerate binary next to it, else `python -m accelerate...`)
    2. accelerate inside sd-scripts venv/.venv (uv and traditional layouts)
    3. python inside sd-scripts venv/.venv via `-m accelerate.commands.launch`
    4. accelerate on PATH (system-wide install, common on RunPod/containers)
    5. the Python running ComfyUI via `-m accelerate.commands.launch`

    Returns a complete launcher command prefix (e.g. ['accelerate', 'launch']
    or ['python', '-m', 'accelerate.commands.launch']); the caller appends
    the launch arguments and the training script.
    """
    import shutil as _shutil

    def _accel_cmd_from_python(python_path):
        return [python_path, "-m", "accelerate.commands.launch"]

    # 1. Custom python exe
    if custom_python_exe and custom_python_exe.strip():
        custom_python = custom_python_exe.strip()
        if not os.path.exists(custom_python):
            raise FileNotFoundError(f"Custom python.exe not found at: {custom_python}")
        venv_scripts_dir = os.path.dirname(custom_python)
        if sys.platform == "win32":
            accel = os.path.join(venv_scripts_dir, "accelerate.exe")
        else:
            accel = os.path.join(venv_scripts_dir, "accelerate")
        if os.path.exists(accel):
            return [accel, "launch"]
        # Fall back to module invocation with the custom python
        return _accel_cmd_from_python(custom_python)

    # 2./3. sd-scripts venv layouts
    for venv_folder in (".venv", "venv"):
        if sys.platform == "win32":
            bin_dir = os.path.join(sd_scripts_path, venv_folder, "Scripts")
            accel = os.path.join(bin_dir, "accelerate.exe")
            py = os.path.join(bin_dir, "python.exe")
        else:
            bin_dir = os.path.join(sd_scripts_path, venv_folder, "bin")
            accel = os.path.join(bin_dir, "accelerate")
            py = os.path.join(bin_dir, "python")
        if os.path.exists(accel):
            return [accel, "launch"]
        if os.path.exists(py):
            return _accel_cmd_from_python(py)

    # 4. accelerate on PATH (e.g. pip install into system python on RunPod)
    accel_on_path = _shutil.which("accelerate")
    if accel_on_path:
        return [accel_on_path, "launch"]

    # 5. Last resort: the Python running ComfyUI. Works when sd-scripts
    # requirements were installed into the same environment as ComfyUI.
    try:
        import accelerate  # noqa: F401

        print(
            "[Anima LoRA] No venv/accelerate found - falling back to ComfyUI's own Python environment."
        )
        return _accel_cmd_from_python(sys.executable)
    except ImportError:
        pass

    raise FileNotFoundError(
        f"Could not find 'accelerate' anywhere.\n"
        f"Checked: custom_python_exe, {sd_scripts_path}/venv and /.venv, system PATH, and ComfyUI's Python.\n"
        f"Fix options:\n"
        f"  - Install sd-scripts requirements into a venv inside the sd-scripts folder, OR\n"
        f"  - pip install the sd-scripts requirements (incl. accelerate) into the Python that runs ComfyUI, OR\n"
        f"  - Set custom_python_exe to the python binary of the environment where sd-scripts is installed."
    )


# Load config and cache on module import
_load_anima_config()
_load_anima_cache()


class AnimaLoraTrainer:
    """
    Trains an Anima LoRA from one or more images using kohya sd-scripts.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Get saved settings or use defaults
        if sys.platform == "win32":
            sd_scripts_fallback = "S:\\Auto\\sd-scripts"
        else:
            sd_scripts_fallback = "~/sd-scripts"

        saved = _anima_config.get("trainer_settings", {})

        # Model dropdowns from ComfyUI folders
        # Get list of checkpoints from ComfyUI
        checkpoints = folder_paths.get_filename_list("checkpoints")
        try:
            diffusion_models = folder_paths.get_filename_list("diffusion_models")
        except:
            diffusion_models = []

        diffusion_models = sorted(set(checkpoints) | set(diffusion_models))

        vae_models = folder_paths.get_filename_list("vae")
        try:
            text_encoders = folder_paths.get_filename_list("text_encoders")
        except:
            text_encoders = []
        try:
            clip_models = folder_paths.get_filename_list("clip")
        except:
            clip_models = []
        # Merge text_encoders and clip folders (newer/older ComfyUI conventions)
        te_list = sorted(set(text_encoders) | set(clip_models))
        if not te_list:
            te_list = ["none found - place Qwen3-0.6B in models/text_encoders"]

        return {
            "required": {
                "inputcount": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Number of image inputs. Click 'Update inputs' button after changing.",
                    },
                ),
                "images_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional: Path to folder containing training images. If provided, images from this folder are used instead of image inputs. Caption .txt files with matching names are used if present.",
                    },
                ),
                "sd_scripts_path": (
                    "STRING",
                    {
                        "default": _anima_config.get(
                            "sd_scripts_path", sd_scripts_fallback
                        ),
                        "tooltip": "Path to kohya sd-scripts installation. Must be a recent version with Anima support (anima_train_network.py).",
                    },
                ),
                "diffusion_model_name": (
                    diffusion_models,
                    {
                        "tooltip": "Anima model (.safetensors) from models/diffusion_models or models/checkpoints."
                    },
                ),
                "vae_name": (
                    vae_models,
                    {
                        "tooltip": "Qwen-Image VAE from models/vae (the official Anima weights use the Qwen-Image VAE)."
                    },
                ),
                "text_encoder_name": (
                    te_list,
                    {
                        "tooltip": "Qwen3-0.6B text encoder (.safetensors) from models/text_encoders (or models/clip)."
                    },
                ),
                "caption": (
                    "STRING",
                    {
                        "default": saved.get(
                            "caption", "anime illustration of subject"
                        ),
                        "multiline": True,
                        "tooltip": "Default caption for all images. Per-image caption inputs override this.",
                    },
                ),
                "training_steps": (
                    "INT",
                    {
                        "default": saved.get("training_steps", 500),
                        "min": 10,
                        "max": 5000,
                        "step": 10,
                        "tooltip": "Number of training steps. 500 is a good starting point. Increase for more images or complex subjects.",
                    },
                ),
                "learning_rate": (
                    "FLOAT",
                    {
                        "default": saved.get("learning_rate", 0.0001),
                        "min": 0.00001,
                        "max": 0.1,
                        "step": 0.00001,
                        "tooltip": "Learning rate. 1e-4 is the documented starting point for Anima (at alpha = rank consider lowering it).",
                    },
                ),
                "lora_rank": (
                    "INT",
                    {
                        "default": saved.get("lora_rank", 16),
                        "min": 4,
                        "max": 128,
                        "step": 4,
                        "tooltip": "LoRA rank/dimension. 8-32 typical for Anima. Higher = more capacity but larger file and more VRAM.",
                    },
                ),
                "vram_mode": (
                    ["Min (512px)", "Low (768px)", "Max (1024px)"],
                    {
                        "default": saved.get("vram_mode", "Low (768px)"),
                        "tooltip": "VRAM optimization preset. Min/Low enable block swapping and VAE chunking to reduce VRAM usage.",
                    },
                ),
                "timestep_sampling": (
                    ["sigmoid", "shift", "flux_shift", "uniform", "sigma"],
                    {
                        "default": saved.get("timestep_sampling", "sigmoid"),
                        "tooltip": "Timestep sampling method for Rectified Flow training. 'sigmoid' is the recommended default.",
                    },
                ),
                "discrete_flow_shift": (
                    "FLOAT",
                    {
                        "default": saved.get("discrete_flow_shift", 1.0),
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Discrete flow shift. Only used when timestep_sampling is 'shift'. Default 1.0.",
                    },
                ),
                "train_llm_adapter": (
                    "BOOLEAN",
                    {
                        "default": saved.get("train_llm_adapter", False),
                        "tooltip": "Also train LoRA on the LLM Adapter (Qwen3 -> T5 bridge). Uses a lowered adapter learning rate (5e-5) for stability. Off by default.",
                    },
                ),
                "save_every_n_epochs": (
                    "INT",
                    {
                        "default": saved.get("save_every_n_epochs", 0),
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Save an intermediate LoRA every N epochs (0 = off). NOTE: with few images one epoch is only a handful of steps, so this can produce MANY files - use save_last_n_epochs to limit, or prefer save_every_n_steps. Files: {name}-000001.safetensors etc. in the output folder.",
                    },
                ),
                "save_last_n_epochs": (
                    "INT",
                    {
                        "default": saved.get("save_last_n_epochs", 0),
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Keep only the last N epoch checkpoints, older ones are deleted automatically (0 = keep all). Only used when save_every_n_epochs > 0.",
                    },
                ),
                "save_every_n_steps": (
                    "INT",
                    {
                        "default": saved.get("save_every_n_steps", 0),
                        "min": 0,
                        "max": 5000,
                        "step": 10,
                        "tooltip": "Save an intermediate LoRA every N steps (0 = off). Usually more practical than epochs for step-based training. Files: {name}-step00000100.safetensors etc. in the output folder.",
                    },
                ),
                "keep_lora": (
                    "BOOLEAN",
                    {
                        "default": saved.get("keep_lora", True),
                        "tooltip": "If True, keeps the trained LoRA file.",
                    },
                ),
                "output_name": (
                    "STRING",
                    {
                        "default": saved.get("output_name", "MyAnimaLora"),
                        "tooltip": "Custom name for the output LoRA. Timestamp will be appended.",
                    },
                ),
                "custom_python_exe": (
                    "STRING",
                    {
                        "default": saved.get("custom_python_exe", ""),
                        "tooltip": "Advanced: Optionally enter the full path to a custom python.exe (e.g. C:\\my-venv\\Scripts\\python.exe). If empty, uses the venv inside sd_scripts_path. The sd_scripts_path field is still required for locating training scripts.",
                    },
                ),
            },
            "optional": {
                "image_1": (
                    "IMAGE",
                    {"tooltip": "Training image (not needed if images_path is set)."},
                ),
                "caption_1": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Caption for image_1. Overrides default caption.",
                    },
                ),
                "image_2": ("IMAGE", {"tooltip": "Training image."}),
                "caption_2": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Caption for image_2. Overrides default caption.",
                    },
                ),
                "image_3": ("IMAGE", {"tooltip": "Training image."}),
                "caption_3": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Caption for image_3. Overrides default caption.",
                    },
                ),
                "image_4": ("IMAGE", {"tooltip": "Training image."}),
                "caption_4": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Caption for image_4. Overrides default caption.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_path",)
    OUTPUT_TOOLTIPS = ("Path to the trained Anima LoRA file.",)
    FUNCTION = "train_anima_lora"
    CATEGORY = "loaders"
    DESCRIPTION = "Trains an Anima LoRA from images using kohya sd-scripts (anima_train_network.py)."

    def _resolve_text_encoder_path(self, text_encoder_name):
        """Resolve the text encoder path from text_encoders or clip folders."""
        for folder_type in ("text_encoders", "clip"):
            try:
                path = folder_paths.get_full_path(folder_type, text_encoder_name)
                if path and os.path.exists(path):
                    return path
            except:
                continue
        return None

    def train_anima_lora(
        self,
        inputcount,
        images_path,
        sd_scripts_path,
        diffusion_model_name,
        vae_name,
        text_encoder_name,
        caption,
        training_steps,
        learning_rate,
        lora_rank,
        vram_mode,
        timestep_sampling="sigmoid",
        discrete_flow_shift=1.0,
        train_llm_adapter=False,
        save_every_n_epochs=0,
        save_last_n_epochs=0,
        save_every_n_steps=0,
        keep_lora=True,
        output_name="MyAnimaLora",
        custom_python_exe="",
        image_1=None,
        **kwargs,
    ):
        global _anima_lora_cache

        # Resolve model paths
        sd_scripts_path = os.path.expanduser(sd_scripts_path.strip())
        diffusion_model_name_path = folder_paths.get_full_path(
            "diffusion_models", diffusion_model_name
        ) or folder_paths.get_full_path("checkpoints", diffusion_model_name)
        if not diffusion_model_name_path or not os.path.exists(
            diffusion_model_name_path
        ):
            raise FileNotFoundError(
                f"Anima DiT not found in diffusion_models or checkpoints: {diffusion_model_name}"
            )
        vae_path = folder_paths.get_full_path("vae", vae_name)
        qwen3_path = self._resolve_text_encoder_path(text_encoder_name)

        # Check if using folder path for images
        use_folder_path = False
        folder_images = []
        folder_captions = []

        if images_path and images_path.strip():
            images_path = os.path.expanduser(images_path.strip())
            if os.path.isdir(images_path):
                # Find all image files in the folder
                image_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
                for filename in sorted(os.listdir(images_path)):
                    if filename.lower().endswith(image_extensions):
                        img_path = os.path.join(images_path, filename)
                        folder_images.append(img_path)

                        # Look for matching caption file
                        base_name = os.path.splitext(filename)[0]
                        caption_file = os.path.join(images_path, f"{base_name}.txt")
                        if os.path.exists(caption_file):
                            with open(caption_file, "r", encoding="utf-8") as f:
                                folder_captions.append(f.read().strip())
                        else:
                            folder_captions.append(caption)  # Use default caption

                if folder_images:
                    use_folder_path = True
                    print(
                        f"[Anima LoRA] Using {len(folder_images)} images from folder: {images_path}"
                    )
                else:
                    print(
                        f"[Anima LoRA] No images found in folder: {images_path}, falling back to inputs"
                    )
            else:
                print(
                    f"[Anima LoRA] Invalid folder path: {images_path}, falling back to inputs"
                )

        if not use_folder_path:
            # Collect all images and captions from inputs
            # External caption_N inputs override the default caption widget
            all_images = []
            all_captions = []

            # image_1 is optional
            if image_1 is not None:
                all_images.append(image_1)
                cap_1 = kwargs.get("caption_1", "")
                all_captions.append(cap_1 if cap_1 else caption)

            for i in range(2, inputcount + 1):
                img = kwargs.get(f"image_{i}")
                if img is not None:
                    all_images.append(img)
                    # Get per-image caption, fall back to default if empty/missing
                    cap = kwargs.get(f"caption_{i}", "")
                    all_captions.append(cap if cap else caption)

            if not all_images:
                raise ValueError(
                    "No images provided. Either set images_path to a folder containing images, or connect at least one image input."
                )

        num_images = len(folder_images) if use_folder_path else len(all_images)
        print(f"[Anima LoRA] Training with {num_images} image(s)")
        print(
            f"[Anima LoRA] Diffusion Model: {diffusion_model_name} | VAE: {vae_name} | Qwen3: {text_encoder_name}"
        )

        # Get VRAM preset settings
        preset = ANIMA_VRAM_PRESETS.get(vram_mode, ANIMA_VRAM_PRESETS["Low (768px)"])
        print(f"[Anima LoRA] Using VRAM mode: {vram_mode}")

        # Validate paths
        train_script = os.path.join(sd_scripts_path, "anima_train_network.py")

        # Resolve the accelerate launcher (handles venv, system install,
        # custom python, and ComfyUI's own environment as fallback)
        launcher_cmd = _resolve_launcher(sd_scripts_path, custom_python_exe)
        print(f"[Anima LoRA] Launcher: {' '.join(launcher_cmd)}")

        if not os.path.exists(train_script):
            raise FileNotFoundError(
                f"anima_train_network.py not found at: {train_script}\n"
                f"Your sd-scripts installation may be too old. Update sd-scripts (main branch) to get Anima support."
            )
        if not diffusion_model_name_path or not os.path.exists(
            diffusion_model_name_path
        ):
            raise FileNotFoundError(
                f"Anima model not found at: {diffusion_model_name_path}"
            )
        if not vae_path or not os.path.exists(vae_path):
            raise FileNotFoundError(f"Qwen-Image VAE not found at: {vae_path}")
        if not qwen3_path or not os.path.exists(qwen3_path):
            raise FileNotFoundError(
                f"Qwen3-0.6B text encoder not found: {text_encoder_name}\n"
                f"Place it in models/text_encoders (or models/clip)."
            )

        # Save settings
        global _anima_config
        _anima_config["sd_scripts_path"] = sd_scripts_path
        _anima_config["trainer_settings"] = {
            "diffusion_model_name": diffusion_model_name,
            "vae_name": vae_name,
            "text_encoder_name": text_encoder_name,
            "caption": caption,
            "training_steps": training_steps,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "vram_mode": vram_mode,
            "timestep_sampling": timestep_sampling,
            "discrete_flow_shift": discrete_flow_shift,
            "train_llm_adapter": train_llm_adapter,
            "save_every_n_epochs": save_every_n_epochs,
            "save_last_n_epochs": save_last_n_epochs,
            "save_every_n_steps": save_every_n_steps,
            "keep_lora": keep_lora,
            "output_name": output_name,
            "custom_python_exe": custom_python_exe,
        }
        _save_anima_config()

        # Compute hash for caching
        extra = f"{diffusion_model_name}|{timestep_sampling}|{discrete_flow_shift}|{train_llm_adapter}"
        if use_folder_path:
            image_hash = _compute_image_hash(
                folder_images,
                folder_captions,
                training_steps,
                learning_rate,
                lora_rank,
                vram_mode,
                output_name,
                extra=extra,
                use_folder_path=True,
            )
        else:
            image_hash = _compute_image_hash(
                all_images,
                all_captions,
                training_steps,
                learning_rate,
                lora_rank,
                vram_mode,
                output_name,
                extra=extra,
                use_folder_path=False,
            )

        # Check cache
        if keep_lora and image_hash in _anima_lora_cache:
            cached_path = _anima_lora_cache[image_hash]
            if os.path.exists(cached_path):
                print(f"[Anima LoRA] Cache hit! Reusing: {cached_path}")
                return (cached_path,)
            else:
                del _anima_lora_cache[image_hash]
                _save_anima_cache()

        # Generate run name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = (
            f"{output_name}_{timestamp}" if output_name else f"anima_lora_{image_hash}"
        )

        # Output folder
        output_folder = os.path.join(sd_scripts_path, "output")
        os.makedirs(output_folder, exist_ok=True)
        lora_output_path = os.path.join(output_folder, f"{run_name}.safetensors")

        # Auto-increment if file somehow still exists (same second)
        if os.path.exists(lora_output_path):
            counter = 1
            while os.path.exists(
                os.path.join(output_folder, f"{run_name}_{counter}.safetensors")
            ):
                counter += 1
            run_name = f"{run_name}_{counter}"
            lora_output_path = os.path.join(output_folder, f"{run_name}.safetensors")
            print(f"[Anima LoRA] Name exists, using: {run_name}")

        # Create temp directory for images
        temp_dir = tempfile.mkdtemp(prefix="comfy_anima_lora_")
        image_folder = os.path.join(
            temp_dir, "1_subject"
        )  # sd-scripts format: repeats_class
        os.makedirs(image_folder, exist_ok=True)

        try:
            # Save images with captions
            if use_folder_path:
                # Copy images from folder and create caption files
                for idx, (src_path, cap) in enumerate(
                    zip(folder_images, folder_captions)
                ):
                    ext = os.path.splitext(src_path)[1]
                    dest_path = os.path.join(image_folder, f"image_{idx + 1:03d}{ext}")
                    shutil.copy2(src_path, dest_path)

                    caption_path = os.path.join(
                        image_folder, f"image_{idx + 1:03d}.txt"
                    )
                    with open(caption_path, "w", encoding="utf-8") as f:
                        f.write(cap)
            else:
                # Save tensor images
                for idx, img_tensor in enumerate(all_images):
                    img_data = img_tensor[0]
                    img_np = (img_data.cpu().numpy() * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_np)

                    image_path = os.path.join(image_folder, f"image_{idx + 1:03d}.png")
                    img_pil.save(image_path, "PNG")

                    caption_path = os.path.join(
                        image_folder, f"image_{idx + 1:03d}.txt"
                    )
                    with open(caption_path, "w", encoding="utf-8") as f:
                        f.write(all_captions[idx])

            print(f"[Anima LoRA] Saved {num_images} images to {image_folder}")

            # Generate config
            config_content = generate_anima_training_config(
                name=run_name,
                image_folder=temp_dir,  # Parent of the class folder
                output_folder=output_folder,
                diffusion_model_path=diffusion_model_name_path,
                qwen3_path=qwen3_path,
                vae_path=vae_path,
                steps=training_steps,
                learning_rate=learning_rate,
                lora_rank=lora_rank,
                resolution=preset["resolution"],
                batch_size=preset["batch_size"],
                optimizer=preset["optimizer"],
                mixed_precision=preset["mixed_precision"],
                gradient_checkpointing=preset["gradient_checkpointing"],
                cache_latents=preset["cache_latents"],
                cache_text_encoder_outputs=preset["cache_text_encoder_outputs"],
                blocks_to_swap=preset["blocks_to_swap"],
                timestep_sampling=timestep_sampling,
                discrete_flow_shift=discrete_flow_shift,
                train_llm_adapter=train_llm_adapter,
                vae_chunk_size=preset["vae_chunk_size"],
                vae_disable_cache=preset["vae_disable_cache"],
                save_every_n_epochs=save_every_n_epochs,
                save_last_n_epochs=save_last_n_epochs,
                save_every_n_steps=save_every_n_steps,
            )

            config_path = os.path.join(temp_dir, "training_config.toml")
            save_config(config_content, config_path)
            print(f"[Anima LoRA] Config saved to {config_path}")

            # Build command
            cmd = launcher_cmd + [
                "--num_cpu_threads_per_process=2",
                train_script,
                f"--config_file={config_path}",
            ]

            print(f"[Anima LoRA] Starting training: {run_name}")
            print(
                f"[Anima LoRA] Images: {num_images}, Steps: {training_steps}, LR: {learning_rate}, Rank: {lora_rank}"
            )

            # Run training
            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            # Set UTF-8 encoding for subprocess to handle Japanese text in sd-scripts
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=sd_scripts_path,
                startupinfo=startupinfo,
                env=env,
            )

            # Stream output
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    print(f"[sd-scripts] {line}")

            process.wait()

            if process.returncode != 0:
                raise RuntimeError(
                    f"sd-scripts Anima training failed with code {process.returncode}"
                )

            print("[Anima LoRA] Training completed!")

            # List intermediate checkpoints (epoch: {name}-000001, steps: {name}-step00000100)
            checkpoint_files = sorted(
                f
                for f in os.listdir(output_folder)
                if f.startswith(run_name + "-") and f.endswith(".safetensors")
            )
            if checkpoint_files:
                print(
                    f"[Anima LoRA] {len(checkpoint_files)} intermediate checkpoint(s) saved in: {output_folder}"
                )
                for f in checkpoint_files:
                    print(f"[Anima LoRA]   - {f}")

            # Find the trained LoRA (final file is exactly {run_name}.safetensors)
            if not os.path.exists(lora_output_path):
                # Fall back to the newest checkpoint if the final file is missing
                if checkpoint_files:
                    lora_output_path = os.path.join(output_folder, checkpoint_files[-1])
                    print(
                        "[Anima LoRA] Final file missing, using newest checkpoint instead."
                    )
                else:
                    raise FileNotFoundError(f"No LoRA file found in {output_folder}")

            print(f"[Anima LoRA] Found trained LoRA: {lora_output_path}")
            print(
                "[Anima LoRA] Note: LoRAs load directly in ComfyUI. If you trained text encoder or "
                "LLM adapter weights, convert with networks/convert_anima_lora_to_comfy.py from sd-scripts."
            )

            # Handle caching
            if keep_lora:
                _anima_lora_cache[image_hash] = lora_output_path
                _save_anima_cache()
                print(f"[Anima LoRA] LoRA saved and cached at: {lora_output_path}")
            else:
                print(f"[Anima LoRA] LoRA available at: {lora_output_path}")

            return (lora_output_path,)

        finally:
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"[Anima LoRA] Warning: Could not clean up temp dir: {e}")
