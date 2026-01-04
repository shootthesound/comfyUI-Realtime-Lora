"""
SDXL LoRA Trainer Node for ComfyUI

Trains SDXL LoRAs using kohya-ss/sd-scripts.
Completely independent from the AI-Toolkit based trainer.
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

from .sdxl_config_template import (
    generate_sdxl_training_config,
    save_config,
    SDXL_VRAM_PRESETS,
)


# Global config for SDXL trainer
_sdxl_config = {}
_sdxl_config_file = os.path.join(os.path.dirname(__file__), ".sdxl_config.json")

# Global cache for trained LoRAs
_sdxl_lora_cache = {}
_sdxl_cache_file = os.path.join(os.path.dirname(__file__), ".sdxl_lora_cache.json")


def _load_sdxl_config():
    """Load SDXL config from disk."""
    global _sdxl_config
    if os.path.exists(_sdxl_config_file):
        try:
            with open(_sdxl_config_file, 'r', encoding='utf-8') as f:
                _sdxl_config = json.load(f)
        except:
            _sdxl_config = {}


def _save_sdxl_config():
    """Save SDXL config to disk."""
    try:
        with open(_sdxl_config_file, 'w', encoding='utf-8') as f:
            json.dump(_sdxl_config, f, indent=2)
    except:
        pass


def _load_sdxl_cache():
    """Load SDXL LoRA cache from disk."""
    global _sdxl_lora_cache
    if os.path.exists(_sdxl_cache_file):
        try:
            with open(_sdxl_cache_file, 'r', encoding='utf-8') as f:
                _sdxl_lora_cache = json.load(f)
        except:
            _sdxl_lora_cache = {}


def _save_sdxl_cache():
    """Save SDXL LoRA cache to disk."""
    try:
        with open(_sdxl_cache_file, 'w', encoding='utf-8') as f:
            json.dump(_sdxl_lora_cache, f)
    except:
        pass


def _compute_image_hash(images, captions, training_steps, learning_rate, lora_rank, vram_mode, output_name, use_folder_path=False):
    """Compute a hash of all images, captions, and training parameters."""
    hasher = hashlib.sha256()

    if use_folder_path:
        # For folder paths, hash the file paths and modification times
        for img_path in images:
            hasher.update(img_path.encode('utf-8'))
            if os.path.exists(img_path):
                hasher.update(str(os.path.getmtime(img_path)).encode('utf-8'))
    else:
        # For tensor inputs, hash the image data
        for img_tensor in images:
            img_np = (img_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            img_bytes = img_np.tobytes()
            hasher.update(img_bytes)

    # Include all captions in hash
    captions_str = "|".join(captions)
    params_str = f"sdxl|{captions_str}|{training_steps}|{learning_rate}|{lora_rank}|{vram_mode}|{output_name}|{len(images)}"
    hasher.update(params_str.encode('utf-8'))

    return hasher.hexdigest()[:16]


def _get_venv_python_path(sd_scripts_path):
    """Get the Python path for sd-scripts venv based on platform.
    Checks both .venv (uv default) and venv (traditional) folders."""
    venv_folders = [".venv", "venv"]

    for venv_folder in venv_folders:
        if sys.platform == 'win32':
            python_path = os.path.join(sd_scripts_path, venv_folder, "Scripts", "python.exe")
        else:
            python_path = os.path.join(sd_scripts_path, venv_folder, "bin", "python")

        if os.path.exists(python_path):
            return python_path

    # Return traditional path for error messaging
    if sys.platform == 'win32':
        return os.path.join(sd_scripts_path, "venv", "Scripts", "python.exe")
    else:
        return os.path.join(sd_scripts_path, "venv", "bin", "python")


def _get_accelerate_path(sd_scripts_path):
    """Get the accelerate path for sd-scripts venv based on platform.
    Checks both .venv (uv default) and venv (traditional) folders."""
    venv_folders = [".venv", "venv"]

    for venv_folder in venv_folders:
        if sys.platform == 'win32':
            accel_path = os.path.join(sd_scripts_path, venv_folder, "Scripts", "accelerate.exe")
        else:
            accel_path = os.path.join(sd_scripts_path, venv_folder, "bin", "accelerate")

        if os.path.exists(accel_path):
            return accel_path

    # Return traditional path for error messaging
    if sys.platform == 'win32':
        return os.path.join(sd_scripts_path, "venv", "Scripts", "accelerate.exe")
    else:
        return os.path.join(sd_scripts_path, "venv", "bin", "accelerate")


# Load config and cache on module import
_load_sdxl_config()
_load_sdxl_cache()


class SDXLLoraTrainer:
    """
    Trains an SDXL LoRA from one or more images using kohya sd-scripts.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Get saved settings or use defaults
        if sys.platform == 'win32':
            sd_scripts_fallback = 'S:\\Auto\\sd-scripts'
        else:
            sd_scripts_fallback = '~/sd-scripts'

        saved = _sdxl_config.get('trainer_settings', {})

        # Get list of checkpoints from ComfyUI
        checkpoints = folder_paths.get_filename_list("checkpoints")

        return {
            "required": {
                "inputcount": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Number of image inputs. Click 'Update inputs' button after changing."}),
                "images_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: Path to folder containing training images. If provided, images from this folder are used instead of image inputs. Caption .txt files with matching names are used if present."
                }),
                "sd_scripts_path": ("STRING", {
                    "default": _sdxl_config.get('sd_scripts_path', sd_scripts_fallback),
                    "tooltip": "Path to kohya sd-scripts installation."
                }),
                "ckpt_name": (checkpoints, {
                    "tooltip": "SDXL checkpoint to train LoRA on."
                }),
                "caption": ("STRING", {
                    "default": saved.get('caption', "photo of subject"),
                    "multiline": True,
                    "tooltip": "Default caption for all images. Per-image caption inputs override this."
                }),
                "training_steps": ("INT", {
                    "default": saved.get('training_steps', 500),
                    "min": 10,
                    "max": 5000,
                    "step": 10,
                    "tooltip": "Number of training steps. 500 is a good starting point. Increase for more images or complex subjects."
                }),
                "learning_rate": ("FLOAT", {
                    "default": saved.get('learning_rate', 0.0005),
                    "min": 0.00001,
                    "max": 0.1,
                    "step": 0.00001,
                    "tooltip": "Learning rate. 0.0005 trains fast but may overshoot. Experiment with lowering for more stable/slower training."
                }),
                "lora_rank": ("INT", {
                    "default": saved.get('lora_rank', 16),
                    "min": 4,
                    "max": 128,
                    "step": 4,
                    "tooltip": "LoRA rank/dimension. 16-32 typical. Higher = more capacity but larger file and more VRAM."
                }),
                "vram_mode": (["Min (512px)", "Low (768px)", "Max (1024px)"], {
                    "default": saved.get('vram_mode', "Low (768px)"),
                    "tooltip": "VRAM optimization preset. Images are automatically resized to the specified resolution."
                }),
                "keep_lora": ("BOOLEAN", {
                    "default": saved.get('keep_lora', True),
                    "tooltip": "If True, keeps the trained LoRA file."
                }),
                "output_name": ("STRING", {
                    "default": saved.get('output_name', "MyLora"),
                    "tooltip": "Custom name for the output LoRA. Timestamp will be appended."
                }),
                "custom_python_exe": ("STRING", {
                    "default": saved.get('custom_python_exe', ""),
                    "tooltip": "Advanced: Optionally enter the full path to a custom python.exe (e.g. C:\\my-venv\\Scripts\\python.exe). If empty, uses the venv inside sd_scripts_path. The sd_scripts_path field is still required for locating training scripts."
                }),
                "no_half_vae": ("BOOLEAN", {
                    "default": saved.get('no_half_vae', False),
                    "tooltip": "Disable half-precision for VAE. Enable this if you get NaN errors during training with certain models."
                }),
            },
            "optional": {
                "image_1": ("IMAGE", {"tooltip": "Training image (not needed if images_path is set)."}),
                "caption_1": ("STRING", {"forceInput": True, "tooltip": "Caption for image_1. Overrides default caption."}),
                "image_2": ("IMAGE", {"tooltip": "Training image."}),
                "caption_2": ("STRING", {"forceInput": True, "tooltip": "Caption for image_2. Overrides default caption."}),
                "image_3": ("IMAGE", {"tooltip": "Training image."}),
                "caption_3": ("STRING", {"forceInput": True, "tooltip": "Caption for image_3. Overrides default caption."}),
                "image_4": ("IMAGE", {"tooltip": "Training image."}),
                "caption_4": ("STRING", {"forceInput": True, "tooltip": "Caption for image_4. Overrides default caption."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_path",)
    OUTPUT_TOOLTIPS = ("Path to the trained SDXL LoRA file.",)
    FUNCTION = "train_sdxl_lora"
    CATEGORY = "loaders"
    DESCRIPTION = "Trains an SDXL LoRA from images using kohya sd-scripts."

    def train_sdxl_lora(
        self,
        inputcount,
        images_path,
        sd_scripts_path,
        ckpt_name,
        caption,
        training_steps,
        learning_rate,
        lora_rank,
        vram_mode,
        keep_lora=True,
        output_name="MyLora",
        custom_python_exe="",
        no_half_vae=False,
        image_1=None,
        **kwargs
    ):
        global _sdxl_lora_cache

        # Get full path to checkpoint
        model_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        # Check if using folder path for images
        use_folder_path = False
        folder_images = []
        folder_captions = []

        if images_path and images_path.strip():
            images_path = os.path.expanduser(images_path.strip())
            if os.path.isdir(images_path):
                # Find all image files in the folder
                image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
                for filename in sorted(os.listdir(images_path)):
                    if filename.lower().endswith(image_extensions):
                        img_path = os.path.join(images_path, filename)
                        folder_images.append(img_path)

                        # Look for matching caption file
                        base_name = os.path.splitext(filename)[0]
                        caption_file = os.path.join(images_path, f"{base_name}.txt")
                        if os.path.exists(caption_file):
                            with open(caption_file, 'r', encoding='utf-8') as f:
                                folder_captions.append(f.read().strip())
                        else:
                            folder_captions.append(caption)  # Use default caption

                if folder_images:
                    use_folder_path = True
                    print(f"[SDXL LoRA] Using {len(folder_images)} images from folder: {images_path}")
                else:
                    print(f"[SDXL LoRA] No images found in folder: {images_path}, falling back to inputs")
            else:
                print(f"[SDXL LoRA] Invalid folder path: {images_path}, falling back to inputs")

        if not use_folder_path:
            # Collect all images and captions from inputs
            # External caption_N inputs override the default caption widget
            all_images = []
            all_captions = []

            # image_1 is now optional
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
                raise ValueError("No images provided. Either set images_path to a folder containing images, or connect at least one image input.")

        num_images = len(folder_images) if use_folder_path else len(all_images)
        print(f"[SDXL LoRA] Training with {num_images} image(s)")
        print(f"[SDXL LoRA] Using model: {ckpt_name}")

        # Get VRAM preset settings (fallback handles old saved settings)
        preset = SDXL_VRAM_PRESETS.get(vram_mode, SDXL_VRAM_PRESETS["Low (768px)"])
        print(f"[SDXL LoRA] Using VRAM mode: {vram_mode}")

        # Validate paths
        train_script = os.path.join(sd_scripts_path, "sdxl_train_network.py")

        # Use custom python exe if provided, otherwise detect from sd_scripts_path
        if custom_python_exe and custom_python_exe.strip():
            custom_python = custom_python_exe.strip()
            if not os.path.exists(custom_python):
                raise FileNotFoundError(f"Custom python.exe not found at: {custom_python}")
            # Derive accelerate path from same directory as custom python
            venv_scripts_dir = os.path.dirname(custom_python)
            if sys.platform == 'win32':
                accelerate_path = os.path.join(venv_scripts_dir, "accelerate.exe")
            else:
                accelerate_path = os.path.join(venv_scripts_dir, "accelerate")
            if not os.path.exists(accelerate_path):
                raise FileNotFoundError(f"accelerate not found at: {accelerate_path} (expected in same directory as custom python)")
        else:
            accelerate_path = _get_accelerate_path(sd_scripts_path)
            if not os.path.exists(accelerate_path):
                raise FileNotFoundError(f"sd-scripts accelerate not found at: {accelerate_path}")

        if not os.path.exists(train_script):
            raise FileNotFoundError(f"sdxl_train_network.py not found at: {train_script}")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"SDXL model not found at: {model_path}")

        # Save settings
        global _sdxl_config
        _sdxl_config['sd_scripts_path'] = sd_scripts_path
        _sdxl_config['trainer_settings'] = {
            'ckpt_name': ckpt_name,
            'caption': caption,
            'training_steps': training_steps,
            'learning_rate': learning_rate,
            'lora_rank': lora_rank,
            'vram_mode': vram_mode,
            'keep_lora': keep_lora,
            'output_name': output_name,
            'custom_python_exe': custom_python_exe,
            'no_half_vae': no_half_vae,
        }
        _save_sdxl_config()

        # Compute hash for caching
        if use_folder_path:
            image_hash = _compute_image_hash(folder_images, folder_captions, training_steps, learning_rate, lora_rank, vram_mode, output_name, use_folder_path=True)
        else:
            image_hash = _compute_image_hash(all_images, all_captions, training_steps, learning_rate, lora_rank, vram_mode, output_name, use_folder_path=False)

        # Check cache
        if keep_lora and image_hash in _sdxl_lora_cache:
            cached_path = _sdxl_lora_cache[image_hash]
            if os.path.exists(cached_path):
                print(f"[SDXL LoRA] Cache hit! Reusing: {cached_path}")
                return (cached_path,)
            else:
                del _sdxl_lora_cache[image_hash]
                _save_sdxl_cache()

        # Generate run name with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{output_name}_{timestamp}" if output_name else f"sdxl_lora_{image_hash}"

        # Output folder
        output_folder = os.path.join(sd_scripts_path, "output")
        os.makedirs(output_folder, exist_ok=True)
        lora_output_path = os.path.join(output_folder, f"{run_name}.safetensors")

        # Auto-increment if file somehow still exists (same second)
        if os.path.exists(lora_output_path):
            counter = 1
            while os.path.exists(os.path.join(output_folder, f"{run_name}_{counter}.safetensors")):
                counter += 1
            run_name = f"{run_name}_{counter}"
            lora_output_path = os.path.join(output_folder, f"{run_name}.safetensors")
            print(f"[SDXL LoRA] Name exists, using: {run_name}")

        # Create temp directory for images
        temp_dir = tempfile.mkdtemp(prefix="comfy_sdxl_lora_")
        image_folder = os.path.join(temp_dir, "1_subject")  # sd-scripts format: repeats_class
        os.makedirs(image_folder, exist_ok=True)

        try:
            # Save images with captions
            if use_folder_path:
                # Copy images from folder and create caption files
                for idx, (src_path, cap) in enumerate(zip(folder_images, folder_captions)):
                    # Copy image to temp folder
                    ext = os.path.splitext(src_path)[1]
                    dest_path = os.path.join(image_folder, f"image_{idx+1:03d}{ext}")
                    shutil.copy2(src_path, dest_path)

                    # Create caption file
                    caption_path = os.path.join(image_folder, f"image_{idx+1:03d}.txt")
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(cap)
            else:
                # Save tensor images
                for idx, img_tensor in enumerate(all_images):
                    img_data = img_tensor[0]
                    img_np = (img_data.cpu().numpy() * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_np)

                    image_path = os.path.join(image_folder, f"image_{idx+1:03d}.png")
                    img_pil.save(image_path, "PNG")

                    # Use per-image caption
                    caption_path = os.path.join(image_folder, f"image_{idx+1:03d}.txt")
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(all_captions[idx])

            print(f"[SDXL LoRA] Saved {num_images} images to {image_folder}")

            # Generate config
            config_content = generate_sdxl_training_config(
                name=run_name,
                image_folder=temp_dir,  # Parent of the class folder
                output_folder=output_folder,
                model_path=model_path,
                steps=training_steps,
                learning_rate=learning_rate,
                lora_rank=lora_rank,
                lora_alpha=lora_rank,  # alpha = rank for full strength training
                resolution=preset['resolution'],
                batch_size=preset['batch_size'],
                optimizer=preset['optimizer'],
                mixed_precision=preset['mixed_precision'],
                gradient_checkpointing=preset['gradient_checkpointing'],
                cache_latents=preset['cache_latents'],
                cache_text_encoder_outputs=preset['cache_text_encoder_outputs'],
            )

            config_path = os.path.join(temp_dir, "training_config.toml")
            save_config(config_content, config_path)
            print(f"[SDXL LoRA] Config saved to {config_path}")

            # Build command
            cmd = [
                accelerate_path,
                "launch",
                "--num_cpu_threads_per_process=2",
                train_script,
                f"--config_file={config_path}",
            ]

            # Add no_half_vae flag if enabled (fixes NaN errors with some models)
            if no_half_vae:
                cmd.append("--no_half_vae")
                print("[SDXL LoRA] Using --no_half_vae to prevent NaN errors")

            print(f"[SDXL LoRA] Starting training: {run_name}")
            print(f"[SDXL LoRA] Images: {num_images}, Steps: {training_steps}, LR: {learning_rate}, Rank: {lora_rank}")

            # Run training
            startupinfo = None
            if sys.platform == 'win32':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            # Set UTF-8 encoding for subprocess to handle Japanese text in sd-scripts
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
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
                raise RuntimeError(f"sd-scripts training failed with code {process.returncode}")

            print(f"[SDXL LoRA] Training completed!")

            # Find the trained LoRA
            if not os.path.exists(lora_output_path):
                # Check for alternative naming
                possible_files = [f for f in os.listdir(output_folder) if f.startswith(run_name) and f.endswith('.safetensors')]
                if possible_files:
                    lora_output_path = os.path.join(output_folder, possible_files[-1])
                else:
                    raise FileNotFoundError(f"No LoRA file found in {output_folder}")

            print(f"[SDXL LoRA] Found trained LoRA: {lora_output_path}")

            # Handle caching
            if keep_lora:
                _sdxl_lora_cache[image_hash] = lora_output_path
                _save_sdxl_cache()
                print(f"[SDXL LoRA] LoRA saved and cached at: {lora_output_path}")
            else:
                print(f"[SDXL LoRA] LoRA available at: {lora_output_path}")

            return (lora_output_path,)

        finally:
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"[SDXL LoRA] Warning: Could not clean up temp dir: {e}")
