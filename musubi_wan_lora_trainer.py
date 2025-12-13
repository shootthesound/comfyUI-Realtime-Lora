"""
Musubi Tuner Wan 2.2 LoRA Trainer Node for ComfyUI

Trains Wan 2.2 LoRAs using kohya-ss/musubi-tuner.
Supports single-frame training with High/Low noise mode selection.
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

from .musubi_wan_config_template import (
    generate_dataset_config,
    save_config,
    MUSUBI_WAN_VRAM_PRESETS,
    WAN_TIMESTEP_RANGES,
)


# Global config for Musubi Wan trainer
_musubi_wan_config = {}
_musubi_wan_config_file = os.path.join(os.path.dirname(__file__), ".musubi_wan_config.json")

# Global cache for trained LoRAs
_musubi_wan_lora_cache = {}
_musubi_wan_cache_file = os.path.join(os.path.dirname(__file__), ".musubi_wan_lora_cache.json")


def _load_musubi_wan_config():
    """Load Musubi Wan config from disk."""
    global _musubi_wan_config
    if os.path.exists(_musubi_wan_config_file):
        try:
            with open(_musubi_wan_config_file, 'r', encoding='utf-8') as f:
                _musubi_wan_config = json.load(f)
        except:
            _musubi_wan_config = {}


def _save_musubi_wan_config():
    """Save Musubi Wan config to disk."""
    try:
        with open(_musubi_wan_config_file, 'w', encoding='utf-8') as f:
            json.dump(_musubi_wan_config, f, indent=2)
    except:
        pass


def _load_musubi_wan_cache():
    """Load Musubi Wan LoRA cache from disk."""
    global _musubi_wan_lora_cache
    if os.path.exists(_musubi_wan_cache_file):
        try:
            with open(_musubi_wan_cache_file, 'r', encoding='utf-8') as f:
                _musubi_wan_lora_cache = json.load(f)
        except:
            _musubi_wan_lora_cache = {}


def _save_musubi_wan_cache():
    """Save Musubi Wan LoRA cache to disk."""
    try:
        with open(_musubi_wan_cache_file, 'w', encoding='utf-8') as f:
            json.dump(_musubi_wan_lora_cache, f)
    except:
        pass


def _compute_image_hash(images, captions, training_steps, learning_rate, lora_rank, vram_mode, output_name, noise_mode, use_folder_path=False):
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

    # Include all captions and noise mode in hash
    captions_str = "|".join(captions)
    params_str = f"musubi_wan|{noise_mode}|{captions_str}|{training_steps}|{learning_rate}|{lora_rank}|{vram_mode}|{output_name}|{len(images)}"
    hasher.update(params_str.encode('utf-8'))

    return hasher.hexdigest()[:16]


def _get_venv_python_path(musubi_path):
    """Get the Python path for musubi-tuner venv based on platform.
    Checks both .venv (uv default) and venv (traditional) folders."""
    venv_folders = [".venv", "venv"]

    for venv_folder in venv_folders:
        if sys.platform == 'win32':
            python_path = os.path.join(musubi_path, venv_folder, "Scripts", "python.exe")
        else:
            python_path = os.path.join(musubi_path, venv_folder, "bin", "python")

        if os.path.exists(python_path):
            return python_path

    # Return traditional path for error messaging
    if sys.platform == 'win32':
        return os.path.join(musubi_path, "venv", "Scripts", "python.exe")
    else:
        return os.path.join(musubi_path, "venv", "bin", "python")


def _get_accelerate_path(musubi_path):
    """Get the accelerate path for musubi-tuner venv based on platform.
    Checks both .venv (uv default) and venv (traditional) folders."""
    venv_folders = [".venv", "venv"]

    for venv_folder in venv_folders:
        if sys.platform == 'win32':
            accel_path = os.path.join(musubi_path, venv_folder, "Scripts", "accelerate.exe")
        else:
            accel_path = os.path.join(musubi_path, venv_folder, "bin", "accelerate")

        if os.path.exists(accel_path):
            return accel_path

    # Return traditional path for error messaging
    if sys.platform == 'win32':
        return os.path.join(musubi_path, "venv", "Scripts", "accelerate.exe")
    else:
        return os.path.join(musubi_path, "venv", "bin", "accelerate")


def _get_model_path(name, folder_type):
    """Get full path to a model file from ComfyUI folders.
    Returns the name as-is if it's already an absolute path that exists."""
    if not name:
        return ""
    # If it's already an absolute path that exists, use it
    if os.path.isabs(name) and os.path.exists(name):
        return name
    # Try to get from ComfyUI folder
    try:
        return folder_paths.get_full_path(folder_type, name)
    except:
        return name


# Load config and cache on module import
_load_musubi_wan_config()
_load_musubi_wan_cache()


class MusubiWanLoraTrainer:
    """
    Trains a Wan 2.2 LoRA from one or more images using Musubi Tuner.
    Supports single-frame training with High/Low noise mode selection.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Get saved settings or use defaults
        if sys.platform == 'win32':
            musubi_fallback = 'C:\\musubi-tuner'
        else:
            musubi_fallback = '~/musubi-tuner'

        saved = _musubi_wan_config.get('trainer_settings', {})

        # Get available models from ComfyUI folders
        diffusion_models = folder_paths.get_filename_list("diffusion_models")
        vae_models = folder_paths.get_filename_list("vae")
        # Text encoders can be in clip or text_encoders folder
        try:
            text_encoders = folder_paths.get_filename_list("text_encoders")
        except:
            text_encoders = []
        try:
            clip_models = folder_paths.get_filename_list("clip")
        except:
            clip_models = []
        text_encoder_list = sorted(set(text_encoders + clip_models))

        # Get saved model selections (for default)
        saved_dit = saved.get('dit_model', '')
        saved_vae = saved.get('vae_model', '')
        saved_t5 = saved.get('t5_model', '')

        # Build dropdown configs with saved defaults if available
        dit_config = {"tooltip": "Wan 2.2 DiT model. Use High noise model for High Noise training, Low noise model for Low Noise training."}
        if saved_dit and saved_dit in diffusion_models:
            dit_config["default"] = saved_dit

        vae_config = {"tooltip": "Wan VAE model (use Wan2.1 VAE, not Wan2.2_VAE.pth)."}
        if saved_vae and saved_vae in vae_models:
            vae_config["default"] = saved_vae

        t5_config = {"tooltip": "T5 text encoder (models_t5_umt5-xxl-enc-bf16.pth or similar)."}
        if saved_t5 and saved_t5 in text_encoder_list:
            t5_config["default"] = saved_t5

        return {
            "required": {
                "inputcount": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Number of image inputs. Click 'Update inputs' button after changing."}),
                "images_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: Path to folder containing training images. If provided, images from this folder are used instead of image inputs. Caption .txt files with matching names are used if present."
                }),
                "musubi_path": ("STRING", {
                    "default": _musubi_wan_config.get('musubi_path', musubi_fallback),
                    "tooltip": "Path to musubi-tuner installation."
                }),
                "noise_mode": (["High Noise", "Low Noise", "Combo"], {
                    "default": saved.get('noise_mode', "High Noise"),
                    "tooltip": "Training mode. High Noise: use with High model. Low Noise: use with Low model. Combo: full range, use with Low model."
                }),
                "dit_model": (diffusion_models, dit_config),
                "vae_model": (vae_models, vae_config),
                "t5_model": (text_encoder_list, t5_config),
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
                    "tooltip": "Number of training steps. 500 is a good starting point."
                }),
                "learning_rate": ("FLOAT", {
                    "default": saved.get('learning_rate', 0.0003),
                    "min": 0.00001,
                    "max": 0.1,
                    "step": 0.00001,
                    "tooltip": "Learning rate. 3e-4 (0.0003) is recommended for Wan training."
                }),
                "lora_rank": ("INT", {
                    "default": saved.get('lora_rank', 16),
                    "min": 4,
                    "max": 128,
                    "step": 4,
                    "tooltip": "LoRA rank/dimension. 16 is recommended for Wan."
                }),
                "vram_mode": (["Max (1256px)", "Max (1256px) fp8", "Medium (1024px)", "Medium (1024px) fp8", "Low (768px)", "Low (768px) fp8", "Min (512px)", "Min (512px) fp8"], {
                    "default": saved.get('vram_mode', "Low (768px) fp8"),
                    "tooltip": "VRAM optimization preset. Controls resolution, fp8, and gradient checkpointing."
                }),
                "blocks_to_swap": ([str(i) for i in range(40)], {
                    "default": saved.get('blocks_to_swap', "26"),
                    "tooltip": "Number of transformer blocks to offload to CPU (0-39). Higher = less VRAM but slower. 26 is a good balance."
                }),
                "keep_lora": ("BOOLEAN", {
                    "default": saved.get('keep_lora', True),
                    "tooltip": "If True, keeps the trained LoRA file."
                }),
                "output_name": ("STRING", {
                    "default": saved.get('output_name', "MyWanLora"),
                    "tooltip": "Custom name for the output LoRA. Timestamp will be appended."
                }),
                "custom_python_exe": ("STRING", {
                    "default": saved.get('custom_python_exe', ""),
                    "tooltip": "Advanced: Optionally enter the full path to a custom python.exe (e.g. C:\\my-venv\\Scripts\\python.exe). If empty, uses the venv inside musubi_path. The musubi_path field is still required for locating training scripts."
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
    OUTPUT_TOOLTIPS = ("Path to the trained Wan 2.2 LoRA file.",)
    FUNCTION = "train_wan_lora"
    CATEGORY = "loaders"
    DESCRIPTION = "Trains a Wan 2.2 LoRA from images using Musubi Tuner. Supports High/Low noise mode for T2V models."

    def train_wan_lora(
        self,
        inputcount,
        images_path,
        musubi_path,
        noise_mode,
        dit_model,
        vae_model,
        t5_model,
        caption,
        training_steps,
        learning_rate,
        lora_rank,
        vram_mode,
        blocks_to_swap,
        keep_lora=True,
        output_name="MyWanLora",
        custom_python_exe="",
        image_1=None,
        **kwargs
    ):
        global _musubi_wan_lora_cache

        # Expand paths
        musubi_path = os.path.expanduser(musubi_path.strip())

        # Get full paths from ComfyUI folders
        dit_path = _get_model_path(dit_model, "diffusion_models")
        vae_path = _get_model_path(vae_model, "vae")
        # Try text_encoders first, then clip
        t5_path = _get_model_path(t5_model, "text_encoders")
        if not t5_path or not os.path.exists(t5_path):
            t5_path = _get_model_path(t5_model, "clip")

        # Get timestep range for noise mode
        timestep_config = WAN_TIMESTEP_RANGES.get(noise_mode, WAN_TIMESTEP_RANGES["High Noise"])
        min_timestep = timestep_config["min_timestep"]
        max_timestep = timestep_config["max_timestep"]

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
                    print(f"[Musubi Wan] Using {len(folder_images)} images from folder: {images_path}")
                else:
                    print(f"[Musubi Wan] No images found in folder: {images_path}, falling back to inputs")
            else:
                print(f"[Musubi Wan] Invalid folder path: {images_path}, falling back to inputs")

        if not use_folder_path:
            # Collect all images and captions from inputs
            all_images = []
            all_captions = []

            if image_1 is not None:
                all_images.append(image_1)
                cap_1 = kwargs.get("caption_1", "")
                all_captions.append(cap_1 if cap_1 else caption)

            for i in range(2, inputcount + 1):
                img = kwargs.get(f"image_{i}")
                if img is not None:
                    all_images.append(img)
                    cap = kwargs.get(f"caption_{i}", "")
                    all_captions.append(cap if cap else caption)

            if not all_images:
                raise ValueError("No images provided. Either set images_path to a folder containing images, or connect at least one image input.")

        num_images = len(folder_images) if use_folder_path else len(all_images)
        print(f"[Musubi Wan] Training with {num_images} image(s)")
        print(f"[Musubi Wan] Noise Mode: {noise_mode} (timesteps {min_timestep}-{max_timestep})")
        print(f"[Musubi Wan] DiT: {dit_model}")
        print(f"[Musubi Wan] VAE: {vae_model}")
        print(f"[Musubi Wan] T5: {t5_model}")

        # Get VRAM preset settings
        preset = MUSUBI_WAN_VRAM_PRESETS.get(vram_mode, MUSUBI_WAN_VRAM_PRESETS["Low (768px) fp8"])
        blocks_to_swap_int = int(blocks_to_swap)
        print(f"[Musubi Wan] Using VRAM mode: {vram_mode}, blocks_to_swap: {blocks_to_swap_int}")

        # Validate paths
        accelerate_path = _get_accelerate_path(musubi_path)
        train_script = os.path.join(musubi_path, "src", "musubi_tuner", "wan_train_network.py")

        if not os.path.exists(accelerate_path):
            raise FileNotFoundError(f"Musubi Tuner accelerate not found at: {accelerate_path}")
        if not os.path.exists(train_script):
            raise FileNotFoundError(f"wan_train_network.py not found at: {train_script}")
        if not dit_path or not os.path.exists(dit_path):
            raise FileNotFoundError(f"DiT model not found at: {dit_path}")
        if not vae_path or not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE model not found at: {vae_path}")
        if not t5_path or not os.path.exists(t5_path):
            raise FileNotFoundError(f"T5 model not found at: {t5_path}")

        # Save settings
        global _musubi_wan_config
        _musubi_wan_config['musubi_path'] = musubi_path
        _musubi_wan_config['trainer_settings'] = {
            'noise_mode': noise_mode,
            'dit_model': dit_model,
            'vae_model': vae_model,
            't5_model': t5_model,
            'caption': caption,
            'training_steps': training_steps,
            'learning_rate': learning_rate,
            'lora_rank': lora_rank,
            'vram_mode': vram_mode,
            'blocks_to_swap': blocks_to_swap,
            'keep_lora': keep_lora,
            'output_name': output_name,
            'custom_python_exe': custom_python_exe,
        }
        _save_musubi_wan_config()

        # Compute hash for caching
        if use_folder_path:
            image_hash = _compute_image_hash(folder_images, folder_captions, training_steps, learning_rate, lora_rank, vram_mode, output_name, noise_mode, use_folder_path=True)
        else:
            image_hash = _compute_image_hash(all_images, all_captions, training_steps, learning_rate, lora_rank, vram_mode, output_name, noise_mode, use_folder_path=False)

        # Check cache
        if keep_lora and image_hash in _musubi_wan_lora_cache:
            cached_path = _musubi_wan_lora_cache[image_hash]
            if os.path.exists(cached_path):
                print(f"[Musubi Wan] Cache hit! Reusing: {cached_path}")
                return (cached_path,)
            else:
                del _musubi_wan_lora_cache[image_hash]
                _save_musubi_wan_cache()

        # Generate run name with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        noise_suffix = {"High Noise": "high", "Low Noise": "low", "Combo": "combo"}.get(noise_mode, "low")
        run_name = f"{output_name}_{noise_suffix}_{timestamp}" if output_name else f"wan_lora_{noise_suffix}_{image_hash}"

        # Output folder
        output_folder = os.path.join(musubi_path, "output")
        os.makedirs(output_folder, exist_ok=True)
        lora_output_path = os.path.join(output_folder, f"{run_name}.safetensors")

        # Auto-increment if file somehow still exists (same second)
        if os.path.exists(lora_output_path):
            counter = 1
            while os.path.exists(os.path.join(output_folder, f"{run_name}_{counter}.safetensors")):
                counter += 1
            run_name = f"{run_name}_{counter}"
            lora_output_path = os.path.join(output_folder, f"{run_name}.safetensors")
            print(f"[Musubi Wan] Name exists, using: {run_name}")

        # Create temp directory for images
        temp_dir = tempfile.mkdtemp(prefix="comfy_musubi_wan_")
        image_folder = temp_dir  # Musubi uses image_directory directly
        os.makedirs(image_folder, exist_ok=True)

        try:
            # Save images with captions
            if use_folder_path:
                # Copy images from folder and create caption files
                for idx, (src_path, cap) in enumerate(zip(folder_images, folder_captions)):
                    ext = os.path.splitext(src_path)[1]
                    dest_path = os.path.join(image_folder, f"image_{idx+1:03d}{ext}")
                    shutil.copy2(src_path, dest_path)

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

                    caption_path = os.path.join(image_folder, f"image_{idx+1:03d}.txt")
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(all_captions[idx])

            print(f"[Musubi Wan] Saved {num_images} images to {image_folder}")

            # Generate dataset config
            config_content = generate_dataset_config(
                image_folder=image_folder,
                resolution=preset['resolution'],
                batch_size=preset['batch_size'],
                enable_bucket=True,
            )

            config_path = os.path.join(temp_dir, "dataset_config.toml")
            save_config(config_content, config_path)
            print(f"[Musubi Wan] Dataset config saved to {config_path}")

            # Set up subprocess environment
            startupinfo = None
            if sys.platform == 'win32':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            # Use custom python exe if provided, otherwise detect from musubi_path
            if custom_python_exe and custom_python_exe.strip():
                python_path = custom_python_exe.strip()
                if not os.path.exists(python_path):
                    raise FileNotFoundError(f"Custom python.exe not found at: {python_path}")
            else:
                python_path = _get_venv_python_path(musubi_path)

            # Pre-cache latents and text encoder outputs (REQUIRED for Musubi training)
            print(f"[Musubi Wan] Pre-caching latents and text encoder outputs...")

            # Cache latents
            cache_latents_script = os.path.join(musubi_path, "src", "musubi_tuner", "wan_cache_latents.py")
            if not os.path.exists(cache_latents_script):
                raise FileNotFoundError(f"wan_cache_latents.py not found at: {cache_latents_script}")

            print(f"[Musubi Wan] Caching VAE latents...")
            cache_latents_cmd = [
                python_path,
                cache_latents_script,
                f"--dataset_config={config_path}",
                f"--vae={vae_path}",
            ]

            cache_latents_process = subprocess.Popen(
                cache_latents_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=musubi_path,
                startupinfo=startupinfo,
                env=env,
            )

            for line in cache_latents_process.stdout:
                line = line.rstrip()
                if line:
                    print(f"[musubi-tuner] {line}")

            cache_latents_process.wait()
            if cache_latents_process.returncode != 0:
                raise RuntimeError(f"Latent caching failed with code {cache_latents_process.returncode}")

            print(f"[Musubi Wan] VAE latents cached.")

            # Cache text encoder outputs
            cache_te_script = os.path.join(musubi_path, "src", "musubi_tuner", "wan_cache_text_encoder_outputs.py")
            if not os.path.exists(cache_te_script):
                raise FileNotFoundError(f"wan_cache_text_encoder_outputs.py not found at: {cache_te_script}")

            print(f"[Musubi Wan] Caching text encoder outputs...")
            cache_te_cmd = [
                python_path,
                cache_te_script,
                f"--dataset_config={config_path}",
                f"--t5={t5_path}",
                "--batch_size=1",
            ]

            cache_te_process = subprocess.Popen(
                cache_te_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=musubi_path,
                startupinfo=startupinfo,
                env=env,
            )

            for line in cache_te_process.stdout:
                line = line.rstrip()
                if line:
                    print(f"[musubi-tuner] {line}")

            cache_te_process.wait()
            if cache_te_process.returncode != 0:
                raise RuntimeError(f"Text encoder caching failed with code {cache_te_process.returncode}")

            print(f"[Musubi Wan] Text encoder outputs cached.")

            # Build training command
            cmd = [
                accelerate_path,
                "launch",
                "--num_cpu_threads_per_process=1",
                f"--mixed_precision={preset['mixed_precision']}",
                train_script,
                "--task=t2v-A14B",  # Wan 2.2 T2V task
                f"--dit={dit_path}",
                f"--vae={vae_path}",
                f"--t5={t5_path}",
                f"--dataset_config={config_path}",
                "--sdpa",
                f"--mixed_precision={preset['mixed_precision']}",
                "--timestep_sampling=shift",
                "--discrete_flow_shift=3.0",
                f"--min_timestep={min_timestep}",
                f"--max_timestep={max_timestep}",
                f"--optimizer_type={preset['optimizer']}",
                f"--learning_rate={learning_rate}",
                f"--network_module=networks.lora_wan",
                f"--network_dim={lora_rank}",
                f"--network_alpha={lora_rank}",
                f"--max_train_steps={training_steps}",
                "--max_data_loader_n_workers=2",
                "--persistent_data_loader_workers",
                f"--output_dir={output_folder}",
                f"--output_name={run_name}",
                "--seed=42",
            ]

            # Add memory optimization flags
            if preset['gradient_checkpointing']:
                cmd.append("--gradient_checkpointing")

            if preset.get('fp8_base', False):
                cmd.append("--fp8_base")

            if blocks_to_swap_int > 0:
                cmd.append(f"--blocks_to_swap={blocks_to_swap_int}")

            print(f"[Musubi Wan] Starting training: {run_name}")
            print(f"[Musubi Wan] Images: {num_images}, Steps: {training_steps}, LR: {learning_rate}, Rank: {lora_rank}")

            # Run training
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=musubi_path,
                startupinfo=startupinfo,
                env=env,
            )

            # Stream output
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    print(f"[musubi-tuner] {line}")

            process.wait()

            if process.returncode != 0:
                raise RuntimeError(f"Musubi Tuner training failed with code {process.returncode}")

            print(f"[Musubi Wan] Training completed!")

            # Find the trained LoRA
            if not os.path.exists(lora_output_path):
                # Check for alternative naming
                possible_files = [f for f in os.listdir(output_folder) if f.startswith(run_name) and f.endswith('.safetensors')]
                if possible_files:
                    lora_output_path = os.path.join(output_folder, possible_files[-1])
                else:
                    raise FileNotFoundError(f"No LoRA file found in {output_folder}")

            print(f"[Musubi Wan] Found trained LoRA: {lora_output_path}")

            # Handle caching
            if keep_lora:
                _musubi_wan_lora_cache[image_hash] = lora_output_path
                _save_musubi_wan_cache()
                print(f"[Musubi Wan] LoRA saved and cached at: {lora_output_path}")
            else:
                print(f"[Musubi Wan] LoRA available at: {lora_output_path}")

            return (lora_output_path,)

        finally:
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"[Musubi Wan] Warning: Could not clean up temp dir: {e}")
