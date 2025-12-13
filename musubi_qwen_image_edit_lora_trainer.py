"""
Musubi Tuner Qwen Image Edit LoRA Trainer Node for ComfyUI

Trains Qwen Image Edit LoRAs using kohya-ss/musubi-tuner.
For training image editing behaviors with source/target image pairs.
Supports Qwen-Image-Edit and Qwen-Image-Edit-2509.
"""

import os
import sys
import json
import hashlib
import tempfile
import shutil
import subprocess
from datetime import datetime

import folder_paths

from .musubi_qwen_image_config_template import (
    generate_dataset_config,
    save_config,
    MUSUBI_QWEN_IMAGE_VRAM_PRESETS,
)


# Global config for Musubi Qwen Image Edit trainer
_musubi_qwen_edit_config = {}
_musubi_qwen_edit_config_file = os.path.join(os.path.dirname(__file__), ".musubi_qwen_image_edit_config.json")

# Global cache for trained LoRAs
_musubi_qwen_edit_lora_cache = {}
_musubi_qwen_edit_cache_file = os.path.join(os.path.dirname(__file__), ".musubi_qwen_image_edit_lora_cache.json")


def _load_musubi_qwen_edit_config():
    """Load Musubi Qwen Edit config from disk."""
    global _musubi_qwen_edit_config
    if os.path.exists(_musubi_qwen_edit_config_file):
        try:
            with open(_musubi_qwen_edit_config_file, 'r', encoding='utf-8') as f:
                _musubi_qwen_edit_config = json.load(f)
        except:
            _musubi_qwen_edit_config = {}


def _save_musubi_qwen_edit_config():
    """Save Musubi Qwen Edit config to disk."""
    try:
        with open(_musubi_qwen_edit_config_file, 'w', encoding='utf-8') as f:
            json.dump(_musubi_qwen_edit_config, f, indent=2)
    except:
        pass


def _load_musubi_qwen_edit_cache():
    """Load Musubi Qwen Edit LoRA cache from disk."""
    global _musubi_qwen_edit_lora_cache
    if os.path.exists(_musubi_qwen_edit_cache_file):
        try:
            with open(_musubi_qwen_edit_cache_file, 'r', encoding='utf-8') as f:
                _musubi_qwen_edit_lora_cache = json.load(f)
        except:
            _musubi_qwen_edit_lora_cache = {}


def _save_musubi_qwen_edit_cache():
    """Save Musubi Qwen Edit LoRA cache to disk."""
    try:
        with open(_musubi_qwen_edit_cache_file, 'w', encoding='utf-8') as f:
            json.dump(_musubi_qwen_edit_lora_cache, f)
    except:
        pass


def _compute_edit_hash(target_images, control_images, training_steps, learning_rate, lora_rank, vram_mode, output_name, model_mode):
    """Compute a hash of all images and training parameters for edit training."""
    hasher = hashlib.sha256()

    # Hash target images
    for img_path in target_images:
        hasher.update(img_path.encode('utf-8'))
        if os.path.exists(img_path):
            hasher.update(str(os.path.getmtime(img_path)).encode('utf-8'))

    # Hash control images
    for img_path in control_images:
        hasher.update(img_path.encode('utf-8'))
        if os.path.exists(img_path):
            hasher.update(str(os.path.getmtime(img_path)).encode('utf-8'))

    params_str = f"musubi_qwen_edit|{model_mode}|{training_steps}|{learning_rate}|{lora_rank}|{vram_mode}|{output_name}|{len(target_images)}|{len(control_images)}"
    hasher.update(params_str.encode('utf-8'))

    return hasher.hexdigest()[:16]


def _get_venv_python_path(musubi_path):
    """Get the Python path for musubi-tuner venv based on platform."""
    venv_folders = [".venv", "venv"]

    for venv_folder in venv_folders:
        if sys.platform == 'win32':
            python_path = os.path.join(musubi_path, venv_folder, "Scripts", "python.exe")
        else:
            python_path = os.path.join(musubi_path, venv_folder, "bin", "python")

        if os.path.exists(python_path):
            return python_path

    if sys.platform == 'win32':
        return os.path.join(musubi_path, "venv", "Scripts", "python.exe")
    else:
        return os.path.join(musubi_path, "venv", "bin", "python")


def _get_accelerate_path(musubi_path):
    """Get the accelerate path for musubi-tuner venv based on platform."""
    venv_folders = [".venv", "venv"]

    for venv_folder in venv_folders:
        if sys.platform == 'win32':
            accel_path = os.path.join(musubi_path, venv_folder, "Scripts", "accelerate.exe")
        else:
            accel_path = os.path.join(musubi_path, venv_folder, "bin", "accelerate")

        if os.path.exists(accel_path):
            return accel_path

    if sys.platform == 'win32':
        return os.path.join(musubi_path, "venv", "Scripts", "accelerate.exe")
    else:
        return os.path.join(musubi_path, "venv", "bin", "accelerate")


def _get_model_path(name, folder_type):
    """Get full path to a model file from ComfyUI folders."""
    if not name:
        return ""
    if os.path.isabs(name) and os.path.exists(name):
        return name
    try:
        return folder_paths.get_full_path(folder_type, name)
    except:
        return name


# Load config and cache on module import
_load_musubi_qwen_edit_config()
_load_musubi_qwen_edit_cache()


class MusubiQwenImageEditLoraTrainer:
    """
    Trains a Qwen Image Edit LoRA using Musubi Tuner.
    For training image editing behaviors with source/target image pairs.
    Uses folder paths only - no direct image inputs.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        if sys.platform == 'win32':
            musubi_fallback = 'C:\\musubi-tuner'
        else:
            musubi_fallback = '~/musubi-tuner'

        saved = _musubi_qwen_edit_config.get('trainer_settings', {})

        # Get available models from ComfyUI folders
        diffusion_models = folder_paths.get_filename_list("diffusion_models")
        vae_models = folder_paths.get_filename_list("vae")
        try:
            text_encoders = folder_paths.get_filename_list("text_encoders")
        except:
            text_encoders = []
        try:
            clip_models = folder_paths.get_filename_list("clip")
        except:
            clip_models = []
        text_encoder_list = sorted(set(text_encoders + clip_models))

        # Get saved model selections
        saved_dit = saved.get('dit_model', '')
        saved_vae = saved.get('vae_model', '')
        saved_te = saved.get('text_encoder', '')

        # Build dropdown configs with saved defaults
        dit_config = {"tooltip": "Qwen Image Edit DiT model. Use qwen_image_edit_bf16 or qwen_image_edit_2509_bf16."}
        if saved_dit and saved_dit in diffusion_models:
            dit_config["default"] = saved_dit

        vae_config = {"tooltip": "Qwen Image VAE model (qwen_image_vae.safetensors)."}
        if saved_vae and saved_vae in vae_models:
            vae_config["default"] = saved_vae

        te_config = {"tooltip": "Qwen2.5-VL text encoder from text_encoders or clip folder."}
        if saved_te and saved_te in text_encoder_list:
            te_config["default"] = saved_te

        return {
            "required": {
                "images_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to folder containing TARGET images (the edited results). Caption .txt files with matching names are used."
                }),
                "control_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to folder containing CONTROL/SOURCE images (the inputs to edit). Must match target images by filename."
                }),
                "musubi_path": ("STRING", {
                    "default": _musubi_qwen_edit_config.get('musubi_path', musubi_fallback),
                    "tooltip": "Path to musubi-tuner installation."
                }),
                "model_mode": (["Qwen-Image-Edit", "Qwen-Image-Edit-2509"], {
                    "default": saved.get('model_mode', "Qwen-Image-Edit-2509"),
                    "tooltip": "Edit model variant. Edit-2509 is the newer version with improved editing."
                }),
                "dit_model": (diffusion_models, dit_config),
                "vae_model": (vae_models, vae_config),
                "text_encoder": (text_encoder_list, te_config),
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
                    "tooltip": "Learning rate. 3e-4 (0.0003) is recommended."
                }),
                "lora_rank": ("INT", {
                    "default": saved.get('lora_rank', 16),
                    "min": 4,
                    "max": 128,
                    "step": 4,
                    "tooltip": "LoRA rank/dimension. 16 is recommended."
                }),
                "vram_mode": (["Max (1024px)", "Max (1024px) fp8", "Medium (768px)", "Medium (768px) fp8", "Low (512px)", "Low (512px) fp8"], {
                    "default": saved.get('vram_mode', "Medium (768px) fp8"),
                    "tooltip": "VRAM optimization preset."
                }),
                "blocks_to_swap": ([str(i) for i in range(46)], {
                    "default": saved.get('blocks_to_swap', "30"),
                    "tooltip": "Number of transformer blocks to offload to CPU (0-45)."
                }),
                "keep_lora": ("BOOLEAN", {
                    "default": saved.get('keep_lora', True),
                    "tooltip": "If True, keeps the trained LoRA file."
                }),
                "output_name": ("STRING", {
                    "default": saved.get('output_name', "MyQwenEditLora"),
                    "tooltip": "Custom name for the output LoRA. Timestamp will be appended."
                }),
                "custom_python_exe": ("STRING", {
                    "default": saved.get('custom_python_exe', ""),
                    "tooltip": "Advanced: Optionally enter the full path to a custom python.exe (e.g. C:\\my-venv\\Scripts\\python.exe). If empty, uses the venv inside musubi_path. The musubi_path field is still required for locating training scripts."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_path",)
    OUTPUT_TOOLTIPS = ("Path to the trained Qwen Image Edit LoRA file.",)
    FUNCTION = "train_qwen_image_edit_lora"
    CATEGORY = "loaders"
    DESCRIPTION = "Trains a Qwen Image Edit LoRA from image pairs using Musubi Tuner. Requires target and control image folders."

    def train_qwen_image_edit_lora(
        self,
        images_path,
        control_path,
        musubi_path,
        model_mode,
        dit_model,
        vae_model,
        text_encoder,
        training_steps,
        learning_rate,
        lora_rank,
        vram_mode,
        blocks_to_swap,
        keep_lora=True,
        output_name="MyQwenEditLora",
        custom_python_exe="",
    ):
        global _musubi_qwen_edit_lora_cache

        # Expand and validate paths
        musubi_path = os.path.expanduser(musubi_path.strip())
        images_path = os.path.expanduser(images_path.strip())
        control_path = os.path.expanduser(control_path.strip())

        if not images_path or not os.path.isdir(images_path):
            raise ValueError(f"Target images folder not found: {images_path}")
        if not control_path or not os.path.isdir(control_path):
            raise ValueError(f"Control images folder not found: {control_path}")

        # Get full paths from ComfyUI folders
        dit_path = _get_model_path(dit_model, "diffusion_models")
        vae_path = _get_model_path(vae_model, "vae")
        text_encoder_path = _get_model_path(text_encoder, "text_encoders")
        if not text_encoder_path or not os.path.exists(text_encoder_path):
            text_encoder_path = _get_model_path(text_encoder, "clip")

        # Determine edit flag based on mode
        edit_flag = "--edit" if model_mode == "Qwen-Image-Edit" else "--edit_plus"

        # Find images in both folders
        image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        target_images = []
        control_images = []

        for filename in sorted(os.listdir(images_path)):
            if filename.lower().endswith(image_extensions):
                target_path = os.path.join(images_path, filename)
                target_images.append(target_path)

                # Find matching control image
                base_name = os.path.splitext(filename)[0]
                control_found = False
                for ext in image_extensions:
                    control_candidate = os.path.join(control_path, base_name + ext)
                    if os.path.exists(control_candidate):
                        control_images.append(control_candidate)
                        control_found = True
                        break

                if not control_found:
                    raise ValueError(f"No matching control image found for: {filename}")

        if not target_images:
            raise ValueError(f"No images found in target folder: {images_path}")

        print(f"[Musubi Qwen Edit] Training with {len(target_images)} image pair(s)")
        print(f"[Musubi Qwen Edit] Mode: {model_mode}")
        print(f"[Musubi Qwen Edit] Target folder: {images_path}")
        print(f"[Musubi Qwen Edit] Control folder: {control_path}")
        print(f"[Musubi Qwen Edit] DiT: {dit_model}")

        # Get VRAM preset settings
        preset = MUSUBI_QWEN_IMAGE_VRAM_PRESETS.get(vram_mode, MUSUBI_QWEN_IMAGE_VRAM_PRESETS["Medium (768px) fp8"])
        blocks_to_swap_int = int(blocks_to_swap)
        print(f"[Musubi Qwen Edit] Using VRAM mode: {vram_mode}, blocks_to_swap: {blocks_to_swap_int}")

        # Validate musubi paths
        accelerate_path = _get_accelerate_path(musubi_path)
        train_script = os.path.join(musubi_path, "src", "musubi_tuner", "qwen_image_train_network.py")

        if not os.path.exists(accelerate_path):
            raise FileNotFoundError(f"Musubi Tuner accelerate not found at: {accelerate_path}")
        if not os.path.exists(train_script):
            raise FileNotFoundError(f"qwen_image_train_network.py not found at: {train_script}")
        if not dit_path or not os.path.exists(dit_path):
            raise FileNotFoundError(f"DiT model not found at: {dit_path}")
        if not vae_path or not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE model not found at: {vae_path}")
        if not text_encoder_path or not os.path.exists(text_encoder_path):
            raise FileNotFoundError(f"Text encoder not found at: {text_encoder_path}")

        # Save settings
        global _musubi_qwen_edit_config
        _musubi_qwen_edit_config['musubi_path'] = musubi_path
        _musubi_qwen_edit_config['trainer_settings'] = {
            'model_mode': model_mode,
            'dit_model': dit_model,
            'vae_model': vae_model,
            'text_encoder': text_encoder,
            'training_steps': training_steps,
            'learning_rate': learning_rate,
            'lora_rank': lora_rank,
            'vram_mode': vram_mode,
            'blocks_to_swap': blocks_to_swap,
            'keep_lora': keep_lora,
            'output_name': output_name,
            'custom_python_exe': custom_python_exe,
        }
        _save_musubi_qwen_edit_config()

        # Compute hash for caching
        image_hash = _compute_edit_hash(target_images, control_images, training_steps, learning_rate, lora_rank, vram_mode, output_name, model_mode)

        # Check cache
        if keep_lora and image_hash in _musubi_qwen_edit_lora_cache:
            cached_path = _musubi_qwen_edit_lora_cache[image_hash]
            if os.path.exists(cached_path):
                print(f"[Musubi Qwen Edit] Cache hit! Reusing: {cached_path}")
                return (cached_path,)
            else:
                del _musubi_qwen_edit_lora_cache[image_hash]
                _save_musubi_qwen_edit_cache()

        # Generate run name with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{output_name}_{timestamp}" if output_name else f"qwen_edit_lora_{image_hash}"

        # Output folder
        output_folder = os.path.join(musubi_path, "output")
        os.makedirs(output_folder, exist_ok=True)
        lora_output_path = os.path.join(output_folder, f"{run_name}.safetensors")

        if os.path.exists(lora_output_path):
            counter = 1
            while os.path.exists(os.path.join(output_folder, f"{run_name}_{counter}.safetensors")):
                counter += 1
            run_name = f"{run_name}_{counter}"
            lora_output_path = os.path.join(output_folder, f"{run_name}.safetensors")

        # Generate dataset config with control_directory
        # Musubi expects target images in image_directory and control images in control_directory
        config_content = generate_dataset_config(
            image_folder=images_path,
            resolution=preset['resolution'],
            batch_size=preset['batch_size'],
            enable_bucket=True,
            control_directory=control_path,
        )

        # Save config to temp location
        temp_dir = tempfile.mkdtemp(prefix="comfy_musubi_qwen_edit_")
        config_path = os.path.join(temp_dir, "dataset_config.toml")
        save_config(config_content, config_path)
        print(f"[Musubi Qwen Edit] Dataset config saved to {config_path}")

        try:
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

            # Pre-cache latents
            print(f"[Musubi Qwen Edit] Pre-caching latents and text encoder outputs...")

            cache_latents_script = os.path.join(musubi_path, "src", "musubi_tuner", "qwen_image_cache_latents.py")
            if not os.path.exists(cache_latents_script):
                raise FileNotFoundError(f"qwen_image_cache_latents.py not found at: {cache_latents_script}")

            print(f"[Musubi Qwen Edit] Caching VAE latents...")
            cache_latents_cmd = [
                python_path,
                cache_latents_script,
                f"--dataset_config={config_path}",
                f"--vae={vae_path}",
                edit_flag,
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

            print(f"[Musubi Qwen Edit] VAE latents cached.")

            # Cache text encoder outputs
            cache_te_script = os.path.join(musubi_path, "src", "musubi_tuner", "qwen_image_cache_text_encoder_outputs.py")
            if not os.path.exists(cache_te_script):
                raise FileNotFoundError(f"qwen_image_cache_text_encoder_outputs.py not found at: {cache_te_script}")

            print(f"[Musubi Qwen Edit] Caching text encoder outputs...")
            cache_te_cmd = [
                python_path,
                cache_te_script,
                f"--dataset_config={config_path}",
                f"--text_encoder={text_encoder_path}",
                "--batch_size=1",
                edit_flag,
            ]

            if preset.get('fp8_vl', False):
                cache_te_cmd.append("--fp8_vl")

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

            print(f"[Musubi Qwen Edit] Text encoder outputs cached.")

            # Build training command
            cmd = [
                accelerate_path,
                "launch",
                "--num_cpu_threads_per_process=1",
                f"--mixed_precision={preset['mixed_precision']}",
                train_script,
                f"--dit={dit_path}",
                f"--vae={vae_path}",
                f"--text_encoder={text_encoder_path}",
                f"--dataset_config={config_path}",
                "--sdpa",
                f"--mixed_precision={preset['mixed_precision']}",
                "--timestep_sampling=shift",
                "--weighting_scheme=none",
                "--discrete_flow_shift=2.2",
                f"--optimizer_type={preset['optimizer']}",
                f"--learning_rate={learning_rate}",
                f"--network_module=networks.lora_qwen_image",
                f"--network_dim={lora_rank}",
                f"--network_alpha={lora_rank}",
                f"--max_train_steps={training_steps}",
                "--max_data_loader_n_workers=2",
                "--persistent_data_loader_workers",
                f"--output_dir={output_folder}",
                f"--output_name={run_name}",
                "--seed=42",
                edit_flag,
            ]

            # Add memory optimization flags
            if preset['gradient_checkpointing']:
                cmd.append("--gradient_checkpointing")

            if preset['fp8_scaled']:
                cmd.append("--fp8_base")
                cmd.append("--fp8_scaled")

            if preset.get('fp8_vl', False):
                cmd.append("--fp8_vl")

            if blocks_to_swap_int > 0:
                cmd.append(f"--blocks_to_swap={blocks_to_swap_int}")

            print(f"[Musubi Qwen Edit] Starting training: {run_name}")
            print(f"[Musubi Qwen Edit] Image pairs: {len(target_images)}, Steps: {training_steps}, LR: {learning_rate}, Rank: {lora_rank}")

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

            for line in process.stdout:
                line = line.rstrip()
                if line:
                    print(f"[musubi-tuner] {line}")

            process.wait()

            if process.returncode != 0:
                raise RuntimeError(f"Musubi Tuner training failed with code {process.returncode}")

            print(f"[Musubi Qwen Edit] Training completed!")

            # Find the trained LoRA
            if not os.path.exists(lora_output_path):
                possible_files = [f for f in os.listdir(output_folder) if f.startswith(run_name) and f.endswith('.safetensors')]
                if possible_files:
                    lora_output_path = os.path.join(output_folder, possible_files[-1])
                else:
                    raise FileNotFoundError(f"No LoRA file found in {output_folder}")

            print(f"[Musubi Qwen Edit] Found trained LoRA: {lora_output_path}")

            # Handle caching
            if keep_lora:
                _musubi_qwen_edit_lora_cache[image_hash] = lora_output_path
                _save_musubi_qwen_edit_cache()
                print(f"[Musubi Qwen Edit] LoRA saved and cached at: {lora_output_path}")
            else:
                print(f"[Musubi Qwen Edit] LoRA available at: {lora_output_path}")

            return (lora_output_path,)

        finally:
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"[Musubi Qwen Edit] Warning: Could not clean up temp dir: {e}")
