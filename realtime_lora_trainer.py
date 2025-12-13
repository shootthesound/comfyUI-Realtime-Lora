"""
Realtime LoRA Trainer Node for ComfyUI

Trains LoRAs on-the-fly from one or more images during generation.
Supports Z-Image, FLUX, Wan, and Qwen models.
Uses the user's AI-Toolkit installation via subprocess (no dependency conflicts).
"""

import os
import sys
import subprocess
import tempfile
import shutil
import glob
import hashlib
import json
from datetime import datetime
import numpy as np
from PIL import Image

import torch
import comfy.sd
import comfy.utils

from .config_template import generate_training_config, save_config, generate_unique_name, ARCHITECTURE_CONFIGS


def _get_venv_python_path(ai_toolkit_path: str) -> str:
    """
    Get the correct Python executable path for the AI-Toolkit venv,
    handling cross-platform differences and different venv folder names.

    Checks both .venv (uv default) and venv (traditional) folders.
    Windows: .venv/Scripts/python.exe or venv/Scripts/python.exe
    Linux/Mac: .venv/bin/python or venv/bin/python
    """
    # Check both .venv (uv) and venv (traditional) folders
    venv_folders = [".venv", "venv"]

    for venv_folder in venv_folders:
        if sys.platform == 'win32':
            python_path = os.path.join(ai_toolkit_path, venv_folder, "Scripts", "python.exe")
        else:
            python_path = os.path.join(ai_toolkit_path, venv_folder, "bin", "python")

        if os.path.exists(python_path):
            return python_path

    # Return the traditional path for error messaging (will fail with helpful error later)
    if sys.platform == 'win32':
        return os.path.join(ai_toolkit_path, "venv", "Scripts", "python.exe")
    else:
        return os.path.join(ai_toolkit_path, "venv", "bin", "python")


# Global cache: maps image+params hash to trained LoRA path
_lora_cache = {}
_cache_file = os.path.join(os.path.dirname(__file__), ".lora_cache.json")

# Global config: stores user preferences like AI-Toolkit path
_config = {}
_config_file = os.path.join(os.path.dirname(__file__), ".config.json")



def _load_cache():
    """Load cache from disk."""
    global _lora_cache
    if os.path.exists(_cache_file):
        try:
            with open(_cache_file, 'r', encoding='utf-8') as f:
                _lora_cache = json.load(f)
        except:
            _lora_cache = {}


def _save_cache():
    """Save cache to disk."""
    try:
        with open(_cache_file, 'w', encoding='utf-8') as f:
            json.dump(_lora_cache, f)
    except:
        pass


def _load_config():
    """Load config from disk."""
    global _config
    if os.path.exists(_config_file):
        try:
            with open(_config_file, 'r', encoding='utf-8') as f:
                _config = json.load(f)
        except:
            _config = {}


def _save_config():
    """Save config to disk."""
    try:
        with open(_config_file, 'w', encoding='utf-8') as f:
            json.dump(_config, f, indent=2)
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
        # Hash all images (tensor inputs)
        for img_tensor in images:
            img_np = (img_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            img_bytes = img_np.tobytes()
            hasher.update(img_bytes)

    # Hash training params (include all captions)
    captions_str = "|".join(captions)
    params_str = f"{captions_str}|{training_steps}|{learning_rate}|{lora_rank}|{vram_mode}|{output_name}|{len(images)}"
    hasher.update(params_str.encode('utf-8'))

    return hasher.hexdigest()[:16]


# Load cache and config on module import
_load_cache()
_load_config()


class RealtimeLoraTrainer:
    """
    Trains a LoRA in real-time from one or more images, then applies it to the model.

    Supports multiple architectures: Z-Image, FLUX, Wan, and Qwen models.

    This node:
    1. Saves input images to a temp folder
    2. Creates caption files for each image
    3. Generates architecture-specific training config
    4. Runs AI-Toolkit training via subprocess
    5. Loads the trained LoRA
    6. Applies it to the model and passes it through

    Supports up to 10 images via optional inputs (image_2 through image_10).
    All images share the same caption.
    """

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        # Get saved settings or use defaults
        if sys.platform == 'win32':
            ai_toolkit_fallback = 'C:\\Ai-toolkit'
        else:
            ai_toolkit_fallback = '~/ai-toolkit'

        # Load all saved settings with fallback defaults
        saved = _config.get('trainer_settings', {})

        return {
            "required": {
                "inputcount": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Number of image inputs. Click 'Update inputs' button after changing."}),
                "images_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: Path to folder containing training images. If provided, images from this folder are used instead of image inputs. Caption .txt files with matching names are used if present."
                }),
                "architecture": (["Z-Image Turbo", "FLUX.1-dev", "Wan 2.2 High", "Wan 2.2 Low", "Wan 2.2 Combo"], {
                    "default": saved.get('architecture', "Z-Image Turbo"),
                    "tooltip": "Model architecture to train on. Model is auto-downloaded from HuggingFace."
                }),
                "ai_toolkit_path": ("STRING", {
                    "default": _config.get('ai_toolkit_path', ai_toolkit_fallback),
                    "tooltip": "Path to your AI-Toolkit installation. Changes are saved automatically."
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
                "vram_mode": (["Max (1256px)", "Medium (1024px)", "Low (768px)", "Min (512px)"], {
                    "default": saved.get('vram_mode', "Low (768px)"),
                    "tooltip": "VRAM optimization preset. Images are automatically resized to the specified resolution."
                }),
                "keep_lora": ("BOOLEAN", {
                    "default": saved.get('keep_lora', True),
                    "tooltip": "If True, keeps the trained LoRA file. If False, deletes after use."
                }),
                "output_name": ("STRING", {
                    "default": saved.get('output_name', "MyLora"),
                    "tooltip": "Custom name for the output LoRA. Timestamp will be appended."
                }),
                "custom_python_exe": ("STRING", {
                    "default": saved.get('custom_python_exe', ""),
                    "tooltip": "Advanced: Optionally enter the full path to a custom python.exe (e.g. C:\\my-venv\\Scripts\\python.exe). If empty, uses the venv inside ai_toolkit_path. The ai_toolkit_path field is still required for locating training scripts."
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
    OUTPUT_TOOLTIPS = ("Path to the trained LoRA file. Connect to ApplyTrainedLora node.",)
    FUNCTION = "train_lora"
    CATEGORY = "loaders"
    DESCRIPTION = "Trains a LoRA on-the-fly from one or more images using AI-Toolkit. Connect output to ApplyTrainedLora node to apply. Supports Z-Image, FLUX, Wan, and Qwen models."

    def train_lora(
        self,
        inputcount,
        images_path,
        architecture,
        ai_toolkit_path,
        caption,
        training_steps,
        learning_rate,
        lora_rank,
        vram_mode,
        keep_lora=True,
        output_name="MyLora",
        custom_python_exe="",
        image_1=None,
        **kwargs
    ):
        global _lora_cache

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
                    print(f"[Realtime LoRA] Using {len(folder_images)} images from folder: {images_path}")
                else:
                    print(f"[Realtime LoRA] No images found in folder: {images_path}, falling back to inputs")
            else:
                print(f"[Realtime LoRA] Invalid folder path: {images_path}, falling back to inputs")

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
        print(f"[Realtime LoRA] Training with {num_images} image(s)")

        # Apply VRAM mode settings
        low_vram = False
        quantize = False
        layer_offloading = False
        gradient_accumulation = 1
        max_resolution_cap = 1256  # Default cap for Max mode

        # Large models always need quantization on consumer GPUs
        # FLUX: 12B params (~24GB), Wan: 14B params (~28GB)
        if architecture in ["FLUX.1-dev", "Wan 2.2 High", "Wan 2.2 Low", "Wan 2.2 Combo"]:
            quantize = True
            low_vram = True
            print(f"[Realtime LoRA] {architecture} requires quantization, enabling low_vram and quantize")

        if vram_mode == "Low (768px)":
            low_vram = True
            quantize = True
            max_resolution_cap = 768
            print(f"[Realtime LoRA] Low mode: low_vram=True, quantize=True, max_res=768")
        elif vram_mode == "Medium (1024px)":
            low_vram = True
            quantize = True
            max_resolution_cap = 1024
            print(f"[Realtime LoRA] Medium mode: low_vram=True, quantize=True, max_res=1024")
        elif vram_mode == "Min (512px)":
            low_vram = True
            quantize = True
            # Only enable layer offloading for architectures that support it (Wan)
            layer_offloading = architecture.startswith("Wan")
            gradient_accumulation = 2
            max_resolution_cap = 512
            print(f"[Realtime LoRA] Min mode: All optimizations enabled, max_res=512")

        # Validate AI-Toolkit path
        run_script = os.path.join(ai_toolkit_path, "run.py")

        # Use custom python exe if provided, otherwise detect from ai_toolkit_path
        if custom_python_exe and custom_python_exe.strip():
            venv_python = custom_python_exe.strip()
            if not os.path.exists(venv_python):
                raise FileNotFoundError(f"Custom python.exe not found at: {venv_python}")
        else:
            venv_python = _get_venv_python_path(ai_toolkit_path)
            if not os.path.exists(venv_python):
                raise FileNotFoundError(f"AI-Toolkit venv not found. Checked .venv and venv folders in: {ai_toolkit_path}")

        if not os.path.exists(run_script):
            raise FileNotFoundError(f"AI-Toolkit run.py not found at: {run_script}")

        # Save all settings to config for future use
        global _config
        _config['ai_toolkit_path'] = ai_toolkit_path
        _config['trainer_settings'] = {
            'architecture': architecture,
            'caption': caption,
            'training_steps': training_steps,
            'learning_rate': learning_rate,
            'lora_rank': lora_rank,
            'vram_mode': vram_mode,
            'keep_lora': keep_lora,
            'output_name': output_name,
            'custom_python_exe': custom_python_exe,
        }
        _save_config()

        # Get model path from architecture config
        arch_config = ARCHITECTURE_CONFIGS[architecture]
        model_path = arch_config['default_path']
        print(f"[Realtime LoRA] Using {architecture} model: {model_path}")

        # Compute hash of all images + training params for caching
        if use_folder_path:
            image_hash = _compute_image_hash(folder_images, folder_captions, training_steps, learning_rate, lora_rank, vram_mode, output_name, use_folder_path=True)
        else:
            image_hash = _compute_image_hash(all_images, all_captions, training_steps, learning_rate, lora_rank, vram_mode, output_name, use_folder_path=False)

        # Check cache: if we've trained this exact set before and the file exists, reuse it
        if keep_lora and image_hash in _lora_cache:
            cached_path = _lora_cache[image_hash]
            if os.path.exists(cached_path):
                print(f"[Realtime LoRA] Cache hit! Reusing: {cached_path}")
                return (cached_path,)
            else:
                # Cached file was deleted, remove from cache
                del _lora_cache[image_hash]
                _save_cache()

        # Generate unique name for this training run with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{output_name}_{timestamp}" if output_name else f"realtime_{image_hash}"

        # Output folder for trained LoRA (inside AI-Toolkit's output folder)
        output_folder = os.path.join(ai_toolkit_path, "output")
        lora_output_dir = os.path.join(output_folder, run_name)

        # Auto-increment name if output folder somehow still exists (same second)
        if os.path.exists(lora_output_dir):
            counter = 1
            while os.path.exists(os.path.join(output_folder, f"{run_name}_{counter}")):
                counter += 1
            run_name = f"{run_name}_{counter}"
            lora_output_dir = os.path.join(output_folder, run_name)
            print(f"[Realtime LoRA] Name exists, using: {run_name}")

        # Create temp directory for training data
        temp_dir = tempfile.mkdtemp(prefix="comfy_realtime_lora_")
        image_folder = os.path.join(temp_dir, "images")
        os.makedirs(image_folder, exist_ok=True)

        try:
            # Step 1: Save all input images with their captions
            max_resolution = 0

            if use_folder_path:
                # Copy images from folder and create caption files
                for idx, (src_path, cap) in enumerate(zip(folder_images, folder_captions)):
                    # Load image to get resolution
                    img_pil = Image.open(src_path)
                    max_resolution = max(max_resolution, img_pil.width, img_pil.height)

                    # Copy image to temp folder
                    ext = os.path.splitext(src_path)[1]
                    dest_path = os.path.join(image_folder, f"training_image_{idx+1:03d}{ext}")
                    shutil.copy2(src_path, dest_path)

                    # Create caption file
                    caption_path = os.path.join(image_folder, f"training_image_{idx+1:03d}.txt")
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(cap)
            else:
                # Save tensor images
                for idx, img_tensor in enumerate(all_images):
                    # ComfyUI images are [B, H, W, C] tensors with values 0-1
                    img_data = img_tensor[0]  # Take first image if batch
                    img_np = (img_data.cpu().numpy() * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_np)

                    # Track max resolution for config
                    max_resolution = max(max_resolution, img_pil.width, img_pil.height)

                    # Save image
                    image_path = os.path.join(image_folder, f"training_image_{idx+1:03d}.png")
                    img_pil.save(image_path, "PNG")

                    # Create caption file (per-image caption)
                    caption_path = os.path.join(image_folder, f"training_image_{idx+1:03d}.txt")
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(all_captions[idx])

            # Apply resolution cap if VRAM mode is enabled
            if max_resolution_cap is not None:
                max_resolution = min(max_resolution, max_resolution_cap)
                print(f"[Realtime LoRA] Resolution capped at {max_resolution}px for VRAM mode")

            # Step 2: Generate training config
            config = generate_training_config(
                name=run_name,
                image_folder=image_folder,
                output_folder=output_folder,
                architecture=architecture,
                model_path=model_path,
                steps=training_steps,
                learning_rate=learning_rate,
                lora_rank=lora_rank,
                resolution=max_resolution,
                low_vram=low_vram,
                quantize=quantize,
                layer_offloading=layer_offloading,
                gradient_accumulation_steps=gradient_accumulation,
            )

            config_path = os.path.join(temp_dir, "training_config.yaml")
            save_config(config, config_path)

            print(f"[Realtime LoRA] Starting training: {run_name}")
            print(f"[Realtime LoRA] Images: {num_images}, Steps: {training_steps}, LR: {learning_rate}, Rank: {lora_rank}")
            if vram_mode != "Max (1256px)":
                print(f"[Realtime LoRA] VRAM Mode: {vram_mode}")

            # Step 3: Run AI-Toolkit training via subprocess
            startupinfo = None
            if sys.platform == 'win32':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            # Set UTF-8 encoding for subprocess to handle non-ASCII output
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            process = subprocess.Popen(
                [venv_python, run_script, config_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                startupinfo=startupinfo,
                cwd=ai_toolkit_path,
                env=env,
            )

            # Stream output for progress visibility
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    # Show all output from AI-Toolkit
                    print(f"[AI-Toolkit] {line}")

            process.wait()

            if process.returncode != 0:
                raise RuntimeError(f"AI-Toolkit training failed with code {process.returncode}")

            print(f"[Realtime LoRA] Training completed!")

            # Step 4: Find the trained LoRA file
            lora_files = glob.glob(os.path.join(lora_output_dir, "*.safetensors"))
            if not lora_files:
                raise FileNotFoundError(f"No LoRA file found in {lora_output_dir}")

            # Get the latest/final one
            lora_path = max(lora_files, key=os.path.getmtime)
            print(f"[Realtime LoRA] Found trained LoRA: {lora_path}")

            # Step 5: Handle caching and return path
            if keep_lora:
                # Add to cache
                _lora_cache[image_hash] = lora_path
                _save_cache()
                print(f"[Realtime LoRA] LoRA saved and cached at: {lora_path}")
            else:
                print(f"[Realtime LoRA] LoRA available at: {lora_path} (will be cleaned up after workflow)")

            return (lora_path,)

        finally:
            # Always clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"[Realtime LoRA] Warning: Could not clean up temp dir: {e}")


class ApplyTrainedLora:
    """
    Applies a trained LoRA from a path to a model.
    Designed to work with RealtimeLoraTrainer - connect lora_path output to this node.
    """

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model to apply the LoRA to."}),
                "lora_path": ("STRING", {"forceInput": True, "tooltip": "Path to the LoRA file (from RealtimeLoraTrainer)."}),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "How strongly to apply the LoRA to the model."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("model", "lora_name", "lora_path")
    OUTPUT_TOOLTIPS = ("The model with the trained LoRA applied.", "Name of the loaded LoRA.", "Full path to the LoRA file.")
    FUNCTION = "apply_lora"
    CATEGORY = "loaders"
    DESCRIPTION = "Applies a trained LoRA to a model. Connect the lora_path from RealtimeLoraTrainer."

    def apply_lora(self, model, lora_path, strength):
        # Extract LoRA name from path for display
        lora_name = ""
        if lora_path:
            lora_name = os.path.basename(os.path.dirname(lora_path))
            if not lora_name or lora_name == "output":
                lora_name = os.path.basename(lora_path)

        if not lora_path or not os.path.exists(lora_path):
            print(f"[ApplyTrainedLora] No valid path provided, passing through unchanged")
            return (model, "", "")

        print(f"[ApplyTrainedLora] Loading: {lora_name}")

        if strength == 0:
            print(f"[ApplyTrainedLora] Strength is 0, skipping")
            return (model, lora_name, lora_path)

        # Load and cache the LoRA
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)[0]
        print(f"[ApplyTrainedLora] Applied LoRA with strength {strength}")
        return (model_lora, lora_name, lora_path)
