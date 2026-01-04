# ComfyUI Realtime LoRA Toolkit

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support-yellow.svg)](https://buymeacoffee.com/lorasandlenses)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Train**, **analyze**, **selectively load by block**, and **edit base models** for **SDXL, SD 1.5, FLUX, Z-Image, Qwen Image, Qwen Image Edit, and Wan 2.2** directly inside ComfyUI. One unified interface across three training backends, plus powerful analysis, block-level loading, strength scheduling, and model editing tools.

> **Version 2 Now Available:** Combined analyzer + selective loader nodes with strength scheduling and LoRA saving!
>
> [![▶ Watch V2 Demo](https://img.youtube.com/vi/3CdyGxvYeHo/0.jpg)](https://www.youtube.com/watch?v=3CdyGxvYeHo)

## At a Glance

| Backend | Models | Best For |
|---------|--------|----------|
| **sd-scripts** | SDXL, SD 1.5 | Fast training, mature workflows, broad checkpoint compatibility |
| **Musubi Tuner** | Z-Image, Qwen Image, Qwen Image Edit, Wan 2.2 | Cutting-edge models, smaller LoRAs, excellent VRAM efficiency |
| **AI-Toolkit** | FLUX.1-dev, Z-Image, Wan 2.2 alternative training pipeline |

**7 architectures. 3 training backends. 29 nodes total.**

- 8 trainer nodes
- 11 selective loaders (5 V1 + 6 V2 combined)
- 2 analyzers (V1 + V2)
- 6 model layer editors
- 3 utility nodes

## What's New

### V2 Combined Analyzer + Selective Loader Nodes

The V2 nodes combine analysis and selective loading into a single node, with powerful new features:

- **Strength Scheduling** - Apply LoRA at different strengths during generation using keyframes
- **Save Refined LoRA** - Export your block-filtered LoRA as a new file
- **40+ Schedule Presets** - Fades, bell curves, pulses, step functions, and inverted versions
- **Extra Z-Image Blocks** - Control context_refiner, noise_refiner, final_layer, x_embedder
- **LoKR/LoHa Support** - Proper handling of decomposed LoRA formats (common for Z-Image)

### Model Layer Editor Nodes

Scale individual blocks of your **base model** before applying LoRAs:

- 6 nodes covering all architectures (SDXL, SD 1.5, FLUX, Z-Image, Wan, Qwen)
- Presets for common adjustments
- User preset system (save/delete your own configurations)
- Save modified models to disk

### Utility Nodes

| Node | Description |
|------|-------------|
| **Clippy Reloaded** | Load images directly from clipboard - copy from browser/screenshot/editor, queue, done |
| **Image of the Day** | 8 sources (Bing, NASA, Unsplash, Pexels, Wikimedia, Lorem Picsum, Random Dog/Cat) with API key persistence |
| **Scheduled LoRA Loader** | Standalone strength scheduling for any LoRA |

See [Utility Nodes](#utility-nodes) section below for full details.

### SDXL Training Improvements

- Added `--no_half_vae` option to fix NaN errors with some SDXL checkpoints
- Fixed batch_size to 1 for all VRAM presets for stability

### Supported LoRA Formats

The analyzer and selective loaders detect and properly handle multiple LoRA formats:

- **Standard LoRA** (lora_up/lora_down)
- **LoKR** (Low-Rank Kronecker) - common for Z-Image LoRAs
- **LoHa** (Low-Rank Hadamard)
- **GLoRA**
- AI-Toolkit, Kohya sd-scripts, and Musubi Tuner formats

The V2 nodes display the detected type in the analysis (e.g., "ZIMAGE (LoKR)").

## Use Cases

- **Subject consistency** - Train on a character or face, use across multiple generations
- **Style transfer** - Capture an art style from a few reference images
- **Rapid prototyping** - Test a LoRA concept in minutes before committing to longer training
- **Video keyframes** - Train on first/last frames for Wan video temporal consistency
- **Image editing behaviors** - Use Qwen Image Edit to teach specific transformations with paired images

## What This Does

This node trains LoRAs on-the-fly from your images without leaving ComfyUI. SDXL and SD 1.5 training is particularly fast - a few minutes on a decent GPU, or under 2 minutes for SD 1.5 on modern hardware. This makes it practical to train a quick LoRA and immediately use it for img2img variations, style transfer, or subject consistency within the same workflow.

**Personal note:** I think SDXL is due for a revival. It trains fast, runs on reasonable hardware, and the results are solid. For quick iteration - testing a concept before committing to a longer train, locking down a subject for consistency, or training on frames for Wan video work - SDXL hits a sweet spot that newer models don't always match. Sometimes the "old" tool is still the right one.

## Supported Models

**Via Kohya sd-scripts:**
- SDXL (any checkpoint) - tested with Juggernaut XL Ragnarok, base SDXL will work too
- SD 1.5 (any checkpoint) - blazingly fast, ~2 mins for 500 steps on a 5090

**Via Musubi Tuner:**
- Z-Image - faster training, smaller LoRA files, no diffusers dependency. Requires the de-distilled model for training, but trained LoRAs work with the regular distilled Z-Image Turbo model.
- Qwen Image - text-to-image generation. Supports Qwen-Image, Qwen-Image-Edit, and Qwen-Image-Edit-2509 models for style/subject LoRAs.
- Qwen Image Edit - for training image editing behaviors with source/target image pairs. Uses folder paths for paired training data.
- Wan 2.2 - single-frame image training with High/Low/Combo noise modes. Separate block offloading control for fine-tuned VRAM management.

**Via AI-Toolkit:**
- Z-Image Turbo
- FLUX.1-dev
- Wan 2.2 (High/Low/Combo)

**Note on Wan 2.2 modes:** Wan uses a two-stage noise model - High handles early denoising steps, Low handles later steps. You can train separate LoRAs for each, or use Combo mode which trains a single LoRA across all noise steps that works with both High and Low models.

**Technical note:** When using High or Low mode, the example workflows still pass the LoRA to both models but at zero strength for the one you didn't train. This prevents ComfyUI from loading the base model into memory before training starts - a workaround to avoid unnecessary VRAM usage.

## Requirements

> **JUST WANT TO ANALYZE, EDIT, OR LOAD LORAS?**
>
> **The Analyzers, Selective Loaders, Model Layer Editors, and Utility Nodes work out of the box - no additional installation required!**
>
> The training backend requirements below **ONLY apply if you want to train LoRAs**. Skip this section if you're just using the analysis, editing, and utility features.

**Python version:** Both AI-Toolkit and sd-scripts work best with Python 3.10-3.12. Python 3.10 is the safest bet. Avoid 3.13 for now.

**For training nodes only** - install the backend(s) you need:

**For SDXL / SD 1.5 training:**
1. Install sd-scripts: https://github.com/kohya-ss/sd-scripts
2. Follow their install instructions

**For Musubi Tuner models (Z-Image, Qwen Image, Wan 2.2):**
1. Install Musubi Tuner: https://github.com/kohya-ss/musubi-tuner
2. Follow their install instructions
3. Download the required models:

   **Z-Image:** Download the de-distilled model from https://huggingface.co/ostris/Z-Image-De-Turbo/tree/main - save to `models/diffusion_models`. Your trained LoRAs will work with the regular distilled Z-Image Turbo model.

   **Qwen Image:** Download bf16 models (not fp8) from Comfy-Org or from the links in the exampe workflows:
   - DiT: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI (qwen_image_bf16.safetensors) or https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI for Edit models
   - VAE: qwen_image_vae.safetensors
   - Text Encoder: qwen_2.5_vl_7b.safetensors (from clip folder)
   - Note: Pre-quantized fp8 models don't work for training - use bf16 versions.

   **Wan 2.2:** Download fp16 models from Comfy-Org or from the links in the exampe workflows:
   - DiT: wan2.2_t2v_14B_fp16.safetensors (High or Low noise variant)
   - VAE: wan_2.2_vae.safetensors
   - T5: models_t5_umt5-xxl-enc-bf16.pth

**For FLUX/Z-Image/Wan training (AI-Toolkit):**
1. Install AI-Toolkit: https://github.com/ostris/ai-toolkit
2. Follow their install instructions

**RTX 50-series GPUs (AI-Toolkit only):** Blackwell GPUs (RTX 5080/5090) require PyTorch 2.7+ with CUDA 12.8 support. The standard AI-Toolkit installation may not work out of the box. A community installer is available at https://github.com/omgitsgb/ostris-ai-toolkit-50gpu-installer that handles the correct PyTorch/CUDA versions. Note: sd-scripts for SDXL/SD1.5 training & Musubi Training is unaffected - this applies only to AI-Toolkit.

You don't need to open the training environments after installation. The node just needs the path to where you installed them.

## Installation

Clone this repo into your ComfyUI custom_nodes folder:

```
cd ComfyUI/custom_nodes
git clone https://github.com/ShootTheSound/comfyUI-Realtime-Lora
```

Restart ComfyUI.

## Nodes

Search for these in ComfyUI:

- **Realtime LoRA Trainer** - Trains using AI-Toolkit (FLUX, Z-Image, Wan)
- **Realtime LoRA Trainer (Z-Image - Musubi Tuner)** - Trains Z-Image using Musubi Tuner (recommended)
- **Realtime LoRA Trainer (Qwen Image - Musubi Tuner)** - Trains Qwen Image/Edit models for style/subject LoRAs
- **Realtime LoRA Trainer (Qwen Image Edit - Musubi Tuner)** - Trains edit behaviors with source/target image pairs
- **Realtime LoRA Trainer (Wan 2.2 - Musubi Tuner)** - Trains Wan 2.2 with High/Low/Combo noise modes
- **Realtime LoRA Trainer (SDXL - sd-scripts)** - Trains using sd-scripts (SDXL)
- **Realtime LoRA Trainer (SD 1.5 - sd-scripts)** - Trains using sd-scripts (SD 1.5)
- **Apply Trained LoRA** - Applies the trained LoRA to your model

**Analysis & Selective Loading (V1):**
- **LoRA Loader + Analyzer** - Loads a LoRA and analyzes block-level impact (outputs analysis JSON for selective loaders)
- **Selective LoRA Loader (SDXL)** - Load SDXL LoRAs with per-block toggles and strength sliders
- **Selective LoRA Loader (Z-Image)** - Load Z-Image LoRAs with per-layer toggles (30 layers)
- **Selective LoRA Loader (FLUX)** - Load FLUX LoRAs with per-block toggles (57 blocks: 19 double + 38 single)
- **Selective LoRA Loader (Wan)** - Load Wan LoRAs with per-block toggles (40 blocks)
- **Selective LoRA Loader (Qwen)** - Load Qwen LoRAs with per-block toggles (60 blocks)

**V2 Combined Analyzer + Selective Loader (Recommended):**

These combine analysis and selective loading in one node, with strength scheduling and LoRA saving:

- **Z-Image Analyzer + Selective Loader V2** - 30 layers + extra blocks (context_refiner, noise_refiner, final_layer, x_embedder)
- **SDXL Analyzer + Selective Loader V2** - 14 blocks with strength scheduling
- **FLUX Analyzer + Selective Loader V2** - 57 blocks with strength scheduling
- **Wan Analyzer + Selective Loader V2** - 40 blocks with strength scheduling
- **Qwen Analyzer + Selective Loader V2** - 60 blocks with strength scheduling
- **LoRA Loader + Analyzer V2** - Basic V2 analyzer without selective loading

**Model Layer Editors:**

Scale individual blocks of your base model before LoRA application:

- **SDXL Model Layer Editor**
- **SD 1.5 Model Layer Editor**
- **FLUX Model Layer Editor**
- **Z-Image Model Layer Editor**
- **Wan Model Layer Editor**
- **Qwen Model Layer Editor**

**Utility Nodes:**

- **Clippy Reloaded (Load Image from Clipboard)** - Paste images directly into ComfyUI
- **Image of the Day** - Random inspiration images from Unsplash, Pexels, or NASA APOD
- **LoRA Loader (Scheduled)** - Standalone strength scheduling for any LoRA

## Getting Started

There are critical example workflows with useful info included in the custom_nodes/comfyUI-Realtime-Lora folder. Open one in ComfyUI and:

1. Paste the path to your training backend installation (sd-scripts, Musubi Tuner, or AI-Toolkit)
2. For SDXL/SD1.5: select your checkpoint from the dropdown
3. For Musubi Tuner Z-Image: select your de-distilled model, VAE, and text encoder from the dropdowns
4. For AI-Toolkit models: the first run will download the model from HuggingFace automatically

**First run with AI-Toolkit:** The model will download to your HuggingFace cache folder. On Windows this is `C:\Users\%USERNAME%\.cache\huggingface\hub`. You can watch that folder to monitor download progress - these models are large (several GB).

## Basic Usage

1. Add the trainer node for your model type
2. Connect your training image(s)
3. Set the path to your training backend installation
4. Queue the workflow
5. Connect the lora_path output to the Apply Trained LoRA node

## Example Workflows

Workflows are included in the `workflows/` folder, organized by category.

> **Note:** Many examples use Z-Image, but equivalent nodes exist for all architectures (SDXL, SD 1.5, FLUX, Wan, Qwen). Just search for the same node name with your architecture - e.g., "SDXL Analyzer + Selective Loader V2" or "FLUX Model Layer Editor".

**Training Workflows** (`workflows/Training/`):
- `SDXLDemo-SDSCRIPTS.json` - SDXL training demo
- `SD_1.5-SDSCRIPTS.json` - SD 1.5 training demo
- `Z-ImageTurbo_MUSUBI-TUNER-VERSION.json` - Z-Image via Musubi Tuner
- `Z-ImageTurbo_AiToolkitVersion.json` - Z-Image via AI-Toolkit
- `FluxDev-Ai-Toolkit.json` - FLUX training
- `Wan2.2 High Noise_MUSUBI-TUNER-VERSION.json` - Wan High noise mode
- `Wan2.2 Low Noise_MUSUBI-TUNER-VERSION.json` - Wan Low noise mode
- `Wan2.2 Combo Mode_MUSUBI-TUNER-VERSION.json` - Wan Combo mode
- `Qwen image and Qwen image edit 2509 Lora-MUSUBI.json` - Qwen Image training
- `Qwen image Edit with Control images pairs-MUSUBI.json` - Qwen Edit with paired images

**Analysis & Editing Workflows** (`workflows/Lora Analysis, Block and Model Editing/`):
- `Lora Analysis and Block Control Demo - Z-Image.json` - V1 selective loader demo
- `Lora Analysis and Block Control Demo- ADVANCED - Z-Image.json` - V2 with strength scheduling
- `Scheduled Lora Loader Demo - zimage.json` - Standalone strength scheduling
- `Model Layer Editor - Zimage.json` - Base model editing demo

**Utility Workflows** (root `workflows/` folder):
- `Clippy Reloaded - image from clipboard - zimage.json` - Clipboard loading demo
- `Image of the Day Node - Z-Image.json` - Random image inspiration demo

## Features

- Train from 1 to 100+ images
- Per-image captions (optional)
- Folder input for batch training with .txt caption files
- Automatic caching - identical inputs skip training and reuse the LoRA
- VRAM presets for different GPU sizes
- Settings are saved between sessions

## LoRA Analysis & Selective Loading

Beyond training, this toolkit includes tools for understanding and fine-tuning how LoRAs affect your generations.

**[▶ Watch Demo: LoRA Analysis & Selective Block Loading](https://www.youtube.com/watch?v=3CdyGxvYeHo)**

[![LoRA Analysis Demo](https://img.youtube.com/vi/3CdyGxvYeHo/0.jpg)](https://www.youtube.com/watch?v=3CdyGxvYeHo)

### LoRA Loader + Analyzer

The analyzer loads any LoRA and shows you which blocks have the most impact. It calculates a "strength" score (0-100%) for each block based on the weight magnitudes in that block. High-impact blocks are where the LoRA learned the most - these are often the blocks responsible for the subject's face, style, or composition.

**Outputs:**
- `model` / `clip` - The model with LoRA applied
- `analysis` - Human-readable text breakdown
- `analysis_json` - JSON data for selective loaders (enables impact-colored checkboxes)
- `lora_path` - Path to the loaded LoRA (can connect to selective loaders)

### Selective LoRA Loaders

Each architecture has its own selective loader with toggles and strength sliders for every block or layer. This lets you:

- **Disable low-impact blocks** to reduce LoRA influence on parts of the image
- **Focus on specific blocks** (e.g., face blocks, style blocks, composition blocks)
- **Fine-tune strength per-block** instead of using a single global strength

**Presets included:**
- Default (all on at 1.0)
- All Off
- Half Strength
- Architecture-specific presets (High Impact, Face Focus, Style Only, etc.)

### Impact-Colored Checkboxes

Connect the `analysis_json` output from the Analyzer to a Selective Loader's `analysis_json` input. The checkboxes will color-code by impact:

- **Blue** = Low impact (0-30%)
- **Cyan/Green** = Medium impact (30-60%)
- **Yellow/Orange** = High impact (60-90%)
- **Red** = Very high impact (90-100%)

This makes it easy to see at a glance which blocks matter most for your LoRA.

Disabled blocks show dimmed colors so you can still see their impact level.

### V2 Strength Scheduling

The V2 nodes support strength scheduling - varying LoRA strength during generation:

**Format:** `step:strength` pairs, comma-separated. Steps are 0-1 (percentage of generation).

**Examples:**
- `0:0, 1:1` - Linear fade in (off at start, full at end)
- `0:1, 0.5:1, 0.6:0, 1:0` - Full strength for first half, then off
- `0:0, 0.5:1, 1:0` - Bell curve (peak at middle)

**40+ Presets included:**
- Basic fades (Linear In/Out)
- Ease curves (slow start/end)
- Bell curves (peak middle, wide, sharp)
- Structure LoRA favorites (High Start → Cut)
- Detail LoRA favorites (Low Start → Ramp Late)
- Step functions and pulses
- Inverted versions of all presets

### Usage Notes

- **Analyzer standalone**: The LoRA Loader + Analyzer works on its own as a drop-in replacement for ComfyUI's standard LoRA loader. The analysis outputs are optional - you can ignore them and just use the model/clip outputs.

- **Path override**: When you connect a path to a Selective Loader's `lora_path` input, the dropdown selection is ignored. This lets you analyze one LoRA and selectively load it in one step.

- **Trainer → Selective Loader**: The `lora_path` output from any trainer node is compatible with the Selective Loader's path input. Train a LoRA and immediately load it with per-block control - useful for testing which blocks matter for your freshly trained subject.

## Utility Nodes

### Clippy Reloaded (Load Image from Clipboard)

*Resurrected from a graveyard near Seattle, Clippy returns to help with your image loading needs.*

Load images directly from your system clipboard - no file saving required.

**How to use:**
1. Copy an image from anywhere (browser right-click → Copy Image, screenshot, image editor)
2. Queue the workflow
3. Image loads directly into ComfyUI

**Features:**
- Works with browser images, screenshots, image editors, any app that copies to clipboard
- Handles RGBA images (composites onto white background)
- Handles file paths copied to clipboard (opens the file)
- Shows image preview in the node
- Clippy provides... commentary on your image choices

**Works with:**
- Right-click → Copy Image from browsers
- Screenshots (Win+Shift+S, Cmd+Shift+4, PrtScn)
- Copy from Photoshop, GIMP, etc.
- Any app that copies images to clipboard

### Image of the Day

Fetch random or daily images from various online sources for inspiration or testing.

**Sources (5 no-key required, 3 need free API key):**

| Source | API Key | Notes |
|--------|---------|-------|
| Lorem Picsum | No | Random photos, customizable size, seed for reproducibility |
| Bing Daily | No | Microsoft's daily wallpaper |
| Wikimedia POTD | No | Wikipedia's Picture of the Day |
| Random Dog | No | Random dog photos by breed |
| Random Cat | No | Random cat photos |
| NASA APOD | Yes (free) | Astronomy Picture of the Day - get key at api.nasa.gov |
| Unsplash | Yes (free) | High-quality random photos - get key at unsplash.com/developers |
| Pexels | Yes (free) | Curated stock photos - get key at pexels.com/api |

**Features:**
- **API key persistence** - Enter your key once, it's saved for future sessions
- **1-hour caching** - Avoids hammering APIs, respects rate limits
- **Width/height control** - For Lorem Picsum source
- **Seed option** - Get reproducible "random" images

**Outputs:**
- `image` - The loaded image
- `title` - Image title/name
- `description` - Image description (when available)
- `source_url` - Link to original source
- `copyright` - Attribution/copyright info

### Scheduled LoRA Loader

A standalone LoRA loader with the same 40+ strength scheduling presets as the V2 nodes, plus schedule chaining for advanced workflows.

**Features:**
- **40+ schedule presets** - Same as V2 nodes (fades, bells, pulses, steps, etc.)
- **Schedule chaining** - Pass schedules between nodes via `schedule_out` → `schedule_in`
- **Inverted schedule output** - Automatic `schedule_inv` for crossfade effects

**Outputs:**
- `model` - Model with LoRA applied
- `positive` / `negative` - Conditioning (with hooks if scheduled)
- `schedule_out` - The active schedule string (for chaining)
- `schedule_inv` - Inverted schedule (1 - strength at each keyframe)

**LoRA Crossfading:**

Connect two Scheduled LoRA Loaders to crossfade between LoRAs during generation:

1. **LoRA A** uses "Linear Out (1→0)" - fades from full strength to zero
2. Connect `schedule_inv` from LoRA A → `schedule_in` on LoRA B
3. **LoRA B** automatically receives "0:0, 1:1" - fades from zero to full strength

Result: LoRA A fades out while LoRA B fades in, creating a smooth transition.

**Use cases:**
- Crossfade between structure LoRA (early) and detail LoRA (late)
- Blend styles progressively during generation
- Chain multiple LoRAs with complementary schedules

## Defaults (Z-Image example)

- 400 training steps
- Learning rate 0.0002
- LoRA rank 16
- Low VRAM mode (768px)

These defaults are starting points for experimentation, not ideal values. Every subject and style is different.

**Learning rate advice:**
- 0.0002 trains fast but can overshoot, causing artifacts or burning in the subject too hard
- Try lowering to 0.0001 or 0.00005 for more stable, gradual training
- If your LoRA looks overcooked or the subject bleeds into everything, lower the learning rate
- If your LoRA is too weak after 400-500 steps, try more steps before raising the learning rate, its already high in the example workflows.


## Support

If this tool saves you time or fits into your workflow, consider [buying me a coffee](https://buymeacoffee.com/lorasandlenses).

I'm currently between contracts due to family circumstances, which has given me time to build and maintain this project. Your support helps me keep developing it.

Members get early access to new builds before public release.


## Credits

This project makes use of these excellent training tools for the training nodes:

- **AI-Toolkit** by ostris: https://github.com/ostris/ai-toolkit
- **sd-scripts** by kohya-ss: https://github.com/kohya-ss/sd-scripts
- **Musubi Tuner** by kohya-ss: https://github.com/kohya-ss/musubi-tuner

The training is done by these projects. This node just makes them accessible from within ComfyUI in a user centric manner. Essentially i want to democratize training and make it easier to get into creativly.

## Author

Peter Neill - [ShootTheSound.com](https://shootthesound.com) / [UltrawideWallpapers.net](https://ultrawidewallpapers.net)

Background in music industry photography and video. Built this node to make LoRA training accessible to creators who just want to get things done without diving into command line tools.

Feedback is welcome - open an issue or reach out.

## License

MIT
