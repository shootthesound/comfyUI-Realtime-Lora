# ComfyUI Realtime LoRA Trainer

Train LoRAs directly inside ComfyUI. Capture a face, a style, or a subject from your reference images and apply it to new generations - all within the same workflow. Useful for subject consistency across a project, style matching, or rapid iteration when you need a LoRA but don't want to context-switch into separate training tools. 


## What This Does

This node trains LoRAs on-the-fly from your images without leaving ComfyUI. SDXL training is particularly fast - a few minutes on a decent GPU. This makes it practical to train a quick LoRA from an image and immediately use it for img2img variations, style transfer, or subject consistency within the same workflow.

No config files to edit, no command line. Just connect images and go.

If this node saves you time or helps your workflow, consider [buying me a coffee](https://buymeacoffee.com/lorasandlenses). Due to family circumstances I'm short on work right now, which is partly why I've had the time to build this. Donations are genuinely appreciated and help me keep developing this tool.

## Supported Models

**Via Kohya sd-scripts:**
- SDXL (any checkpoint) - tested with Juggernaut XL Ragnarok, base SDXL will work too

**Via AI-Toolkit:**
- Z-Image Turbo
- FLUX.1-dev
- Wan 2.2 (High/Low/Combo)

**Note on Wan 2.2 modes:** Wan uses a two-stage noise model - High handles early denoising steps, Low handles later steps. You can train separate LoRAs for each, or use Combo mode which trains a single LoRA across all noise steps that works with both High and Low models.

## Requirements

You need to install the training backend(s) separately:

**For SDXL training:**
1. Install sd-scripts: https://github.com/kohya-ss/sd-scripts
2. Follow their install instructions
3. Run `accelerate config` in the venv (just press Enter to accept defaults for each question)

**For FLUX/Z-Image/Wan training:**
1. Install AI-Toolkit: https://github.com/ostris/ai-toolkit
2. Follow their install instructions

You don't need to open either environment after installation. The node just needs the path to where you installed them.

## Installation

Clone this repo into your ComfyUI custom_nodes folder:

```
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/comfyUI-Realtime-Lora
```

Restart ComfyUI.

## Nodes

**Realtime LoRA Trainer** - Trains using AI-Toolkit (FLUX, Z-Image, Wan)

**Realtime LoRA Trainer (SDXL)** - Trains using sd-scripts (SDXL)

**Apply Trained LoRA** - Applies the trained LoRA to your model

## Getting Started

There are example workflows included in the custom_nodes/comfyUI-Realtime-Lora folder. Open one in ComfyUI and:

1. Paste the path to your sd-scripts or AI-Toolkit installation into the node
2. For SDXL: select your checkpoint from the dropdown
3. For AI-Toolkit models: the first run will download the model from HuggingFace automatically

**First run with AI-Toolkit:** The model will download to your HuggingFace cache folder. On Windows this is typically `C:\Users\YourName\.cache\huggingface\hub`. You can watch that folder to monitor download progress - these models are large (several GB).

## Basic Usage

1. Add the trainer node for your model type
2. Connect your training image(s)
3. Set the path to your AI-Toolkit or sd-scripts installation
4. Queue the workflow
5. Connect the lora_path output to the Apply Trained LoRA node

## Features

- Train from 1 to 100+ images
- Per-image captions (optional)
- Folder input for batch training with .txt caption files
- Automatic caching - identical inputs skip training and reuse the LoRA
- VRAM presets for different GPU sizes
- Settings are saved between sessions

## Defaults

- 500 training steps
- Learning rate 0.0005
- LoRA rank 16
- Low VRAM mode (768px)

These defaults are starting points for experimentation, not ideal values. Every subject and style is different.

**Learning rate advice:**
- 0.0005 trains fast but can overshoot, causing artifacts or burning in the subject too hard
- Try lowering to 0.0001 or 0.00005 for more stable, gradual training
- If your LoRA looks overcooked or the subject bleeds into everything, lower the learning rate
- If your LoRA is too weak after 500 steps, try more steps before raising the learning rate

## Credits

This project is a thin wrapper that calls these excellent training tools:

- **AI-Toolkit** by ostris: https://github.com/ostris/ai-toolkit
- **sd-scripts** by kohya-ss: https://github.com/kohya-ss/sd-scripts

All the heavy lifting is done by these projects. This node just makes them accessible from within ComfyUI.

## Author

Peter Neill - [ShootTheSound.com](https://shootthesound.com) / [UltrawideWallpapers.net](https://ultrawidewallpapers.net)

Background in music industry photography and video. Built this node to make LoRA training accessible to creators who just want to get things done without diving into command line tools.

Feedback is welcome - open an issue or reach out. There's a roadmap for future development including SD 1.5 support, an export/import presets system, and more.

## License

MIT
