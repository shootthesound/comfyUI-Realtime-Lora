"""
ComfyUI Realtime LoRA Trainer

Trains LoRAs on-the-fly from images during generation.
Supports Z-Image, FLUX, Wan models via AI-Toolkit.
Also supports SDXL and SD 1.5 via kohya sd-scripts.

Includes LoRA Layer Analyzer and Selective LoRA Loader for analyzing
and loading specific blocks/layers from LoRA files.
"""

from .realtime_lora_trainer import RealtimeLoraTrainer, ApplyTrainedLora
from .sdxl_lora_trainer import SDXLLoraTrainer
from .sd15_lora_trainer import SD15LoraTrainer
from .musubi_zimage_lora_trainer import MusubiZImageLoraTrainer
from .musubi_qwen_image_lora_trainer import MusubiQwenImageLoraTrainer
from .musubi_qwen_image_edit_lora_trainer import MusubiQwenImageEditLoraTrainer
from .musubi_wan_lora_trainer import MusubiWanLoraTrainer
from .lora_analyzer import LoRALoaderWithAnalysis
from .lora_analyzer_v2 import NODE_CLASS_MAPPINGS as V2_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as V2_NODE_DISPLAY_NAME_MAPPINGS
from .selective_lora_loader import SDXLSelectiveLoRALoader, ZImageSelectiveLoRALoader, FLUXSelectiveLoRALoader, WanSelectiveLoRALoader, QwenSelectiveLoRALoader
from .scheduled_lora_loader import ScheduledLoRALoader
from .clipboard_image_loader import ClippyRebornImageLoader
from .image_of_day import ImageOfDayLoader
from .model_layer_analyzer import NODE_CLASS_MAPPINGS as MODEL_LAYER_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MODEL_LAYER_NODE_DISPLAY_NAME_MAPPINGS

# Web directory for JavaScript extensions
WEB_DIRECTORY = "./web/js"

NODE_CLASS_MAPPINGS = {
    "RealtimeLoraTrainer": RealtimeLoraTrainer,
    "ApplyTrainedLora": ApplyTrainedLora,
    "SDXLLoraTrainer": SDXLLoraTrainer,
    "SD15LoraTrainer": SD15LoraTrainer,
    "MusubiZImageLoraTrainer": MusubiZImageLoraTrainer,
    "MusubiQwenImageLoraTrainer": MusubiQwenImageLoraTrainer,
    "MusubiQwenImageEditLoraTrainer": MusubiQwenImageEditLoraTrainer,
    "MusubiWanLoraTrainer": MusubiWanLoraTrainer,
    "LoRALoaderWithAnalysis": LoRALoaderWithAnalysis,
    "SDXLSelectiveLoRALoader": SDXLSelectiveLoRALoader,
    "ZImageSelectiveLoRALoader": ZImageSelectiveLoRALoader,
    "FLUXSelectiveLoRALoader": FLUXSelectiveLoRALoader,
    "WanSelectiveLoRALoader": WanSelectiveLoRALoader,
    "QwenSelectiveLoRALoader": QwenSelectiveLoRALoader,
    "ScheduledLoRALoader": ScheduledLoRALoader,
    "ClippyRebornImageLoader": ClippyRebornImageLoader,
    "ImageOfDayLoader": ImageOfDayLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RealtimeLoraTrainer": "Realtime LoRA Trainer",
    "ApplyTrainedLora": "Apply Trained LoRA",
    "SDXLLoraTrainer": "Realtime LoRA Trainer (SDXL - sd-scripts)",
    "SD15LoraTrainer": "Realtime LoRA Trainer (SD 1.5 - sd-scripts)",
    "MusubiZImageLoraTrainer": "Realtime LoRA Trainer (Z-Image - Musubi Tuner)",
    "MusubiQwenImageLoraTrainer": "Realtime LoRA Trainer (Qwen Image - Musubi Tuner)",
    "MusubiQwenImageEditLoraTrainer": "Realtime LoRA Trainer (Qwen Image Edit - Musubi Tuner)",
    "MusubiWanLoraTrainer": "Realtime LoRA Trainer (Wan 2.2 - Musubi Tuner)",
    "LoRALoaderWithAnalysis": "LoRA Loader + Analyzer",
    "SDXLSelectiveLoRALoader": "Selective LoRA Loader (SDXL)",
    "ZImageSelectiveLoRALoader": "Selective LoRA Loader (Z-Image)",
    "FLUXSelectiveLoRALoader": "Selective LoRA Loader (FLUX)",
    "WanSelectiveLoRALoader": "Selective LoRA Loader (Wan)",
    "QwenSelectiveLoRALoader": "Selective LoRA Loader (Qwen)",
    "ScheduledLoRALoader": "LoRA Loader (Scheduled)",
    "ClippyRebornImageLoader": "Clippy Reloaded (Load Image from Clipboard)",
    "ImageOfDayLoader": "Image of the Day",
}

# Merge V2 analyzer nodes (includes combined analyzer+selective loaders)
NODE_CLASS_MAPPINGS.update(V2_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(V2_NODE_DISPLAY_NAME_MAPPINGS)

# Merge Model Layer Analyzer/Editor nodes
NODE_CLASS_MAPPINGS.update(MODEL_LAYER_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(MODEL_LAYER_NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
