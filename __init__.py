"""
ComfyUI Realtime LoRA Trainer
Trains LoRAs on-the-fly from images during generation.
Supports Z-Image, FLUX, Wan models via AI-Toolkit.
Also supports SDXL and SD 1.5 via kohya sd-scripts.
Includes LoRA Layer Analyzer and Selective LoRA Loader for analyzing
and loading specific blocks/layers from LoRA files.
and Flux Klein / VAE / Qwen3-8B / Qwen3-4B / Z-Image debiasing & inspection tools.
"""

from .realtime_lora_trainer import RealtimeLoraTrainer, ApplyTrainedLora
from .sdxl_lora_trainer import SDXLLoraTrainer
from .sd15_lora_trainer import SD15LoraTrainer
from .anima_lora_trainer import AnimaLoraTrainer
from .musubi_zimage_lora_trainer import MusubiZImageLoraTrainer
from .musubi_zimage_base_lora_trainer import MusubiZImageBaseLoraTrainer
from .musubi_flux_klein_lora_trainer import MusubiFluxKleinLoraTrainer
from .musubi_qwen_image_lora_trainer import MusubiQwenImageLoraTrainer
from .musubi_qwen_image_edit_lora_trainer import MusubiQwenImageEditLoraTrainer
from .musubi_wan_lora_trainer import MusubiWanLoraTrainer
from .lora_analyzer import LoRALoaderWithAnalysis
from .lora_analyzer_v2 import (
    NODE_CLASS_MAPPINGS as V2_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as V2_NODE_DISPLAY_NAME_MAPPINGS,
)
from .selective_lora_loader import (
    SDXLSelectiveLoRALoader,
    ZImageSelectiveLoRALoader,
    FLUXSelectiveLoRALoader,
    WanSelectiveLoRALoader,
    QwenSelectiveLoRALoader,
)
from .scheduled_lora_loader import ScheduledLoRALoader
from .model_layer_analyzer import (
    NODE_CLASS_MAPPINGS as MODEL_LAYER_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MODEL_LAYER_NODE_DISPLAY_NAME_MAPPINGS,
)

# Flux Klein Debiaser Pack
from .flux_klein_debiaser_node import (
    NODE_CLASS_MAPPINGS as FK_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as FK_NODE_DISPLAY_NAME_MAPPINGS,
)
from .flux_vae_debiaser_node import (
    NODE_CLASS_MAPPINGS as FV_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as FV_NODE_DISPLAY_NAME_MAPPINGS,
)
from .flux_vae_inspector_node import (
    NODE_CLASS_MAPPINGS as FVI_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as FVI_NODE_DISPLAY_NAME_MAPPINGS,
)
from .qwen3_8b_text_encoder_debiaser_node import (
    NODE_CLASS_MAPPINGS as QW_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as QW_NODE_DISPLAY_NAME_MAPPINGS,
)
from .qwen3_8b_text_encoder_inspector_node import (
    NODE_CLASS_MAPPINGS as QWI_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as QWI_NODE_DISPLAY_NAME_MAPPINGS,
)

# Qwen3-4B Text Encoder Debiaser & Inspector
from .qwen3_4b_text_encoder_debiaser_node import (
    NODE_CLASS_MAPPINGS as QW4_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as QW4_NODE_DISPLAY_NAME_MAPPINGS,
)
from .qwen3_4b_text_encoder_inspector_node import (
    NODE_CLASS_MAPPINGS as QW4I_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as QW4I_NODE_DISPLAY_NAME_MAPPINGS,
)

# Z-Image Deep Debiaser
from .zimage_deep_debiaser_node import (
    NODE_CLASS_MAPPINGS as ZI_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as ZI_NODE_DISPLAY_NAME_MAPPINGS,
)

# Web directory for JavaScript extensions
WEB_DIRECTORY = "./web/js"

NODE_CLASS_MAPPINGS = {
    "RealtimeLoraTrainer": RealtimeLoraTrainer,
    "ApplyTrainedLora": ApplyTrainedLora,
    "SDXLLoraTrainer": SDXLLoraTrainer,
    "SD15LoraTrainer": SD15LoraTrainer,
    "AnimaLoraTrainer": AnimaLoraTrainer,
    "MusubiZImageLoraTrainer": MusubiZImageLoraTrainer,
    "MusubiZImageBaseLoraTrainer": MusubiZImageBaseLoraTrainer,
    "MusubiFluxKleinLoraTrainer": MusubiFluxKleinLoraTrainer,
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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RealtimeLoraTrainer": "Realtime LoRA Trainer",
    "ApplyTrainedLora": "Apply Trained LoRA",
    "SDXLLoraTrainer": "Realtime LoRA Trainer (SDXL - sd-scripts)",
    "SD15LoraTrainer": "Realtime LoRA Trainer (SD 1.5 - sd-scripts)",
    "AnimaLoraTrainer": "Realtime LoRA Trainer (Anima - sd-scripts)",
    "MusubiZImageLoraTrainer": "Realtime LoRA Trainer (Z-Image - Musubi Tuner)",
    "MusubiZImageBaseLoraTrainer": "Realtime LoRA Trainer (Z-Image Base - Musubi Tuner)",
    "MusubiFluxKleinLoraTrainer": "Realtime LoRA Trainer (FLUX Klein - Musubi Tuner)",
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
}

# Merge V2 analyzer nodes (includes combined analyzer+selective loaders)
NODE_CLASS_MAPPINGS.update(V2_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(V2_NODE_DISPLAY_NAME_MAPPINGS)
# Merge Model Layer Analyzer/Editor nodes
NODE_CLASS_MAPPINGS.update(MODEL_LAYER_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(MODEL_LAYER_NODE_DISPLAY_NAME_MAPPINGS)

# Flux Klein / VAE / Qwen3-8B debiaser & inspector nodes
NODE_CLASS_MAPPINGS.update(FK_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(FV_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(FVI_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(QW_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(QWI_NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS.update(FK_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(FV_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(FVI_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(QW_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(QWI_NODE_DISPLAY_NAME_MAPPINGS)

# Qwen3-4B Text Encoder debiaser & inspector nodes
NODE_CLASS_MAPPINGS.update(QW4_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(QW4I_NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS.update(QW4_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(QW4I_NODE_DISPLAY_NAME_MAPPINGS)

# Z-Image Deep Debiaser node
NODE_CLASS_MAPPINGS.update(ZI_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(ZI_NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
