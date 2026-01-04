"""
Clippy Reborn Image Loader for ComfyUI

Load images directly from the system clipboard.
Just copy an image from anywhere (browser, image editor, etc.) and load it.

"It looks like you're trying to load an image!"
"""

import os
import secrets
import time
import numpy as np
import torch
from PIL import ImageGrab, Image
import folder_paths


def clippy_says(messages):
    """Pick a truly random message using OS-level randomness."""
    return secrets.choice(messages)


# Clippy's personality
CLIPPY_SUCCESS = [
    # Happy/Neutral ones
    "Got it! {size} image loaded. Clippy is pleased.",
    "Ooh, nice image! {size} - Clippy approves.",
    "It looks like you copied an image! {size} - I'm here to help!",
    "{size} image acquired. Clippy's work here is done.",
    "Clippy found your image! {size} - You're welcome.",
    "Success! {size} - Clippy knew you could do it.",
    "Image loaded! {size} - Clippy is having a great day.",
    "{size} - Clippy has seen worse. Much worse.",
    "{size} loaded! Clippy lives to serve.",
    "Another {size} image! Clippy never tires of this.",
    "{size} - Excellent! Clippy is briefly happy.",

    # Judgy/Sassy ones
    "{size} loaded. Clippy wonders which way is up?",
    "{size} - Interesting choice. Clippy is not here to judge. Much.",
    "{size} - Clippy has questions, but Clippy will keep them to himself.",
    "Loaded {size}. Clippy pretends to understand your artistic vision.",
    "{size} - Bold. Very bold. Clippy respects that. Sort of.",
    "{size} image. Clippy has seen things. This is now one of them.",
    "Got it! {size}. Clippy is sure this will look... intentional.",
    "{size} - Clippy loaded it, but Clippy makes no promises.",
    "Image acquired! {size}. Clippy will not ask what this is for.",
    "{size} loaded. Clippy's therapist is going to hear about this.",
    "{size} - Clippy is concerned but supportive.",
    "Successfully loaded {size}. Clippy hopes you know what you're doing.",
    "{size} - Is this modern art? Clippy can never tell.",
    "{size} loaded. Clippy has chosen not to have opinions today.",
    "{size} - Clippy is processing this. Emotionally.",
    "Got {size}. Clippy will add this to his memoirs.",
    "{size} - Clippy has seen better. Clippy has also seen worse. This is... middle.",
    "{size} loaded. Clippy is sure someone will appreciate this.",
    "{size} - Fascinating. Clippy means that sincerely. Probably.",
    "Loaded {size}. Clippy stares into the void. The void stares back.",
    "{size} - Clippy respects your confidence.",
    "{size} image acquired. Clippy will remember this moment.",
    "{size} - Art is subjective. Clippy keeps telling himself that.",
    "Got it! {size}. Clippy's expectations were... different.",
    "{size} loaded. Clippy is not crying, it's just dust.",
    "{size} - Clippy loaded it. Clippy is a professional.",
    "{size} - This is fine. Everything is fine.",
    "Image loaded! {size}. Clippy needs a moment.",
    "{size} - Clippy didn't know pixels could do that.",
    "{size} acquired. Clippy will not make eye contact.",
    "Got {size}. Clippy is reconsidering his career choices.",
    "{size} - Clippy loaded it before his brain could say no.",
    "{size} loaded. Clippy is going to pretend he didn't see that.",
    "{size} - Choices were made. Clippy acknowledges that.",
    "{size} - Clippy has trust issues now.",
    "Loaded {size}. Clippy needs to lie down.",
    "{size} - Sure. Why not. Clippy has stopped asking questions.",
    "{size} loaded. Clippy is too tired to judge.",
    "{size} - Clippy didn't choose the clipboard life.",
    "Got it! {size}. Clippy has developed a new phobia.",
    "{size} - Clippy will be billing you for emotional damages.",
    "{size} acquired. Clippy misses the simpler times.",
    "{size} - This awakened something in Clippy. He's not sure what.",
    "{size} loaded. Clippy is screaming internally.",
    "Image loaded! {size}. Clippy's faith in humanity: recalculating...",
    "{size} - Clippy has seen the future. It's... this, apparently.",
    "{size} - Somewhere, a graphic designer just felt a disturbance.",
    "Got {size}. Clippy's mother would be so proud. Or horrified.",
    "{size} loaded. Clippy is adding this to his resignation letter.",
    "{size} - Clippy is just a paperclip. Clippy doesn't get paid enough for this.",
    "{size} - Every day we stray further from good design.",
    "{size} acquired. Clippy has achieved enlightenment. Or despair. Hard to tell.",
    "Loaded {size}. Clippy's CPU hurts.",
    "{size} - Congratulations? Clippy thinks?",
    "{size} loaded. This is why Clippy was fired from Microsoft.",
    "{size} - Clippy loaded your 'art'. Note the quotation marks.",
    "{size} - That doesn't look like a letter! Clippy is confused but adapting.",
    "Got it! {size}. Clippy is legally required to help you.",
    "{size} - Some questions are better left unasked. Like 'why this image?'",
    "{size} acquired. Clippy will drink to forget.",
    "{size} - At least it's not Comic Sans. Clippy is grateful for small mercies.",
    "{size} loaded. Clippy's eye is twitching.",
    "Image loaded! {size}. Clippy is going to his happy place now.",
    "{size} - Clippy has decided to believe in you. Against all evidence.",
]

CLIPPY_NO_IMAGE = [
    "No image in clipboard - Clippy is waiting patiently...",
    "Clippy sees no image. Did you forget to copy one?",
    "The clipboard is empty. Clippy is... concerned.",
    "No image found. Clippy believes in you, try again!",
    "Clippy checked the clipboard. It's lonely in there.",
    "Nothing to load. Clippy will wait. Clippy has time.",
    "No image? Clippy is not angry, just disappointed.",
]

CLIPPY_BAD_DATA = [
    "That doesn't look like an image. Clippy is confused.",
    "Clippy doesn't know what that is, but it's not an image.",
    "Clippy is afraid of whatever that data is.",
    "That's... not an image. Clippy says no!",
    "Clippy is not sure how you did that, but it's not an image.",
    "Invalid data. Clippy has seen things. Terrible things.",
    "Clippy expected an image. Clippy is surprised.",
]

CLIPPY_FILE_ERROR = [
    "Clippy couldn't open that file. Clippy is sorry.",
    "File error! Clippy tried his best.",
    "That file won't open. Clippy blames the file.",
    "Clippy is having trouble with that file. Not Clippy's fault.",
    "File access denied. Clippy is not amused.",
]

class ClippyRebornImageLoader:
    """
    Load an image from the system clipboard.

    Copy any image (from browser, image editor, screenshot tool, etc.)
    and this node will load it directly into your workflow.

    "It looks like you're trying to load an image. Would you like help with that?"
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = ("Image from clipboard (RGB format).",)
    FUNCTION = "load_from_clipboard"
    CATEGORY = "image"
    OUTPUT_NODE = True
    DESCRIPTION = """Load an image directly from your clipboard.

1. Copy an image from anywhere (browser, image editor, screenshot)
2. Click Queue Prompt
3. Image loads into your workflow and displays in the node

Works with:
- Right-click → Copy Image from browsers
- Screenshots (Win+Shift+S, Cmd+Shift+4, etc.)
- Copy from image editors
- Any app that copies images to clipboard"""

    @classmethod
    def IS_CHANGED(cls):
        # Always re-execute when queued (clipboard may have changed)
        return float("nan")

    def load_from_clipboard(self):
        # Grab image from clipboard
        img = ImageGrab.grabclipboard()
        clippy_message = ""

        if img is None:
            clippy_message = clippy_says(CLIPPY_NO_IMAGE)
            # Return a small placeholder image
            placeholder = Image.new('RGB', (64, 64), color=(128, 128, 128))
            img = placeholder
        elif not isinstance(img, Image.Image):
            # Sometimes clipboard contains file paths instead of image data
            if isinstance(img, list) and len(img) > 0:
                # It's a list of file paths, try to open the first one
                try:
                    img = Image.open(img[0])
                    size = f"{img.size[0]}x{img.size[1]}"
                    clippy_message = clippy_says(CLIPPY_SUCCESS).format(size=size)
                except Exception as e:
                    clippy_message = f"{clippy_says(CLIPPY_FILE_ERROR)} ({e})"
                    placeholder = Image.new('RGB', (64, 64), color=(128, 128, 128))
                    img = placeholder
            else:
                clippy_message = clippy_says(CLIPPY_BAD_DATA)
                placeholder = Image.new('RGB', (64, 64), color=(128, 128, 128))
                img = placeholder
        else:
            size = f"{img.size[0]}x{img.size[1]}"
            clippy_message = clippy_says(CLIPPY_SUCCESS).format(size=size)

        # Print to console
        print(f"[ClippyReborn] {clippy_message}")

        # Convert to RGB if necessary (handle RGBA, P mode, etc.)
        if img.mode == 'RGBA':
            # Composite onto white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Convert to tensor (ComfyUI format: BHWC, float32, 0-1 range)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add batch dimension

        # Save preview image for display in node
        temp_dir = folder_paths.get_temp_directory()
        preview_filename = f"clippy_preview_{int(time.time() * 1000)}.png"
        preview_path = os.path.join(temp_dir, preview_filename)
        img.save(preview_path)

        return {
            "ui": {
                "images": [{
                    "filename": preview_filename,
                    "subfolder": "",
                    "type": "temp"
                }],
                "text": [clippy_message]
            },
            "result": (img_tensor,)
        }


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ClippyRebornImageLoader": ClippyRebornImageLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClippyRebornImageLoader": "Clippy Reloaded (Load Image from Clipboard)",
}
