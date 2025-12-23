"""
Image of the Day Loader for ComfyUI

Fetches images from various "image of the day" and random image services.
No pip installs required - uses stdlib + ComfyUI builtins.
"""

import os
import json
import hashlib
import time
import tempfile
import re
import random
from io import BytesIO
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image


# Cache settings
CACHE_DIR = os.path.join(tempfile.gettempdir(), "comfyui_image_of_day_cache")
CACHE_DURATION = 3600  # 1 hour

# Config file for persistent API keys
CONFIG_FILE = os.path.join(os.path.dirname(__file__), ".image_of_day_config.json")


def load_api_keys() -> dict:
    """Load saved API keys from config file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('api_keys', {})
        except (json.JSONDecodeError, IOError, OSError):
            pass
    return {}


def save_api_keys(api_keys: dict):
    """Save API keys to config file."""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump({'api_keys': api_keys}, f, indent=2)
    except (IOError, OSError):
        pass


# Load saved keys at module load
_saved_api_keys = load_api_keys()


def get_cache_path(key: str) -> str:
    """Get cache file path for a key."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    hash_key = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hash_key}.json")


def get_cached(key: str):
    """Get cached response if valid."""
    cache_path = get_cache_path(key)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if time.time() - data.get('timestamp', 0) < CACHE_DURATION:
                return data.get('content')
        except (json.JSONDecodeError, IOError, OSError, KeyError):
            pass
    return None


def set_cached(key: str, content):
    """Cache a response."""
    cache_path = get_cache_path(key)
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({'timestamp': time.time(), 'content': content}, f)
    except (IOError, OSError, TypeError):
        pass


def fetch_url(url: str, headers: dict = None) -> bytes:
    """Fetch URL content."""
    req = Request(url, headers=headers or {})
    req.add_header('User-Agent', 'ComfyUI-ImageOfDay/1.0')
    with urlopen(req, timeout=30) as response:
        return response.read()


def fetch_json(url: str, headers: dict = None) -> dict:
    """Fetch and parse JSON from URL."""
    data = fetch_url(url, headers)
    return json.loads(data.decode('utf-8'))


def load_image_from_url(url: str) -> Image.Image:
    """Load PIL Image from URL."""
    data = fetch_url(url)
    return Image.open(BytesIO(data)).convert('RGB')


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI tensor format (BHWC)."""
    img_array = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_array).unsqueeze(0)


class ImageOfDayLoader:
    """
    Load images from various "Image of the Day" services.

    Sources include NASA APOD, Bing Daily, Wikimedia, Lorem Picsum,
    Unsplash, Pexels, and fun options like Random Dog/Cat.

    Most sources work without API keys. Unsplash and Pexels
    require free API keys from their developer portals.

    Note: NASA APOD uses DEMO_KEY by default (30 requests/hour limit).
    For heavy use, get a free API key at api.nasa.gov.
    """

    # Sources ordered: no-API first, API-required last
    SOURCES = [
        "Lorem Picsum",
        "Bing Daily",
        "Wikimedia POTD",
        "Random Dog",
        "Random Cat",
        "NASA APOD (API)",
        "Unsplash (API)",
        "Pexels (API)",
    ]

    # Which sources need API keys
    API_SOURCES = ["NASA APOD (API)", "Unsplash (API)", "Pexels (API)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (cls.SOURCES, {
                    "default": "Lorem Picsum",
                    "tooltip": "Image source. Sources marked (API) require a free API key."
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "API key for selected source. NASA: api.nasa.gov | Unsplash: unsplash.com/developers | Pexels: pexels.com/api"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "tooltip": "Image width (for Lorem Picsum)"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "tooltip": "Image height (for Lorem Picsum)"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "tooltip": "Seed for reproducible random (0 = random)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "title", "description", "source_url", "copyright")
    FUNCTION = "load_image"
    CATEGORY = "image"

    def load_image(self, source, api_key="", width=1024, height=1024, seed=0):
        """Load image from selected source."""
        global _saved_api_keys

        # Treat "no key needed" as empty (set by JS for non-API sources)
        if api_key == "no key needed":
            api_key = ""

        # Use saved API key if none provided
        if not api_key and source in self.API_SOURCES:
            api_key = _saved_api_keys.get(source, "")

        # Save API key if provided for an API source
        if api_key and source in self.API_SOURCES:
            if _saved_api_keys.get(source) != api_key:
                _saved_api_keys[source] = api_key
                save_api_keys(_saved_api_keys)

        try:
            if source == "Lorem Picsum":
                return self._fetch_lorem_picsum(width, height, seed)
            elif source == "Bing Daily":
                return self._fetch_bing_daily()
            elif source == "Wikimedia POTD":
                return self._fetch_wikimedia_potd()
            elif source == "Random Dog":
                return self._fetch_random_dog()
            elif source == "Random Cat":
                return self._fetch_random_cat()
            elif source == "NASA APOD (API)":
                return self._fetch_nasa_apod(api_key)
            elif source == "Unsplash (API)":
                return self._fetch_unsplash(api_key)
            elif source == "Pexels (API)":
                return self._fetch_pexels(api_key)
            else:
                return self._error_result(f"Unknown source: {source}")
        except Exception as e:
            return self._error_result(str(e))

    def _error_result(self, message: str):
        """Return error placeholder."""
        print(f"[ImageOfDay] Error: {message}")
        # Create a simple gray placeholder
        img = Image.new('RGB', (512, 512), color=(128, 128, 128))
        tensor = pil_to_tensor(img)
        return (tensor, "Error", message, "", "")

    def _fetch_nasa_apod(self, api_key: str = ""):
        """Fetch NASA Astronomy Picture of the Day."""
        if not api_key:
            return self._error_result("NASA APOD requires an API key. Get one free at api.nasa.gov")

        # Try today first, then fall back to previous days if not an image
        from datetime import datetime, timedelta

        for days_back in range(4):  # Try today + 3 previous days
            target_date = datetime.now() - timedelta(days=days_back)
            date_str = target_date.strftime('%Y-%m-%d')
            cache_key = f"nasa_apod_{date_str}_{api_key[:8]}"

            cached = get_cached(cache_key)
            # Validate cache has expected keys and is an image
            if cached and cached.get('media_type') == 'image' and ('url' in cached or 'hdurl' in cached):
                data = cached
            else:
                # Include thumbs=True to get video thumbnails, and specific date
                url = f"https://api.nasa.gov/planetary/apod?api_key={api_key}&thumbs=True&date={date_str}"
                try:
                    data = fetch_json(url)
                except HTTPError as e:
                    return self._error_result(f"NASA API HTTP error: {e.code} - Check your API key")
                except Exception as e:
                    return self._error_result(f"NASA API request failed: {e}")

                # Cache all responses
                set_cached(cache_key, data)

            # Check for API error response
            if 'error' in data:
                error_msg = data.get('error', {}).get('message', str(data.get('error')))
                return self._error_result(f"NASA API error: {error_msg}")

            # Check for error code/message format
            if 'code' in data and 'msg' in data:
                return self._error_result(f"NASA API error: {data.get('msg')}")

            # Check media type - only accept images
            media_type = data.get('media_type', 'image')
            if media_type == 'image':
                img_url = data.get('hdurl') or data.get('url', '')
                if img_url:
                    break  # Found a good image
            elif media_type == 'video':
                # Try thumbnail for videos
                img_url = data.get('thumbnail_url', '')
                if img_url:
                    break  # Video with thumbnail is acceptable
            # If 'other' or no URL, try previous day
            continue
        else:
            # Exhausted all attempts
            return self._error_result("No image APOD found in the last 4 days")

        if not img_url:
            return self._error_result("No image URL in NASA response")

        img = load_image_from_url(img_url)
        tensor = pil_to_tensor(img)

        return (
            tensor,
            data.get('title', 'NASA APOD'),
            data.get('explanation', '')[:500],  # Truncate long descriptions
            data.get('url', img_url),
            data.get('copyright', 'Public Domain')
        )

    def _fetch_bing_daily(self):
        """Fetch Bing's daily wallpaper."""
        cache_key = f"bing_daily_{time.strftime('%Y-%m-%d')}"

        cached = get_cached(cache_key)
        if cached:
            data = cached
        else:
            url = "https://www.bing.com/HPImageArchive.aspx?format=js&idx=0&n=1&mkt=en-US"
            data = fetch_json(url)
            set_cached(cache_key, data)

        if not data.get('images'):
            return self._error_result("No images in Bing response")

        image_data = data['images'][0]
        url_path = image_data.get('url', '')
        if not url_path:
            return self._error_result("No image URL in Bing response")
        img_url = "https://www.bing.com" + url_path

        img = load_image_from_url(img_url)
        tensor = pil_to_tensor(img)

        # Parse title from copyright string
        copyright_str = image_data.get('copyright', '')
        title = image_data.get('title', '')
        if not title and '(' in copyright_str:
            title = copyright_str.split('(')[0].strip()

        return (
            tensor,
            title or "Bing Daily",
            image_data.get('title', ''),
            f"https://www.bing.com{image_data.get('copyrightlink', '')}",
            copyright_str
        )

    def _fetch_wikimedia_potd(self):
        """Fetch Wikimedia Commons Picture of the Day via REST API."""
        today = time.strftime('%Y/%m/%d')
        cache_key = f"wikimedia_potd_{today.replace('/', '-')}"

        cached = get_cached(cache_key)
        if cached and cached.get('url'):
            img_url = cached.get('url')
            title = cached.get('title')
            desc = cached.get('description')
        else:
            # Use Wikipedia REST API for featured content (includes POTD)
            url = f"https://api.wikimedia.org/feed/v1/wikipedia/en/featured/{today}"
            try:
                data = fetch_json(url)
            except Exception as e:
                return self._error_result(f"Wikimedia API error: {e}")

            # Get image of the day from the response
            potd = data.get('image')
            if not potd:
                return self._error_result("No image in Wikimedia featured response")

            title = potd.get('title', 'Wikimedia POTD')
            # Clean up title (remove "File:" prefix)
            if title.startswith('File:'):
                title = title[5:]

            desc = potd.get('description', {}).get('text', '')

            # Get the image URL - try thumbnail first, then original
            thumbnail = potd.get('thumbnail')
            original = potd.get('image')
            if thumbnail and thumbnail.get('source'):
                img_url = thumbnail.get('source')
                # Try to get higher res by modifying thumb URL
                if '/thumb/' in img_url and 'px-' in img_url:
                    # Replace size with larger one (e.g., 320px -> 1280px)
                    img_url = re.sub(r'/(\d+)px-', '/1280px-', img_url)
            elif original and original.get('source'):
                img_url = original.get('source')
            else:
                return self._error_result("No image URL in Wikimedia POTD")

            set_cached(cache_key, {'url': img_url, 'title': title, 'description': desc})

        img = load_image_from_url(img_url)
        tensor = pil_to_tensor(img)

        return (
            tensor,
            title,
            desc[:500] if desc else "",
            img_url,
            "Wikimedia Commons (check individual license)"
        )

    def _fetch_lorem_picsum(self, width: int, height: int, seed: int):
        """Fetch random image from Lorem Picsum."""
        if seed > 0:
            url = f"https://picsum.photos/seed/{seed}/{width}/{height}"
            cache_key = f"picsum_{seed}_{width}_{height}"
        else:
            url = f"https://picsum.photos/{width}/{height}"
            cache_key = None  # Don't cache random

        # For seeded requests, we can cache
        if cache_key:
            cached = get_cached(cache_key)
            if cached:
                img_url = cached.get('url', url)
                info = cached
            else:
                img_url = url
                info = {'url': url, 'author': 'Unknown', 'id': str(seed)}
                set_cached(cache_key, info)
        else:
            img_url = url
            info = {'author': 'Unknown', 'id': 'random'}

        img = load_image_from_url(img_url)
        tensor = pil_to_tensor(img)

        return (
            tensor,
            f"Lorem Picsum #{info.get('id', 'random')}",
            f"Random photo from Unsplash via Lorem Picsum",
            "https://picsum.photos",
            f"Photo by {info.get('author', 'Unknown')} (Unsplash)"
        )

    def _fetch_unsplash(self, api_key: str):
        """Fetch random image from Unsplash."""
        if not api_key:
            return self._error_result("Unsplash requires an API key. Get one free at unsplash.com/developers")

        url = "https://api.unsplash.com/photos/random"
        headers = {"Authorization": f"Client-ID {api_key}"}

        data = fetch_json(url, headers)

        img_url = data.get('urls', {}).get('regular', '')
        if not img_url:
            return self._error_result("No image URL in Unsplash response")

        img = load_image_from_url(img_url)
        tensor = pil_to_tensor(img)

        user = data.get('user', {})

        return (
            tensor,
            data.get('description') or data.get('alt_description') or "Unsplash Photo",
            data.get('alt_description', ''),
            data.get('links', {}).get('html', img_url),
            f"Photo by {user.get('name', 'Unknown')} on Unsplash"
        )

    def _fetch_pexels(self, api_key: str):
        """Fetch curated image from Pexels."""
        if not api_key:
            return self._error_result("Pexels requires an API key. Get one free at pexels.com/api")

        # Use random page to get different images each time
        page = random.randint(1, 100)
        url = f"https://api.pexels.com/v1/curated?per_page=1&page={page}"
        headers = {"Authorization": api_key}

        data = fetch_json(url, headers)

        photos = data.get('photos', [])
        if not photos:
            return self._error_result("No photos in Pexels response")

        photo = photos[0]
        img_url = photo.get('src', {}).get('large2x') or photo.get('src', {}).get('original', '')

        if not img_url:
            return self._error_result("No image URL in Pexels response")

        img = load_image_from_url(img_url)
        tensor = pil_to_tensor(img)

        return (
            tensor,
            photo.get('alt') or "Pexels Photo",
            "",
            photo.get('url', img_url),
            f"Photo by {photo.get('photographer', 'Unknown')} on Pexels"
        )

    def _fetch_random_dog(self):
        """Fetch random dog image."""
        url = "https://dog.ceo/api/breeds/image/random"
        data = fetch_json(url)

        if data.get('status') != 'success':
            return self._error_result("Dog API request failed")

        img_url = data.get('message', '')
        if not img_url:
            return self._error_result("No image URL in Dog API response")

        # Extract breed from URL
        # URL format: https://images.dog.ceo/breeds/{breed}/image.jpg
        breed = "Unknown"
        try:
            if '/breeds/' in img_url:
                parts = img_url.split('/breeds/')
                if len(parts) > 1:
                    breed_parts = parts[1].split('/')
                    if breed_parts:
                        breed = breed_parts[0].replace('-', ' ').title()
        except (IndexError, AttributeError):
            breed = "Unknown"

        img = load_image_from_url(img_url)
        tensor = pil_to_tensor(img)

        return (
            tensor,
            f"Random Dog: {breed}",
            f"A lovely {breed}",
            "https://dog.ceo",
            "Dog CEO API (Stanford Dogs Dataset)"
        )

    def _fetch_random_cat(self):
        """Fetch random cat image."""
        url = "https://api.thecatapi.com/v1/images/search"
        data = fetch_json(url)

        if not data or not isinstance(data, list):
            return self._error_result("Cat API request failed")

        img_url = data[0].get('url', '')
        if not img_url:
            return self._error_result("No image URL in Cat API response")

        img = load_image_from_url(img_url)
        tensor = pil_to_tensor(img)

        return (
            tensor,
            "Random Cat",
            "A lovely cat",
            "https://thecatapi.com",
            "The Cat API"
        )
