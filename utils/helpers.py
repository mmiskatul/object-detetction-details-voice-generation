# utils/helpers.py
import hashlib
import os

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_image_hash(image_bytes: bytes) -> str:
    """Return MD5 hash of image bytes"""
    return hashlib.md5(image_bytes).hexdigest()

def get_cached_audio_path(image_hash: str) -> str:
    """Return the cached audio file path"""
    return os.path.join(CACHE_DIR, f"{image_hash}.wav")
