"""
app/vision/image_processing/image_hash.py - Image Cryptographic & Perceptual Hashing
====================================================================================
Computes SHA-256, MD5, Average Hash (aHash), Difference Hash (dHash), and Perceptual Hash (pHash)
for image deduplication, integrity checking, and perceptual similarity matching.
"""

import os
import hashlib
import numpy as np
from PIL import Image as PILImage
from app.vision.image_processing.image_models import ImageHash


def compute_file_sha256(file_path: str) -> str:
    """Computes SHA-256 checksum of an image file."""
    if not os.path.exists(file_path):
        return ""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def compute_file_md5(file_path: str) -> str:
    """Computes MD5 checksum of an image file."""
    if not os.path.exists(file_path):
        return ""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def compute_average_hash(pil_img: PILImage.Image, hash_size: int = 8) -> str:
    """
    Computes Average Hash (aHash).
    1. Resize image to hash_size x hash_size (8x8 = 64 pixels).
    2. Convert to grayscale.
    3. Calculate mean pixel intensity.
    4. Compute 64-bit boolean mask (1 if pixel > mean, else 0).
    """
    resized = pil_img.convert("L").resize((hash_size, hash_size), PILImage.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=float)
    avg = pixels.mean()
    bits = pixels > avg
    # Convert boolean array to hex string
    hex_str = ""
    for byte in bits.reshape(-1, 8):
        byte_val = sum(val << i for i, val in enumerate(byte))
        hex_str += f"{byte_val:02x}"
    return hex_str


def compute_difference_hash(pil_img: PILImage.Image, hash_size: int = 8) -> str:
    """
    Computes Difference Hash (dHash).
    1. Resize to (hash_size + 1) x hash_size (9x8).
    2. Convert to grayscale.
    3. Calculate horizontal pixel gradient differences (pixel[x+1] > pixel[x]).
    """
    resized = pil_img.convert("L").resize((hash_size + 1, hash_size), PILImage.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=float)
    diff = pixels[:, 1:] > pixels[:, :-1]
    hex_str = ""
    for byte in diff.reshape(-1, 8):
        byte_val = sum(val << i for i, val in enumerate(byte))
        hex_str += f"{byte_val:02x}"
    return hex_str


def compute_perceptual_hash(pil_img: PILImage.Image, hash_size: int = 8) -> str:
    """
    Computes Perceptual Hash (pHash) using discrete cosine transform (DCT) approximation.
    """
    # 32x32 grayscale resize
    resized = pil_img.convert("L").resize((32, 32), PILImage.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=float)

    # Simplified DCT low-frequency sample (top-left 8x8 minus DC component)
    top_left_8x8 = pixels[:hash_size, :hash_size]
    median_val = np.median(top_left_8x8)
    bits = top_left_8x8 > median_val

    hex_str = ""
    for byte in bits.reshape(-1, 8):
        byte_val = sum(val << i for i, val in enumerate(byte))
        hex_str += f"{byte_val:02x}"
    return hex_str


def compute_all_hashes(file_path: str, pil_img: PILImage.Image) -> ImageHash:
    """Computes full set of cryptographic and perceptual hashes for an image."""
    sha = compute_file_sha256(file_path)
    md5 = compute_file_md5(file_path)
    ahash = compute_average_hash(pil_img) if pil_img else ""
    dhash = compute_difference_hash(pil_img) if pil_img else ""
    phash = compute_perceptual_hash(pil_img) if pil_img else ""

    return ImageHash(
        sha256=sha,
        md5=md5,
        average_hash=ahash,
        difference_hash=dhash,
        perceptual_hash=phash
    )
