"""
app/vision/multimodal_utils.py - Multimodal Context Formatting & Token Utilities
=================================================================================
Provides multi-image context merging, token budget estimation, and context truncation routines.
"""

import os
import logging
from typing import List
from app.vision.multimodal_models import ImageContextData

logger = logging.getLogger("sana_ai.vision.multimodal.utils")


def merge_image_contexts(images: List[ImageContextData], max_chars: int = 2500) -> str:
    """
    Merges single or multiple ImageContextData objects into a unified,
    structured ChatML prompt context block.
    """
    if not images:
        return ""

    lines = ["=== ATTACHED MULTIMODAL IMAGE CONTEXT ==="]
    lines.append(f"Total Attached Images: {len(images)}")

    for idx, img in enumerate(images, start=1):
        file_basename = os.path.basename(img.file_path)
        lines.append(f"\n--- Image Asset #{idx}: {file_basename} [{img.role.value.upper()}] ---")
        lines.append(f"Category: {img.analysis.category.value.upper()}")
        
        if img.vision.short_caption:
            lines.append(f"Visual Scene: {img.vision.short_caption}")
        if img.analysis.domain_summary:
            lines.append(f"Domain Insights: {img.analysis.domain_summary}")
        if img.ocr.raw_text:
            ocr_snippet = img.ocr.raw_text[:400] + ("..." if len(img.ocr.raw_text) > 400 else "")
            lines.append(f"Extracted Text (OCR):\n{ocr_snippet}")

    lines.append("=== END ATTACHED IMAGE CONTEXT ===")
    merged_text = "\n".join(lines)

    if len(merged_text) > max_chars:
        merged_text = merged_text[:max_chars] + "\n...[Image Context Truncated to Fit Token Budget]...\n=== END ATTACHED IMAGE CONTEXT ==="

    return merged_text


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator (approx. 4 characters per token)."""
    if not text:
        return 0
    return len(text) // 4
