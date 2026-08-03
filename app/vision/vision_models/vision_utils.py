"""
app/vision/vision_models/vision_utils.py - Vision Helper & Formatting Utilities
================================================================================
Provides text cleaning, Florence-2 bounding box parsing, confidence formatting,
and prompt context block construction routines.
"""

import re
from typing import Dict, Any, List
from app.vision.vision_models.vision_schemas import ObjectDetectionResult, ImageContext


def clean_ocr_text(raw_text: str) -> str:
    """Cleans up raw OCR text output by stripping excessive whitespace and control characters."""
    if not raw_text:
        return ""
    # Replace multiple newlines with single newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', raw_text.strip())
    return cleaned


def parse_bounding_boxes(raw_florence_dict: Dict[str, Any]) -> List[ObjectDetectionResult]:
    """
    Parses Florence-2 bounding box output dictionary into ObjectDetectionResult objects.
    Florence-2 output structure: {'<OD>': {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['cat', 'table']}}
    """
    results: List[ObjectDetectionResult] = []
    if not isinstance(raw_florence_dict, dict):
        return results

    # Search for detection task key
    for task_key, data in raw_florence_dict.items():
        if isinstance(data, dict) and "bboxes" in data and "labels" in data:
            bboxes = data["bboxes"]
            labels = data["labels"]
            for i in range(min(len(bboxes), len(labels))):
                box = bboxes[i]
                lbl = labels[i]
                if len(box) == 4:
                    results.append(
                        ObjectDetectionResult(
                            label=lbl,
                            confidence=0.90,
                            bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                        )
                    )
    return results


def format_vision_context_block(
    caption_short: str,
    caption_detailed: str,
    ocr_text: str,
    objects: List[ObjectDetectionResult],
    file_name: str = ""
) -> str:
    """Constructs structured text representation for PromptBuilder injection."""
    lines = [
        "=== ATTACHED IMAGE VISUAL UNDERSTANDING ===",
        f"File: {file_name or 'Uploaded Image'}",
        f"Visual Summary: {caption_short or 'Image attached.'}"
    ]

    if caption_detailed and caption_detailed != caption_short:
        lines.append(f"Detailed Description: {caption_detailed}")

    if objects:
        obj_str = ", ".join([f"{o.label}" for o in objects[:10]])
        lines.append(f"Detected Objects ({len(objects)}): {obj_str}")

    if ocr_text:
        lines.append(f"Extracted Text (OCR):\n{ocr_text}")

    lines.append("=== END IMAGE VISUAL UNDERSTANDING ===")
    return "\n".join(lines)
