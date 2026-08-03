r"""
test_florence2_load.py

Standalone isolation test for Florence-2 model loading.
Run this directly (NOT through the bot) to check whether the
language-model weights (encoder/decoder embed_tokens, lm_head)
actually load from the checkpoint, or come back MISSING like
they did in the bot's log.

Usage:
    (.venv) PS D:\PersonalAI_Bot> python test_florence2_load.py
"""

import os
import sys
import time

# Ensure UTF-8 output encoding on Windows stdout if possible
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "microsoft/Florence-2-base"  # note: capital F-2, HF is case-sensitive on some mirrors
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# If you're loading from a local cached path instead of the hub,
# swap MODEL_ID for that path here, e.g.:
# MODEL_ID = r"D:\PersonalAI_Bot\models\florence2"


def patch_florence_config_if_needed():
    """Patches local HuggingFace cache for Florence-2 remote code compatibility if present."""
    try:
        from scripts.patch_florence2 import patch_florence2_files
        patch_florence2_files()
    except Exception:
        pass


def main():
    print(f"Device target: {DEVICE}")
    print(f"Loading from: {MODEL_ID}")
    print("-" * 60)

    # Patch remote code in cache if needed for newer transformers compatibility
    patch_florence_config_if_needed()

    t0 = time.time()
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"Processor loaded in {time.time() - t0:.2f}s")

    t1 = time.time()
    print("\nLoading model (watch for a LOAD REPORT / MISSING keys table below)...")
    torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    # HF Transformers 5.x supports `dtype=...` (preferred) or fallback to `torch_dtype=...`
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            dtype=torch_dtype,
        ).to(DEVICE).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(DEVICE).eval()

    print(f"Model loaded in {time.time() - t1:.2f}s")

    # --- Check 1: did any critical weights come back missing? ---
    print("\n" + "=" * 60)
    print("CHECK 1: Verifying no MISSING weights in state dict")
    print("=" * 60)
    missing_found = False
    for name, param in model.named_parameters():
        if torch.all(param == 0):
            # Zero-initialized params are a strong sign of a MISSING key
            # that got randomly (but degenerately) initialized.
            if "embed_tokens" in name or "lm_head" in name:
                print(f"  [WARN] Suspicious all-zero param: {name}")
                missing_found = True

    if not missing_found:
        print("  [OK] No obviously zero-initialized embed_tokens/lm_head params found.")
    else:
        print("  [FAIL] Found suspicious params — checkpoint is likely incomplete.")
        print("     Try deleting the local HF cache for this model and re-downloading:")
        print(r"     rmdir /s /q %USERPROFILE%\.cache\huggingface\hub\models--microsoft--Florence-2-base")

    # --- Check 2: run an actual inference and see if output is coherent ---
    print("\n" + "=" * 60)
    print("CHECK 2: Running a real inference (caption task)")
    print("=" * 60)

    # Use local test image if present, or create a synthetic placeholder image
    test_image_path = r"D:\PersonalAI_Bot\logs\temp_media\photo_7074001001_155.jpg"

    try:
        from PIL import Image
        if os.path.exists(test_image_path):
            image = Image.open(test_image_path).convert("RGB")
            # Florence-2 DaViT vision tower requires square inputs (h*w == num_tokens)
            if image.width != image.height:
                image = image.resize((768, 768))
            print(f"  Loaded test image from: {test_image_path}")
        else:
            print(f"  [WARN] Test image not found at {test_image_path}, creating fallback test image.")
            image = Image.new("RGB", (768, 768), color=(128, 128, 128))
    except Exception as e:
        print(f"  [FAIL] Failed to load or create test image: {e}")
        sys.exit(1)

    prompt = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch_dtype)

    t2 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            num_beams=3,
            do_sample=False,
        )
    gen_time = time.time() - t2

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text, task=prompt, image_size=(image.width, image.height)
    )

    print(f"  Inference time: {gen_time:.2f}s")
    print(f"  Raw output: {generated_text[:200]}")
    print(f"  Parsed output: {parsed}")

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    caption = parsed.get(prompt, "")
    if isinstance(caption, str) and (not caption or len(caption.strip()) < 5 or caption.strip().lower() in ("", "the", "a", "an")):
        print("  [FAIL] Output looks garbled/empty -> checkpoint's LM head is likely untrained/random.")
        print("     This confirms the MISSING weights issue from your bot log.")
    else:
        print("  [OK] Output looks coherent. Checkpoint loading is fine in isolation.")
        print("     -> The bug is more likely in how your bot's model_loader/VisionManager")
        print("        wires this model into the pipeline (e.g. reloading per-request,")
        print("        or a mismatched processor/model pairing).")


if __name__ == "__main__":
    main()
