"""
app/rag/document_processing/document_cleaner.py - Document Text Sanitizer & Normalizer
========================================================================================

1. PURPOSE:
-----------
Sanitizes and normalizes extracted raw document text prior to chunking and embedding.

2. WHY IT EXISTS (DATA QUALITY FOR VECTOR EMBEDDINGS):
------------------------------------------------------
Raw extracted document text contains null bytes, control codes, non-standard Unicode glyphs, irregular line breaks,
and trailing spaces. Feeding messy text into sentence transformer embedding models reduces similarity vector precision.
`DocumentCleaner` produces clean, standardized text while preserving structural paragraph breaks.

3. RESPONSIBILITIES:
--------------------
- Perform Unicode NFKC normalization.
- Strip null bytes (`\x00`) and non-printable control characters.
- Collapse excessive blank lines into clean paragraph breaks (`\n\n`).
- Trim trailing line spaces.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `CleaningResult` from `document_models.py`.
- Called by `document_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import re
import logging
import unicodedata
from app.rag.document_processing.document_models import CleaningResult

logger = logging.getLogger("sana_ai.rag.document.cleaner")


class DocumentCleaner:
    """
    Document text sanitization and Unicode normalization engine.
    """

    def clean_text(self, raw_text: str) -> CleaningResult:
        """
        Cleans and normalizes raw text string.

        Args:
            raw_text (str): Raw extracted document text string.

        Returns:
            CleaningResult: CleaningResult object containing original length, cleaned length, and text.
        """
        if not raw_text:
            return CleaningResult(original_length=0, cleaned_length=0, text="")

        orig_len = len(raw_text)

        try:
            # 1. Unicode NFKC Normalization
            normalized = unicodedata.normalize("NFKC", raw_text)

            # 2. Strip null bytes and control codes (preserving newlines and tabs)
            cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", normalized)

            # 3. Replace carriage returns (\r\n -> \n, \r -> \n)
            cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

            # 4. Collapse multiple horizontal spaces into single space per line
            lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.split("\n")]

            # 5. Collapse 3+ consecutive newlines into double newlines (paragraph boundary)
            rejoined = "\n".join(lines)
            cleaned_text = re.sub(r"\n{3,}", "\n\n", rejoined).strip()

            cleaned_len = len(cleaned_text)
            logger.debug(f"🧹 Cleaned Document Text | Original: {orig_len} chars ──► Cleaned: {cleaned_len} chars")

            return CleaningResult(
                original_length=orig_len,
                cleaned_length=cleaned_len,
                text=cleaned_text
            )
        except Exception as err:
            logger.warning(f"Error during document text cleaning ({err}). Returning basic trimmed text.")
            basic_text = raw_text.strip()
            return CleaningResult(
                original_length=orig_len,
                cleaned_length=len(basic_text),
                text=basic_text
            )
