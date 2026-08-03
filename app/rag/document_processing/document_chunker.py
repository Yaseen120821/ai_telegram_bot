"""
app/rag/document_processing/document_chunker.py - Recursive Document Chunking Engine
========================================================================================

1. PURPOSE:
-----------
Splits cleaned document text into small, self-contained, overlapping text passages (`Chunk` objects) optimized for LLM context windows.

2. WHY IT EXISTS (RECURSIVE CHUNKING WITH OVERLAP):
---------------------------------------------------
Splitting text at arbitrary byte positions cuts sentences and code blocks in half. `DocumentChunker` implements
**Recursive Character Chunking** with configurable overlap: it attempts to split text on double newlines (`\n\n`),
then single newlines (`\n`), then period boundaries (`. `), then spaces (` `), preserving semantic continuity across chunk edges.

3. RESPONSIBILITIES:
--------------------
- Split text using separator hierarchy (`["\n\n", "\n", ". ", " ", ""]`).
- Enforce target chunk size (`DEFAULT_CHUNK_SIZE = 500`) and overlap window (`DEFAULT_CHUNK_OVERLAP = 50`).
- Compute start/end character offsets, chunk index, estimated tokens, and MD5 content hashes.
- Return structured `ChunkCollection` dataclasses.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `DocumentMetadata`, `Chunk`, `ChunkCollection` from `document_models.py`.
- Uses `DEFAULT_CHUNK_SIZE`, `DEFAULT_CHUNK_OVERLAP` from `document_config.py`.
- Called by `document_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import hashlib
import logging
from typing import List, Optional

from app.rag.document_processing.document_config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, MIN_CHUNK_SIZE
from app.rag.document_processing.document_models import DocumentMetadata, Chunk, ChunkCollection
from app.rag.document_processing.document_utils import DocumentUtils

logger = logging.getLogger("sana_ai.rag.document.chunker")


class DocumentChunker:
    """
    Recursive character text chunking engine with configurable overlap.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ) -> None:
        """
        Initializes chunking parameters.

        Args:
            chunk_size (int): Target maximum characters per chunk.
            chunk_overlap (int): Overlap characters copied between consecutive chunks.
        """
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.separators: List[str] = ["\n\n", "\n", ". ", " ", ""]

    def create_chunks(
        self,
        doc_id: str,
        cleaned_text: str,
        metadata: Optional[DocumentMetadata] = None
    ) -> ChunkCollection:
        """
        Splits cleaned text into an overlapping list of Chunk objects.

        Args:
            doc_id (str): Parent document identifier string.
            cleaned_text (str): Cleaned document text.
            metadata (Optional[DocumentMetadata]): Parent document metadata.

        Returns:
            ChunkCollection: ChunkCollection dataclass container.
        """
        if not cleaned_text or len(cleaned_text.strip()) == 0:
            return ChunkCollection(doc_id=doc_id, chunks=[], total_chunks=0)

        raw_passages = self._recursive_split(cleaned_text, self.chunk_size, self.chunk_overlap)
        chunks: List[Chunk] = []

        start_offset = 0
        for idx, passage in enumerate(raw_passages):
            if len(passage.strip()) < MIN_CHUNK_SIZE and idx > 0 and idx < len(raw_passages) - 1:
                continue

            # Calculate character offsets in cleaned_text
            pos = cleaned_text.find(passage, start_offset)
            if pos != -1:
                start_char = pos
                start_offset = pos + min(10, len(passage))
            else:
                start_char = start_offset

            end_char = start_char + len(passage)
            chunk_id = DocumentUtils.generate_chunk_id(doc_id, idx)
            content_hash = hashlib.md5(passage.encode("utf-8")).hexdigest()
            est_tokens = DocumentUtils.estimate_tokens(passage)

            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=passage,
                chunk_index=idx,
                start_char=start_char,
                end_char=end_char,
                estimated_tokens=est_tokens,
                hash_digest=content_hash,
                metadata=metadata
            )
            chunks.append(chunk)

        logger.debug(
            f"✂️ Generated {len(chunks)} Overlapping Chunks for Doc ID: '{doc_id}' "
            f"[Chunk Size: {self.chunk_size} | Overlap: {self.chunk_overlap}]"
        )
        return ChunkCollection(doc_id=doc_id, chunks=chunks, total_chunks=len(chunks))

    def _recursive_split(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Internal recursive splitting implementation.

        Args:
            text (str): Input text string.
            chunk_size (int): Max chunk size.
            chunk_overlap (int): Overlap size.

        Returns:
            List[str]: List of text passage strings.
        """
        if len(text) <= chunk_size:
            return [text]

        # 1. Find highest-order separator that exists in text
        separator = ""
        for sep in self.separators:
            if sep == "" or sep in text:
                separator = sep
                break

        # 2. Split text by separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # 3. Combine splits into chunks up to chunk_size with overlap
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for s in splits:
            s_len = len(s) + len(separator)
            if current_len + s_len > chunk_size and current_chunk:
                joined = separator.join(current_chunk)
                chunks.append(joined)

                # Calculate overlap window
                overlap_text = joined[-chunk_overlap:] if chunk_overlap > 0 else ""
                current_chunk = [overlap_text, s] if overlap_text else [s]
                current_len = sum(len(x) + len(separator) for x in current_chunk)
            else:
                current_chunk.append(s)
                current_len += s_len

        if current_chunk:
            joined = separator.join(current_chunk)
            if joined.strip():
                chunks.append(joined)

        return chunks
