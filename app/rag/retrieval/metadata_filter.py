"""
app/rag/retrieval/metadata_filter.py - Retrieval Metadata Filter Engine
==========================================================================

1. PURPOSE:
-----------
Applies structured metadata filtering rules (e.g., file type, folder path, file extension) to retrieved candidate chunks.

2. WHY IT EXISTS (PRECISION REFINEMENT):
----------------------------------------
When a user requests information specifically from source code files or PDF reports, `MetadataFilter` discards
candidates that do not match the target document attributes, boosting search precision.

3. RESPONSIBILITIES:
--------------------
- Filter candidate chunks based on metadata rules (`file_type`, `extension`, `filename`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `retriever.py`.

5. COMPLETE CODE:
-----------------
"""

import logging
from typing import List, Dict, Any

from app.rag.retrieval.retrieval_models import RetrievedChunk

logger = logging.getLogger("sana_ai.rag.retrieval.metadata_filter")


class MetadataFilter:
    """
    Metadata filtering engine evaluating candidate chunks against filter rules.
    """

    def apply_filters(
        self,
        chunks: List[RetrievedChunk],
        filters: Dict[str, Any]
    ) -> List[RetrievedChunk]:
        """
        Filters candidate chunks based on key-value metadata conditions.

        Args:
            chunks (List[RetrievedChunk]): Input candidate chunks list.
            filters (Dict[str, Any]): Filter attribute rules dictionary.

        Returns:
            List[RetrievedChunk]: Filtered candidate chunks list.
        """
        if not chunks or not filters:
            return chunks

        filtered: List[RetrievedChunk] = []
        for chunk in chunks:
            if not chunk.metadata:
                filtered.append(chunk)
                continue

            match = True
            for key, val in filters.items():
                if key == "file_type" and chunk.metadata.file_type != str(val):
                    match = False
                    break
                elif key == "filename" and str(val).lower() not in chunk.metadata.filename.lower():
                    match = False
                    break
                elif key == "min_file_size" and chunk.metadata.file_size_bytes < int(val):
                    match = False
                    break

            if match:
                filtered.append(chunk)

        logger.debug(f"🛡️ MetadataFilter: {len(chunks)} candidate chunks ──► {len(filtered)} matched filters")
        return filtered
