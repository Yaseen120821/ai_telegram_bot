"""
app/rag/vector_store/vector_utils.py - Vector Store Utility Helpers
=====================================================================
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger("sana_ai.rag.vector_store.utils")


class VectorStoreUtils:
    """
    Utility helpers for ID mapping and vector store serialization.
    """

    @staticmethod
    def map_id_to_index(record_id: str, index_pos: int) -> Dict[str, Any]:
        """Maps a string record ID to an internal integer index position."""
        return {"id": record_id, "position": index_pos}
