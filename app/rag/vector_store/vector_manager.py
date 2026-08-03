"""
app/rag/vector_store/vector_manager.py - Central Vector Store Manager Facade
=============================================================================
"""

import logging
import threading
from typing import Optional, List, Dict, Any

from app.rag.vector_store.faiss_manager import FAISSIndexManager
from app.rag.vector_store.vector_models import VectorIndexMetadata, IndexSearchRequest, IndexSearchResult

logger = logging.getLogger("sana_ai.rag.vector_store.manager")


class VectorManager:
    """
    Thread-safe Singleton facade orchestrating vector storage and FAISS indices.
    """
    _instance: Optional["VectorManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        if VectorManager._instance is not None:
            raise RuntimeError("VectorManager is a Singleton! Use VectorManager.get_instance().")
        self.faiss_mgr: FAISSIndexManager = FAISSIndexManager()
        logger.info("🏛️ VectorManager Singleton Architecture initialized successfully.")

    @classmethod
    def get_instance(cls) -> "VectorManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def add_vectors(self, vectors: List[List[float]], metadata_list: List[Dict[str, Any]]) -> bool:
        return self.faiss_mgr.add_vectors(vectors, metadata_list)

    def search_similar(self, query_vector: List[float], top_k: int = 5) -> IndexSearchResult:
        req = IndexSearchRequest(query_vector=query_vector, top_k=top_k)
        return self.faiss_mgr.search(req)

    def get_index_metadata(self) -> VectorIndexMetadata:
        return self.faiss_mgr.get_metadata()
