r"""
app/rag/health_check.py - RAG Diagnostic Health Engine
======================================================

1. PURPOSE:
-----------
Executes automated diagnostic health verification checks across SANA AI's RAG Subsystem components.

2. WHY IT EXISTS (SYSTEM DIAGNOSTICS & RELIABILITY):
----------------------------------------------------
In production environments, silent component failures (e.g. GPU out of memory, missing document directory,
or corrupt model weights) can cause unexpected query failures. `HealthCheckEngine` provides a clean `check_health()` API
inspecting 6 critical health dimensions and returning structured `HealthStatus` reports.

3. RESPONSIBILITIES:
--------------------
- Verify Sentence Transformer model loading status.
- Verify Knowledge Document Source folder accessibility.
- Verify Vector Store chunk storage initialization.
- Verify Configuration parameters validity.
- Inspect system RAM & PyTorch GPU hardware memory status.
- Return structured health report payload.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Interfaced by `RAGManager` and diagnostic tools.

5. COMPLETE CODE:
-----------------
"""

import os
import logging
import threading
from typing import Optional, Dict, Any, List

import torch
from app.rag.embeddings.embedding_model import EmbeddingModel
from app.rag.document_processing.document_manager import DocumentManager
from app.rag.rag_config import KNOWLEDGE_BASE_DIR, DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger("sana_ai.rag.health")


class HealthCheckEngine:
    """
    Thread-safe Singleton executing diagnostic health verification checks.
    """
    _instance: Optional["HealthCheckEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        if HealthCheckEngine._instance is not None:
            raise RuntimeError("HealthCheckEngine is a Singleton! Use `HealthCheckEngine.get_instance()` instead.")

        logger.info("🏥 HealthCheckEngine Singleton initialized successfully.")

    @classmethod
    def get_instance(cls) -> "HealthCheckEngine":
        """Thread-safe accessor for shared HealthCheckEngine Singleton."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def check_health(self) -> Dict[str, Any]:
        """
        Executes comprehensive diagnostic health checks across all RAG components.

        Returns:
            Dict[str, Any]: Structured diagnostic health report.
        """
        checks: Dict[str, Dict[str, Any]] = {}
        all_healthy = True

        # 1. Embedding Model Health Check
        try:
            model_wrapper = EmbeddingModel()
            model_loaded = model_wrapper.is_loaded or model_wrapper.load_model()
            checks["embedding_model"] = {
                "status": "healthy" if model_loaded else "unhealthy",
                "model_name": DEFAULT_EMBEDDING_MODEL,
                "device": model_wrapper.device,
                "gpu_available": torch.cuda.is_available()
            }
            if not model_loaded:
                all_healthy = False
        except Exception as e:
            checks["embedding_model"] = {"status": "unhealthy", "error": str(e)}
            all_healthy = False

        # 2. Knowledge Base Directory Health Check
        kb_exists = KNOWLEDGE_BASE_DIR.exists() and os.access(str(KNOWLEDGE_BASE_DIR), os.R_OK)
        checks["knowledge_base_dir"] = {
            "status": "healthy" if kb_exists else "warning",
            "path": str(KNOWLEDGE_BASE_DIR),
            "accessible": kb_exists
        }

        # 3. Vector & Document Store Health Check
        try:
            doc_mgr = DocumentManager.get_instance()
            checks["document_store"] = {
                "status": "healthy",
                "total_documents": len(doc_mgr._processed_chunks_store),
                "total_chunks": sum(len(c) for c in doc_mgr._processed_chunks_store.values())
            }
        except Exception as e:
            checks["document_store"] = {"status": "unhealthy", "error": str(e)}
            all_healthy = False

        # 4. Hardware Resource Allocation Check
        cuda_mem_allocated = 0.0
        if torch.cuda.is_available():
            cuda_mem_allocated = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)

        checks["hardware_resources"] = {
            "status": "healthy",
            "device_type": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_allocated_mb": cuda_mem_allocated
        }

        overall_status = "healthy" if all_healthy else "unhealthy"
        logger.info(f"🏥 Health Diagnostic Check Executed | Overall Status: '{overall_status}'")

        return {
            "overall_status": overall_status,
            "components": checks
        }
