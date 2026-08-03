"""
app/rag/embeddings/embedding_utils.py - Embedding Math & Vector Utility Engine
================================================================================

1. PURPOSE:
-----------
Provides mathematical vector operations (L2 unit vector normalization, Cosine similarity calculation, L2 distance,
dimension validation, and memory estimation) for the Embedding Subsystem.

2. WHY IT EXISTS:
-----------------
Decouples linear algebra calculations from model loaders and generators. Ensures vectors are correctly L2-normalized
and provides pure Python & NumPy fallback functions for similarity calculation.

3. RESPONSIBILITIES:
--------------------
- Perform L2 unit vector normalization (`normalize_vector`).
- Compute Cosine Similarity between two dense float vectors (`cosine_similarity`).
- Compute Euclidean L2 distance (`l2_distance`).
- Validate vector dimensions (`validate_dimension`).
- Estimate RAM memory footprint (`estimate_memory_bytes`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `embedding_validator.py`, `embedding_generator.py`, `embedding_manager.py`, and `tests/test_embeddings.py`.

5. COMPLETE CODE:
-----------------
"""

import math
import logging
from typing import List, Tuple

logger = logging.getLogger("sana_ai.rag.embeddings.utils")


class EmbeddingUtils:
    """
    Mathematical vector helper utilities for normalization and similarity calculation.
    """

    @staticmethod
    def normalize_vector(vector: List[float]) -> Tuple[List[float], float]:
        """
        Normalizes a dense vector to L2 unit length (norm = 1.0).

        Args:
            vector (List[float]): Raw floating-point vector.

        Returns:
            Tuple[List[float], float]: (normalized_vector, l2_norm).
        """
        if not vector:
            return [], 0.0

        sq_sum = sum(v * v for v in vector)
        norm = math.sqrt(sq_sum)

        if norm == 0.0 or math.isnan(norm) or math.isinf(norm):
            return vector, norm

        normalized = [v / norm for v in vector]
        return normalized, norm

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """
        Calculates Cosine Similarity between two dense float vectors.

        Args:
            vec_a (List[float]): Vector A.
            vec_b (List[float]): Vector B.

        Returns:
            float: Cosine similarity score (-1.0 to 1.0).
        """
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        sim = dot_product / (norm_a * norm_b)
        return max(-1.0, min(1.0, float(sim)))

    @staticmethod
    def l2_distance(vec_a: List[float], vec_b: List[float]) -> float:
        """
        Calculates Euclidean (L2) distance between two dense float vectors.

        Args:
            vec_a (List[float]): Vector A.
            vec_b (List[float]): Vector B.

        Returns:
            float: Euclidean distance value (>= 0.0).
        """
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return float("inf")

        sq_diff = sum((a - b) ** 2 for a, b in zip(vec_a, vec_b))
        return math.sqrt(sq_diff)

    @staticmethod
    def validate_dimension(vector: List[float], expected_dim: int) -> bool:
        """
        Verifies vector dimension matches expected dimension count.

        Args:
            vector (List[float]): Target vector.
            expected_dim (int): Expected dimension (e.g. 384).

        Returns:
            bool: True if length matches expected_dim.
        """
        if not vector:
            return False
        return len(vector) == expected_dim

    @staticmethod
    def estimate_memory_bytes(num_vectors: int, dim: int) -> int:
        """
        Estimates memory footprint in bytes for N float32 vectors.

        Args:
            num_vectors (int): Count of vectors.
            dim (int): Vector dimension count.

        Returns:
            int: Memory footprint in bytes.
        """
        return num_vectors * dim * 4
