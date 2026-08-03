"""
app/rag/embeddings/embedding_validator.py - Dense Vector Validator
====================================================================

1. PURPOSE:
-----------
Validates dense vector floating-point arrays prior to vector database storage or mathematical distance search.

2. WHY IT EXISTS (DEFENSIVE NUMERICAL VALIDATION):
--------------------------------------------------
Corrupted transformer forward passes, zero-mask inputs, or model loading bugs can produce zero-magnitude vectors,
`NaN` values, or `Inf` floats. Storing invalid vectors in FAISS index causes hard C++ segfaults or search failures.
`EmbeddingValidator` acts as a numerical gatekeeper.

3. RESPONSIBILITIES:
--------------------
- Verify vector dimension matches `EMBEDDING_DIMENSION` (384).
- Guard against empty vectors.
- Detect `NaN` (Not a Number) floating-point values.
- Detect `Inf` (Infinite) floating-point values.
- Detect zero-magnitude vectors (norm = 0.0).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `EMBEDDING_DIMENSION` from `embedding_config.py`.
- Uses `VectorStatus`, `ValidationResult` from `embedding_types.py` & `embedding_models.py`.
- Used by `embedding_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import math
import logging
from typing import List

from app.rag.embeddings.embedding_config import EMBEDDING_DIMENSION
from app.rag.embeddings.embedding_types import VectorStatus
from app.rag.embeddings.embedding_models import ValidationResult

logger = logging.getLogger("sana_ai.rag.embeddings.validator")


class EmbeddingValidator:
    """
    Validation engine verifying numerical vector properties and dimensions.
    """

    def validate_vector(
        self,
        vector: List[float],
        expected_dim: int = EMBEDDING_DIMENSION
    ) -> ValidationResult:
        """
        Performs thorough validation checks on a dense float vector.

        Args:
            vector (List[float]): Dense vector floats.
            expected_dim (int): Target expected dimension.

        Returns:
            ValidationResult: ValidationResult dataclass instance.
        """
        # 1. Empty check
        if not vector or len(vector) == 0:
            return ValidationResult(
                status=VectorStatus.EMPTY.value,
                is_valid=False,
                reason="Vector is empty (0 dimensions)."
            )

        # 2. Dimension check
        if len(vector) != expected_dim:
            return ValidationResult(
                status=VectorStatus.INVALID_DIMENSION.value,
                is_valid=False,
                reason=f"Dimension mismatch: expected {expected_dim}, got {len(vector)}."
            )

        # 3. NaN and Inf check
        sq_sum = 0.0
        for idx, val in enumerate(vector):
            if math.isnan(val):
                return ValidationResult(
                    status=VectorStatus.NAN_DETECTED.value,
                    is_valid=False,
                    reason=f"NaN value detected at vector index {idx}."
                )
            if math.isinf(val):
                return ValidationResult(
                    status=VectorStatus.INF_DETECTED.value,
                    is_valid=False,
                    reason=f"Infinite (Inf) value detected at vector index {idx}."
                )
            sq_sum += val * val

        # 4. Zero-magnitude vector check
        if sq_sum == 0.0:
            return ValidationResult(
                status=VectorStatus.ZERO_VECTOR.value,
                is_valid=False,
                reason="Vector is all zeros (zero magnitude norm)."
            )

        return ValidationResult(
            status=VectorStatus.VALID.value,
            is_valid=True,
            reason="Vector passed mathematical validation checks."
        )
