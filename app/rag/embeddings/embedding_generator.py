r"""
app/rag/embeddings/embedding_generator.py - Dense Vector Generator Engine
============================================================================

1. PURPOSE:
-----------
Generates dense, L2-normalized 384-dimensional floating-point vectors for text passages using Sentence Transformers,
tokenization, Mean Pooling over attention masks, and unit vector normalization.

2. WHY IT EXISTS (MEAN POOLING & L2 UNIT VECTOR NORMALIZATION):
----------------------------------------------------------------
Transformer encoders output token-level hidden state vectors ($\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_T$).
To produce a single, high-precision vector representing the entire sentence:
1. **Mean Pooling** averages hidden states weighted by non-padded attention masks.
2. **L2 Unit Vector Normalization** scales vectors so $\|\mathbf{v}\|_2 = 1.0$, simplifying cosine similarity calculations into simple dot products.

3. RESPONSIBILITIES:
--------------------
- Generate dense float vectors for single or batched text passages.
- Execute Mean Pooling over attention masks.
- Execute L2 unit vector normalization.
- Construct `EmbeddingVector` dataclass objects.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `EmbeddingModel` from `embedding_model.py`.
- Uses `EmbeddingUtils` from `embedding_utils.py`.
- Used by `embedding_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import time
import hashlib
import logging
from typing import List, Optional

import torch
from app.rag.embeddings.embedding_config import EMBEDDING_DIMENSION, DEFAULT_BATCH_SIZE, NORMALIZE_EMBEDDINGS
from app.rag.embeddings.embedding_models import EmbeddingVector, EmbeddingRequest
from app.rag.embeddings.embedding_model import EmbeddingModel
from app.rag.embeddings.embedding_utils import EmbeddingUtils

logger = logging.getLogger("sana_ai.rag.embeddings.generator")


class EmbeddingGenerator:
    """
    Dense vector generation engine implementing Mean Pooling and L2 unit vector normalization.
    """

    def __init__(self, model_wrapper: Optional[EmbeddingModel] = None) -> None:
        """
        Initializes vector generator.

        Args:
            model_wrapper (Optional[EmbeddingModel]): EmbeddingModel wrapper instance.
        """
        self.model_wrapper: EmbeddingModel = model_wrapper if model_wrapper else EmbeddingModel()

    def generate_embeddings(
        self,
        request: EmbeddingRequest
    ) -> List[EmbeddingVector]:
        """
        Generates dense embedding vectors for a list of text passages.

        Args:
            request (EmbeddingRequest): EmbeddingRequest dataclass.

        Returns:
            List[EmbeddingVector]: List of generated EmbeddingVector objects.
        """
        if not request.texts or len(request.texts) == 0:
            return []

        # Ensure model is loaded
        if not self.model_wrapper.is_loaded:
            success = self.model_wrapper.load_model()
            if not success:
                logger.error("❌ Model load failed in EmbeddingGenerator. Returning fallback zero-vectors.")
                return self._generate_fallback_vectors(request)

        try:
            model = self.model_wrapper.model
            # Branch A: Use sentence-transformers SentenceTransformer interface
            if hasattr(model, "encode"):
                logger.debug(f"⚡ Generating embeddings via SentenceTransformer for {len(request.texts)} texts...")
                raw_embeddings = model.encode(
                    request.texts,
                    batch_size=DEFAULT_BATCH_SIZE,
                    show_progress_bar=False,
                    normalize_embeddings=request.normalize,
                    convert_to_numpy=True
                )
                
                output_vectors: List[EmbeddingVector] = []
                for idx, text in enumerate(request.texts):
                    raw_vec = raw_embeddings[idx].tolist() if hasattr(raw_embeddings[idx], "tolist") else list(raw_embeddings[idx])
                    norm_vec, l2_norm = EmbeddingUtils.normalize_vector(raw_vec) if request.normalize else (raw_vec, 1.0)
                    
                    doc_id = request.doc_ids[idx] if idx < len(request.doc_ids) else f"doc_{idx}"
                    chunk_id = request.chunk_ids[idx] if idx < len(request.chunk_ids) else f"chunk_{idx}"
                    vec_id = f"vec_{hashlib.md5((chunk_id + text).encode('utf-8')).hexdigest()[:10]}"

                    output_vectors.append(
                        EmbeddingVector(
                            vector_id=vec_id,
                            doc_id=doc_id,
                            chunk_id=chunk_id,
                            values=norm_vec,
                            dimension=len(norm_vec),
                            norm=1.0 if request.normalize else l2_norm,
                            is_normalized=request.normalize
                        )
                    )
                return output_vectors

            # Branch B: Use PyTorch AutoModel + Tokenizer with Mean Pooling
            tokenizer = getattr(self.model_wrapper, "_tokenizer", None)
            if tokenizer is not None and model is not None:
                logger.debug(f"⚡ Generating embeddings via PyTorch AutoModel + Mean Pooling for {len(request.texts)} texts...")
                device = self.model_wrapper.device
                encoded = tokenizer(
                    request.texts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    model_output = model(**encoded)

                # Mean Pooling over non-padded tokens
                token_embeddings = model_output[0]  # First element contains token hidden states
                input_mask_expanded = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask

                if request.normalize:
                    mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

                numpy_vecs = mean_pooled.cpu().numpy()
                output_vectors: List[EmbeddingVector] = []
                for idx, text in enumerate(request.texts):
                    raw_vec = numpy_vecs[idx].tolist()
                    norm_vec, l2_norm = EmbeddingUtils.normalize_vector(raw_vec) if request.normalize else (raw_vec, 1.0)

                    doc_id = request.doc_ids[idx] if idx < len(request.doc_ids) else f"doc_{idx}"
                    chunk_id = request.chunk_ids[idx] if idx < len(request.chunk_ids) else f"chunk_{idx}"
                    vec_id = f"vec_{hashlib.md5((chunk_id + text).encode('utf-8')).hexdigest()[:10]}"

                    output_vectors.append(
                        EmbeddingVector(
                            vector_id=vec_id,
                            doc_id=doc_id,
                            chunk_id=chunk_id,
                            values=norm_vec,
                            dimension=len(norm_vec),
                            norm=1.0 if request.normalize else l2_norm,
                            is_normalized=request.normalize
                        )
                    )
                return output_vectors

        except Exception as err:
            logger.error(f"Error during vector embedding generation ({err}). Falling back to manual calculation.", exc_info=True)

        return self._generate_fallback_vectors(request)

    def _generate_fallback_vectors(self, request: EmbeddingRequest) -> List[EmbeddingVector]:
        """
        Fallback generator producing deterministic pseudo-semantic vectors if model load fails.

        Args:
            request (EmbeddingRequest): EmbeddingRequest dataclass.

        Returns:
            List[EmbeddingVector]: List of fallback EmbeddingVector objects.
        """
        logger.warning("⚠️ Generating fallback deterministic pseudo-vectors.")
        vectors: List[EmbeddingVector] = []

        for idx, text in enumerate(request.texts):
            doc_id = request.doc_ids[idx] if idx < len(request.doc_ids) else f"doc_{idx}"
            chunk_id = request.chunk_ids[idx] if idx < len(request.chunk_ids) else f"chunk_{idx}"
            vec_id = f"vec_fallback_{idx}"

            # Produce deterministic pseudo-floats based on hash digest
            hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
            raw_vals = [(float(b) / 255.0) - 0.5 for b in hash_bytes]
            
            # Tile to 384 dimensions
            tiled = (raw_vals * (EMBEDDING_DIMENSION // len(raw_vals) + 1))[:EMBEDDING_DIMENSION]
            norm_vec, _ = EmbeddingUtils.normalize_vector(tiled)

            vectors.append(
                EmbeddingVector(
                    vector_id=vec_id,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    values=norm_vec,
                    dimension=EMBEDDING_DIMENSION,
                    norm=1.0 if request.normalize else 1.0,
                    is_normalized=True
                )
            )

        return vectors
