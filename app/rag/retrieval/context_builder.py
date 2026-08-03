r"""
app/rag/retrieval/context_builder.py - RAG Knowledge Context Builder
======================================================================

1. PURPOSE:
-----------
Assembles formatted, source-attributed Knowledge Context blocks for ChatML System Prompt injection within strict token budgets.

2. WHY IT EXISTS (PROMPT TOKEN BUDGETING & SOURCE CITATIONS):
-------------------------------------------------------------
LLM context windows are finite resources shared between System Persona, RAG Knowledge, Memories, Emotion Guidance,
and Conversation History. `ContextBuilder` enforces a strict token budget (`MAX_CONTEXT_TOKENS = 1000`) and formats
passages with clear source citations (`[1] Source: filename.pdf (Relevance Match: 92%)`).

3. RESPONSIBILITIES:
--------------------
- Format candidate chunks into ChatML System Prompt context text.
- Enforce token budget limits (`MAX_CONTEXT_TOKENS = 1000`).
- Extract unique `KnowledgeSource` citation records.
- Return structured `KnowledgeContext` objects.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `MAX_CONTEXT_TOKENS`, `MAX_SOURCES_PER_PROMPT` from `retrieval_config.py`.
- Used by `retrieval_manager.py` and `app/llm/prompt_builder.py`.

5. COMPLETE CODE:
-----------------
"""

import logging
from typing import List, Dict

from app.rag.rag_models import KnowledgeSource
from app.rag.document_processing.document_types import DocumentType
from app.rag.retrieval.retrieval_config import MAX_CONTEXT_TOKENS, MAX_SOURCES_PER_PROMPT
from app.rag.retrieval.retrieval_models import RetrievedChunk, KnowledgeContext
from app.rag.retrieval.retrieval_utils import RetrievalUtils

logger = logging.getLogger("sana_ai.rag.retrieval.context_builder")


class ContextBuilder:
    """
    RAG Knowledge Context block construction engine enforcing System Prompt token budgets.
    """

    def build_context(self, query: str, chunks: List[RetrievedChunk]) -> KnowledgeContext:
        """
        Formats candidate chunks into a structured ChatML System Prompt knowledge context block.

        Args:
            query (str): User question statement.
            chunks (List[RetrievedChunk]): List of retrieved candidate chunks.

        Returns:
            KnowledgeContext: KnowledgeContext dataclass container.
        """
        if not chunks:
            return KnowledgeContext(
                query=query,
                formatted_text="",
                sources=[],
                chunks_count=0,
                estimated_tokens=0
            )

        lines = ["=== RELEVANT RETRIEVED KNOWLEDGE DOCUMENTS (RAG) ==="]
        included_chunks: List[RetrievedChunk] = []
        sources_map: Dict[str, KnowledgeSource] = {}
        total_tokens = RetrievalUtils.estimate_context_tokens(lines[0])

        for idx, chunk in enumerate(chunks[:MAX_SOURCES_PER_PROMPT], 1):
            doc_name = chunk.metadata.filename if chunk.metadata else f"Document_{chunk.doc_id}"
            score_pct = int(chunk.similarity_score * 100) if chunk.similarity_score <= 1.0 else int(chunk.similarity_score)
            
            header = f"[{idx}] Source: {doc_name} (Relevance Match: {score_pct}%)"
            body = chunk.content.strip()
            block_text = f"{header}\n{body}\n"
            block_tokens = RetrievalUtils.estimate_context_tokens(block_text)

            if total_tokens + block_tokens > MAX_CONTEXT_TOKENS:
                logger.info(f"✂️ ContextBuilder token budget reached ({total_tokens} + {block_tokens} > {MAX_CONTEXT_TOKENS} tokens). Stopping chunk injection.")
                break

            lines.append(header)
            lines.append(body)
            lines.append("")
            total_tokens += block_tokens
            included_chunks.append(chunk)

            # Record KnowledgeSource citation
            doc_id = chunk.doc_id
            if doc_id not in sources_map:
                file_type = chunk.metadata.file_type if chunk.metadata else DocumentType.UNKNOWN.value
                sources_map[doc_id] = KnowledgeSource(
                    source_id=doc_id,
                    name=doc_name,
                    type=file_type,
                    total_chunks=1
                )
            else:
                sources_map[doc_id].total_chunks += 1

        if included_chunks:
            lines.append("Instructions: Ground your answer primarily in the retrieved knowledge passages above. Cite source filenames where helpful.")

        formatted_text = "\n".join(lines).strip()
        final_tokens = RetrievalUtils.estimate_context_tokens(formatted_text)

        logger.debug(
            f"📝 Built Knowledge Context | Chunks Included: {len(included_chunks)}/{len(chunks)} | "
            f"Sources Cited: {len(sources_map)} | Tokens Used: ~{final_tokens}/{MAX_CONTEXT_TOKENS}"
        )

        return KnowledgeContext(
            query=query,
            formatted_text=formatted_text,
            sources=list(sources_map.values()),
            chunks_count=len(included_chunks),
            estimated_tokens=final_tokens
        )
