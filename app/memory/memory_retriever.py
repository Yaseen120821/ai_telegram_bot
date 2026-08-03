r"""
app/memory/memory_retriever.py - Long-Term Memory Search, Ranking & Retrieval Engine
======================================================================================

1. PURPOSE:
-----------
Retrieves, filters, ranks, and formats active long-term memories stored in SQLite for a specified Telegram `user_id`.
Integrates an in-memory cache to minimize disk I/O and calculates context relevance scores for LLM prompt injection.

2. WHY IT EXISTS (CONTEXT SIZE MANAGEMENT & PERFORMANCE):
---------------------------------------------------------
Loading thousands of database records into an LLM prompt overflows context limits and increases latency.
`MemoryRetriever` scores candidate memories using multi-factor relevance ranking and returns top facts bounded by a limit.

3. RESPONSIBILITIES:
--------------------
- Maintain a thread-safe in-memory cache to eliminate redundant SQLite disk reads.
- Rank candidate memories using a multi-factor score: Keyword Match + Importance + Confidence + Access Count + Recency.
- Format recalled memories into a clean text block for System Prompt injection with explicit user profile identity rules.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `MemoryStore` from `app/memory/memory_store.py`.
- Called by `app/memory/memory_manager.py` and `app/llm/prompt_builder.py`.

5. COMPLETE CODE:
-----------------
"""

import time
import logging
from typing import List, Optional, Dict, Tuple
from app.memory.memory_store import MemoryStore
from app.memory.memory_models import MemoryItem, SearchResult

logger = logging.getLogger("sana_ai.memory.retriever")

DEFAULT_MEMORY_LIMIT = 10
CACHE_TTL_SECONDS = 300.0  # 5-minute in-memory cache TTL


class MemoryRetriever:
    """
    Retrieval, caching, and relevance ranking engine for persistent MemoryItem records.
    """

    def __init__(self, memory_store: Optional[MemoryStore] = None) -> None:
        """
        Initializes MemoryRetriever with an active MemoryStore and in-memory TTL cache.

        Args:
            memory_store (Optional[MemoryStore]): MemoryStore instance.
        """
        self.memory_store: MemoryStore = memory_store or MemoryStore()
        # In-memory cache mapping: user_id -> (timestamp, List[MemoryItem])
        self._cache: Dict[str, Tuple[float, List[MemoryItem]]] = {}

    def invalidate_cache(self, user_id: Optional[str] = None) -> None:
        """
        Invalidates the in-memory cache for a specific user or clears the entire cache.

        Args:
            user_id (Optional[str]): Telegram User ID string. If None, clears entire cache.
        """
        if user_id:
            u_id = str(user_id)
            if u_id in self._cache:
                del self._cache[u_id]
                logger.debug(f"⚡ In-Memory Cache Invalidated for User ID: {u_id}")
        else:
            self._cache.clear()
            logger.debug("⚡ In-Memory Cache Cleared completely.")

    def get_user_memories_cached(self, user_id: str) -> List[MemoryItem]:
        """
        Fetches active memories for user_id, utilizing the in-memory TTL cache on Cache Hit.

        Args:
            user_id (str): Telegram User ID string.

        Returns:
            List[MemoryItem]: List of active MemoryItem records.
        """
        u_id = str(user_id)
        now = time.time()

        # 1. Check Cache Hit
        if u_id in self._cache:
            cached_time, cached_items = self._cache[u_id]
            if now - cached_time < CACHE_TTL_SECONDS:
                logger.debug(f"⚡ Cache HIT for User ID {u_id} [{len(cached_items)} memories returned from RAM]")
                return cached_items

        # 2. Cache MISS: Fetch from SQLite disk storage
        logger.debug(f"🐢 Cache MISS for User ID {u_id} -> Querying SQLite Database...")
        fresh_items = self.memory_store.get_memories_by_user(u_id)
        
        # 3. Refresh Cache
        self._cache[u_id] = (now, fresh_items)
        return fresh_items

    def calculate_relevance_score(self, item: MemoryItem, query: Optional[str] = None) -> float:
        """
        Calculates a multi-factor relevance score for a MemoryItem against an incoming query:
        Score = (Importance * 1.0) + (Confidence * 2.0) + (Access Count * 0.1) + Keyword Bonus (5.0 if query match).

        Args:
            item (MemoryItem): Memory item to evaluate.
            query (Optional[str]): Incoming user prompt query.

        Returns:
            float: Combined numerical relevance score.
        """
        score = (item.importance * 1.0) + (item.confidence * 2.0) + (item.access_count * 0.1)

        if query and query.strip():
            q_terms = [t for t in query.lower().split() if len(t) > 2]
            key_text = f"{item.category} {item.memory_key} {item.memory_value}".lower()
            matches = sum(1 for term in q_terms if term in key_text)
            if matches > 0:
                score += (matches * 5.0)  # Significant relevance boost for matching query terms

        return score

    def rank_memories(self, memories: List[MemoryItem], query: Optional[str] = None) -> List[MemoryItem]:
        """
        Ranks candidate memories in descending order based on their multi-factor relevance score.

        Args:
            memories (List[MemoryItem]): Candidate memories.
            query (Optional[str]): Incoming user query string.

        Returns:
            List[MemoryItem]: Ranked list of memories.
        """
        return sorted(memories, key=lambda m: self.calculate_relevance_score(m, query), reverse=True)

    def retrieve_by_user(
        self,
        user_id: str,
        query: Optional[str] = None,
        limit: int = DEFAULT_MEMORY_LIMIT
    ) -> List[MemoryItem]:
        """
        Retrieves top N ranked memories stored for user_id.

        Args:
            user_id (str): Telegram User ID string.
            query (Optional[str]): Optional user prompt for relevance scoring.
            limit (int): Maximum records threshold (default: 10).

        Returns:
            List[MemoryItem]: Bounded list of top-ranked MemoryItem objects.
        """
        raw_memories = self.get_user_memories_cached(str(user_id))
        ranked = self.rank_memories(raw_memories, query=query)
        return ranked[:limit]

    def retrieve_by_category(self, user_id: str, category: str, limit: int = DEFAULT_MEMORY_LIMIT) -> List[MemoryItem]:
        """
        Retrieves memories filtered by user_id and taxonomy category.

        Args:
            user_id (str): Telegram User ID string.
            category (str): Taxonomy category string.
            limit (int): Maximum records threshold.

        Returns:
            List[MemoryItem]: Filtered memory records.
        """
        raw_memories = self.memory_store.get_memories_by_user(str(user_id), category=category)
        ranked = self.rank_memories(raw_memories)
        return ranked[:limit]

    def search_memories(self, user_id: str, query: str, limit: int = DEFAULT_MEMORY_LIMIT) -> SearchResult:
        """
        Searches user memories matching a keyword query string against category, key, or value.

        Args:
            user_id (str): Telegram User ID string.
            query (str): Search keyword filter.
            limit (int): Maximum returned items limit.

        Returns:
            SearchResult: SearchResult object containing matching ranked items.
        """
        all_memories = self.get_user_memories_cached(str(user_id))
        q_lower = query.strip().lower()

        matching = [
            item for item in all_memories
            if q_lower in item.category.lower() or q_lower in item.memory_key.lower() or q_lower in item.memory_value.lower()
        ]

        ranked = self.rank_memories(matching, query=query)[:limit]
        return SearchResult(items=ranked, total_count=len(matching), query=query)

    def get_formatted_memory_context(
        self,
        user_id: str,
        query: Optional[str] = None,
        limit: int = DEFAULT_MEMORY_LIMIT
    ) -> str:
        """
        Retrieves top N relevant memories and formats them into a clean Markdown text block.

        Args:
            user_id (str): Telegram User ID string.
            query (Optional[str]): Optional user query string.
            limit (int): Upper bound limit.

        Returns:
            str: Clean, human-readable prompt memory block string (or empty string if no memories exist).
        """
        memories = self.retrieve_by_user(str(user_id), query=query, limit=limit)

        if not memories:
            return ""

        lines = [
            "=== RECALLED USER LONG-TERM MEMORIES & PROFILE ===",
            f"The following persistent facts describe the USER interacting with you (User ID: {user_id}):"
        ]

        for item in memories:
            lines.append(f"• [{item.category}] {item.memory_key}: {item.memory_value}")

        lines.append(
            "\nInstructions for Using Stored User Facts:\n"
            "- The facts above describe the USER (the human). Use them to answer questions about the user accurately.\n"
            "- If the user asks 'What is my name?', use the [profile] name fact stored above (e.g. 'Your name is Yaseen.').\n"
            "- Do NOT confuse the user's name with your assistant name (SANA AI)."
        )
        
        context_str = "\n".join(lines)
        logger.debug(f"Formatted Long-Term Memory Context [Length: {len(context_str)} chars | {len(memories)} facts]")
        return context_str
