"""
Semantic deduplication using sentence-transformers (all-MiniLM-L6-v2).

Maintains a rolling window of recent cluster embeddings and uses cosine
similarity to detect near-duplicate items that exact hashing would miss
(e.g. different headlines describing the same product launch).

Embeddings are persisted to the ``UniqueCluster.embedding_vector`` DB column
so that the window survives process restarts.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Deque, Optional, Tuple

import numpy as np

from signal_hunter.core.config import settings
from signal_hunter.core.observability import items_deduped_semantic_total

logger = logging.getLogger(__name__)

# Sentinel so the model is only loaded once.
_MODEL = None
_MODEL_LOCK = asyncio.Lock()

# Shared thread pool for all embedding calls — avoids spinning up a new pool per call.
_EMBED_EXECUTOR: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="embed"
)


async def _get_model():
    """Lazily load the sentence-transformer model on the first call."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    async with _MODEL_LOCK:
        # Double-checked locking – another coroutine may have loaded it while
        # we were waiting for the lock.
        if _MODEL is not None:
            return _MODEL

        loop = asyncio.get_running_loop()
        _MODEL = await loop.run_in_executor(_EMBED_EXECUTOR, _load_model_sync)

    return _MODEL


def _load_model_sync():
    """Synchronous model load – runs in a thread-pool executor."""
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer("all-MiniLM-L6-v2")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SemanticDeduplicator:
    """
    Maintains a rolling window of (cluster_id, embedding, category) tuples and
    provides fast nearest-neighbour lookup via brute-force cosine similarity.

    The window size is controlled by ``settings.SEMANTIC_WINDOW_SIZE``.  When
    the window is full the oldest entry is automatically evicted.

    Embeddings are persisted to the ``UniqueCluster.embedding_vector`` DB column
    so that calling ``hydrate_from_db()`` on startup restores the window.

    Thread safety is guaranteed by an ``asyncio.Lock`` around all mutations and
    reads of the internal deque.
    """

    def __init__(self) -> None:
        # Window stores elements as: (cluster_id, embedding, category)
        self._window: Deque[Tuple[str, np.ndarray, str]] = deque()
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def find_similar(
        self,
        title: str,
        snippet: str,
        threshold: Optional[float] = None,
    ) -> Optional[Tuple[str, float]]:
        """
        Embed ``title + ' ' + snippet`` and search the window for the closest
        existing cluster.
        """
        model = await _get_model()
        query_embedding = await _embed(model, f"{title} {snippet}")

        async with self._lock:
            window_snapshot = list(self._window)

        if not window_snapshot:
            return None

        # Vectorized similarity matching
        cluster_ids = [cid for cid, _, _ in window_snapshot]
        categories = [cat for _, _, cat in window_snapshot]
        embeddings = np.stack([emb for _, emb, _ in window_snapshot])

        norm_query = np.linalg.norm(query_embedding)
        norm_embeddings = np.linalg.norm(embeddings, axis=1)

        if norm_query == 0.0:
            return None

        # Avoid zero-division warnings on degraded inputs
        norm_embeddings = np.where(norm_embeddings == 0.0, 1.0, norm_embeddings)

        # Compute cosine similarity for all vectors concurrently
        similarities = np.dot(embeddings, query_embedding) / (norm_query * norm_embeddings)

        best_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])
        best_cluster_id = cluster_ids[best_idx]
        best_category = categories[best_idx]

        # Category-conditional thresholds logic
        if threshold is None:
            # Announcing products / funding (SAAS, AI_TOOL) require looser matching to group variants
            # Memes/trends require stricter matching to prevent false groupings
            threshold_map = {
                "SAAS": 0.90,
                "AI_TOOL": 0.90,
                "HARDWARE": 0.92,
                "MEME": 0.96,
                "OTHER": 0.94,
                "pending": settings.SEMANTIC_DEDUP_THRESHOLD,
            }
            threshold = threshold_map.get(best_category, settings.SEMANTIC_DEDUP_THRESHOLD)

        if best_similarity >= threshold:
            items_deduped_semantic_total.inc()
            return best_cluster_id, best_similarity

        return None

    async def add_to_window(
        self,
        cluster_id: str,
        title: str,
        snippet: str,
        category: str = "pending",
    ) -> None:
        """
        Embed ``title + ' ' + snippet`` and append the result to the rolling
        window, evicting the oldest entry when the window is full.

        The embedding vector is also persisted to the ``UniqueCluster.embedding_vector``
        DB column so it can be reloaded after a restart.
        """
        model = await _get_model()
        embedding = await _embed(model, f"{title} {snippet}")

        async with self._lock:
            self._window.append((cluster_id, embedding, category))
            max_size: int = settings.SEMANTIC_WINDOW_SIZE
            while len(self._window) > max_size:
                self._window.popleft()

        # Persist the embedding vector to the DB column for restart recovery.
        await self._persist_embedding(cluster_id, embedding)

    async def _persist_embedding(self, cluster_id: str, embedding: np.ndarray) -> None:
        """Write the embedding vector to the UniqueCluster.embedding_vector column."""
        try:
            from sqlalchemy import update
            from signal_hunter.core.database import AsyncSessionLocal, UniqueCluster

            serialized = json.dumps(embedding.tolist())
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    await session.execute(
                        update(UniqueCluster)
                        .where(UniqueCluster.id == cluster_id)
                        .values(embedding_vector=serialized)
                    )
        except Exception as e:
            logger.warning(
                "[SemanticDeduplicator] Failed to persist embedding for %s: %s", cluster_id, e
            )

    async def hydrate_from_db(self) -> None:
        """
        Reload the rolling window from the DB on startup.

        Fetches clusters seen within the past 72 hours that have a stored
        ``embedding_vector`` and re-populates the in-memory deque.  This
        ensures the deduplicator remembers clusters from before the last restart.
        """
        logger.info("[SemanticDeduplicator] Hydrating semantic window from DB...")
        try:
            from sqlalchemy import select
            from signal_hunter.core.database import AsyncSessionLocal, UniqueCluster

            cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=72)
            async with AsyncSessionLocal() as session:
                stmt = (
                    select(UniqueCluster)
                    .where(UniqueCluster.last_seen >= cutoff)
                    .where(UniqueCluster.embedding_vector.isnot(None))
                    .order_by(UniqueCluster.last_seen.asc())
                )
                result = await session.execute(stmt)
                clusters = result.scalars().all()

            loaded = 0
            async with self._lock:
                for cluster in clusters:
                    try:
                        vec = np.array(json.loads(cluster.embedding_vector), dtype=np.float32)
                        self._window.append((cluster.id, vec, cluster.category or "pending"))
                        loaded += 1
                        if len(self._window) > settings.SEMANTIC_WINDOW_SIZE:
                            self._window.popleft()
                    except Exception as e:
                        logger.debug(
                            "[SemanticDeduplicator] Skipping invalid embedding for %s: %s",
                            cluster.id,
                            e,
                        )

            logger.info("[SemanticDeduplicator] Hydrated %d clusters into semantic window.", loaded)
        except Exception as e:
            logger.error("[SemanticDeduplicator] Error during DB hydration: %s", e)

    async def decay_window(self) -> None:
        """Evict cluster embeddings from the window that are older than 72 hours."""
        logger.info("[SemanticDeduplicator] Running semantic window decay/cleanup...")
        try:
            from sqlalchemy import select
            from signal_hunter.core.database import AsyncSessionLocal, UniqueCluster

            cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=72)
            async with AsyncSessionLocal() as session:
                stmt = select(UniqueCluster.id).where(UniqueCluster.last_seen >= cutoff)
                result = await session.execute(stmt)
                recent_ids = set(result.scalars().all())

            async with self._lock:
                initial_len = len(self._window)
                self._window = deque(
                    [item for item in self._window if item[0] in recent_ids],
                    maxlen=settings.SEMANTIC_WINDOW_SIZE,
                )
                logger.info(
                    "[SemanticDeduplicator] Window decayed. Size: %d -> %d",
                    initial_len,
                    len(self._window),
                )
        except Exception as e:
            logger.error("[SemanticDeduplicator] Error during window decay: %s", e)

    async def update_window_category(self, cluster_id: str, category: str) -> None:
        """Update the category of a cluster in the sliding window."""
        async with self._lock:
            for i, item in enumerate(self._window):
                if item[0] == cluster_id:
                    self._window[i] = (item[0], item[1], category)
                    break

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def window_size(self) -> int:
        """Current number of entries in the rolling window."""
        return len(self._window)


async def _embed(model, text: str) -> np.ndarray:
    """Run the (synchronous) sentence-transformer encode in the shared thread executor."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _EMBED_EXECUTOR,
        lambda: model.encode(text, normalize_embeddings=False),
    )
    return np.array(result, dtype=np.float32)


# ---------------------------------------------------------------------------
# Module-level singleton.
# ---------------------------------------------------------------------------
semantic_deduplicator: SemanticDeduplicator = SemanticDeduplicator()
