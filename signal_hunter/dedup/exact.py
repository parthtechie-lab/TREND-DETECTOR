"""
Exact deduplication using SHA-256 hashing with an LRU OrderedDict cache.

Normalises the (title, source, date) triple and caches the hash so that
repeated items from the same source on the same day are dropped immediately
without touching the database.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections import OrderedDict
from datetime import datetime

from signal_hunter.core.observability import items_deduped_exact_total

# Maximum number of entries to keep in the in-process LRU cache.
_CACHE_MAX_SIZE: int = 50_000


def _normalize(text: str) -> str:
    """Lowercase, strip, and collapse all internal whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _make_key(title: str, source: str, fetched_at: datetime) -> str:
    """
    Build a deterministic SHA-256 cache key for the (title, source, date) triple.

    The date component is truncated to YYYYMMDD so that items on the same
    calendar day are considered identical regardless of the exact timestamp.
    """
    date_str = fetched_at.strftime("%Y%m%d")
    raw = f"{_normalize(title)}:{_normalize(source)}:{date_str}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ExactDeduplicator:
    """
    Thread-safe exact deduplicator backed by an in-memory LRU cache.

    The cache is an ``OrderedDict`` used in insertion order; when the cache
    exceeds ``_CACHE_MAX_SIZE`` the oldest entry is evicted (LRU eviction via
    ``popitem(last=False)``).
    """

    def __init__(self, max_size: int = 50_000) -> None:
        self._cache: OrderedDict[str, bool] = OrderedDict()
        self._max_size = max_size
        self._lock: asyncio.Lock = asyncio.Lock()

    async def hydrate_from_db(self) -> None:
        """Hydrate the exact deduplication cache by loading raw items from the database."""
        from datetime import datetime, timezone, timedelta
        import logging
        from sqlalchemy import select
        from signal_hunter.core.database import AsyncSessionLocal, RawItem as DBRawItem

        logger = logging.getLogger(__name__)
        logger.info("[ExactDeduplicator] Hydrating exact deduplicator cache from DB...")
        try:
            cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=72)
            async with AsyncSessionLocal() as session:
                stmt = select(DBRawItem.title, DBRawItem.source, DBRawItem.fetched_at).where(
                    DBRawItem.fetched_at >= cutoff
                )
                result = await session.execute(stmt)
                items = result.all()

                async with self._lock:
                    for title, source, fetched_at in items:
                        key = _make_key(title, source, fetched_at)
                        self._cache[key] = True
                        if len(self._cache) > self._max_size:
                            self._cache.popitem(last=False)
                logger.info("[ExactDeduplicator] Hydrated %d keys from the database.", len(items))
        except Exception as e:
            logger.error("[ExactDeduplicator] Failed to hydrate cache: %s", e)

    async def is_duplicate(
        self,
        title: str,
        source: str,
        fetched_at: datetime,
    ) -> bool:
        """
        Return ``True`` when the item has been seen before.

        On a cache hit the Prometheus counter ``items_deduped_exact_total`` is
        incremented and the entry is moved to the *most-recently-used* position.
        On a cache miss the key is inserted and the cache is trimmed if
        necessary.

        Parameters
        ----------
        title:
            Human-readable headline / title of the ingested item.
        source:
            Identifier of the data source (e.g. ``"hackernews"``).
        fetched_at:
            The timestamp recorded when the item was fetched; only the date
            portion is used for key construction.
        """
        key = _make_key(title, source, fetched_at)

        async with self._lock:
            if key in self._cache:
                # Promote to MRU position.
                self._cache.move_to_end(key)
                items_deduped_exact_total.inc()
                return True

            # Cache miss – record this key.
            self._cache[key] = True
            self._cache.move_to_end(key)

            # Evict the LRU entry when the cache is full.
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

            return False

    @property
    def cache_size(self) -> int:
        """Current number of entries held in the LRU cache."""
        return len(self._cache)


# ---------------------------------------------------------------------------
# Module-level singleton – import and use this directly.
# ---------------------------------------------------------------------------
exact_deduplicator: ExactDeduplicator = ExactDeduplicator()
