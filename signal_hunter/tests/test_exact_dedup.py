"""Signal Hunter AI — unit tests for exact hash-based deduplication."""

import pytest
from datetime import datetime, timezone
from signal_hunter.dedup.exact import ExactDeduplicator


@pytest.mark.asyncio
async def test_exact_deduplication():
    """Verify exact hash-based deduplication catches duplicate titles from the same source."""
    dedup = ExactDeduplicator(max_size=100)
    now = datetime.now(timezone.utc)

    title = "OpenHands: Staged autonomy for self-healing crawlers"
    source = "reddit"

    # First fetch: should not be marked duplicate
    is_dup1 = await dedup.is_duplicate(title, source, now)
    assert is_dup1 is False

    # Second fetch on same day: should be detected as duplicate
    is_dup2 = await dedup.is_duplicate(title, source, now)
    assert is_dup2 is True

    # Same title, different source: should NOT be marked duplicate
    is_dup3 = await dedup.is_duplicate(title, "youtube", now)
    assert is_dup3 is False


@pytest.mark.asyncio
async def test_exact_deduplication_normalization():
    """Verify that minor spacing and casing differences are normalized before hashing."""
    dedup = ExactDeduplicator(max_size=100)
    now = datetime.now(timezone.utc)

    title1 = "   OpenHands:  Staged Autonomy   "
    title2 = "openhands: staged autonomy"
    source = "reddit"

    is_dup1 = await dedup.is_duplicate(title1, source, now)
    assert is_dup1 is False

    is_dup2 = await dedup.is_duplicate(title2, source, now)
    assert is_dup2 is True


@pytest.mark.asyncio
async def test_exact_deduplication_eviction():
    """Verify that the LRU cache evicts oldest keys when max_size is exceeded."""
    dedup = ExactDeduplicator(max_size=2)
    now = datetime.now(timezone.utc)

    assert await dedup.is_duplicate("Title A", "reddit", now) is False
    assert await dedup.is_duplicate("Title B", "reddit", now) is False

    # Trigger eviction of Title A
    assert await dedup.is_duplicate("Title C", "reddit", now) is False

    # Title B should still be in cache
    assert await dedup.is_duplicate("Title B", "reddit", now) is True

    # Title A should have been evicted and be seen as new
    assert await dedup.is_duplicate("Title A", "reddit", now) is False
