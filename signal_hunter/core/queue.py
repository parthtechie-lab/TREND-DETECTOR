"""Signal Hunter AI — asyncio queue manager.

Three bounded asyncio queues act as the internal bus between pipeline stages:

  raw_items_queue   →  de-duplicator   →  unique_items_queue
  unique_items_queue →  AI workers     →  scored_items_queue
  scored_items_queue →  alerter / DB writer

When a queue is full the item is **dropped** (non-blocking) so that slow
downstream stages never back-pressure ingestors.  Dropped-item counts are
tracked per queue and exposed via :func:`get_drop_counts`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from signal_hunter.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Queue instances
# ---------------------------------------------------------------------------

raw_items_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
    maxsize=settings.RAW_QUEUE_MAXSIZE
)
"""Holds raw items as they arrive from ingestors before de-duplication."""

unique_items_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
    maxsize=settings.UNIQUE_QUEUE_MAXSIZE
)
"""Holds de-duplicated cluster payloads waiting for AI classification."""

scored_items_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
    maxsize=settings.SCORED_QUEUE_MAXSIZE
)
"""Holds classified/scored items waiting to be alerted or persisted."""

# ---------------------------------------------------------------------------
# Drop counters
# ---------------------------------------------------------------------------

_drop_counters: dict[str, int] = {
    "raw": 0,
    "unique": 0,
    "scored": 0,
}
"""Running tally of items dropped due to full queues, keyed by queue name."""

# ---------------------------------------------------------------------------
# Push helpers
# ---------------------------------------------------------------------------


async def push_raw(item: dict[str, Any]) -> None:
    """Enqueue *item* onto :data:`raw_items_queue`.

    If the queue is full the item is **silently dropped** and a warning is
    logged.  This is intentional: ingestors must not block waiting for
    downstream capacity.

    :param item: Arbitrary dict representing a raw ingested item.
    """
    try:
        raw_items_queue.put_nowait(item)
    except asyncio.QueueFull:
        _drop_counters["raw"] += 1
        logger.warning(
            "raw_items_queue full (maxsize=%d) — item dropped. "
            "Total raw drops: %d.  Consider increasing RAW_QUEUE_MAXSIZE "
            "or scaling AI workers.",
            settings.RAW_QUEUE_MAXSIZE,
            _drop_counters["raw"],
        )


async def push_unique(item: dict[str, Any]) -> None:
    """Enqueue *item* onto :data:`unique_items_queue`.

    If the queue is full the item is dropped and a warning is logged.

    :param item: Dict representing a de-duplicated cluster payload.
    """
    try:
        unique_items_queue.put_nowait(item)
    except asyncio.QueueFull:
        _drop_counters["unique"] += 1
        logger.warning(
            "unique_items_queue full (maxsize=%d) — item dropped. "
            "Total unique drops: %d.  Consider increasing UNIQUE_QUEUE_MAXSIZE "
            "or AI_WORKER_COUNT.",
            settings.UNIQUE_QUEUE_MAXSIZE,
            _drop_counters["unique"],
        )


async def push_scored(item: dict[str, Any]) -> None:
    """Enqueue *item* onto :data:`scored_items_queue`.

    If the queue is full the item is dropped and a warning is logged.

    :param item: Dict representing a classified/scored cluster payload.
    """
    try:
        scored_items_queue.put_nowait(item)
    except asyncio.QueueFull:
        _drop_counters["scored"] += 1
        logger.warning(
            "scored_items_queue full (maxsize=%d) — item dropped. "
            "Total scored drops: %d.  Consider increasing SCORED_QUEUE_MAXSIZE "
            "or alerter throughput.",
            settings.SCORED_QUEUE_MAXSIZE,
            _drop_counters["scored"],
        )


# ---------------------------------------------------------------------------
# Observability helpers
# ---------------------------------------------------------------------------


def get_queue_depths() -> dict[str, int]:
    """Return the current number of items waiting in each queue.

    :returns: ``{'raw': int, 'unique': int, 'scored': int}``
    """
    return {
        "raw": raw_items_queue.qsize(),
        "unique": unique_items_queue.qsize(),
        "scored": scored_items_queue.qsize(),
    }


def get_drop_counts() -> dict[str, int]:
    """Return the cumulative number of dropped items per queue since startup.

    :returns: ``{'raw': int, 'unique': int, 'scored': int}``
    """
    return dict(_drop_counters)
