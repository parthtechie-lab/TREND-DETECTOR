"""
Ingestion scheduler for Signal Hunter AI.

Starts all source pollers as concurrent asyncio tasks, enforcing
concurrency limits via per-tier semaphores and staggering launch times
to avoid thundering-herd bursts on startup.

Tier model
----------
* Tier A — lightweight API pollers (Reddit, YouTube, Product Hunt)
  Semaphore(2): at most 2 Tier-A pollers run a poll *cycle* simultaneously.
* Tier C — heavyweight browser pollers (TikTok)
  Semaphore(1): only one Tier-C poller runs a poll *cycle* at a time.

Note: semaphores wrap individual poll *cycles*, not the entire lifespan of a
poller.  A poller that is sleeping between cycles does NOT hold its semaphore.
"""
import asyncio
import logging

from signal_hunter.core.config import settings
from signal_hunter.core.queue import raw_items_queue
from signal_hunter.ingestion.reddit import RedditPoller
from signal_hunter.ingestion.youtube import YouTubePoller
from signal_hunter.ingestion.product_hunt import ProductHuntPoller
from signal_hunter.ingestion.tiktok import TikTokPoller
from signal_hunter.ingestion.hacker_news import HackerNewsPoller
from signal_hunter.ingestion.base import SourcePoller
from signal_hunter.selfheal.monitor import health_monitor

logger = logging.getLogger(__name__)


_TIER_A_SEMAPHORE_LIMIT: int = 2
_TIER_C_SEMAPHORE_LIMIT: int = 1

# Stagger delay in seconds between each poller's first iteration.
_STAGGER_DELAYS: list[int] = [0, 5, 10, 15, 20]


def _make_gated_runner(
    poller: SourcePoller,
    queue: asyncio.Queue,
    semaphore: asyncio.Semaphore,
    start_delay: int,
) -> asyncio.Task:
    """
    Return a coroutine that:
    1. Waits *start_delay* seconds before the first cycle.
    2. Acquires *semaphore* around each individual poll cycle so that
       concurrency across same-tier pollers is bounded.
    3. Runs forever, never propagating exceptions (they are already
       handled inside SourcePoller.run_forever).

    We patch the poller's ``run_forever`` behaviour by wrapping it:
    instead of calling ``run_forever`` directly (which does not use a
    semaphore), we override the loop here to honour the semaphore at
    the cycle boundary.

    Implementation note: because SourcePoller.run_forever already
    contains the main retry/backoff logic, we replicate only the
    cycle-gating here by wrapping poll() with the semaphore.
    """

    async def _gated_loop() -> None:
        source = poller.source_name

        if start_delay > 0:
            logger.info(
                "[scheduler] Staggering %s poller by %ds before first poll.",
                source,
                start_delay,
            )
            await asyncio.sleep(start_delay)

        logger.info("[scheduler] Poller %s is now running (tier=%s).", source, poller.tier)

        while True:
            import time as _time
            import random as _random

            cycle_start = _time.monotonic()
            success = False
            try:
                async with semaphore:
                    logger.debug(
                        "[scheduler] Semaphore acquired for %s (tier=%s).",
                        source,
                        poller.tier,
                    )
                    items_yielded = 0
                    async for item in poller.poll():
                        try:
                            queue.put_nowait(item.__dict__)
                            items_yielded += 1
                        except asyncio.QueueFull:
                            logger.warning(
                                "[scheduler][%s] Queue full — dropping item %s.",
                                source,
                                item.external_id,
                            )
                    logger.debug(
                        "[scheduler] Semaphore released for %s.", source
                    )
                items_this_cycle = items_yielded
                poller._on_success()
                success = True
                # Update the source_health DB table so the self-heal monitor
                # has accurate data to detect degraded sources.
                try:
                    await health_monitor.update_health(source, success=True, items_count=items_this_cycle)
                except Exception as _he:
                    logger.warning("[scheduler] Failed to update health for %s: %s", source, _he)
            except asyncio.CancelledError:
                logger.info("[scheduler] Poller %s cancelled.", source)
                raise
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[scheduler] Unhandled exception in %s poll cycle.", source
                )
                poller._on_failure()
                try:
                    await health_monitor.update_health(source, success=False)
                except Exception as _he:
                    logger.warning("[scheduler] Failed to update health for %s: %s", source, _he)


            elapsed = _time.monotonic() - cycle_start
            sleep_target = (
                poller.poll_interval_seconds
                + _random.uniform(-0.15, 0.15) * poller.poll_interval_seconds
                if success
                else poller._backoff_seconds
            )
            sleep_duration = max(0.0, sleep_target - elapsed)

            logger.debug(
                "[scheduler] %s sleeping %.2fs (success=%s).",
                source,
                sleep_duration,
                success,
            )
            await asyncio.sleep(sleep_duration)

    return _gated_loop()


async def run_ingestion() -> None:
    """
    Start all source pollers and run them indefinitely.

    This coroutine blocks until all pollers complete (which should be
    never under normal operation) or until the task is cancelled.

    Usage::

        asyncio.run(run_ingestion())
    """
    # ------------------------------------------------------------------
    # Semaphore construction
    # ------------------------------------------------------------------
    tier_a_semaphore = asyncio.Semaphore(_TIER_A_SEMAPHORE_LIMIT)
    tier_b_semaphore = asyncio.Semaphore(2)  # Limit Tier-B poll concurrency
    tier_c_semaphore = asyncio.Semaphore(_TIER_C_SEMAPHORE_LIMIT)

    tier_semaphores: dict[str, asyncio.Semaphore] = {
        "A": tier_a_semaphore,
        "B": tier_b_semaphore,
        "C": tier_c_semaphore,
    }

    # ------------------------------------------------------------------
    # Poller registry
    # ------------------------------------------------------------------
    pollers: list[SourcePoller] = [
        RedditPoller(),
        YouTubePoller(),
        ProductHuntPoller(),
        HackerNewsPoller(),
        TikTokPoller(),
    ]

    logger.info(
        "[scheduler] Starting ingestion with %d pollers: %s",
        len(pollers),
        ", ".join(p.source_name for p in pollers),
    )
    logger.info(
        "[scheduler] Tier-A concurrency limit: %d | Tier-C concurrency limit: %d",
        _TIER_A_SEMAPHORE_LIMIT,
        _TIER_C_SEMAPHORE_LIMIT,
    )

    # ------------------------------------------------------------------
    # Build gated coroutines with staggered starts
    # ------------------------------------------------------------------
    coroutines = [
        _make_gated_runner(
            poller=poller,
            queue=raw_items_queue,
            semaphore=tier_semaphores[poller.tier],
            start_delay=_STAGGER_DELAYS[i],
        )
        for i, poller in enumerate(pollers)
    ]

    # ------------------------------------------------------------------
    # Launch all coroutines concurrently
    # ------------------------------------------------------------------
    results = await asyncio.gather(*coroutines, return_exceptions=True)

    # Log any unexpected terminal exceptions from pollers.
    for poller, result in zip(pollers, results):
        if isinstance(result, Exception) and not isinstance(
            result, asyncio.CancelledError
        ):
            logger.error(
                "[scheduler] Poller %s terminated with an unhandled exception: %s",
                poller.source_name,
                result,
                exc_info=result,
            )

    logger.info("[scheduler] run_ingestion() exiting — all pollers have stopped.")
