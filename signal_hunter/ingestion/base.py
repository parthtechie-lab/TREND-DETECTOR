"""
Abstract base class for all source pollers.
Each poller runs as an independent asyncio coroutine,
never blocking other pollers.
"""
import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from signal_hunter.core.observability import source_health_consecutive_failures

logger = logging.getLogger(__name__)


@dataclass
class RawItem:
    """Represents a single piece of raw ingested content from any source."""

    source: str
    external_id: str
    title: str
    body: str
    url: str
    author: str
    fetched_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )

    def __post_init__(self) -> None:
        # Standardise fetched_at to timezone-naive UTC
        if self.fetched_at.tzinfo is not None:
            self.fetched_at = self.fetched_at.astimezone(timezone.utc).replace(tzinfo=None)


class SourcePoller(ABC):
    """
    Abstract base class for all source pollers.

    Subclasses must implement:
        - source_name (property)
        - poll_interval_seconds (property)
        - tier (property)
        - poll() (async generator)
    """

    def __init__(self) -> None:
        self._consecutive_failures: int = 0
        self._backoff_seconds: float = 5.0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique identifier for this data source (e.g. 'reddit', 'youtube')."""

    @property
    @abstractmethod
    def poll_interval_seconds(self) -> int:
        """How often, in seconds, to run a polling cycle."""

    @property
    @abstractmethod
    def tier(self) -> str:
        """
        Concurrency tier label for the scheduler semaphore.

        Use 'A' for lightweight API callers and 'C' for heavyweight
        browser-driven pollers.
        """

    @abstractmethod
    async def poll(self) -> AsyncIterator[RawItem]:
        """
        Yield RawItems discovered during one polling cycle.

        This is an async generator; callers iterate with ``async for``.
        """
        # Make type-checkers happy — subclasses must use ``yield``.
        yield  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run_forever(self, queue: asyncio.Queue) -> None:
        """
        Run the poller indefinitely, pushing items onto *queue*.

        Design notes:
        - poll() is invoked once per cycle; each yielded RawItem is
          immediately pushed to the queue via put_nowait so that slow
          producers never starve fast consumers.
        - A QueueFull exception is caught per-item and logged at WARNING
          level to avoid losing the rest of the batch.
        - Any other exception from poll() is caught, logged with a full
          traceback, and treated as a failure cycle (backoff + Prometheus
          counter increment).
        - Jitter (±15 % of poll_interval_seconds) is applied every cycle
          to desynchronise multiple pollers started at the same time.
        """
        logger.info(
            "[%s] Poller starting — interval=%ds tier=%s",
            self.source_name,
            self.poll_interval_seconds,
            self.tier,
        )

        while True:
            cycle_start = time.monotonic()
            success = False
            items_count = 0
            try:
                async for item in self.poll():
                    items_count += 1
                    try:
                        queue.put_nowait(item.__dict__)
                    except asyncio.QueueFull:
                        logger.warning(
                            "[%s] Queue is full — dropping item external_id=%s",
                            self.source_name,
                            item.external_id,
                        )
                self._on_success()
                success = True
                try:
                    from signal_hunter.selfheal.monitor import health_monitor
                    await health_monitor.update_health(self.source_name, success=True, items_count=items_count)
                except Exception as ex:
                    logger.error("[%s] Failed to log health on success: %s", self.source_name, ex)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[%s] Unhandled exception during poll cycle — scheduling backoff.",
                    self.source_name,
                )
                self._on_failure()
                try:
                    from signal_hunter.selfheal.monitor import health_monitor
                    await health_monitor.update_health(self.source_name, success=False)
                except Exception as ex:
                    logger.error("[%s] Failed to log health on failure: %s", self.source_name, ex)

            elapsed = time.monotonic() - cycle_start
            sleep_target = (
                self.poll_interval_seconds + self._jitter()
                if success
                else self._backoff_seconds
            )
            # Never sleep a negative amount if polling took longer than the interval.
            sleep_duration = max(0.0, sleep_target - elapsed)

            logger.debug(
                "[%s] Cycle complete in %.2fs — sleeping %.2fs "
                "(failures=%d backoff=%.0fs).",
                self.source_name,
                elapsed,
                sleep_duration,
                self._consecutive_failures,
                self._backoff_seconds,
            )
            await asyncio.sleep(sleep_duration)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_random_user_agent(self) -> str:
        """Return a random User-Agent string from the predefined pool."""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        ]
        return random.choice(user_agents)

    def get_proxy(self) -> Optional[str]:
        """
        Return a proxy URL string if configured.
        
        Enterprise system extension: can be configured to read from an env
        variable or setting like settings.PROXY_URL / settings.PROXY_LIST.
        """
        return getattr(settings, "PROXY_URL", None)

    def _jitter(self) -> float:
        """Return a random jitter of ±15 % of poll_interval_seconds."""
        return random.uniform(-0.15, 0.15) * self.poll_interval_seconds

    def _on_success(self) -> None:
        """Reset failure counters and Prometheus gauge on a successful cycle."""
        self._consecutive_failures = 0
        self._backoff_seconds = 5.0
        source_health_consecutive_failures.labels(source=self.source_name).set(0)
        logger.debug("[%s] Poll cycle succeeded.", self.source_name)

    def _on_failure(self) -> None:
        """
        Increment failure counters, double the backoff (capped at 300 s),
        and update the Prometheus gauge.
        """
        self._consecutive_failures += 1
        self._backoff_seconds = min(self._backoff_seconds * 2, 300.0)
        source_health_consecutive_failures.labels(source=self.source_name).set(
            self._consecutive_failures
        )
        logger.warning(
            "[%s] Consecutive failures: %d — next backoff: %.0fs.",
            self.source_name,
            self._consecutive_failures,
            self._backoff_seconds,
        )

# Import settings dynamically at runtime to avoid circular dependencies
from signal_hunter.core.config import settings

