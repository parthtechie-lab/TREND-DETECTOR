"""
Persistent state-store tracker for API limits and quotas.
"""
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Resolve quota file path relative to the package root for portability.
_DEFAULT_QUOTA_FILE = str(Path(__file__).parent.parent / "quota_usage.json")


class QuotaTracker:
    """Tracks per-source API quota and token usage persistently."""

    def __init__(self, filename: str = "") -> None:
        self.filename = filename or _DEFAULT_QUOTA_FILE
        self.limits = {
            "youtube": 10000,
            "reddit": 5000,
            "product_hunt": 10000,
            "hacker_news": 20000,
        }
        self.usage = {}
        self.current_date = datetime.now(timezone.utc).date().isoformat()
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("date") == self.current_date:
                    self.usage = data.get("usage", {})
                else:
                    self.usage = {}
            except Exception as e:
                logger.error("Failed to load quota file: %s", e)
        else:
            self.usage = {}

    def _save_sync(self) -> None:
        """Synchronous file write — always run via run_in_executor from async contexts."""
        try:
            with open(self.filename, "w", encoding="utf-8") as f:
                json.dump({
                    "date": self.current_date,
                    "usage": self.usage
                }, f, indent=2)
        except Exception as e:
            logger.error("Failed to save quota file: %s", e)

    def _save(self) -> None:
        """Save quota state. Schedules an async write if an event loop is running."""
        try:
            loop = asyncio.get_running_loop()
            # We are inside an async context — offload blocking I/O to executor.
            loop.run_in_executor(None, self._save_sync)
        except RuntimeError:
            # No running loop (e.g. called during startup sync init) — write synchronously.
            self._save_sync()

    def check_reset(self) -> None:
        now_date = datetime.now(timezone.utc).date().isoformat()
        if now_date != self.current_date:
            self.current_date = now_date
            self.usage = {}
            self._save()
            logger.info("Quota tracker reset for new day: %s", self.current_date)

    def get_usage(self, source: str) -> int:
        self.check_reset()
        return self.usage.get(source, 0)

    def has_quota(self, source: str, cost: int = 1) -> bool:
        self.check_reset()
        limit = self.limits.get(source, 10000)
        current = self.usage.get(source, 0)
        return (current + cost) <= limit

    def consume(self, source: str, cost: int = 1) -> None:
        self.check_reset()
        self.usage[source] = self.usage.get(source, 0) + cost
        self._save()
        limit = self.limits.get(source, 10000)
        current = self.usage[source]

        # Log warnings if approaching limits
        if current >= limit * 0.8:
            logger.warning(
                "[QuotaTracker] Source %s is at %d/%d (%.1f%%) of its daily quota!",
                source, current, limit, (current / limit) * 100
            )

quota_tracker = QuotaTracker()

