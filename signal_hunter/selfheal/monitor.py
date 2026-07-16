"""Signal Hunter AI — crawler self-healing & health monitor.

Tracks source scraping reliability, logs health metrics in SQL, and issues alerts
on crawler degradation without autonomous live-patch deployments.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy import select

from signal_hunter.core.database import AsyncSessionLocal, SourceHealth
from signal_hunter.core.config import settings
from signal_hunter.alerting.dispatcher import dispatcher
from signal_hunter.core.observability import source_health_consecutive_failures

logger = logging.getLogger(__name__)


class SourceHealthMonitor:
    """Monitors ingestion health and alerts operators to degradations."""

    def _utcnow(self) -> datetime:
        """Return timezone-naive UTC datetime."""
        return datetime.now(timezone.utc).replace(tzinfo=None)

    async def update_health(
        self, source_name: str, success: bool, items_count: int = 0
    ) -> None:
        """Upsert SourceHealth details in the database."""
        async with AsyncSessionLocal() as session:
            async with session.begin():
                stmt = select(SourceHealth).where(SourceHealth.source_name == source_name)
                result = await session.execute(stmt)
                record = result.scalars().first()

                now = self._utcnow()
                if not record:
                    record = SourceHealth(
                        source_name=source_name,
                        consecutive_failures=0,
                        total_items_24h=0,
                        is_degraded=False,
                        updated_at=now,
                    )
                    session.add(record)

                if success:
                    record.consecutive_failures = 0
                    record.last_success = now
                    record.total_items_24h += items_count
                    record.is_degraded = False
                else:
                    record.consecutive_failures += 1
                    if record.consecutive_failures >= 5:
                        record.is_degraded = True
                        if record.consecutive_failures == 5:
                            import os
                            import json
                            import time
                            incident_dir = os.path.join(os.getcwd(), ".ase-incidents")
                            os.makedirs(incident_dir, exist_ok=True)
                            incident_file = os.path.join(incident_dir, f"incident-{source_name}.json")
                            incident_data = {
                                "incident_id": f"inc-{source_name}-{int(time.time())}",
                                "source": source_name,
                                "consecutive_failures": record.consecutive_failures,
                                "detected_at": now.isoformat(),
                                "status": "open",
                                "description": f"Source {source_name} has failed 5 consecutive times. Suspected scrapers / credentials issue.",
                            }
                            try:
                                with open(incident_file, "w", encoding="utf-8") as f:
                                    json.dump(incident_data, f, indent=2)
                                logger.critical(
                                    "[ASE_INCIDENT] Critical failure on source %s. Incident file written to %s",
                                    source_name,
                                    incident_file
                                )
                            except Exception as ex:
                                logger.error("Failed to write incident file: %s", ex)

                record.updated_at = now
                logger.debug(
                    "Source health updated for %s: success=%s (consecutive_failures=%d)",
                    source_name,
                    success,
                    record.consecutive_failures,
                )

                # Update Prometheus gauge
                source_health_consecutive_failures.labels(source=source_name).set(
                    record.consecutive_failures
                )

    async def check_for_degradation(self) -> None:
        """Query degraded sources and fire Telegram notices if appropriate."""
        logger.info("Running scheduled check for degraded ingestion sources...")
        
        # Periodically decay/clean the semantic deduplication window
        try:
            from signal_hunter.dedup.semantic import semantic_deduplicator
            await semantic_deduplicator.decay_window()
        except Exception as ex:
            logger.error("Failed to decay semantic window: %s", ex)
        now = self._utcnow()
        cutoff = now - timedelta(hours=24)

        async with AsyncSessionLocal() as session:
            stmt = select(SourceHealth).where(SourceHealth.is_degraded == True)  # noqa: E712
            result = await session.execute(stmt)
            degraded_sources = result.scalars().all()

        for source in degraded_sources:
            # Rate limit alerts: Only alert once per 24 hours per source
            if source.last_degradation_alert and source.last_degradation_alert >= cutoff:
                logger.debug(
                    "Source %s is degraded but alerted recently (at %s). Skipping alert.",
                    source.source_name,
                    source.last_degradation_alert,
                )
                continue

            alert_text = (
                f"🚨 *INGESTION WARNING* \\| 📡 *{source.source_name.upper()}*\n\n"
                f"⚠️ Source has failed *{source.consecutive_failures}* consecutive times\\.\n"
                f"Last successful fetch: *{source.last_success or 'Never'}* UTC\\.\n\n"
                f"ℹ️ Playwright or API rate limit credentials might require rotation\\."
            )

            # Update the database stamp first to prevent alert double-fires
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    stmt = select(SourceHealth).where(SourceHealth.id == source.id)
                    res = await session.execute(stmt)
                    db_src = res.scalars().first()
                    if db_src:
                        db_src.last_degradation_alert = now

            logger.warning(
                "Source %s is degraded (%d failures). Dispatching Telegram warning.",
                source.source_name,
                source.consecutive_failures,
            )
            await dispatcher.send_message(alert_text)

    async def reset_24h_counters(self) -> None:
        """Reset total_items_24h values in the database."""
        logger.info("Resetting 24-hour source count metrics...")
        async with AsyncSessionLocal() as session:
            async with session.begin():
                stmt = select(SourceHealth)
                result = await session.execute(stmt)
                records = result.scalars().all()
                for rec in records:
                    rec.total_items_24h = 0
                    rec.updated_at = self._utcnow()

    async def run_forever(self) -> None:
        """Run health checks periodically and reset counts daily."""
        logger.info("Starting health monitor coroutine...")
        last_reset_day = datetime.now(timezone.utc).day

        while True:
            try:
                await self.check_for_degradation()

                # Check if calendar day rolled over to reset 24h limits
                current_day = datetime.now(timezone.utc).day
                if current_day != last_reset_day:
                    await self.reset_24h_counters()
                    last_reset_day = current_day

                await asyncio.sleep(300)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Exception in health monitor loop: %s", e, exc_info=True)


health_monitor = SourceHealthMonitor()
