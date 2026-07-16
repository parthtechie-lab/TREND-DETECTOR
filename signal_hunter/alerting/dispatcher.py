"""Signal Hunter AI — Telegram alert dispatcher.

Manages message delivery, rate limits (HTTP 429), and batch buffering for
lower-confidence digest items.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import aiohttp
from sqlalchemy import select

from signal_hunter.core.config import settings
from signal_hunter.core.database import AsyncSessionLocal, AlertSent, UniqueCluster, Classification, RawItem
from signal_hunter.core.observability import alerts_sent_total
from signal_hunter.ai.models import ScoredItem, ClassificationResult, ClassificationOutput, ItemCategory
from signal_hunter.alerting.formatter import formatter
from signal_hunter.ai.summarizer import summarizer

logger = logging.getLogger(__name__)


class _TokenBucket:
    """Client-side token-bucket rate limiter for Telegram API."""

    def __init__(self, rate_per_minute: int) -> None:
        self._capacity: float = float(rate_per_minute)
        self._tokens: float = float(rate_per_minute)
        self._rate: float = rate_per_minute / 60.0
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    self._capacity,
                    self._tokens + elapsed * self._rate,
                )
                self._last_refill = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            await asyncio.sleep(0.1)


class AlertDispatcher:
    """Dispatches market intelligence notifications to Telegram with rate-limit recovery."""

    def __init__(self) -> None:
        self.digest_buffer: deque[ScoredItem] = deque()
        self.last_digest_time: float = time.time()
        self._lock = asyncio.Lock()
        self._rate_limiter = _TokenBucket(rate_per_minute=30)
        # Single persistent HTTP session for Telegram calls (avoid per-call TCP handshake)
        self._http_session: Optional[aiohttp.ClientSession] = None

    async def send_message(self, text: str, cluster_id: str = "", alert_type: str = "") -> Optional[int]:
        """Send a message using the Telegram Bot API.

        Recovers from 429 (Too Many Requests) by backing off exponentially.
        Returns:
            The message ID (int) if successful, else None.
        """
        if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
            logger.warning("Telegram token or Chat ID not configured. Skipping send.")
            return None

        # Enforce rate limit before calling external API
        await self._rate_limiter.acquire()

        url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": settings.TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "MarkdownV2",
            "disable_web_page_preview": True,
        }

        retries = 3
        backoff_delays = [5, 10, 20]
        msg_id: Optional[int] = None
        last_error_reason = "Unknown failure"

        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        session = self._http_session
        if True:  # keep indentation consistent — session is now reused
            for attempt in range(retries + 1):
                try:
                    async with session.post(url, json=payload, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            msg_id = data.get("result", {}).get("message_id")
                            logger.info("Alert dispatched to Telegram. Message ID: %s", msg_id)
                            return msg_id
                        elif resp.status == 429:
                            retry_after = 5
                            try:
                                data = await resp.json()
                                retry_after = data.get("parameters", {}).get("retry_after", 5)
                            except Exception:
                                pass
                            logger.warning(
                                "Telegram returned HTTP 429. Backing off for %d seconds...",
                                retry_after,
                            )
                            last_error_reason = f"HTTP 429: Rate limited, retry after {retry_after}s"
                            await asyncio.sleep(retry_after)
                        else:
                            resp_text = await resp.text()
                            logger.error(
                                "Failed to send Telegram message (HTTP %d): %s",
                                resp.status,
                                resp_text,
                            )
                            last_error_reason = f"HTTP {resp.status}: {resp_text}"
                            break
                except asyncio.TimeoutError:
                    logger.warning("Timeout sending Telegram message (attempt %d/%d).", attempt + 1, retries + 1)
                    last_error_reason = "TimeoutError"
                except Exception as e:
                    logger.error("Exception in Telegram dispatch: %s", e, exc_info=True)
                    last_error_reason = f"Exception: {e}"

                if attempt < retries:
                    delay = backoff_delays[attempt]
                    logger.info("Retrying in %d seconds...", delay)
                    await asyncio.sleep(delay)

        # Log failed alert to dead_letter_alerts table for auditing
        if msg_id is None:
            try:
                from signal_hunter.core.database import DeadLetterAlert
                async with AsyncSessionLocal() as session:
                    async with session.begin():
                        dl_alert = DeadLetterAlert(
                            cluster_id=cluster_id or "unknown",
                            alert_type=alert_type or "unknown",
                            message_payload=text,
                            failure_reason=last_error_reason,
                        )
                        session.add(dl_alert)
                logger.error("[dispatcher] Logged failed alert delivery to dead_letter_alerts table.")
            except Exception as e:
                logger.error("[dispatcher] Failed to write dead letter alert to database: %s", e)

        return None

    def _is_quiet_hours(self) -> bool:
        """Check if the current time falls within the quiet hours range configured in watchlist.json."""
        wl = settings.watchlist
        qh = wl.get("quiet_hours")
        if not qh:
            return False

        try:
            from zoneinfo import ZoneInfo
            tz_str = qh.get("timezone", "Asia/Kolkata")
            tz = ZoneInfo(tz_str)
            now = datetime.now(tz)
            
            start_str = qh.get("start", "22:00")
            end_str = qh.get("end", "08:00")
            
            start_h, start_m = map(int, start_str.split(":"))
            end_h, end_m = map(int, end_str.split(":"))
            
            now_mins = now.hour * 60 + now.minute
            start_mins = start_h * 60 + start_m
            end_mins = end_h * 60 + end_m
            
            if start_mins > end_mins:
                # Spans across midnight (e.g. 22:00 to 08:00)
                return now_mins >= start_mins or now_mins < end_mins
            else:
                # Same day (e.g. 09:00 to 17:00)
                return start_mins <= now_mins < end_mins
        except Exception as e:
            logger.warning("[dispatcher] Error checking quiet hours: %s", e)
            return False

    async def recover_unalerted_from_db(self) -> None:
        """Query database for validated classifications that have not been alerted and load them."""
        logger.info("[dispatcher] Bootstrapping: recovering unalerted items from DB...")
        try:
            async with AsyncSessionLocal() as session:
                # 1. Fetch all cluster IDs that have already had alerts sent
                alerted_stmt = select(AlertSent.cluster_id)
                alerted_result = await session.execute(alerted_stmt)
                alerted_cluster_ids = set(alerted_result.scalars().all())

                # 2. Fetch classifications that passed validation and are not alerted
                # Limit to the last 24 hours to avoid a thundering herd on old data
                cutoff_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=24)
                from sqlalchemy.orm import selectinload
                stmt = (
                    select(Classification)
                    .options(selectinload(Classification.cluster))
                    .where(Classification.passed_validation == True)
                    .where(Classification.classified_at >= cutoff_time)
                    .where(~Classification.cluster_id.in_(alerted_cluster_ids))
                )
                result = await session.execute(stmt)
                classifications = result.scalars().all()

                for cls in classifications:
                    cluster = cls.cluster
                    if not cluster:
                        continue
                    
                    # Get the earliest raw item of this cluster
                    raw_item_stmt = (
                        select(RawItem)
                        .where(RawItem.cluster_id == cluster.id)
                        .order_by(RawItem.fetched_at.asc())
                        .limit(1)
                    )
                    raw_item_result = await session.execute(raw_item_stmt)
                    raw_item = raw_item_result.scalars().first()
                    if not raw_item:
                        continue

                    try:
                        cat_enum = ItemCategory(cls.category)
                    except ValueError:
                        cat_enum = ItemCategory.OTHER

                    scored_item = ScoredItem(
                        cluster_id=cluster.id,
                        raw_item_id=raw_item.id,
                        classification=ClassificationResult(
                            output=ClassificationOutput(
                                product_name=cls.product_name,
                                category=cat_enum,
                                evidence_quote=cls.evidence_quote,
                                confidence=cls.confidence,
                                trend_signal_present=cls.trend_signal_present,
                            ),
                            raw_model_response=cls.raw_model_output,
                            passed_validation=cls.passed_validation,
                            validation_failure_reason=None,
                            model_version=cls.model_version,
                        ),
                        source=raw_item.source,
                        title=cluster.canonical_title,
                        url=raw_item.url or "",
                        body=raw_item.body or "",
                        fetched_at=raw_item.fetched_at.replace(tzinfo=timezone.utc),
                        corroborated=(cluster.source_count >= settings.CORROBORATION_MIN_SOURCES),
                        source_count=cluster.source_count,
                    )

                    async with self._lock:
                        if not any(i.cluster_id == cluster.id for i in self.digest_buffer):
                            self.digest_buffer.append(scored_item)
                            logger.info(
                                "[dispatcher] Recovered unalerted cluster %s (title=%s) from database.",
                                cluster.id,
                                cluster.canonical_title,
                            )
        except Exception as e:
            logger.error("[dispatcher] Error during unalerted items recovery: %s", e, exc_info=True)

    async def _send_realtime_alert(self, item: ScoredItem) -> bool:
        """Generate summary and send a realtime alert for the item.

        Returns True if sent successfully, False otherwise.
        """
        cluster_id = item.cluster_id
        logger.info(
            "Item in cluster %s qualified for REAL-TIME alert (conf=%.2f, corroborated=True). Generating summary...",
            cluster_id,
            item.classification.output.confidence,
        )
        try:
            # Retrieve corroboration sources for summary
            from signal_hunter.corroboration.engine import corroboration_engine

            sources_list = await corroboration_engine.get_cluster_sources(cluster_id)
            # Build snippet list: body excerpts from each corroborating source.
            # Arg order for summarizer.summarize() is (canonical_title, snippets).
            snippets = [src.get("body", "")[:300] for src in sources_list]
            if not snippets:
                snippets = [item.body[:300]]

            summary = await summarizer.summarize(item.title, snippets)

            # Build and send message
            text = formatter.format_realtime(item, summary)
            msg_id = await self.send_message(text, cluster_id=cluster_id, alert_type="realtime")

            if msg_id is not None:
                # Record in AlertSent
                async with AsyncSessionLocal() as session:
                    async with session.begin():
                        alert = AlertSent(
                            cluster_id=cluster_id,
                            telegram_message_id=msg_id,
                            alert_type="realtime",
                        )
                        session.add(alert)
                logger.info("Logged realtime alert for cluster %s in database.", cluster_id)
                alerts_sent_total.labels(alert_type="realtime").inc()
                return True
            else:
                logger.warning("Failed to send Telegram message for realtime alert %s.", cluster_id)
                return False
        except Exception as e:
            logger.error("Exception in _send_realtime_alert for cluster %s: %s", cluster_id, e, exc_info=True)
            return False

    async def dispatch(self, item: ScoredItem) -> None:
        """Process a classified scored item.

        Determines if it qualifies for an immediate real-time alert or
        buffers it for the periodic batch digest.
        """
        cluster_id = item.cluster_id
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=6)

        # 1. Deduplication check: Has this cluster had an alert sent in the last 6 hours?
        async with AsyncSessionLocal() as session:
            stmt = (
                select(AlertSent)
                .where(AlertSent.cluster_id == cluster_id)
                .where(AlertSent.sent_at >= cutoff)
            )
            result = await session.execute(stmt)
            existing_alert = result.scalars().first()

        is_realtime_qualified = (
            item.classification.output.confidence >= settings.CONFIDENCE_THRESHOLD_ALERT 
            and item.corroborated
        )

        if existing_alert:
            # Upgrade path: if we qualify for a realtime alert, but only a digest alert was sent,
            # we allow upgrading to a realtime alert. Otherwise, we skip.
            if is_realtime_qualified and existing_alert.alert_type == "digest":
                logger.info("[dispatcher] Upgrading cluster %s alert from digest to realtime.", cluster_id)
            else:
                logger.info(
                    "Cluster %s already had a %s alert sent within the last 6h. Skipping alert.",
                    cluster_id,
                    existing_alert.alert_type,
                )
                return

        # 2. Check thresholds for real-time alert (high confidence AND corroborated)
        if is_realtime_qualified:
            # Check quiet hours constraint
            if self._is_quiet_hours():
                wl = settings.watchlist
                qh = wl.get("quiet_hours", {})
                bypass = qh.get("bypass_for_high_priority", True)
                if not bypass:
                    logger.info(
                        "[dispatcher] Real-time alert for cluster %s suppressed due to quiet hours. Buffering for digest.",
                        cluster_id
                    )
                    async with self._lock:
                        if item not in self.digest_buffer:
                            self.digest_buffer.append(item)
                    logger.debug("[dispatcher] Buffered item for digest due to quiet hours (cluster=%s).", cluster_id)
                    return

            # Attempt to send realtime alert
            success = await self._send_realtime_alert(item)
            if not success:
                logger.warning(
                    "[dispatcher] Real-time alert delivery failed for cluster %s. Buffering to prevent message skip.",
                    cluster_id,
                )
                async with self._lock:
                    if item not in self.digest_buffer:
                        self.digest_buffer.append(item)

        # 3. Otherwise, check store threshold for digest buffering
        elif item.classification.output.confidence >= settings.CONFIDENCE_THRESHOLD_STORE:
            async with self._lock:
                # Add to digest buffer if not already present
                if item not in self.digest_buffer:
                    self.digest_buffer.append(item)
                    logger.info(
                        "Buffered item %s for digest (conf=%.2f). Buffer size: %d",
                        item.raw_item_id,
                        item.classification.output.confidence,
                        len(self.digest_buffer),
                    )

    async def run_digest_loop(self) -> None:
        """Periodic loop that packages buffered digest items and delivers them."""
        logger.info("Starting Telegram digest loop (interval = %ds)...", settings.ALERT_BATCH_WINDOW_SECONDS)
        
        # Recover any unalerted items on startup
        await self.recover_unalerted_from_db()

        while True:
            try:
                await asyncio.sleep(settings.ALERT_BATCH_WINDOW_SECONDS)
                
                # Check quiet hours constraint before packaging digest
                if self._is_quiet_hours():
                    logger.info("[dispatcher] Digest loop suppressed during quiet hours.")
                    continue

                async with self._lock:
                    if not self.digest_buffer:
                        continue

                    # Separate real-time items from normal digest items
                    realtime_items: List[ScoredItem] = []
                    digest_items: List[ScoredItem] = []
                    while self.digest_buffer:
                        buffered_item = self.digest_buffer.popleft()
                        is_realtime = (
                            buffered_item.classification.output.confidence >= settings.CONFIDENCE_THRESHOLD_ALERT
                            and buffered_item.corroborated
                        )
                        if is_realtime:
                            realtime_items.append(buffered_item)
                        else:
                            digest_items.append(buffered_item)

                    # Drain the digest items into list (max 10 items per message)
                    items_to_send: List[ScoredItem] = []
                    for _ in range(min(10, len(digest_items))):
                        if digest_items:
                            items_to_send.append(digest_items.pop(0))

                    # Put back any remaining digest items
                    for item in reversed(digest_items):
                        self.digest_buffer.appendleft(item)

                # 1. Dispatch deferred/suppressed realtime alerts individually (with full summaries!)
                for rt_item in realtime_items:
                    logger.info(
                        "[dispatcher] Releasing suppressed realtime alert for cluster %s after quiet hours.",
                        rt_item.cluster_id,
                    )
                    success = await self._send_realtime_alert(rt_item)
                    if not success:
                        # Put back to buffer for retry if it failed again
                        async with self._lock:
                            self.digest_buffer.append(rt_item)

                # 2. Package and send remaining digest items
                if items_to_send:
                    logger.info("Sending digest with %d items...", len(items_to_send))
                    text = formatter.format_digest(items_to_send)
                    digest_cluster_ids = ",".join([item.cluster_id for item in items_to_send])
                    msg_id = await self.send_message(text, cluster_id=digest_cluster_ids, alert_type="digest")

                    if msg_id is not None:
                        # Record in AlertSent
                        async with AsyncSessionLocal() as session:
                            async with session.begin():
                                for item in items_to_send:
                                    alert = AlertSent(
                                        cluster_id=item.cluster_id,
                                        telegram_message_id=msg_id,
                                        alert_type="digest",
                                    )
                                    session.add(alert)
                        alerts_sent_total.labels(alert_type="digest").inc()
                        self.last_digest_time = time.time()
                    else:
                        logger.warning("Failed to send Telegram digest. Restoring items to buffer.")
                        async with self._lock:
                            for item in reversed(items_to_send):
                                self.digest_buffer.appendleft(item)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in digest loop: %s", e, exc_info=True)


dispatcher = AlertDispatcher()
