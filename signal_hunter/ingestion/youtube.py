"""
YouTube Data API v3 source poller using aiohttp.

Searches for videos matching a set of curated queries published within
the last 24 hours.  Results are deduplicated by video ID within a single
cycle to prevent the same video appearing under multiple queries from
being emitted more than once.
"""
import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Set

import aiohttp

from signal_hunter.core.config import settings
from signal_hunter.core.observability import items_ingested_total
from signal_hunter.ingestion.base import RawItem, SourcePoller

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.googleapis.com/youtube/v3/search"
_MAX_RESULTS_PER_QUERY: int = 20
_REQUEST_TIMEOUT_SECONDS: int = 30




def _make_external_id(video_id: str) -> str:
    """Return a stable SHA-256 hex digest for a YouTube video ID."""
    return hashlib.sha256(video_id.encode()).hexdigest()


def _published_after_param() -> str:
    """Return an RFC-3339 timestamp for 24 hours ago, as required by the API."""
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=24)
    # YouTube expects 'Z' suffix, not '+00:00'.
    return cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")


class YouTubePoller(SourcePoller):
    """Polls the YouTube Data API v3 for recently published tech/AI videos."""

    # ------------------------------------------------------------------
    # SourcePoller interface
    # ------------------------------------------------------------------

    @property
    def source_name(self) -> str:
        return "youtube"

    @property
    def poll_interval_seconds(self) -> int:
        return 600  # 10 minutes — API quota is precious

    @property
    def tier(self) -> str:
        return "A"

    # ------------------------------------------------------------------
    # Core polling logic
    # ------------------------------------------------------------------

    async def poll(self) -> AsyncIterator[RawItem]:  # type: ignore[override]
        """
        Yield RawItems for videos matching each SEARCH_QUERY published in
        the last 24 hours.

        A single aiohttp.ClientSession is reused across all queries in
        the cycle to benefit from keep-alive connection pooling.
        """
        published_after = _published_after_param()
        seen_ids: Set[str] = set()

        from signal_hunter.ingestion.quota_tracker import quota_tracker

        timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for query in settings.youtube_queries:
                if not quota_tracker.has_quota("youtube", 100):
                    logger.warning("[youtube] Daily API quota threshold reached. Skipping query %r.", query)
                    return

                params = {
                    "part": "snippet",
                    "type": "video",
                    "q": query,
                    "maxResults": _MAX_RESULTS_PER_QUERY,
                    "publishedAfter": published_after,
                    "key": settings.YOUTUBE_API_KEY,
                }

                try:
                    async with session.get(_BASE_URL, params=params) as response:
                        if response.status == 200:
                            quota_tracker.consume("youtube", 100)
                        elif response.status == 403:
                            # Daily quota exhausted — no point trying other queries.
                            logger.error(
                                "[youtube] HTTP 403 — API quota likely exhausted. "
                                "Skipping remaining queries for this cycle."
                            )
                            return

                        elif response.status == 429:
                            logger.warning(
                                "[youtube] HTTP 429 — rate limited. "
                                "Skipping remaining queries for this cycle."
                            )
                            return

                        response.raise_for_status()
                        data = await response.json()

                except aiohttp.ClientResponseError as exc:
                    logger.error(
                        "[youtube] HTTP error for query %r: %s %s",
                        query,
                        exc.status,
                        exc.message,
                    )
                    continue

                except aiohttp.ClientError as exc:
                    logger.error(
                        "[youtube] Network error for query %r: %s",
                        query,
                        exc,
                    )
                    continue

                except asyncio.CancelledError:
                    raise

                except Exception:
                    logger.exception(
                        "[youtube] Unexpected error for query %r.", query
                    )
                    continue

                for item_data in data.get("items", []):
                    video_id = item_data.get("id", {}).get("videoId")
                    if not video_id:
                        continue

                    external_id = _make_external_id(video_id)
                    if external_id in seen_ids:
                        continue
                    seen_ids.add(external_id)

                    snippet = item_data.get("snippet", {})
                    title = snippet.get("title", "")
                    channel_title = snippet.get("channelTitle", "")
                    published_at = snippet.get("publishedAt", "")
                    description = snippet.get("description", "")

                    body = json.dumps(
                        {
                            "description": description[:500],
                            "channelTitle": channel_title,
                            "publishedAt": published_at,
                        },
                        ensure_ascii=False,
                    )

                    item = RawItem(
                        source="youtube",
                        external_id=external_id,
                        title=title,
                        body=body,
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        author=channel_title,
                        fetched_at=datetime.now(tz=timezone.utc),
                    )
                    items_ingested_total.labels(source="youtube").inc()
                    yield item
