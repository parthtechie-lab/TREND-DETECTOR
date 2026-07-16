"""
Hacker News source poller using the official Firebase API.

Polls top new story items from Hacker News every 5 minutes.
Deduplicates within a single cycle to avoid duplicate submissions.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Set

import aiohttp

from signal_hunter.core.observability import items_ingested_total
from signal_hunter.ingestion.base import RawItem, SourcePoller

logger = logging.getLogger(__name__)

_NEW_STORIES_URL = "https://hacker-news.firebaseio.com/v0/newstories.json"
_ITEM_URL_TEMPLATE = "https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
_MAX_STORIES_PER_CYCLE: int = 30
_REQUEST_TIMEOUT_SECONDS: int = 15

# Keywords for topic pre-filtering. Only stories matching at least one keyword
# (case-insensitive substring match on the title) are sent through the AI pipeline.
# This avoids wasting LLM credits on politics, sports, and other off-topic HN content.
_TOPIC_KEYWORDS: frozenset[str] = frozenset({
    "ai", "llm", "gpt", "ml", "saas", "startup", "launch", "product hunt",
    "software", "tool", "api", "agent", "openai", "claude", "gemini",
    "funding", "seed", "series a", "vc", "venture", "raise", "acquisition",
    "app", "platform", "developer", "devtool", "sdk", "framework",
    "automation", "nocode", "low-code", "b2b", "b2c", "revenue", "mrr", "arr",
    "enterprise", "cloud", "microservices", "neural", "model", "inference",
    "show hn", "ask hn", "open source", "github",
})


def _is_relevant(title: str) -> bool:
    """Return True if the story title matches at least one topic keyword."""
    lower = title.lower()
    return any(kw in lower for kw in _TOPIC_KEYWORDS)


def _make_external_id(item_id: int) -> str:
    """Return a stable SHA-256 hex digest for a Hacker News item ID."""
    return hashlib.sha256(str(item_id).encode()).hexdigest()


class HackerNewsPoller(SourcePoller):
    """Polls the official Hacker News API for new tech and SaaS stories."""

    @property
    def source_name(self) -> str:
        return "hacker_news"

    @property
    def poll_interval_seconds(self) -> int:
        return 300  # 5 minutes

    @property
    def tier(self) -> str:
        return "A"

    async def _fetch_item(self, session: aiohttp.ClientSession, item_id: int) -> dict | None:
        """Fetch item details for a specific Hacker News story ID."""
        url = _ITEM_URL_TEMPLATE.format(item_id=item_id)
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                logger.warning(
                    "[hacker_news] Failed to fetch item %d: HTTP %d",
                    item_id,
                    response.status,
                )
        except Exception as e:
            logger.warning("[hacker_news] Exception fetching item %d: %s", item_id, e)
        return None

    async def poll(self) -> AsyncIterator[RawItem]:  # type: ignore[override]
        """Yield RawItems for recent stories posted on Hacker News."""
        timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS)
        seen_ids: Set[int] = set()

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                # Fetch new story IDs
                async with session.get(_NEW_STORIES_URL) as response:
                    if response.status != 200:
                        logger.error(
                            "[hacker_news] Failed to fetch new stories list: HTTP %d",
                            response.status,
                        )
                        return
                    story_ids = await response.json()
            except Exception as e:
                logger.error("[hacker_news] Exception fetching new stories: %s", e)
                return

            if not story_ids:
                return

            # Target only the most recent N stories
            target_ids = story_ids[:_MAX_STORIES_PER_CYCLE]
            
            # Fetch details in parallel
            tasks = [self._fetch_item(session, item_id) for item_id in target_ids]
            items = await asyncio.gather(*tasks)

            for item in items:
                if not item:
                    continue

                item_id = item.get("id")
                if not item_id or item_id in seen_ids:
                    continue
                seen_ids.add(item_id)

                # We only track stories (not jobs or comments)
                item_type = item.get("type")
                if item_type != "story":
                    continue

                title = item.get("title", "")
                url = item.get("url") or f"https://news.ycombinator.com/item?id={item_id}"
                author = item.get("by", "anonymous")
                score = item.get("score", 0)
                comments_count = item.get("descendants", 0)
                text = item.get("text", "")

                # Skip off-topic stories early to avoid wasting LLM classification credits.
                if not _is_relevant(title):
                    logger.debug("[hacker_news] Skipping off-topic story: %r", title)
                    continue

                # Construct body JSON payload
                body_payload = json.dumps({
                    "text": text[:500],
                    "score": score,
                    "comments_count": comments_count,
                    "author": author,
                    "item_id": item_id
                }, ensure_ascii=False)

                yield RawItem(
                    source=self.source_name,
                    external_id=_make_external_id(item_id),
                    title=title,
                    body=body_payload,
                    url=url,
                    author=author,
                    fetched_at=datetime.now(timezone.utc)
                )

                # Increment ingested metrics
                items_ingested_total.labels(source=self.source_name).inc()
