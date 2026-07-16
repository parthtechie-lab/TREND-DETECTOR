"""
Product Hunt GraphQL source poller using aiohttp.

Fetches today's top posts via the official Product Hunt v2 GraphQL API,
enriching each post with its tagline, description, vote count, and topic
tags.
"""
import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

import aiohttp

from signal_hunter.core.config import settings
from signal_hunter.core.observability import items_ingested_total
from signal_hunter.ingestion.base import RawItem, SourcePoller

logger = logging.getLogger(__name__)

_GRAPHQL_ENDPOINT = "https://api.producthunt.com/v2/api/graphql"
_REQUEST_TIMEOUT_SECONDS: int = 30
_RATE_LIMIT_SLEEP: int = 120  # seconds to back off on HTTP 429

# Fetch today's posts ordered by ranking, including topic tags.
_GQL_QUERY = """
query TodaysPosts($first: Int!) {
  posts(first: $first, order: RANKING) {
    edges {
      node {
        id
        slug
        name
        tagline
        description
        url
        votesCount
        createdAt
        topics {
          edges {
            node {
              name
            }
          }
        }
      }
    }
  }
}
"""

_POSTS_PER_CYCLE: int = 30


def _make_external_id(slug: str) -> str:
    """Return a stable SHA-256 hex digest for a Product Hunt post slug."""
    return hashlib.sha256(slug.encode()).hexdigest()


def _extract_topics(post_node: dict) -> list[str]:
    """Safely extract topic names from the GraphQL edges structure."""
    try:
        return [
            edge["node"]["name"]
            for edge in post_node.get("topics", {}).get("edges", [])
        ]
    except (KeyError, TypeError):
        return []


class ProductHuntPoller(SourcePoller):
    """Polls the Product Hunt GraphQL API for today's top posts."""

    # ------------------------------------------------------------------
    # SourcePoller interface
    # ------------------------------------------------------------------

    @property
    def source_name(self) -> str:
        return "product_hunt"

    @property
    def poll_interval_seconds(self) -> int:
        return 900  # 15 minutes — PH posts don't arrive that frequently

    @property
    def tier(self) -> str:
        return "A"

    # ------------------------------------------------------------------
    # Core polling logic
    # ------------------------------------------------------------------

    async def poll(self) -> AsyncIterator[RawItem]:  # type: ignore[override]
        """Yield RawItems for today's Product Hunt posts."""
        headers = {
            "Authorization": f"Bearer {settings.PRODUCT_HUNT_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "query": _GQL_QUERY,
            "variables": {"first": _POSTS_PER_CYCLE},
        }

        timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(
                    _GRAPHQL_ENDPOINT,
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status == 401:
                        logger.error(
                            "[product_hunt] HTTP 401 — invalid or expired API key. "
                            "Skipping cycle."
                        )
                        return

                    if response.status == 429:
                        logger.warning(
                            "[product_hunt] HTTP 429 — rate limited. "
                            "Sleeping %ds before next cycle.",
                            _RATE_LIMIT_SLEEP,
                        )
                        await asyncio.sleep(_RATE_LIMIT_SLEEP)
                        return

                    response.raise_for_status()
                    data = await response.json()

            except aiohttp.ClientResponseError as exc:
                logger.error(
                    "[product_hunt] HTTP error: %s %s", exc.status, exc.message
                )
                return

            except aiohttp.ClientError as exc:
                logger.error("[product_hunt] Network error: %s", exc)
                return

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception("[product_hunt] Unexpected error during request.")
                return

        # ------------------------------------------------------------------
        # Parse the GraphQL response
        # ------------------------------------------------------------------
        errors = data.get("errors")
        if errors:
            logger.error("[product_hunt] GraphQL errors: %s", errors)
            return

        edges = (
            data.get("data", {}).get("posts", {}).get("edges", []) or []
        )
        if not edges:
            logger.info("[product_hunt] No posts returned in this cycle.")
            return

        for edge in edges:
            post = edge.get("node")
            if not post:
                continue

            slug = post.get("slug") or post.get("id", "")
            if not slug:
                logger.debug("[product_hunt] Skipping post with no slug/id.")
                continue

            external_id = _make_external_id(slug)
            topics = _extract_topics(post)

            body = json.dumps(
                {
                    "tagline": post.get("tagline", ""),
                    "description": (post.get("description") or "")[:500],
                    "votesCount": post.get("votesCount", 0),
                    "topics": topics,
                },
                ensure_ascii=False,
            )

            item = RawItem(
                source="product_hunt",
                external_id=external_id,
                title=post.get("name", ""),
                body=body,
                url=post.get("url", f"https://www.producthunt.com/posts/{slug}"),
                author="",  # PH API does not expose maker names at list level
                fetched_at=datetime.now(tz=timezone.utc),
            )
            items_ingested_total.labels(source="product_hunt").inc()
            yield item
