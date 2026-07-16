"""
Reddit source poller using asyncpraw.

Polls new and hot posts from a curated list of SaaS / startup / AI
subreddits every 5 minutes.  Each post is deduplicated by a SHA-256
digest of the post ID and subreddit name so the same post seen in both
`new` and `hot` listings is only emitted once per cycle.
"""
import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional, Set

import asyncpraw
import asyncpraw.exceptions
import asyncprawcore.exceptions

from signal_hunter.core.config import settings
from signal_hunter.core.observability import items_ingested_total
from signal_hunter.ingestion.base import RawItem, SourcePoller

logger = logging.getLogger(__name__)

_POSTS_PER_LISTING: int = 25
_RATE_LIMIT_SLEEP: int = 60  # seconds to sleep on HTTP 429


def _make_external_id(post_id: str, subreddit: str) -> str:
    """Return a stable SHA-256 hex digest for a Reddit post."""
    payload = f"{post_id}:{subreddit}".encode()
    return hashlib.sha256(payload).hexdigest()


def _build_body(post: asyncpraw.models.Submission, subreddit: str) -> str:
    """Serialise relevant post metadata to a JSON string."""
    return json.dumps(
        {
            "selftext": (post.selftext or "")[:500],
            "score": post.score,
            "num_comments": post.num_comments,
            "flair": post.link_flair_text,
            "subreddit": subreddit,
        },
        ensure_ascii=False,
    )


class RedditPoller(SourcePoller):
    """Polls Reddit via asyncpraw for startup / AI-related posts."""

    # ------------------------------------------------------------------
    # SourcePoller interface
    # ------------------------------------------------------------------

    @property
    def source_name(self) -> str:
        return "reddit"

    @property
    def poll_interval_seconds(self) -> int:
        return 300  # 5 minutes

    @property
    def tier(self) -> str:
        return "A"

    # ------------------------------------------------------------------
    # Lazy asyncpraw client
    # ------------------------------------------------------------------

    _client: Optional[asyncpraw.Reddit] = None

    def _get_client(self) -> asyncpraw.Reddit:
        """Return the shared asyncpraw.Reddit instance, creating it lazily."""
        if self._client is None:
            self._client = asyncpraw.Reddit(
                client_id=settings.REDDIT_CLIENT_ID,
                client_secret=settings.REDDIT_CLIENT_SECRET,
                user_agent=settings.REDDIT_USER_AGENT,
                # Read-only mode — no username/password needed for public posts.
                ratelimit_seconds=5,
            )
            logger.info("[reddit] asyncpraw client initialised (read-only).")
        return self._client

    # ------------------------------------------------------------------
    # Core polling logic
    # ------------------------------------------------------------------

    async def poll(self) -> AsyncIterator[RawItem]:  # type: ignore[override]
        """
        Yield RawItems from the 'new' and 'hot' listings of every
        configured subreddit.

        Deduplication within a single cycle is handled via a local set of
        seen external_ids so that a post appearing in both listings is
        only yielded once.
        """
        client = self._get_client()
        seen_ids: Set[str] = set()

        from signal_hunter.ingestion.quota_tracker import quota_tracker

        for subreddit_name in settings.subreddits:
            for listing in ("new", "hot"):
                if not quota_tracker.has_quota("reddit", 1):
                    logger.warning("[reddit] Quota limit reached. Skipping subreddits queries.")
                    return

                try:
                    quota_tracker.consume("reddit", 1)
                    subreddit = await client.subreddit(subreddit_name)
                    listing_fn = (
                        subreddit.new if listing == "new" else subreddit.hot
                    )
                    async for post in listing_fn(limit=_POSTS_PER_LISTING):
                        external_id = _make_external_id(post.id, subreddit_name)
                        if external_id in seen_ids:
                            continue
                        seen_ids.add(external_id)

                        item = RawItem(
                            source="reddit",
                            external_id=external_id,
                            title=post.title or "",
                            body=_build_body(post, subreddit_name),
                            url=f"https://www.reddit.com{post.permalink}",
                            author=str(post.author) if post.author else "[deleted]",
                            fetched_at=datetime.now(tz=timezone.utc),
                        )
                        items_ingested_total.labels(source="reddit").inc()
                        yield item

                except asyncpraw.exceptions.RedditAPIException as exc:
                    # Reddit-level API errors (e.g. invalid subreddit, auth issues)
                    logger.error(
                        "[reddit] RedditAPIException for r/%s (%s): %s",
                        subreddit_name,
                        listing,
                        exc,
                    )
                    # Continue to the next subreddit rather than aborting the cycle.
                    continue

                except asyncprawcore.exceptions.ResponseException as exc:
                    if exc.response.status == 429:
                        logger.warning(
                            "[reddit] Rate-limited (HTTP 429) on r/%s — sleeping %ds.",
                            subreddit_name,
                            _RATE_LIMIT_SLEEP,
                        )
                        await asyncio.sleep(_RATE_LIMIT_SLEEP)
                    else:
                        logger.error(
                            "[reddit] ResponseException for r/%s (%s): HTTP %s — %s",
                            subreddit_name,
                            listing,
                            exc.response.status,
                            exc,
                        )
                    continue

                except asyncio.CancelledError:
                    # Propagate cancellation immediately — do not swallow it.
                    raise

                except Exception:
                    logger.exception(
                        "[reddit] Unexpected error polling r/%s (%s).",
                        subreddit_name,
                        listing,
                    )
                    continue
