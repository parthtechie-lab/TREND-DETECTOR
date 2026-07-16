"""
TikTok Tier-C Playwright poller.

Uses a headless Chromium browser to scrape trending videos from
TikTok's public trending page.  This poller is gated behind the
``settings.PLAYWRIGHT_ENABLED`` flag because it consumes significantly
more memory and CPU than API-based pollers.

Safety guardrails
-----------------
* CAPTCHA detection — if a CAPTCHA iframe is detected, the browser is
  closed immediately and the cycle is abandoned to avoid triggering
  automated-access detection.
* Random human-like delays between interactions prevent triggering
  TikTok's bot-detection heuristics.
* A TimeoutError while waiting for the video card selector is treated as
  a degraded cycle (the page layout may have changed); we log a warning
  and return without crashing.
"""
import asyncio
import hashlib
import json
import logging
import random
from datetime import datetime, timezone
from typing import AsyncIterator

from signal_hunter.core.config import settings
from signal_hunter.core.observability import items_ingested_total
from signal_hunter.ingestion.base import RawItem, SourcePoller

logger = logging.getLogger(__name__)

_TIKTOK_TRENDING_URL = "https://www.tiktok.com/trending"
_MAX_VIDEOS: int = 15
_PAGE_TIMEOUT_MS: int = 30_000      # wait for page load
_ELEMENT_TIMEOUT_MS: int = 15_000   # wait for video cards
_CAPTCHA_SELECTOR = "iframe[src*='captcha']"

# Selectors for video cards — TikTok changes these periodically.
_CARD_SELECTORS = [
    "[data-e2e='trending-item']",
    "[data-e2e='video-card']",
    "div.tiktok-trending-item",
    "div[class*='DivItemContainer']",
]

# Selectors to extract the video title / description from a card.
_TITLE_SELECTORS = [
    "[aria-label]",
    "[data-e2e='video-desc']",
    "span[class*='SpanText']",
    "p[class*='PDesc']",
]


def _make_external_id(key: str) -> str:
    """Return a stable SHA-256 hex digest for a TikTok video URL or title."""
    return hashlib.sha256(key.encode()).hexdigest()


async def _human_sleep(min_s: float = 0.5, max_s: float = 3.0) -> None:
    """Await a random delay to mimic human interaction pacing."""
    await asyncio.sleep(random.uniform(min_s, max_s))


class TikTokPoller(SourcePoller):
    """
    Tier-C Playwright-based poller that scrapes TikTok trending videos.

    Only active when ``settings.PLAYWRIGHT_ENABLED`` is truthy.
    """

    # ------------------------------------------------------------------
    # SourcePoller interface
    # ------------------------------------------------------------------

    @property
    def source_name(self) -> str:
        return "tiktok"

    @property
    def poll_interval_seconds(self) -> int:
        return 1800  # 30 minutes — browser is expensive

    @property
    def tier(self) -> str:
        return "C"

    # ------------------------------------------------------------------
    # Core polling logic
    # ------------------------------------------------------------------

    async def poll(self) -> AsyncIterator[RawItem]:  # type: ignore[override]
        """
        Yield up to _MAX_VIDEOS RawItems from TikTok's trending page.

        If ``settings.PLAYWRIGHT_ENABLED`` is falsy the generator logs a
        warning and returns immediately without launching a browser.
        """
        if not settings.PLAYWRIGHT_ENABLED:
            logger.warning(
                "[tiktok] PLAYWRIGHT_ENABLED is False — skipping browser poll. "
                "Set PLAYWRIGHT_ENABLED=true to activate this poller."
            )
            return

        # Playwright is imported lazily so that environments without it
        # installed don't fail at import time when the poller is disabled.
        try:
            from playwright.async_api import (
                TimeoutError as PlaywrightTimeoutError,
                async_playwright,
            )
        except ImportError:
            logger.error(
                "[tiktok] playwright package is not installed. "
                "Install it with: pip install playwright && playwright install chromium"
            )
            return

        from signal_hunter.ingestion.browser_pool import browser_pool

        # Acquire a page first — if this fails, abort immediately.
        page = None
        try:
            page = await browser_pool.get_page(proxy=self.get_proxy())
        except Exception as e:
            logger.error("[tiktok] Failed to acquire page from browser pool: %s", e)
            self._on_failure()
            return

        # -------------------------------------------------------------------
        # Navigation and scraping — runs only when page acquisition succeeded.
        # The try/finally guarantees the page is always released back to the
        # pool regardless of what happens inside.
        # -------------------------------------------------------------------
        try:
            # --------------------------------------------------------
            # Navigate to TikTok trending
            # --------------------------------------------------------
            logger.info("[tiktok] Navigating to %s", _TIKTOK_TRENDING_URL)
            await page.goto(
                _TIKTOK_TRENDING_URL,
                timeout=_PAGE_TIMEOUT_MS,
                wait_until="domcontentloaded",
            )
            await _human_sleep(1.0, 2.5)

            # --------------------------------------------------------
            # CAPTCHA check
            # --------------------------------------------------------
            captcha_found = await page.query_selector(_CAPTCHA_SELECTOR)
            if captcha_found:
                logger.warning(
                    "[tiktok] CAPTCHA iframe detected — aborting cycle to avoid "
                    "triggering bot-detection. Will retry next cycle."
                )
                return

            # --------------------------------------------------------
            # Wait for video cards
            # --------------------------------------------------------
            card_selector: str | None = None
            for selector in _CARD_SELECTORS:
                try:
                    await page.wait_for_selector(
                        selector, timeout=_ELEMENT_TIMEOUT_MS
                    )
                    card_selector = selector
                    logger.debug(
                        "[tiktok] Video card selector matched: %s", selector
                    )
                    break
                except PlaywrightTimeoutError:
                    continue

            if card_selector is None:
                logger.warning(
                    "[tiktok] No video card selector matched within %dms — "
                    "the page layout may have changed. Marking cycle as degraded.",
                    _ELEMENT_TIMEOUT_MS,
                )
                self._on_failure()
                return

            # --------------------------------------------------------
            # Extract video items
            # --------------------------------------------------------
            cards = await page.query_selector_all(card_selector)
            logger.info(
                "[tiktok] Found %d video cards (extracting up to %d).",
                len(cards),
                _MAX_VIDEOS,
            )

            yielded = 0
            for card in cards[:_MAX_VIDEOS]:
                await _human_sleep(0.5, 3.0)

                title = ""
                description = ""

                # Try multiple selectors to extract the title.
                for title_sel in _TITLE_SELECTORS:
                    el = await card.query_selector(title_sel)
                    if el:
                        text = (await el.inner_text()).strip()
                        if text:
                            title = text
                            break

                # Fallback: read aria-label directly on the card element.
                if not title:
                    aria = await card.get_attribute("aria-label")
                    if aria:
                        title = aria.strip()

                if not title:
                    logger.debug(
                        "[tiktok] Could not extract title from card — skipping."
                    )
                    continue

                # Try to grab a description (may be the same as title on TikTok).
                desc_el = await card.query_selector("[data-e2e='video-desc']")
                if desc_el:
                    description = (await desc_el.inner_text()).strip()

                # Try to extract a link for the URL field.
                # Use the video URL as the stable external_id (more reliable than
                # the title, which can vary across scrapes for the same video).
                url = _TIKTOK_TRENDING_URL  # fallback
                link_el = await card.query_selector("a[href*='/video/']")
                if link_el:
                    href = await link_el.get_attribute("href")
                    if href:
                        url = (
                            href
                            if href.startswith("http")
                            else f"https://www.tiktok.com{href}"
                        )

                # Prefer URL-based dedup key; fall back to title hash only when
                # no specific video URL could be extracted.
                dedup_key = url if url != _TIKTOK_TRENDING_URL else title
                external_id = _make_external_id(dedup_key)

                body = json.dumps(
                    {"description": description or title},
                    ensure_ascii=False,
                )

                item = RawItem(
                    source="tiktok",
                    external_id=external_id,
                    title=title,
                    body=body,
                    url=url,
                    author="",  # Not readily available from the trending page
                    fetched_at=datetime.now(tz=timezone.utc),
                )
                items_ingested_total.labels(source="tiktok").inc()
                yield item
                yielded += 1

            logger.info("[tiktok] Emitted %d items this cycle.", yielded)

        except PlaywrightTimeoutError:
            logger.warning(
                "[tiktok] Playwright TimeoutError while loading page — "
                "marking cycle as degraded."
            )
            self._on_failure()
            return

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception("[tiktok] Unexpected error during Playwright session.")
            raise  # re-raise so the scheduler can handle it

        finally:
            # Always release the page back to the pool to prevent leaks.
            if page is not None:
                await browser_pool.release_page(page)
                logger.debug("[tiktok] Page released back to browser pool.")
