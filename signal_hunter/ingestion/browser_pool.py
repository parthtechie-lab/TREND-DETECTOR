"""
Headless browser pool manager for resource-safe scraping.

Maintains a single browser instance globally and recycles it after 50 page
allocations or 1 hour of active time to prevent memory leaks.
"""
import asyncio
import logging
import time
import random
from typing import Optional

logger = logging.getLogger(__name__)

class BrowserPool:
    """Headless browser pool manager.

    Maintains a global browser instance and recycles/re-creates it
    after 50 page allocations or 1 hour of lifetime to avoid memory leaks.
    """

    def __init__(self) -> None:
        self._playwright = None
        self._browser = None
        self._pages_created: int = 0
        self._started_at: float = 0.0
        self._lock = asyncio.Lock()

    async def get_page(self, proxy: Optional[str] = None):
        """Acquire a new page from the browser pool, recycling the browser if necessary."""
        from playwright.async_api import async_playwright

        async with self._lock:
            now = time.monotonic()
            # Recycle browser if it exceeds page count limit or lifetime limit
            if (
                self._browser is not None
                and (self._pages_created >= 50 or (now - self._started_at) >= 3600)
            ):
                logger.info(
                    "[BrowserPool] Recycling browser instance (pages=%d, age=%.0fs)...",
                    self._pages_created,
                    now - self._started_at,
                )
                await self._close_unlocked()

            # Launch a new browser instance if one is not active
            if self._browser is None:
                logger.info("[BrowserPool] Launching new browser instance...")
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-blink-features=AutomationControlled",
                    ],
                )
                self._pages_created = 0
                self._started_at = now

            # Select a random User-Agent to rotate footprinting
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
            ]
            ua = random.choice(user_agents)

            # Create context configuration
            context_args = {
                "user_agent": ua,
                "viewport": {"width": 1280, "height": 800},
                "locale": "en-US",
                "timezone_id": "UTC",
            }
            if proxy:
                context_args["proxy"] = {"server": proxy}

            context = await self._browser.new_context(**context_args)
            page = await context.new_page()
            self._pages_created += 1
            return page

    async def release_page(self, page) -> None:
        """Release and close a page and its associated browser context."""
        try:
            context = page.context
            await page.close()
            await context.close()
        except Exception as e:
            logger.error("[BrowserPool] Error releasing page/context: %s", e)

    async def close(self) -> None:
        """Explicitly shut down the browser pool."""
        async with self._lock:
            await self._close_unlocked()

    async def _close_unlocked(self) -> None:
        try:
            if self._browser:
                await self._browser.close()
        except Exception as e:
            logger.error("[BrowserPool] Error closing browser: %s", e)
        finally:
            self._browser = None

        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            logger.error("[BrowserPool] Error stopping playwright: %s", e)
        finally:
            self._playwright = None

        self._pages_created = 0
        self._started_at = 0.0


browser_pool = BrowserPool()
