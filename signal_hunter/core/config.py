"""Signal Hunter AI — centralised application settings.

All configuration is read from environment variables and/or a .env file at
startup.  Every module that needs a setting should import the pre-built
``settings`` singleton at the bottom of this module rather than constructing
its own ``Settings()`` instance.
"""

from __future__ import annotations

import functools
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings backed by environment variables / .env."""

    # ------------------------------------------------------------------ #
    # Reddit OAuth credentials                                             #
    # ------------------------------------------------------------------ #
    REDDIT_CLIENT_ID: str = Field(default="", description="Reddit OAuth2 client ID.")
    REDDIT_CLIENT_SECRET: str = Field(
        default="", description="Reddit OAuth2 client secret."
    )
    REDDIT_USERNAME: str = Field(
        default="", description="Reddit account username for script-type apps."
    )
    REDDIT_PASSWORD: str = Field(
        default="", description="Reddit account password for script-type apps."
    )
    REDDIT_USER_AGENT: str = Field(
        default="signal-hunter/1.0",
        description="User-Agent string sent with every Reddit API request.",
    )

    # ------------------------------------------------------------------ #
    # YouTube Data API v3                                                  #
    # ------------------------------------------------------------------ #
    YOUTUBE_API_KEY: str = Field(default="", description="Google / YouTube Data API v3 key.")

    # ------------------------------------------------------------------ #
    # Product Hunt API                                                     #
    # ------------------------------------------------------------------ #
    PRODUCT_HUNT_API_KEY: str = Field(default="", description="Product Hunt API key.")
    PRODUCT_HUNT_API_SECRET: str = Field(
        default="", description="Product Hunt API secret."
    )

    # ------------------------------------------------------------------ #
    # OpenAI / LLM                                                         #
    # ------------------------------------------------------------------ #
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key.")
    OPENAI_BASE_URL: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the OpenAI-compatible API endpoint.",
    )
    OPENAI_FALLBACK_API_KEY: str = Field(default="", description="OpenAI fallback API key.")
    OPENAI_FALLBACK_BASE_URL: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the fallback API endpoint.",
    )
    FALLBACK_MODEL: str = Field(
        default="gpt-4o-mini",
        description="Model used for fallback classification.",
    )
    CLASSIFIER_MODEL: str = Field(
        default="gpt-4o-mini",
        description="Model used for fast classification / scoring.",
    )
    CLASSIFIER_RATE_LIMIT_RPM: int = Field(
        default=60,
        ge=1,
        description="Max requests-per-minute for the OpenAI classifier (token-bucket rate limiter).",
    )
    SUMMARIZER_MODEL: str = Field(
        default="gpt-4o",
        description="Model used for high-quality summarisation and evidence extraction.",
    )

    # ------------------------------------------------------------------ #
    # Telegram                                                             #
    # ------------------------------------------------------------------ #
    TELEGRAM_BOT_TOKEN: str = Field(
        default="", description="Telegram bot token issued by @BotFather."
    )
    TELEGRAM_CHAT_ID: str = Field(
        default="",
        description="Target chat / channel ID where alerts are delivered.",
    )

    # ------------------------------------------------------------------ #
    # Database                                                             #
    # ------------------------------------------------------------------ #
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./signal_hunter.db",
        description="SQLAlchemy async database URL.",
    )

    # ------------------------------------------------------------------ #
    # Confidence thresholds                                                #
    # ------------------------------------------------------------------ #
    CONFIDENCE_THRESHOLD_ALERT: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score required to trigger a Telegram alert.",
    )
    CONFIDENCE_THRESHOLD_STORE: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score required to persist a classification.",
    )

    # ------------------------------------------------------------------ #
    # Alerting                                                             #
    # ------------------------------------------------------------------ #
    ALERT_BATCH_WINDOW_SECONDS: int = Field(
        default=300,
        ge=0,
        description="Seconds to wait before flushing a batch of alerts.",
    )

    # ------------------------------------------------------------------ #
    # Queue sizing                                                         #
    # ------------------------------------------------------------------ #
    RAW_QUEUE_MAXSIZE: int = Field(
        default=500,
        ge=1,
        description="Max items held in the raw-ingestion queue before back-pressure.",
    )
    UNIQUE_QUEUE_MAXSIZE: int = Field(
        default=200,
        ge=1,
        description="Max items held in the de-duplicated queue.",
    )
    SCORED_QUEUE_MAXSIZE: int = Field(
        default=200,
        ge=1,
        description="Max items held in the scored / classified queue.",
    )

    # ------------------------------------------------------------------ #
    # Worker concurrency                                                   #
    # ------------------------------------------------------------------ #
    AI_WORKER_COUNT: int = Field(
        default=3,
        ge=1,
        description="Number of concurrent AI classification worker coroutines.",
    )

    # ------------------------------------------------------------------ #
    # Semantic de-duplication                                              #
    # ------------------------------------------------------------------ #
    SEMANTIC_DEDUP_THRESHOLD: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Cosine-similarity threshold above which two items are considered duplicates.",
    )
    SEMANTIC_WINDOW_SIZE: int = Field(
        default=500,
        ge=1,
        description="How many recent items to keep in the rolling embedding window.",
    )

    # ------------------------------------------------------------------ #
    # Corroboration                                                        #
    # ------------------------------------------------------------------ #
    CORROBORATION_MIN_SOURCES: int = Field(
        default=2,
        ge=1,
        description="Minimum distinct sources needed to corroborate a signal.",
    )
    CORROBORATION_WINDOW_HOURS: int = Field(
        default=48,
        ge=1,
        description="Look-back window (hours) when counting corroborating sources.",
    )

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    # ------------------------------------------------------------------ #
    # Feature flags                                                        #
    # ------------------------------------------------------------------ #
    PLAYWRIGHT_ENABLED: bool = Field(
        default=False,
        description="Enable Playwright-based scrapers for JS-rendered pages.",
    )

    # ------------------------------------------------------------------ #
    # Server ports                                                         #
    # ------------------------------------------------------------------ #
    METRICS_PORT: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Port on which the Prometheus metrics HTTP server listens.",
    )
    DASHBOARD_PORT: int = Field(
        default=8080,
        ge=1024,
        le=65535,
        description="Port on which the optional web dashboard listens.",
    )

    WATCHLIST_FILE: str = Field(
        default="",
        description="Path to the JSON watchlist configuration file. Defaults to watchlist.json next to the package root.",
    )

    def _resolve_watchlist_path(self) -> Path:
        """Resolve the watchlist file path, falling back to the package-relative default."""
        if self.WATCHLIST_FILE:
            return Path(self.WATCHLIST_FILE)
        # Resolve relative to this config file: signal_hunter/core/config.py → signal_hunter/ → watchlist.json
        return Path(__file__).parent.parent / "watchlist.json"

    @functools.cached_property
    def watchlist(self) -> dict:
        """Load and parse the watchlist JSON configuration file (cached after first read)."""
        import json
        path = self._resolve_watchlist_path()
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    @property
    def subreddits(self) -> list[str]:
        """Dynamic list of subreddits to monitor from the watchlist config."""
        wl = self.watchlist
        if wl and "subreddits" in wl:
            return wl["subreddits"]
        return ["SaaS", "startups", "artificial", "MachineLearning", "Entrepreneur", "ProductHunter", "webdev"]

    @property
    def youtube_queries(self) -> list[str]:
        """Dynamic list of YouTube search queries to monitor from the watchlist config."""
        wl = self.watchlist
        if wl and "youtube_queries" in wl:
            return wl["youtube_queries"]
        return ["AI tools 2025", "SaaS startup launch", "new AI software", "tech product launch 2025"]

    # ------------------------------------------------------------------ #
    # Pydantic-settings configuration                                      #
    # ------------------------------------------------------------------ #
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere instead of Settings()
# ---------------------------------------------------------------------------
settings: Settings = Settings()
