"""Signal Hunter AI — Demo Runner.

Launches the application in an interactive mock mode. Automatically patches:
1. Ingestion pollers to inject realistic, fast-paced trend signals (including
   semantic duplicates to demonstrate embedding matching).
2. The OpenAI client to return grounded, structured, and validated JSON outputs.
3. Telegram dispatcher to safely log and print warnings instead of crashing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import sys
from datetime import datetime, timezone
from typing import AsyncIterator, List

# Add parent directory to python path if run directly
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

# Mock out heavy and uninstalled libraries BEFORE importing application modules
from unittest.mock import MagicMock, AsyncMock
import numpy as np

sys.modules['asyncpraw'] = MagicMock()
sys.modules['asyncpraw.exceptions'] = MagicMock()
sys.modules['asyncprawcore'] = MagicMock()
sys.modules['asyncprawcore.exceptions'] = MagicMock()

# Mock sentence-transformers to return realistic embeddings based on keyword detection
class MockSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass
    def encode(self, text, *args, **kwargs):
        vec = np.zeros(384, dtype=np.float32)
        text_lower = text.lower()
        if "devin" in text_lower:
            vec[0] = 1.0
        elif "cursor" in text_lower:
            vec[1] = 1.0
        elif "bolt.new" in text_lower:
            vec[2] = 1.0
        elif "vllm" in text_lower:
            vec[3] = 1.0
        elif "streamlit" in text_lower:
            vec[4] = 1.0
        elif "langchain" in text_lower:
            vec[5] = 1.0
        else:
            # Deterministic hash of the text to a mock index between 6 and 380
            import hashlib
            idx = int(hashlib.md5(text.encode()).hexdigest(), 16) % 375 + 6
            vec[idx] = 1.0
        return vec

mock_st_mod = MagicMock()
mock_st_mod.SentenceTransformer = MockSentenceTransformer
sys.modules['sentence_transformers'] = mock_st_mod

sys.modules['playwright'] = MagicMock()
sys.modules['playwright.async_api'] = MagicMock()

# Import app components
import signal_hunter.main as app_main
from signal_hunter.core.config import settings
from signal_hunter.ingestion.base import SourcePoller, RawItem
from signal_hunter.ai.models import ItemCategory
from signal_hunter.alerting.dispatcher import dispatcher

logger = logging.getLogger("signal_hunter.demo")

# ---------------------------------------------------------------------------
# Simulated Ingestion Feed Data
# ---------------------------------------------------------------------------
# Includes semantic duplicates to demonstrate embedding matching!
MOCK_FEED = [
    # Signal 1: Devin AI (AI_TOOL)
    {
        "source": "reddit",
        "title": "Devin AI has completely blown my mind",
        "body": "I just watched the Devin demo. An AI that can write code, fix bugs, and deploy models independently is insane.",
        "url": "https://reddit.com/r/artificial/comments/devin",
        "author": "dev_pioneer"
    },
    {
        "source": "youtube",
        "title": "Devin: The Autonomous AI Software Engineer",
        "body": "Cognition launched Devin, the first autonomous AI agent capable of engineering software from scratch.",
        "url": "https://youtube.com/watch?v=devin_ai",
        "author": "TechVlog"
    },
    # Signal 2: Cursor (AI_TOOL)
    {
        "source": "reddit",
        "title": "Why Cursor is replacing VS Code for me",
        "body": "The Copilot++ tab autocomplete and natural language edits in Cursor editor are way ahead of standard copilot.",
        "url": "https://reddit.com/r/webdev/comments/cursor",
        "author": "js_ninja"
    },
    {
        "source": "product_hunt",
        "title": "Cursor Code Editor",
        "body": "An AI-first code editor designed for pair programming. Auto-composes entire code blocks based on context.",
        "url": "https://producthunt.com/posts/cursor",
        "author": "anysphere"
    },
    # Signal 3: Bolt.new (SAAS)
    {
        "source": "youtube",
        "title": "Building a SaaS in 5 minutes with Bolt.new",
        "body": "Bolt.new allows you to spin up a full-stack Next.js app in seconds with zero config. Complete walkthrough and tutorial.",
        "url": "https://youtube.com/watch?v=boltnew",
        "author": "CodeCraft"
    },
    {
        "source": "reddit",
        "title": "Bolt.new is actually wild",
        "body": "You can prompt, build, run, and deploy full stack web apps directly in the browser with Bolt.new. No local node_modules required.",
        "url": "https://reddit.com/r/SaaS/comments/boltnew",
        "author": "startup_guy"
    },
    # Signal 4: vLLM (AI_TOOL)
    {
        "source": "tiktok",
        "title": "vLLM is supercharging LLM inference speeds!",
        "body": "If you are running open weights models like Llama-3, you must use vLLM. The PagedAttention memory management decreases latency.",
        "url": "https://tiktok.com/@techguru/video/vllm",
        "author": "ml_guru"
    },
    # Signal 5: Streamlit (OTHER)
    {
        "source": "reddit",
        "title": "Streamlit makes internal tools so easy",
        "body": "We built a company dashboard using Streamlit in two hours. Simple Python code, very clean UI.",
        "url": "https://reddit.com/r/SaaS/comments/streamlit",
        "author": "data_wizard"
    },
    # Signal 6: LangChain (AI_TOOL)
    {
        "source": "product_hunt",
        "title": "LangChain v0.3 Launch",
        "body": "Production ready LLM orchestration. Standardized tool calling, streaming improvements, and simplified state management.",
        "url": "https://producthunt.com/posts/langchain",
        "author": "hwchase17"
    }
]

# ---------------------------------------------------------------------------
# Mock OpenAI Client & Response Generator
# ---------------------------------------------------------------------------
class MockChatCompletions:
    async def create(self, model, response_format=None, messages=None, **kwargs):
        # Extract user prompt content
        user_message = next(msg["content"] for msg in messages if msg["role"] == "user")
        
        # Determine which mock signal we are processing
        product_name = None
        category = ItemCategory.OTHER
        evidence_quote = None
        confidence = 0.3
        trend_signal_present = False

        # Grounding match: Find a matching keyword and build a verbatim quote
        if "devin" in user_message.lower():
            product_name = "Devin AI"
            category = ItemCategory.AI_TOOL
            if "blown my mind" in user_message.lower():
                evidence_quote = "An AI that can write code, fix bugs, and deploy models independently is insane."
            else:
                evidence_quote = "the first autonomous AI agent capable of engineering software from scratch."
            confidence = 0.95
            trend_signal_present = True
        elif "cursor" in user_message.lower():
            product_name = "Cursor Editor"
            category = ItemCategory.AI_TOOL
            if "replacing vs code" in user_message.lower():
                evidence_quote = "The Copilot++ tab autocomplete and natural language edits in Cursor editor are way ahead"
            else:
                evidence_quote = "An AI-first code editor designed for pair programming."
            confidence = 0.92
            trend_signal_present = True
        elif "bolt.new" in user_message.lower():
            product_name = "Bolt.new"
            category = ItemCategory.SAAS
            if "5 minutes" in user_message.lower():
                evidence_quote = "spin up a full-stack Next.js app in seconds with zero config."
            else:
                evidence_quote = "You can prompt, build, run, and deploy full stack web apps directly in the browser"
            confidence = 0.89
            trend_signal_present = True
        elif "vllm" in user_message.lower():
            product_name = "vLLM"
            category = ItemCategory.AI_TOOL
            evidence_quote = "The PagedAttention memory management decreases latency."
            confidence = 0.87
            trend_signal_present = True
        elif "streamlit" in user_message.lower():
            product_name = "Streamlit"
            category = ItemCategory.OTHER
            evidence_quote = "We built a company dashboard using Streamlit in two hours."
            confidence = 0.78
            trend_signal_present = True
        elif "langchain" in user_message.lower():
            product_name = "LangChain"
            category = ItemCategory.AI_TOOL
            evidence_quote = "Standardized tool calling, streaming improvements, and simplified state management."
            confidence = 0.82
            trend_signal_present = True

        # Construct raw response JSON matching classifier system expectations
        response_data = {
            "product_name": product_name,
            "category": category.value,
            "evidence_quote": evidence_quote,
            "confidence": confidence,
            "trend_signal_present": trend_signal_present
        }

        # Mock OpenAI Choice response structure
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(response_data)
        
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 150
        mock_usage.completion_tokens = 45

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        # Sleep briefly to simulate network roundtrip latency
        await asyncio.sleep(0.3)
        return mock_response

class MockOpenAIClient:
    def __init__(self, *args, **kwargs):
        self.chat = MagicMock()
        self.chat.completions = MockChatCompletions()

# ---------------------------------------------------------------------------
# Mock Source Pollers
# ---------------------------------------------------------------------------
class MockRedditPoller(SourcePoller):
    @property
    def source_name(self) -> str: return "reddit"
    @property
    def poll_interval_seconds(self) -> int: return 10
    @property
    def tier(self) -> str: return "A"
    
    async def poll(self) -> AsyncIterator[RawItem]:
        # Filter mock items for reddit
        items = [x for x in MOCK_FEED if x["source"] == "reddit"]
        # Select one at random
        item = random.choice(items)
        yield RawItem(
            source=self.source_name,
            external_id=f"mock_{self.source_name}_{random.randint(1000, 9999)}",
            title=item["title"],
            body=item["body"],
            url=item["url"],
            author=item["author"],
            fetched_at=datetime.now(timezone.utc)
        )

class MockYouTubePoller(SourcePoller):
    @property
    def source_name(self) -> str: return "youtube"
    @property
    def poll_interval_seconds(self) -> int: return 12
    @property
    def tier(self) -> str: return "A"
    
    async def poll(self) -> AsyncIterator[RawItem]:
        items = [x for x in MOCK_FEED if x["source"] == "youtube"]
        item = random.choice(items)
        yield RawItem(
            source=self.source_name,
            external_id=f"mock_{self.source_name}_{random.randint(1000, 9999)}",
            title=item["title"],
            body=item["body"],
            url=item["url"],
            author=item["author"],
            fetched_at=datetime.now(timezone.utc)
        )

class MockProductHuntPoller(SourcePoller):
    @property
    def source_name(self) -> str: return "product_hunt"
    @property
    def poll_interval_seconds(self) -> int: return 15
    @property
    def tier(self) -> str: return "B"
    
    async def poll(self) -> AsyncIterator[RawItem]:
        items = [x for x in MOCK_FEED if x["source"] == "product_hunt"]
        item = random.choice(items)
        yield RawItem(
            source=self.source_name,
            external_id=f"mock_{self.source_name}_{random.randint(1000, 9999)}",
            title=item["title"],
            body=item["body"],
            url=item["url"],
            author=item["author"],
            fetched_at=datetime.now(timezone.utc)
        )

class MockTikTokPoller(SourcePoller):
    @property
    def source_name(self) -> str: return "tiktok"
    @property
    def poll_interval_seconds(self) -> int: return 20
    @property
    def tier(self) -> str: return "C"
    
    async def poll(self) -> AsyncIterator[RawItem]:
        items = [x for x in MOCK_FEED if x["source"] == "tiktok"]
        item = random.choice(items)
        yield RawItem(
            source=self.source_name,
            external_id=f"mock_{self.source_name}_{random.randint(1000, 9999)}",
            title=item["title"],
            body=item["body"],
            url=item["url"],
            author=item["author"],
            fetched_at=datetime.now(timezone.utc)
        )


class MockHackerNewsPoller(SourcePoller):
    @property
    def source_name(self) -> str: return "hacker_news"
    @property
    def poll_interval_seconds(self) -> int: return 18
    @property
    def tier(self) -> str: return "A"
    
    async def poll(self) -> AsyncIterator[RawItem]:
        # Use items from reddit or product_hunt to simulate hacker_news posts
        items = [x for x in MOCK_FEED if x["source"] in ("reddit", "product_hunt")]
        item = random.choice(items)
        yield RawItem(
            source=self.source_name,
            external_id=f"mock_{self.source_name}_{random.randint(1000, 9999)}",
            title=item["title"],
            body=item["body"],
            url=item["url"],
            author=item["author"],
            fetched_at=datetime.now(timezone.utc)
        )

# Mocked Escalated Summarizer function
async def mock_summarize(cluster_sources: list, canonical_title: str) -> str:
    # Build a simple summary explaining the trend
    return (
        f"Simulated analysis for {canonical_title}: "
        f"Early traction observed across multiple sources. Feedback indicates high developer interest "
        f"and strong growth metrics in production-level deployments."
    )

# Mocked Telegram Message sender (to avoid throwing exceptions without API keys)
async def mock_send_message(text: str) -> int:
    msg_id = random.randint(10000, 99999)
    print(f"\n=======================================================")
    print(f"📣 [DEMO TELEGRAM ALERT DISPATCHED (ID #{msg_id})]")
    print(text)
    print(f"=======================================================\n")
    return msg_id

# ---------------------------------------------------------------------------
# Monkey Patching & Seeding
# ---------------------------------------------------------------------------
def patch_application():
    # 1. Override the OpenAI client in the classifier and summarizer modules
    import openai
    openai.AsyncOpenAI = MockOpenAIClient
    
    import signal_hunter.ai.classifier as cls_mod
    cls_mod.AsyncOpenAI = MockOpenAIClient
    # Reset singleton client to re-initialize with mock
    if cls_mod.classifier is not None:
        cls_mod.classifier._client = None
        cls_mod.classifier._get_client()

    import signal_hunter.ai.summarizer as sum_mod
    sum_mod.AsyncOpenAI = MockOpenAIClient
    sum_mod.summarizer.summarize = mock_summarize

    # 2. Patch the Ingestion scheduler to use mock pollers
    import signal_hunter.ingestion.scheduler as sched_mod
    sched_mod.RedditPoller = MockRedditPoller
    sched_mod.YouTubePoller = MockYouTubePoller
    sched_mod.ProductHuntPoller = MockProductHuntPoller
    sched_mod.HackerNewsPoller = MockHackerNewsPoller
    sched_mod.TikTokPoller = MockTikTokPoller
    
    # Speed up stagger start delays for demo responsiveness
    sched_mod._STAGGER_DELAYS = [0, 1, 2, 3, 4]

    # 3. Patch AlertDispatcher to capture and output messages to console instead of Telegram API
    dispatcher.send_message = mock_send_message
    
    # 4. Enable Playwright/TikTok scraper override for demo visibility
    settings.PLAYWRIGHT_ENABLED = True
    
    # Speed up batch digest interval to 30 seconds for quick testing
    settings.ALERT_BATCH_WINDOW_SECONDS = 30
    
    # Reset log level to INFO
    settings.LOG_LEVEL = "INFO"

async def seed_database():
    """Seed initial source health statuses into the local database."""
    from signal_hunter.core.database import AsyncSessionLocal, SourceHealth
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        async with session.begin():
            # Check if source_health already seeded
            stmt = select(SourceHealth)
            result = await session.execute(stmt)
            if not result.scalars().first():
                for source in ["reddit", "youtube", "product_hunt", "tiktok"]:
                    health = SourceHealth(
                        source_name=source,
                        last_success=datetime.now(timezone.utc).replace(tzinfo=None),
                        consecutive_failures=0,
                        total_items_24h=0,
                        is_degraded=False,
                        updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
                    )
                    session.add(health)
                logger.info("Successfully seeded database source health indicators.")

async def start_demo():
    logger.info("Setting up Mock patches for Demo mode...")
    patch_application()

    # Pre-initialize DB tables and seed data
    from signal_hunter.core.database import init_db
    await init_db()
    await seed_database()

    # Print friendly instructions
    print("\n" + "="*80)
    print("🚀 SIGNAL HUNTER AI DEMO RUNNING")
    print("="*80)
    print(f"📊 Web Dashboard is live at:     http://localhost:{settings.DASHBOARD_PORT}")
    print(f"📈 Prometheus Metrics live at:   http://localhost:{settings.METRICS_PORT}/metrics")
    print("💬 Telegram alerts are routed to this console output.")
    print("="*80 + "\n")

    # Start main application gather loop
    await app_main.main()

if __name__ == "__main__":
    try:
        asyncio.run(start_demo())
    except KeyboardInterrupt:
        logger.info("Demo runner terminated by operator.")
