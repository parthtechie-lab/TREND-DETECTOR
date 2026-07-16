"""Signal Hunter AI — unit tests for Hacker News source poller."""

import pytest
from unittest.mock import patch, AsyncMock

from signal_hunter.ingestion.hacker_news import HackerNewsPoller


@pytest.mark.asyncio
async def test_hacker_news_polling_flow():
    """Verify that the HN poller queries the top stories and maps details correctly."""
    poller = HackerNewsPoller()

    story_ids = [99001]
    story_details = {
        "id": 99001,
        "type": "story",
        "title": "Show HN: OpenHands — Autonomous Healing Crawlers",
        "by": "saas_founder",
        "score": 120,
        "descendants": 24,
        "url": "https://github.com/openhands/openhands",
        "text": "An autonomous self-healing crawler framework."
    }

    async def mock_fetch(session, item_id):
        if item_id == 99001:
            return story_details
        return None

    with patch.object(HackerNewsPoller, "_fetch_item", side_effect=mock_fetch), \
         patch("aiohttp.ClientSession.get") as mock_get:
        
        # Mock GET response for newstories.json
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=story_ids)
        mock_get.return_value.__aenter__.return_value = mock_resp

        items = []
        async for item in poller.poll():
            items.append(item)

        # Assertions
        assert len(items) == 1
        hn_item = items[0]
        assert hn_item.source == "hacker_news"
        assert hn_item.title == "Show HN: OpenHands — Autonomous Healing Crawlers"
        assert hn_item.author == "saas_founder"
        assert hn_item.url == "https://github.com/openhands/openhands"
        
        # Verify JSON body structure
        import json
        body_data = json.loads(hn_item.body)
        assert body_data["score"] == 120
        assert body_data["comments_count"] == 24
        assert body_data["author"] == "saas_founder"
        assert body_data["item_id"] == 99001
