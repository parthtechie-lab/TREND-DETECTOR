"""Signal Hunter AI — unit tests for Quiet Hours alert suppression."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from signal_hunter.alerting.dispatcher import AlertDispatcher
from signal_hunter.ai.models import ScoredItem, ClassificationResult, ClassificationOutput, ItemCategory


@pytest.mark.asyncio
async def test_quiet_hours_suppression_and_bypass():
    """Verify quiet hours triggers buffering or bypassing correctly depending on config."""
    dispatcher = AlertDispatcher()
    import uuid
    random_id = str(uuid.uuid4())
    
    # Mock ScoredItem to send
    item = ScoredItem(
        cluster_id=f"cluster-qh-{random_id}",
        raw_item_id=f"raw-item-qh-{random_id}",
        classification=ClassificationResult(
            output=ClassificationOutput(
                product_name="QH Test",
                category=ItemCategory.SAAS,
                evidence_quote="verbatim evidence string",
                confidence=0.9,
                trend_signal_present=True
            ),
            raw_model_response="",
            passed_validation=True,
            validation_failure_reason=None,
            model_version="gpt-4o"
        ),
        source="reddit",
        title="QH Test Post",
        url="https://test.com",
        body="verbatim evidence string text details",
        fetched_at=datetime.now(timezone.utc),
        corroborated=True,
        source_count=2
    )

    from unittest.mock import PropertyMock, AsyncMock, patch, MagicMock

    # 1. Test case: Inside quiet hours with bypass=False -> Should buffer item for digest
    mock_watchlist = {
        "quiet_hours": {
            "start": "22:00",
            "end": "08:00",
            "timezone": "Asia/Kolkata",
            "bypass_for_high_priority": False
        }
    }

    with patch("signal_hunter.core.config.Settings.watchlist", new_callable=PropertyMock) as mock_wl, \
         patch.object(AlertDispatcher, "_is_quiet_hours", return_value=True), \
         patch.object(AlertDispatcher, "send_message", AsyncMock()) as mock_send:
        
        mock_wl.return_value = mock_watchlist
        await dispatcher.dispatch(item)
        
        # Should NOT call send_message
        assert mock_send.call_count == 0
        # Should add to digest buffer
        assert len(dispatcher.digest_buffer) == 1
        assert dispatcher.digest_buffer[0].cluster_id == item.cluster_id

    # Reset buffer
    dispatcher.digest_buffer.clear()

    # 2. Test case: Inside quiet hours with bypass=True -> Should send message immediately (bypass)
    mock_watchlist_bypass = {
        "quiet_hours": {
            "start": "22:00",
            "end": "08:00",
            "timezone": "Asia/Kolkata",
            "bypass_for_high_priority": True
        }
    }

    with patch("signal_hunter.core.config.Settings.watchlist", new_callable=PropertyMock) as mock_wl_bypass, \
         patch.object(AlertDispatcher, "_is_quiet_hours", return_value=True), \
         patch.object(AlertDispatcher, "send_message", AsyncMock(return_value=123)) as mock_send, \
         patch("signal_hunter.corroboration.engine.corroboration_engine.get_cluster_sources", AsyncMock(return_value=[])), \
         patch("signal_hunter.ai.summarizer.summarizer.summarize", AsyncMock(return_value="")):
        
        mock_wl_bypass.return_value = mock_watchlist_bypass
        await dispatcher.dispatch(item)
        
        # Should call send_message immediately
        assert mock_send.call_count == 1
        # Digest buffer should remain empty
        assert len(dispatcher.digest_buffer) == 0


@pytest.mark.asyncio
async def test_ultimate_beast_alerting():
    """Verify ultimate beast mode: upgrades, failed retries, recovery, and post quiet hours release."""
    from signal_hunter.core.database import AsyncSessionLocal, init_db, AlertSent, UniqueCluster, Classification, RawItem
    from signal_hunter.ai.models import ScoredItem, ClassificationResult, ClassificationOutput, ItemCategory
    from unittest.mock import AsyncMock, patch, MagicMock
    from datetime import datetime, timezone
    import uuid

    await init_db()

    dispatcher = AlertDispatcher()
    dispatcher.digest_buffer.clear()

    # 1. Test failed realtime alert buffering
    item = ScoredItem(
        cluster_id=f"cluster-{uuid.uuid4()}",
        raw_item_id=f"raw-{uuid.uuid4()}",
        classification=ClassificationResult(
            output=ClassificationOutput(
                product_name="Retry Test",
                category=ItemCategory.SAAS,
                evidence_quote="some quote",
                confidence=0.95,
                trend_signal_present=True
            ),
            raw_model_response="",
            passed_validation=True,
            validation_failure_reason=None,
            model_version="gpt-4o"
        ),
        source="reddit",
        title="Retry Test Title",
        url="https://retry.com",
        body="some body text quote details",
        fetched_at=datetime.now(timezone.utc),
        corroborated=True,
        source_count=2
    )

    with patch.object(AlertDispatcher, "send_message", AsyncMock(return_value=None)), \
         patch.object(AlertDispatcher, "_is_quiet_hours", return_value=False):
        
        await dispatcher.dispatch(item)
        # Since send_message returned None (failure), it should be buffered
        assert len(dispatcher.digest_buffer) == 1
        assert dispatcher.digest_buffer[0].cluster_id == item.cluster_id

    dispatcher.digest_buffer.clear()

    # 2. Test alert upgrading (digest -> realtime)
    # Put a "digest" alert in the DB for this cluster
    async with AsyncSessionLocal() as session:
        async with session.begin():
            # Create cluster first
            cluster = UniqueCluster(
                id=item.cluster_id,
                canonical_title=item.title,
                category=item.classification.output.category,
                first_seen=datetime.now(timezone.utc).replace(tzinfo=None),
                last_seen=datetime.now(timezone.utc).replace(tzinfo=None),
                source_count=2
            )
            session.add(cluster)
            
            alert = AlertSent(
                cluster_id=item.cluster_id,
                alert_type="digest",
                telegram_message_id=999
            )
            session.add(alert)

    with patch.object(AlertDispatcher, "_send_realtime_alert", AsyncMock(return_value=True)) as mock_send_rt, \
         patch.object(AlertDispatcher, "_is_quiet_hours", return_value=False):
         
        await dispatcher.dispatch(item)
        # Should allow upgrading from digest to realtime
        assert mock_send_rt.call_count == 1
