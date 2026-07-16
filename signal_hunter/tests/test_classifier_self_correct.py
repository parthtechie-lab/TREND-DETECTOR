"""Signal Hunter AI — unit tests for LLM validation self-correction loop."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from signal_hunter.ai.classifier import AsyncClassifier


@pytest.mark.asyncio
async def test_classifier_self_correction_flow():
    """Verify that the classifier retries and corrects paraphrased quotes."""
    classifier = AsyncClassifier()

    # Define mock response outputs
    # Choice 1 contains a paraphrased quote ("released today" instead of "releasing today")
    choice_fail = MagicMock()
    choice_fail.message.content = json.dumps({
        "product_name": "Devin AI",
        "category": "AI_TOOL",
        "evidence_quote": "Devin AI is released today",
        "confidence": 0.9,
        "trend_signal_present": True
    })

    # Choice 2 contains the corrected verbatim quote
    choice_success = MagicMock()
    choice_success.message.content = json.dumps({
        "product_name": "Devin AI",
        "category": "AI_TOOL",
        "evidence_quote": "Devin AI is releasing today",
        "confidence": 0.95,
        "trend_signal_present": True
    })

    # Set up mock responses sequentially
    mock_responses = [
        MagicMock(choices=[choice_fail], usage=MagicMock(prompt_tokens=100, completion_tokens=40)),
        MagicMock(choices=[choice_success], usage=MagicMock(prompt_tokens=180, completion_tokens=45))
    ]

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=mock_responses)

    with patch.object(AsyncClassifier, "_get_client", return_value=mock_client):
        # Trigger classification
        result = await classifier.classify(
            title="Devin AI Announcement",
            body="The new tool Devin AI is releasing today.",
            url=""
        )

        # Assertions
        assert mock_client.chat.completions.create.call_count == 2
        assert result.passed_validation is True
        assert result.output.evidence_quote == "Devin AI is releasing today"
        assert result.output.product_name == "Devin AI"
        assert result.output.confidence == 0.855
