"""Signal Hunter AI — unit tests for deterministic hallucination filter."""

import pytest
from signal_hunter.ai.models import ClassificationOutput, ItemCategory
from signal_hunter.ai.validator import validator


def test_validator_pass_on_exact_match():
    """Verify validation passes when the evidence quote exists in the source text."""
    output = ClassificationOutput(
        product_name="OpenHands",
        category=ItemCategory.SAAS,
        evidence_quote="OpenHands rewrites the selector",
        confidence=0.9,
        trend_signal_present=True
    )
    source_text = "The new tool OpenHands rewrites the selector to automatically fix selector breakdowns."
    passed, reason = validator.validate(output, source_text)
    assert passed is True
    assert reason is None


def test_validator_fail_on_missing_quote_positive_trend():
    """Verify validation fails if a positive trend is flagged but the evidence quote is missing."""
    output = ClassificationOutput(
        product_name="OpenHands",
        category=ItemCategory.SAAS,
        evidence_quote=None,
        confidence=0.9,
        trend_signal_present=True
    )
    source_text = "The new tool OpenHands rewrites the selector."
    passed, reason = validator.validate(output, source_text)
    assert passed is False
    assert "missing" in reason.lower()


def test_validator_pass_on_missing_quote_negative_trend():
    """Verify validation passes if no trend is present, even without an evidence quote."""
    output = ClassificationOutput(
        product_name=None,
        category=ItemCategory.OTHER,
        evidence_quote=None,
        confidence=0.3,
        trend_signal_present=False
    )
    source_text = "Nothing interesting here, just some spam."
    passed, reason = validator.validate(output, source_text)
    assert passed is True
    assert reason is None


def test_validator_fail_on_mismatch():
    """Verify validation fails if the evidence quote does not exist in the source text (hallucination)."""
    output = ClassificationOutput(
        product_name="SuperApp",
        category=ItemCategory.AI_TOOL,
        evidence_quote="SuperApp is exploding in growth",
        confidence=0.85,
        trend_signal_present=True
    )
    source_text = "I tried using SuperApp yesterday. It was okay, but a bit buggy."
    passed, reason = validator.validate(output, source_text)
    assert passed is False
    assert "not found" in reason.lower()


def test_validator_fail_on_short_quote_high_conf():
    """Verify validation fails if a high-confidence claim uses a quote that is too short."""
    output = ClassificationOutput(
        product_name="FastTool",
        category=ItemCategory.AI_TOOL,
        evidence_quote="Fast",
        confidence=0.95,
        trend_signal_present=True
    )
    source_text = "FastTool is a fast new utility for developers."
    passed, reason = validator.validate(output, source_text)
    assert passed is False
    assert "too short" in reason.lower()
