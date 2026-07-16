"""
Pydantic models for AI classification outputs used throughout Signal Hunter.

Defines the data contract between the classifier, validator, corroboration
engine, alerting dispatcher, and the dashboard API.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, field_validator
from pydantic import ConfigDict


class ItemCategory(str, Enum):
    """High-level product / content category assigned by the classifier."""

    AI_TOOL = "AI_TOOL"
    SAAS = "SAAS"
    HARDWARE = "HARDWARE"
    MEME = "MEME"
    OTHER = "OTHER"


class ClassificationOutput(BaseModel):
    """
    Raw structured output produced by the LLM classifier.

    Fields
    ------
    product_name:
        Canonical product / company name extracted from the source text.
        ``None`` when the classifier cannot identify a discrete product.
    category:
        One of the :class:`ItemCategory` enum values.
    evidence_quote:
        A verbatim excerpt from the source text (≤ 200 characters) that
        supports the classification decision.  Required when
        ``trend_signal_present`` is ``True``.
    confidence:
        Classifier confidence in [0, 1].
    trend_signal_present:
        ``True`` when the item contains a detectable market / trend signal
        worthy of downstream corroboration and alerting.
    """

    product_name: Optional[str] = None
    category: ItemCategory
    evidence_quote: Optional[str] = None
    confidence: float
    trend_signal_present: bool

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {v!r}")
        return round(v, 4)

    @field_validator("evidence_quote")
    @classmethod
    def _validate_evidence_quote(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) > 200:
            return v[:200]
        return v


class ClassificationResult(BaseModel):
    """
    Full result envelope wrapping :class:`ClassificationOutput`.

    Carries provenance information (raw model response, validation status,
    model version) so that every DB record is fully auditable.
    """

    output: ClassificationOutput
    raw_model_response: str
    passed_validation: bool
    validation_failure_reason: Optional[str] = None
    model_version: str
    prompt_version: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class ScoredItem(BaseModel):
    """
    A fully classified and potentially corroborated item ready for alerting.

    This is the payload pushed onto ``scored_items_queue`` and broadcast over
    the dashboard WebSocket.
    """

    cluster_id: str
    raw_item_id: str
    classification: ClassificationResult
    source: str
    title: str
    url: str
    body: str
    fetched_at: datetime
    corroborated: bool = False
    source_count: int = 1

    # Allow ORM-style attribute access for SQLAlchemy models (Pydantic v2 pattern).
    model_config = ConfigDict(from_attributes=True)
