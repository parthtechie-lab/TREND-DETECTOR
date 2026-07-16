"""
Deterministic hallucination filter for classifier outputs.

The validator checks that any ``evidence_quote`` claimed by the classifier
actually appears (in normalised form) inside the source text.  This prevents
the model from fabricating quotes and ensures that high-confidence results
have meaningful evidence.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from signal_hunter.ai.models import ClassificationOutput
from signal_hunter.core.observability import hallucination_rejections_total


def _normalize(text: str) -> str:
    """
    Lowercase, collapse whitespace, and strip all non-alphanumeric characters
    except ASCII spaces.

    This makes substring matching robust to minor punctuation and capitalisation
    differences between the model output and the original source text.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class EvidenceValidator:
    """
    Stateless hallucination filter applied to every :class:`ClassificationOutput`.

    The validator is deliberately conservative – it only rejects outputs when
    there is clear evidence of fabrication or structural inconsistency.
    """

    def validate(
        self,
        output: ClassificationOutput,
        source_text: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate *output* against the original *source_text*.

        Returns
        -------
        passed : bool
            ``True`` when the output passes all checks.
        failure_reason : Optional[str]
            Human-readable description of the first failing check, or ``None``
            when ``passed`` is ``True``.

        Validation rules (applied in order)
        ------------------------------------
        1. ``trend_signal_present=True`` with no ``evidence_quote`` → FAIL.
        2. ``evidence_quote`` is ``None`` for a non-trend item → PASS (acceptable).
        3. Normalised ``evidence_quote`` not contained in normalised
           ``source_text`` → FAIL + increment ``hallucination_rejections_total``.
        4. ``confidence > 0.8`` with ``evidence_quote`` shorter than 10 chars
           → FAIL (suspiciously terse for a high-confidence claim).
        5. All other cases → PASS.
        """
        # Rule 1 – trend signal without evidence.
        if output.trend_signal_present and output.evidence_quote is None:
            return False, "missing evidence_quote for positive trend signal"

        # Rule 2 – no evidence for a non-trend item is fine.
        if output.evidence_quote is None:
            return True, None

        norm_quote = _normalize(output.evidence_quote)
        norm_source = _normalize(source_text)

        # Rule 3 – hallucination check.
        if norm_quote not in norm_source:
            hallucination_rejections_total.inc()
            return (
                False,
                f"evidence_quote not found in source_text: {output.evidence_quote!r}",
            )

        # Rule 4 – suspiciously short evidence for a high-confidence prediction.
        if output.confidence > 0.8 and len(output.evidence_quote) < 10:
            return False, "evidence_quote too short for confidence > 0.8"

        return True, None


# ---------------------------------------------------------------------------
# Module-level singleton.
# ---------------------------------------------------------------------------
validator: EvidenceValidator = EvidenceValidator()
