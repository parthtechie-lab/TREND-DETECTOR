"""
Escalated summariser using a stronger OpenAI model.

Called when a cluster has been corroborated by multiple sources to produce a
concise 2-3 sentence market-intelligence summary for inclusion in Telegram
alerts.
"""

from __future__ import annotations

import logging
from typing import List

from openai import AsyncOpenAI

from signal_hunter.core.config import settings
from signal_hunter.core.observability import track_ai_call

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a market intelligence analyst. "
    "Write a 2-3 sentence factual summary based ONLY on the provided texts. "
    "Do not add opinions, predictions, or information not present in the source texts. "
    "Be concise, precise, and professional."
)

_SNIPPET_MAX_CHARS: int = 300


class Summarizer:
    """
    Calls a stronger OpenAI model to produce a multi-source narrative summary.

    Each source snippet is truncated to ``_SNIPPET_MAX_CHARS`` characters to
    keep the prompt within a reasonable token budget.
    """

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client

    def _build_user_prompt(
        self,
        canonical_title: str,
        snippets: List[str],
    ) -> str:
        """Format numbered snippets into the user message."""
        numbered = "\n".join(
            f"{i + 1}. {s[:_SNIPPET_MAX_CHARS]}" for i, s in enumerate(snippets)
        )
        return (
            f"Product / topic: {canonical_title}\n\n"
            f"Source texts:\n{numbered}\n\n"
            "Write your 2-3 sentence summary now."
        )

    async def summarize(
        self,
        canonical_title: str,
        snippets: List[str],
    ) -> str:
        """
        Produce a factual multi-source summary.

        Parameters
        ----------
        canonical_title:
            The canonical cluster title used to ground the summary.
        snippets:
            List of short body excerpts from each corroborating source.

        Returns
        -------
        Summary string.  On any error returns a safe fallback message.
        """
        if not snippets:
            return f"Summary unavailable for: {canonical_title}"

        model = settings.SUMMARIZER_MODEL
        client = self._get_client()

        with track_ai_call(model=model) as result:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": self._build_user_prompt(
                                canonical_title, snippets
                            ),
                        },
                    ],
                )
                if response.usage:
                    result["prompt_tokens"] = response.usage.prompt_tokens
                    result["completion_tokens"] = response.usage.completion_tokens
                summary = (response.choices[0].message.content or "").strip()
                if not summary:
                    return f"Summary unavailable for: {canonical_title}"
                return summary

            except Exception as exc:
                result["status"] = "error"
                logger.error(
                    "Summarizer API error for %r: %s", canonical_title, exc
                )
                return f"Summary unavailable for: {canonical_title}"


# ---------------------------------------------------------------------------
# Module-level singleton.
# ---------------------------------------------------------------------------
summarizer: Summarizer = Summarizer()
