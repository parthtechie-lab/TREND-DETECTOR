"""Signal Hunter AI — Telegram alert formatting module.

Escapes all MarkdownV2 characters properly and builds structured, visually
appealing alerts for real-time messages and daily digests.
"""

from __future__ import annotations

import re
from typing import List
from signal_hunter.ai.models import ScoredItem, ItemCategory


class AlertFormatter:
    """Formats market intelligence notifications with robust MarkdownV2 escaping."""

    def _escape_md(self, text: str) -> str:
        """Escape Telegram MarkdownV2 special characters.

        Special chars: '_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+',
        '-', '=', '|', '{', '}', '.', '!'
        """
        if not text:
            return ""
        # Escape any of the character set by prepending with a backslash
        escape_chars = r"_*[]()~`>#+-=|{}.!"
        return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)

    def _confidence_bar(self, confidence: float) -> str:
        """Generate an 8-block horizontal bar representing confidence (0.0 to 1.0)."""
        clamped = max(0.0, min(1.0, confidence))
        filled_count = round(clamped * 8)
        empty_count = 8 - filled_count
        return "█" * filled_count + "░" * empty_count

    def _category_emoji(self, category: str) -> str:
        """Return the matching category emoji."""
        if category == ItemCategory.AI_TOOL:
            return "🤖"
        elif category == ItemCategory.SAAS:
            return "💼"
        elif category == ItemCategory.HARDWARE:
            return "🔩"
        elif category == ItemCategory.MEME:
            return "😂"
        else:
            return "📦"

    def format_realtime(self, item: ScoredItem, summary: str) -> str:
        """Format a single high-confidence corroborated item for real-time alerting.

        Matches Telegram MarkdownV2 spec.
        """
        emoji = self._category_emoji(item.classification.output.category)
        category_name = self._escape_md(item.classification.output.category.value)
        prod_name = self._escape_md(item.classification.output.product_name or "Unknown Product")
        source = self._escape_md(item.source.upper())
        conf_pct = int(item.classification.output.confidence * 100)
        bar = self._confidence_bar(item.classification.output.confidence)
        evidence = self._escape_md(item.classification.output.evidence_quote or "No direct quote available")
        escaped_summary = self._escape_md(summary)
        url = self._escape_md(item.url)

        corrobed_str = ""
        if item.corroborated:
            corrobed_str = f"\n✅ *Corroborated across {item.source_count} sources*\n"

        msg = (
            f"🚨 *SIGNAL DETECTED* \\| {emoji} *{category_name}*\n\n"
            f"🚀 *Product:* {prod_name}\n"
            f"📈 *Confidence:* `{bar}` *{conf_pct}%*\n"
            f"📡 *Source:* {source}\n"
            f"{corrobed_str}\n"
            f"💬 *Evidence Quote:*\n"
            f"› _{evidence}_\n\n"
            f"📝 *Analysis:*\n"
            f"{escaped_summary}\n\n"
            f"🔗 [View Source]({url})"
        )
        return msg

    def format_digest(self, items: List[ScoredItem]) -> str:
        """Format a batch of lower-confidence items into a unified, compact digest."""
        header = f"📊 *Signal Hunter Digest* — {len(items)} Items\n\n"
        lines = []
        for idx, item in enumerate(items, 1):
            emoji = self._category_emoji(item.classification.output.category)
            prod_name = self._escape_md(item.classification.output.product_name or "Unknown Product")
            conf_pct = int(item.classification.output.confidence * 100)
            url = self._escape_md(item.url)
            lines.append(f"{idx}\\. {emoji} *[{prod_name}]({url})* \\- `{conf_pct}%` confidence")

        return header + "\n".join(lines)


formatter = AlertFormatter()
