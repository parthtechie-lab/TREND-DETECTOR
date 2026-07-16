"""Signal Hunter AI — corroboration engine.

Validates whether a semantic cluster is corroborated by checking if it has
appeared across multiple distinct sources within a specific rolling window.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Any

from sqlalchemy import select, func

from signal_hunter.core.config import settings
from signal_hunter.core.database import AsyncSessionLocal, RawItem, UniqueCluster

logger = logging.getLogger(__name__)


class CorroborationEngine:
    """Checks cross-source signal trends by evaluating a sliding time window."""

    async def check_corroboration(self, cluster_id: str) -> Tuple[bool, float]:
        """Verify if a cluster has enough distinct sources within the corroboration window
        using a Weighted Corroboration Index based on source authority.

        Returns:
            A tuple of (is_corroborated, weighted_signal_strength)
        """
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
            hours=settings.CORROBORATION_WINDOW_HOURS
        )

        # Authority weights: Product Hunt and HN represent direct high-signal launches,
        # Reddit is solid general discussion, YouTube/TikTok are slightly more anecdotal.
        SOURCE_WEIGHTS = {
            "product_hunt": 1.5,
            "hacker_news": 1.4,
            "reddit": 1.0,
            "youtube": 0.8,
            "tiktok": 0.6,
        }

        async with AsyncSessionLocal() as session:
            stmt = (
                select(RawItem.source)
                .where(RawItem.cluster_id == cluster_id)
                .where(RawItem.fetched_at >= cutoff)
                .distinct()
            )
            result = await session.execute(stmt)
            sources = result.scalars().all()

        if not sources:
            return False, 0.0

        # Calculate weighted signal strength index
        weighted_score = sum(SOURCE_WEIGHTS.get(src, 1.0) for src in sources)
        
        # Corroborated if either we meet minimum source count, or cumulative authority weight >= 1.8
        is_corroborated = len(sources) >= settings.CORROBORATION_MIN_SOURCES or weighted_score >= 1.8

        logger.debug(
            "Cluster %s sources: %s, weighted score: %.2f. Corroborated: %s",
            cluster_id,
            sources,
            weighted_score,
            is_corroborated,
        )
        return is_corroborated, round(weighted_score, 2)


    async def get_cluster_sources(self, cluster_id: str) -> List[Dict[str, Any]]:
        """Fetch all raw items belonging to a specific cluster.

        Returns:
            A list of dicts containing the items' source, title, body, and url.
        """
        async with AsyncSessionLocal() as session:
            stmt = select(RawItem).where(RawItem.cluster_id == cluster_id)
            result = await session.execute(stmt)
            items = result.scalars().all()

        return [
            {
                "source": item.source,
                "title": item.title,
                "body": item.body,
                "url": item.url,
                "author": item.author,
                "fetched_at": item.fetched_at,
            }
            for item in items
        ]


corroboration_engine = CorroborationEngine()
