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

    async def check_corroboration(self, cluster_id: str) -> Tuple[bool, int]:
        """Verify if a cluster has enough distinct sources within the corroboration window.

        Returns:
            A tuple of (is_corroborated, distinct_source_count)
        """
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
            hours=settings.CORROBORATION_WINDOW_HOURS
        )

        async with AsyncSessionLocal() as session:
            stmt = (
                select(func.count(func.distinct(RawItem.source)))
                .where(RawItem.cluster_id == cluster_id)
                .where(RawItem.fetched_at >= cutoff)
            )
            result = await session.execute(stmt)
            source_count = result.scalar() or 0

        is_corroborated = source_count >= settings.CORROBORATION_MIN_SOURCES
        logger.debug(
            "Cluster %s source count: %d (required: %d). Corroborated: %s",
            cluster_id,
            source_count,
            settings.CORROBORATION_MIN_SOURCES,
            is_corroborated,
        )
        return is_corroborated, source_count

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
