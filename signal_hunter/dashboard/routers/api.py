"""Signal Hunter AI — FastAPI routers for API endpoints.

Serves analytical details, item history, metrics, and health reports.
"""

from __future__ import annotations

import logging
from typing import List, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Query
from sqlalchemy import select, desc

from signal_hunter.core.queue import get_queue_depths, get_drop_counts
from signal_hunter.core.database import AsyncSessionLocal, Classification, UniqueCluster, AlertSent, SourceHealth

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def get_health():
    """Retrieve system health status and pipeline queue sizing details."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "queue_depths": get_queue_depths(),
    }


from signal_hunter.core.observability import (
    hallucination_rejections_total,
    items_ingested_total,
    alerts_sent_total,
)


@router.get("/metrics")
async def get_metrics():
    """Expose queue lengths and counts of items dropped due to full buffers."""
    # Aggregate ingested items
    ingested_sum = 0
    try:
        for metric in items_ingested_total.collect():
            for sample in metric.samples:
                ingested_sum += int(sample.value)
    except Exception:
        pass

    # Aggregate sent alerts
    alerts_sum = 0
    try:
        for metric in alerts_sent_total.collect():
            for sample in metric.samples:
                alerts_sum += int(sample.value)
    except Exception:
        pass

    return {
        "queue_depths": get_queue_depths(),
        "drop_counts": get_drop_counts(),
        "hallucination_rejections": int(hallucination_rejections_total._value.get()),
        "items_ingested": ingested_sum,
        "alerts_sent": alerts_sum,
    }


@router.get("/items")
async def list_items(
    limit: int = Query(default=50, ge=1, le=200),
    source: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    corroborated_only: bool = Query(default=False),
):
    """Fetch categorized trend items with optional source/category filtering."""
    async with AsyncSessionLocal() as session:
        # Build base query joining classifications with their parent cluster
        stmt = (
            select(Classification, UniqueCluster)
            .join(UniqueCluster, Classification.cluster_id == UniqueCluster.id)
        )

        # Push filters into SQL WHERE clause BEFORE applying LIMIT.
        # Previously these were applied in Python after fetching `limit` rows,
        # meaning a filter could produce empty results even when matching rows exist.
        if category:
            stmt = stmt.where(Classification.category == category)

        if corroborated_only:
            # A cluster is corroborated when it has data from >= 2 distinct sources.
            # The `sources` JSON column stores the array of source names.
            # We approximate this check using the UniqueCluster.source_count column.
            stmt = stmt.where(UniqueCluster.source_count >= 2)

        # Note: source filter requires the UniqueCluster.sources JSON column which contains
        # a JSON array like '["reddit", "youtube"]'. We filter in a lightweight Python
        # pass after the SQL query since SQLite doesn't support JSON_CONTAINS portably.
        # However, we fetch 3× the limit to give the Python filter enough items to work with.
        fetch_limit = limit * 3 if source else limit
        stmt = stmt.order_by(desc(Classification.classified_at)).limit(fetch_limit)

        result = await session.execute(stmt)
        rows = result.all()

    items = []
    for classification, cluster in rows:
        sources_list = cluster.sources or []
        if isinstance(sources_list, str):
            import json as _json
            try:
                sources_list = _json.loads(sources_list)
            except Exception:
                sources_list = []

        # Apply source filter (Python-side, limited scope)
        if source and source not in sources_list:
            continue

        is_corroborated = cluster.source_count >= 2

        items.append({
            "id": classification.id,
            "cluster_id": classification.cluster_id,
            "product_name": classification.product_name,
            "category": classification.category,
            "evidence_quote": classification.evidence_quote,
            "confidence": classification.confidence,
            "trend_signal_present": classification.trend_signal_present,
            "classified_at": classification.classified_at.isoformat() if classification.classified_at else None,
            "canonical_title": cluster.canonical_title,
            "sources": sources_list,
            "source_count": cluster.source_count,
            "corroborated": is_corroborated,
        })

        if len(items) >= limit:
            break

    return items



@router.get("/alerts")
async def list_alerts(limit: int = Query(default=20, ge=1, le=100)):
    """Fetch historical record of Telegram messages dispatched by the system."""
    async with AsyncSessionLocal() as session:
        stmt = (
            select(AlertSent, UniqueCluster)
            .join(UniqueCluster, AlertSent.cluster_id == UniqueCluster.id)
            .order_by(desc(AlertSent.sent_at))
            .limit(limit)
        )
        result = await session.execute(stmt)
        rows = result.all()

    alerts = []
    for alert, cluster in rows:
        alerts.append({
            "id": alert.id,
            "cluster_id": alert.cluster_id,
            "canonical_title": cluster.canonical_title,
            "sent_at": alert.sent_at.isoformat() if alert.sent_at else None,
            "telegram_message_id": alert.telegram_message_id,
            "alert_type": alert.alert_type,
        })
    return alerts


@router.get("/sources")
async def list_sources():
    """Retrieve detailed statistics and health indicators for all crawler streams."""
    async with AsyncSessionLocal() as session:
        stmt = select(SourceHealth).order_by(SourceHealth.source_name)
        result = await session.execute(stmt)
        records = result.scalars().all()

    sources = []
    for record in records:
        sources.append({
            "source_name": record.source_name,
            "last_success": record.last_success.isoformat() if record.last_success else None,
            "consecutive_failures": record.consecutive_failures,
            "total_items_24h": record.total_items_24h,
            "is_degraded": record.is_degraded,
            "updated_at": record.updated_at.isoformat() if record.updated_at else None,
        })
    return sources
