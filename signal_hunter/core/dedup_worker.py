"""Signal Hunter AI — de-duplication worker.

Processes raw items from raw_items_queue. Employs a two-stage deduplication pipeline:
1. Exact hash check (fast caching pass)
2. Semantic embedding similarity comparison (SentenceTransformers pass)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
import numpy as np
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select

from signal_hunter.core.queue import raw_items_queue, push_unique
from signal_hunter.core.database import AsyncSessionLocal, RawItem, UniqueCluster
from signal_hunter.dedup.exact import ExactDeduplicator
from signal_hunter.dedup.semantic import semantic_deduplicator
from signal_hunter.core.observability import items_deduped_exact_total, items_deduped_semantic_total

logger = logging.getLogger(__name__)

# Singletons for deduplication processes
exact_dedup = ExactDeduplicator(max_size=50000)
semantic_dedup = semantic_deduplicator



async def dedup_worker() -> None:
    """Consumes items from raw_items_queue, performs deduplication, and stores them in SQL."""
    logger.info("Deduplication worker pipeline active.")
    
    # Pre-warm exact deduplication cache
    await exact_dedup.hydrate_from_db()

    while True:
        try:
            raw_data = await raw_items_queue.get()
            title = raw_data.get("title", "")
            body = raw_data.get("body", "")
            source = raw_data.get("source", "")
            external_id = raw_data.get("external_id", "")
            url = raw_data.get("url", "")
            author = raw_data.get("author", "")
            fetched_at_raw = raw_data.get("fetched_at")

            if isinstance(fetched_at_raw, str):
                try:
                    fetched_at = datetime.fromisoformat(fetched_at_raw)
                except ValueError:
                    fetched_at = datetime.now(timezone.utc)
            elif isinstance(fetched_at_raw, datetime):
                fetched_at = fetched_at_raw
            else:
                fetched_at = datetime.now(timezone.utc)

            # --- Stage 1: Exact de-duplication ---
            if await exact_dedup.is_duplicate(title, source, fetched_at):
                logger.debug(
                    "Dropped exact duplicate item: %s [%s]", title, source
                )
                items_deduped_exact_total.inc()
                raw_items_queue.task_done()
                continue

            # Create a unique database ID for this raw item
            raw_item_uuid = str(uuid.uuid4())

            # --- Stage 2: Semantic de-duplication ---
            # Compare embedding similarity of title + short body context
            snippet = body[:300] if body else ""
            cluster_match = await semantic_dedup.find_similar(title, snippet)
            cluster_id: Optional[str] = None
            current_source_count = 1
            if cluster_match:
                # Semantic match found -> Merge into existing cluster
                cluster_id, similarity_score = cluster_match
                logger.info(
                    "Semantic match (score %.2f) for: %s. Merging into cluster %s.",
                    similarity_score,
                    title,
                    cluster_id,
                )
                items_deduped_semantic_total.inc()

                # Update the cluster entry in the database
                async with AsyncSessionLocal() as session:
                    async with session.begin():
                        stmt = select(UniqueCluster).where(UniqueCluster.id == cluster_id)
                        result = await session.execute(stmt)
                        db_cluster = result.scalars().first()

                        if db_cluster:
                            # Update sources JSON list safely
                            try:
                                src_list = json.loads(db_cluster.sources) if isinstance(db_cluster.sources, str) else db_cluster.sources
                            except Exception:
                                src_list = db_cluster.sources or []
                            if not src_list:
                                src_list = []

                            if source not in src_list:
                                src_list.append(source)
                            db_cluster.sources = src_list
                            db_cluster.source_count = len(src_list)
                            db_cluster.last_seen = fetched_at.replace(tzinfo=None)
                            current_source_count = len(src_list)
            else:
                # No semantic match found -> Instantiate a new cluster
                cluster_id = str(uuid.uuid4())
                logger.info("No semantic match found. Instantiating new cluster %s for: %s", cluster_id, title)
                current_source_count = 1

                # Add to semantic de-duplication rolling memory
                await semantic_dedup.add_to_window(cluster_id, title, snippet)

                # Write the new cluster entry to the database
                async with AsyncSessionLocal() as session:
                    async with session.begin():
                        db_cluster = UniqueCluster(
                            id=cluster_id,
                            canonical_title=title,
                            category="pending",  # Category set by AI classification worker later
                            first_seen=fetched_at.replace(tzinfo=None),
                            last_seen=fetched_at.replace(tzinfo=None),
                            source_count=1,
                            sources=[source],
                            embedding_vector=None,  # Optionally updated later
                        )
                        session.add(db_cluster)

            # Write raw item row linked to the cluster_id
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    db_raw_item = RawItem(
                        id=raw_item_uuid,
                        source=source,
                        external_id=external_id,
                        title=title,
                        body=body,
                        url=url,
                        author=author,
                        fetched_at=fetched_at.replace(tzinfo=None),
                        cluster_id=cluster_id,
                    )
                    session.add(db_raw_item)

            # Push structured data to the next staging queue
            dispatch_item = {
                "cluster_id": cluster_id,
                "raw_item_id": raw_item_uuid,
                "title": title,
                "body": body,
                "source": source,
                "url": url,
                "fetched_at": fetched_at.isoformat(),
                "source_count": current_source_count,
            }
            await push_unique(dispatch_item)
            raw_items_queue.task_done()

        except asyncio.CancelledError:
            logger.info("Deduplication worker shutting down.")
            break
        except Exception as e:
            logger.error("Error in de-duplication worker loop: %s", e, exc_info=True)
            try:
                raw_items_queue.task_done()
            except Exception:
                pass
