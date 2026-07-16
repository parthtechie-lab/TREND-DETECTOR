"""Signal Hunter AI — worker pool.

Runs parallel async workers that process de-duplicated clusters from the queue,
invokes the LLM classification layer, performs structured audits, and updates state.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from datetime import datetime, timezone

from signal_hunter.core.queue import unique_items_queue, push_scored
from signal_hunter.core.config import settings
from signal_hunter.core.database import AsyncSessionLocal, Classification
from signal_hunter.ai.classifier import classifier
from signal_hunter.ai.models import ScoredItem
from signal_hunter.corroboration.engine import corroboration_engine

logger = logging.getLogger(__name__)


async def ai_worker(worker_id: int) -> None:
    """Coroutine processing items from unique_items_queue.

    Performs classification and corroboration checks.
    """
    logger.info("AI Worker %d active.", worker_id)
    while True:
        try:
            # unique_items_queue holds dict objects containing the deduplicated cluster details
            item_data = await unique_items_queue.get()
            logger.debug(
                "AI Worker %d picked up cluster %s (%s)",
                worker_id,
                item_data.get("cluster_id"),
                item_data.get("title"),
            )

            title = item_data.get("title", "")
            body = item_data.get("body", "")
            source = item_data.get("source", "")
            cluster_id = item_data.get("cluster_id")
            raw_item_id = item_data.get("raw_item_id")
            url = item_data.get("url", "")
            fetched_at_raw = item_data.get("fetched_at")

            # Convert datetime if stored as iso string
            if isinstance(fetched_at_raw, str):
                try:
                    fetched_at = datetime.fromisoformat(fetched_at_raw)
                except ValueError:
                    fetched_at = datetime.now(timezone.utc)
            elif isinstance(fetched_at_raw, datetime):
                fetched_at = fetched_at_raw
            else:
                fetched_at = datetime.now(timezone.utc)

            # Extract source count from payload
            source_count = item_data.get("source_count", 1)

            # Invoke classification Layer
            result = await classifier.classify(
                title, body, url=url, cluster_id=cluster_id, source_count=source_count
            )

            # Grounding check: require evidence_quote and structured matching
            if not result.passed_validation:
                logger.warning(
                    "Cluster %s failed LLM validation: %s. Discarding.",
                    cluster_id,
                    result.validation_failure_reason,
                )
                unique_items_queue.task_done()
                continue

            # Confidence Gating: discard if too low
            if result.output.confidence < settings.CONFIDENCE_THRESHOLD_STORE:
                logger.info(
                    "Cluster %s classified confidence %.2f is below STORE threshold (%.2f). Discarding.",
                    cluster_id,
                    result.output.confidence,
                    settings.CONFIDENCE_THRESHOLD_STORE,
                )
                unique_items_queue.task_done()
                continue

            # Write classification record to Database (includes logging raw LLM outputs)
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    db_classification = Classification(
                        cluster_id=cluster_id,
                        product_name=result.output.product_name,
                        category=result.output.category.value,
                        evidence_quote=result.output.evidence_quote,
                        confidence=result.output.confidence,
                        trend_signal_present=result.output.trend_signal_present,
                        raw_model_output=result.raw_model_response,
                        model_version=result.model_version,
                        passed_validation=result.passed_validation,
                        prompt_version=result.prompt_version,
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                    )
                    session.add(db_classification)

            # Update category in semantic deduplicator rolling window
            try:
                from signal_hunter.dedup.semantic import semantic_deduplicator
                await semantic_deduplicator.update_window_category(cluster_id, result.output.category.value)
            except Exception as e:
                logger.error("Failed to update semantic window category: %s", e)

            # Check cross-source corroboration status (rolling window check)
            corroborated, source_count = await corroboration_engine.check_corroboration(cluster_id)

            # Construct ScoredItem pydantic model for ingestion into scored_items_queue
            scored_item = ScoredItem(
                cluster_id=cluster_id,
                raw_item_id=raw_item_id,
                classification=result,
                source=source,
                title=title,
                url=url,
                body=body,
                fetched_at=fetched_at,
                corroborated=corroborated,
                source_count=source_count,
            )

            # Put item into the scored_items_queue for delivery
            await push_scored(scored_item)
            unique_items_queue.task_done()

        except asyncio.CancelledError:
            logger.info("AI Worker %d shutting down.", worker_id)
            break
        except Exception as e:
            logger.error("AI Worker %d encountered exception: %s\n%s", worker_id, e, traceback.format_exc())
            try:
                unique_items_queue.task_done()
            except Exception:
                pass


async def run_ai_workers() -> None:
    """Launch settings.AI_WORKER_COUNT AI classification workers."""
    workers = [
        asyncio.create_task(ai_worker(idx))
        for idx in range(1, settings.AI_WORKER_COUNT + 1)
    ]
    await asyncio.gather(*workers)
