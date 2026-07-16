"""Signal Hunter AI — Main Orchestrator.

Bootstraps the database, metrics engine, and structured logger, then gathers
all async worker pools, schedules cron jobs, and runs the FastAPI dashboard.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
import uvicorn

from signal_hunter.core.config import settings
from signal_hunter.core.database import init_db
from signal_hunter.core.observability import setup_logging, start_metrics_server
from signal_hunter.core.queue import scored_items_queue
from signal_hunter.ingestion.scheduler import run_ingestion
from signal_hunter.core.dedup_worker import dedup_worker
from signal_hunter.core.worker import run_ai_workers
from signal_hunter.alerting.dispatcher import dispatcher
from signal_hunter.selfheal.monitor import health_monitor
from signal_hunter.dashboard.app import create_app, broadcast_scored_item

# Global logger for bootstrap sequence
logger = logging.getLogger("signal_hunter.main")

ASCII_BANNER = r"""
   _____ _                   _   _    _             _              _____ 
  / ____(_)                 | | | |  | |           | |            |_   _|
 | (___  _  __ _ _ __   __ _| | | |__| |_   _ _ __ | |_ ___ _ __    | |  
  \___ \| |/ _` | '_ \ / _` | | |  __  | | | | '_ \| __/ _ \ '__|   | |  
  ____) | | (_| | | | | (_| | | | |  | | |_| | | | | ||  __/ |     _| |_ 
 |_____/|_|\__, |_| |_|\__,_|_| |_|  |_|\__,_|_| |_|\__\___|_|    |_____|
            __/ |                                                        
           |___/       🎯 Unified Market Intelligence System v1.0
"""


async def alert_worker() -> None:
    """Consumes items from scored_items_queue.

    Dispatches them to Telegram and broadcasts them to WebSocket dashboard clients.
    """
    logger.info("Alert dispatcher worker active.")
    while True:
        try:
            scored_item = await scored_items_queue.get()
            logger.info("Scored item received in alert queue: %s", scored_item.title)

            # 1. Process Telegram delivery (real-time or digest buffering)
            await dispatcher.dispatch(scored_item)

            # 2. Push real-time event to the dashboard clients over WebSockets
            await broadcast_scored_item(scored_item)

            scored_items_queue.task_done()
        except asyncio.CancelledError:
            logger.info("Alert worker shutting down.")
            break
        except Exception as e:
            logger.error("Exception in alert worker: %s", e, exc_info=True)
            try:
                scored_items_queue.task_done()
            except Exception:
                pass


async def shutdown(loop: asyncio.AbstractEventLoop, sig: signal.Signals) -> None:
    """Handle graceful shutdown of the event loop and background tasks."""
    logger.warning("Received exit signal %s. Initiating graceful shutdown...", sig.name)

    # Cancel all running tasks except the current one
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    logger.info("Canceling %d background tasks...", len(tasks))
    for task in tasks:
        task.cancel()

    # Wait for tasks to clean up or timeout after 5s
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("All background tasks completed/canceled. Stopping event loop.")
    loop.stop()


def setup_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Register handlers for OS signals."""
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(
                sig, lambda s=sig: asyncio.create_task(shutdown(loop, s))
            )
        except NotImplementedError:
            # Signal handlers are not fully supported on some OS versions/environments
            pass


async def main() -> None:
    """Application entrypoint."""
    # 1. Initialize structured logging
    setup_logging(settings.LOG_LEVEL)

    # 2. Print startup banner
    print(ASCII_BANNER)
    logger.info("Starting Signal Hunter AI process bootstrapping...")

    # 3. Start Prometheus metrics server
    try:
        start_metrics_server(settings.METRICS_PORT)
    except Exception as e:
        logger.error("Failed to start Prometheus HTTP metrics server on port %d: %s", settings.METRICS_PORT, e)

    # 4. Initialize SQLAlchemy Database Tables
    logger.info("Initializing SQL database connection and running schemas...")
    try:
        await init_db()
        logger.info("Database schemas validated successfully.")
    except Exception as e:
        logger.critical("Fatal database initialization error: %s", e, exc_info=True)
        sys.exit(1)

    # 4b. Hydrate semantic deduplication window from DB (restores memory across restarts)
    try:
        from signal_hunter.dedup.semantic import semantic_deduplicator
        await semantic_deduplicator.hydrate_from_db()
    except Exception as e:
        logger.warning("Could not hydrate semantic window from DB (continuing): %s", e)


    # Get active asyncio loop and register termination handlers
    loop = asyncio.get_running_loop()
    setup_signal_handlers(loop)

    # 5. Spin up FastAPI App and Uvicorn Server instance
    fastapi_app = create_app()
    uvicorn_config = uvicorn.Config(
        app=fastapi_app,
        host="0.0.0.0",
        port=settings.DASHBOARD_PORT,
        log_level="warning",
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)

    # 6. Gather all background coroutines and execute in parallel
    logger.info("Launching parallel background workers...")
    try:
        await asyncio.gather(
            run_ingestion(),                     # Ingestion Scheduler
            dedup_worker(),                      # Two-stage Deduplication Pipeline
            run_ai_workers(),                    # LLM Classification Workers
            alert_worker(),                      # Alert Dispatcher Worker
            dispatcher.run_digest_loop(),        # Digest Delivery Loop
            health_monitor.run_forever(),        # Degradation & Health Monitor
            uvicorn_server.serve(),              # FastAPI Dashboard HTTP/WS Server
        )
    except asyncio.CancelledError:
        logger.info("Workers gather loop cancelled. Terminating.")
    except Exception as e:
        logger.critical("Fatal exception in workers orchestrator: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
