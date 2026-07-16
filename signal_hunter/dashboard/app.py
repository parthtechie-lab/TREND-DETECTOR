"""Signal Hunter AI — FastAPI Application.

Integrates database connectivity, routes REST endpoints, mounts client-side
assets, and broadcasts real-time pipeline event streams over WebSocket connections.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from signal_hunter.dashboard.routers.api import router as api_router
from signal_hunter.ai.models import ScoredItem

logger = logging.getLogger(__name__)


class WebSocketLogHandler(logging.Handler):
    """Custom logging handler to stream records to WebSocket clients."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            log_entry = self.format(record)
            payload = {
                "type": "log_entry",
                "data": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": log_entry,
                }
            }
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(ws_broadcast_queue.put(payload))
            except RuntimeError:
                pass
        except Exception:
            pass


class ConnectionManager:
    """Manages active WebSocket connections for real-time telemetry distribution."""

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        """Register a new WebSocket channel."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.debug("WebSocket client connected. Active connections: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a disconnected WebSocket channel."""
        self.active_connections.remove(websocket)
        logger.debug("WebSocket client disconnected. Active connections: %d", len(self.active_connections))

    async def broadcast(self, message: dict) -> None:
        """Deliver a payload to all connected clients."""
        if not self.active_connections:
            return

        logger.debug("Broadcasting update to %d WebSocket clients...", len(self.active_connections))
        # Build send coroutines and gather them to run in parallel
        tasks = [
            connection.send_json(message)
            for connection in self.active_connections
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for ws, res in zip(list(self.active_connections), results):
            if isinstance(res, Exception):
                logger.warning("Error sending WebSocket message. Disconnecting client: %s", res)
                try:
                    self.disconnect(ws)
                except Exception:
                    pass


manager = ConnectionManager()
ws_broadcast_queue: asyncio.Queue[dict] = asyncio.Queue()


async def process_ws_broadcast_queue() -> None:
    """Worker task that drains the ws_broadcast_queue and broadcasts to clients."""
    logger.info("WebSocket broadcast queue processor started.")
    while True:
        try:
            msg = await ws_broadcast_queue.get()
            await manager.broadcast(msg)
            ws_broadcast_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Exception in WebSocket broadcast queue processor: %s", e, exc_info=True)


async def broadcast_scored_item(item: ScoredItem) -> None:
    """Helper method to construct a JSON update and push it onto the broadcast queue."""
    payload = {
        "type": "scored_item",
        "data": {
            "cluster_id": item.cluster_id,
            "raw_item_id": item.raw_item_id,
            "product_name": item.classification.output.product_name,
            "category": item.classification.output.category.value,
            "evidence_quote": item.classification.output.evidence_quote,
            "confidence": item.classification.output.confidence,
            "trend_signal_present": item.classification.output.trend_signal_present,
            "title": item.title,
            "source": item.source,
            "url": item.url,
            "fetched_at": item.fetched_at.isoformat() if item.fetched_at else None,
            "corroborated": item.corroborated,
            "source_count": item.source_count,
        }
    }
    await ws_broadcast_queue.put(payload)


def create_app() -> FastAPI:
    """FastAPI Application Factory."""
    app = FastAPI(
        title="Signal Hunter AI",
        description="Unified Market Intelligence & Trend Detection System Dashboard",
        version="1.0.0",
    )

    # 1. Mount REST routers
    app.include_router(api_router, prefix="/api")

    # 2. WebSocket endpoint for live notifications and telemetry updates
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await manager.connect(websocket)
        try:
            # Keep-alive loop to monitor connection health
            while True:
                # Discard incoming messages as this socket is outbound-only
                await websocket.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception as e:
            logger.error("WebSocket exception: %s", e)
            manager.disconnect(websocket)

    # 3. Mount static folder for single-page dashboard HTML/CSS/JS
    _static_dir = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")

    # Start the broadcast worker on app startup
    @app.on_event("startup")
    async def startup_event():
        asyncio.create_task(process_ws_broadcast_queue())

        # Attach WebSocket log handler to stream logs live to UI
        log_handler = WebSocketLogHandler()
        log_handler.setFormatter(logging.Formatter("%(message)s"))
        log_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(log_handler)

    return app
