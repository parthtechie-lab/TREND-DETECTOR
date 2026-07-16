"""Signal Hunter AI — Prometheus metrics and structured logging.

This module owns **all** instrumentation for the application:

* A suite of Prometheus ``Counter``, ``Gauge``, and ``Histogram`` objects
  exported as module-level singletons that any module can import directly.
* :func:`setup_logging` — configures the root logger with a structured,
  JSON-style format suitable for log aggregation pipelines.
* :func:`start_metrics_server` — starts the Prometheus HTTP exposition server.
* :func:`track_ai_call` — a context manager that transparently times and
  records the outcome of every LLM API call.

Import pattern::

    from signal_hunter.core.observability import items_ingested_total, track_ai_call

    items_ingested_total.labels(source="reddit").inc()

    with track_ai_call("gpt-4o-mini") as result:
        response = await openai_client.chat.completions.create(...)
        result["status"] = "success"
        result["prompt_tokens"] = response.usage.prompt_tokens
        result["completion_tokens"] = response.usage.completion_tokens
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Generator

from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prometheus metrics — module-level singletons
# ---------------------------------------------------------------------------

items_ingested_total: Counter = Counter(
    "signal_hunter_items_ingested_total",
    "Total number of raw items ingested from all data sources.",
    ["source"],
)
"""Labelled by ``source`` (e.g. ``reddit``, ``youtube``, ``product_hunt``)."""

items_deduped_exact_total: Counter = Counter(
    "signal_hunter_items_deduped_exact_total",
    "Total number of items removed by exact (hash/ID-based) de-duplication.",
)

items_deduped_semantic_total: Counter = Counter(
    "signal_hunter_items_deduped_semantic_total",
    "Total number of items removed by semantic (embedding cosine-similarity) de-duplication.",
)

ai_calls_total: Counter = Counter(
    "signal_hunter_ai_calls_total",
    "Total number of LLM API calls made.",
    ["model", "status"],
)
"""Labelled by ``model`` (model identifier) and ``status`` (``success`` or ``error``)."""

ai_latency_seconds: Histogram = Histogram(
    "signal_hunter_ai_latency_seconds",
    "End-to-end latency of LLM API calls in seconds.",
    ["model"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
)
"""Labelled by ``model``. Custom buckets reflect realistic LLM response times."""

hallucination_rejections_total: Counter = Counter(
    "signal_hunter_hallucination_rejections_total",
    "Total number of LLM responses rejected by the hallucination / schema validation step.",
)

alerts_sent_total: Counter = Counter(
    "signal_hunter_alerts_sent_total",
    "Total number of Telegram alerts dispatched.",
    ["alert_type"],
)
"""Labelled by ``alert_type`` (e.g. ``new_signal``, ``corroboration``, ``digest``)."""

queue_depth: Gauge = Gauge(
    "signal_hunter_queue_depth",
    "Current number of items waiting in an internal asyncio queue.",
    ["queue_name"],
)
"""Labelled by ``queue_name`` (``raw``, ``unique``, ``scored``)."""

source_health_consecutive_failures: Gauge = Gauge(
    "signal_hunter_source_health_consecutive_failures",
    "Number of consecutive fetch failures for a data source.",
    ["source"],
)
"""Labelled by ``source``. Resets to 0 on a successful fetch."""

llm_tokens_used_total: Counter = Counter(
    "signal_hunter_llm_tokens_used_total",
    "Cumulative LLM tokens consumed, split by model and token type.",
    ["model", "token_type"],
)
"""Labelled by ``model`` and ``token_type`` (``prompt`` or ``completion``)."""


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Minimal JSON-style log formatter.

    Emits one line per record in the format::

        {"timestamp": "...", "level": "INFO", "logger": "...", "message": "..."}

    Extra key-value pairs attached to the ``LogRecord`` (e.g. via
    ``logger.info("msg", extra={"request_id": "abc"})``) are appended
    automatically.
    """

    _BUILTIN_ATTRS = frozenset(
        {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        import json

        record.message = record.getMessage()
        # ISO-8601 timestamp
        ts = self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S")
        log_obj: dict[str, object] = {
            "timestamp": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
        }

        # Append exception traceback if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Append any extra fields the caller attached
        for key, value in record.__dict__.items():
            if key not in self._BUILTIN_ATTRS and not key.startswith("_"):
                log_obj[key] = value

        return json.dumps(log_obj, default=str, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a structured JSON-style formatter.

    Call this **once** at application startup before any other code runs.

    :param level: Logging level string (``DEBUG``, ``INFO``, ``WARNING``,
                  ``ERROR``, ``CRITICAL``).  Defaults to ``'INFO'``.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    handler.setLevel(numeric_level)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove any existing handlers to avoid duplicate output
    root.handlers.clear()
    root.addHandler(handler)

    logger.debug("Structured logging initialised at level %s.", level)


# ---------------------------------------------------------------------------
# Prometheus HTTP server
# ---------------------------------------------------------------------------


def start_metrics_server(port: int = 9090) -> None:
    """Start the Prometheus metrics exposition HTTP server.

    The server runs in a daemon thread managed by ``prometheus_client`` and
    exposes all registered metrics at ``http://0.0.0.0:<port>/metrics``.

    :param port: TCP port to listen on.  Defaults to ``9090``.
    :raises OSError: If the port is already in use.
    """
    start_http_server(port)
    logger.info(
        "Prometheus metrics available at http://0.0.0.0:%d/metrics", port
    )


# ---------------------------------------------------------------------------
# AI call context manager
# ---------------------------------------------------------------------------


@contextmanager
def track_ai_call(model: str) -> Generator[dict[str, object], None, None]:
    """Time and record the outcome of a single LLM API call.

    Yields a mutable *result* dict into the ``with`` body.  Callers should
    set ``result['status']`` to ``'error'`` on failure, and optionally set
    ``result['prompt_tokens']`` and ``result['completion_tokens']`` so that
    token usage is tracked.

    On exit the context manager:

    * Observes the elapsed wall-clock time on :data:`ai_latency_seconds`.
    * Increments :data:`ai_calls_total` with the resolved ``status``.
    * If token counts are present, increments :data:`llm_tokens_used_total`.

    Example::

        with track_ai_call("gpt-4o-mini") as result:
            try:
                resp = await client.chat.completions.create(...)
                result["prompt_tokens"] = resp.usage.prompt_tokens
                result["completion_tokens"] = resp.usage.completion_tokens
            except Exception:
                result["status"] = "error"
                raise

    :param model: Model identifier string (e.g. ``'gpt-4o-mini'``).
    :yields: A mutable ``dict`` for callers to populate with ``status`` and
             optional token counts.
    """
    result: dict[str, object] = {"status": "success"}
    start = time.perf_counter()
    try:
        yield result
    except Exception:
        # Ensure failures are always reflected in the status even if the
        # caller forgot to set it.
        if result.get("status") == "success":
            result["status"] = "error"
        raise
    finally:
        elapsed = time.perf_counter() - start
        status = str(result.get("status", "success"))

        ai_latency_seconds.labels(model=model).observe(elapsed)
        ai_calls_total.labels(model=model, status=status).inc()

        prompt_tokens = result.get("prompt_tokens")
        completion_tokens = result.get("completion_tokens")
        if isinstance(prompt_tokens, int) and prompt_tokens > 0:
            llm_tokens_used_total.labels(model=model, token_type="prompt").inc(
                prompt_tokens
            )
        if isinstance(completion_tokens, int) and completion_tokens > 0:
            llm_tokens_used_total.labels(model=model, token_type="completion").inc(
                completion_tokens
            )

        logger.debug(
            "AI call completed.",
            extra={
                "model": model,
                "status": status,
                "latency_seconds": round(elapsed, 4),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )
