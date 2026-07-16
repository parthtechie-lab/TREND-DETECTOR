FROM python:3.12-slim

LABEL maintainer="Signal Hunter AI"
LABEL description="Unified Market Intelligence + Trend Detection System"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Optional: install Playwright browsers (only if PLAYWRIGHT_ENABLED=true) ───
# Uncomment if Tier C is required:
# RUN playwright install chromium --with-deps

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Expose ports ──────────────────────────────────────────────────────────────
EXPOSE 8080 9090

# ── Entrypoint ───────────────────────────────────────────────────────────────
CMD ["python", "-m", "signal_hunter.main"]
