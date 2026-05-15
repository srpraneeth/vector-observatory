FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --extra app --no-dev --no-install-project

# ---- runtime ----
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

COPY vector_observatory/ ./vector_observatory/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/movies_demo.parquet ./data/movies_demo.parquet

ENV PYTHONPATH=/app

RUN uv run python scripts/generate_demo.py

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
