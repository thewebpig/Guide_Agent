FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN pip install --no-cache-dir uv==0.12.2

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev \
    && useradd --create-home --uid 10001 guide

# Bake the local embedding model into the image so the first visitor does not
# trigger a network download or an avoidable cold-start timeout.
RUN .venv/bin/python -c "from guide_agent.retrieval import FastEmbedder; next(iter(FastEmbedder().embed(['warmup'])))"

RUN chown -R guide:guide /app

USER guide
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3)"

CMD ["uvicorn", "guide_agent.demo_api:app", "--host", "0.0.0.0", "--port", "8765", "--workers", "1"]
