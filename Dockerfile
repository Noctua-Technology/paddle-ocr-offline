FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

FROM python:3.10-slim-bookworm

RUN groupadd --system --gid 999 app \
    && useradd --system --gid 999 --uid 999 --create-home app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libopenblas-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV FLAGS_enable_pir_api=0
ENV FLAGS_use_mkldnn=0

USER app

CMD ["/app/.venv/bin/fastapi", "run", "/app/src/app.py", "--port", "80"]