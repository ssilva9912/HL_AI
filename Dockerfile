FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY alembic.ini ./
COPY backend ./backend
COPY evaluation ./evaluation
COPY frontend ./frontend
COPY docker ./docker

RUN groupadd --gid 10001 homelab \
    && useradd --uid 10001 --gid homelab --create-home homelab \
    && mkdir -p /app/data/documents /app/data/staging /app/data/index \
    && mkdir -p /home/homelab/.cache \
    && chown -R homelab:homelab /app /home/homelab

USER homelab

EXPOSE 8000 8501
