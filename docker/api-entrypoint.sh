#!/bin/sh
set -eu

alembic upgrade head

exec uvicorn backend.api.app:app \
  --host "${HOMELAB_API_HOST:-0.0.0.0}" \
  --port "${HOMELAB_API_PORT:-8000}"
