#!/usr/bin/env bash
# Start the shared QuickVina HTTP docking service (optional; for parallel / hparam runs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# shellcheck source=/dev/null
source "$PROJECT_ROOT/.env_vars"

HOST="${ORACLE_HOST:-127.0.0.1}"
PORT="${ORACLE_PORT:-5050}"
ENV_NAME="${ORACLE_CONDA_ENV:-mol-grpo}"

echo "Starting benchmark docking oracle on ${HOST}:${PORT} (conda env: ${ENV_NAME})"
echo "Workers: export DOCKING_USE_HTTP_SERVICE=1 && source .env_vars"
exec conda run -n "$ENV_NAME" python "$SCRIPT_DIR/oracle_app.py" \
  --host "$HOST" \
  --port "$PORT"
