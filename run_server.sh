#!/usr/bin/env bash
#
# run_server.sh — launch the marker API server with sane env defaults.
#
# What it does:
#   1. Sets default values for marker's env vars (only if you haven't already set them).
#   2. Loads overrides from a local env file (default: local.env) if present.
#      marker/settings.py reads `local.env` via find_dotenv(), so this matches the
#      project convention; vars exported here are also picked up by settings.
#   3. Launches the FastAPI server (marker_server / server_cli).
#
# Usage:
#   ./run_server.sh                 # host 127.0.0.1, port 8000, 1 worker
#   HOST=0.0.0.0 PORT=8080 ./run_server.sh
#   WORKERS=4 ./run_server.sh       # multiple worker processes (N x model memory)
#   ENV_FILE=.env ./run_server.sh   # read a different env file
#
set -euo pipefail

cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# 1. Defaults (override by exporting before running, or via the env file).
#    `:=` only assigns when the var is unset/empty, so real env vars win.
# ---------------------------------------------------------------------------
: "${HOST:=127.0.0.1}"          # interface the API server binds to
: "${PORT:=8000}"               # port the API server listens on
: "${WORKERS:=1}"               # worker processes; each loads its own model copy (N x RAM/VRAM)
: "${TORCH_DEVICE:=}"           # cpu | cuda | mps ; empty = marker auto-detects
: "${LOGLEVEL:=INFO}"

# Note: all LLM config — service selection (llm_service), API keys, and local-LLM
# params (openai_*/ollama_*) — plus OCR options (force_ocr, etc.) are passed per
# request at the API stage, not via env.

# ---------------------------------------------------------------------------
# 2. Load the env file (KEY=VALUE lines). Defaults above act as fallbacks;
#    anything defined in the file overrides them for this process.
# ---------------------------------------------------------------------------
ENV_FILE="${ENV_FILE:-local.env}"
if [[ -f "$ENV_FILE" ]]; then
  echo "[run_server] loading env from $ENV_FILE"
  set -a                        # auto-export every var defined while sourcing
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
else
  echo "[run_server] no $ENV_FILE found — using defaults / current environment"
fi

# Re-export the vars marker reads so they reach the child process regardless
# of whether they came from defaults, the env file, or the shell.
export HOST PORT TORCH_DEVICE LOGLEVEL

# ---------------------------------------------------------------------------
# 3. Launch the server. Prefer the installed `marker_server` entrypoint;
#    fall back to the module if it isn't on PATH. WORKERS > 1 runs multiple
#    processes (each loads its own copy of the models).
# ---------------------------------------------------------------------------
echo "[run_server] starting marker API server on http://${HOST}:${PORT}  (device: ${TORCH_DEVICE:-auto}, workers: ${WORKERS})"
if command -v marker_server >/dev/null 2>&1; then
  exec marker_server --host "$HOST" --port "$PORT" --workers "$WORKERS"
else
  exec python marker_server.py --host "$HOST" --port "$PORT" --workers "$WORKERS"
fi
