#!/usr/bin/env bash
#
# marker.sh — run the Marker HTTP server + the LLM-Fabric marker-worker on a Marker host.
#
# Single script, one node of the fabric (the document twin of the vLLM "worker.sh"):
#   1. start the Marker server (run_server.sh) on 127.0.0.1:$MARKER_PORT
#   2. wait — once — until it answers /health (first run downloads weights; be patient)
#   3. start the marker-worker, which connects OUTBOUND to central NATS, joins the
#      "marker-workers" queue group on subject "marker.request", and bridges each
#      request to the local Marker server. No inbound ports needed on this host.
#
# Usage:
#   ./marker.sh
# Override anything inline:
#   NATS_URL=wss://nats.materials.wiki:8443 WORKER_ID=marker-$(hostname) ./marker.sh
set -euo pipefail
cd "$(dirname "$0")"

# --- activate the venv that has BOTH marker AND the marker-worker installed ---
#   pip install -e /path/to/llm-fabric/shared -e /path/to/llm-fabric/services/marker_worker
source /work/10417/susy/ls6/AI/vllm/.venv/bin/activate

# --- TLS / proxy env (same as the vLLM worker box) ---
export SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())')
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
export CURL_CA_BUNDLE=$SSL_CERT_FILE

# --- central NATS (outbound). Dedicated :8443 entrypoint behind Traefik. ---
export NATS_URL=${NATS_URL:-wss://nats.materials.wiki:8443}
# If this host can only egress via an HTTP CONNECT proxy, uncomment:
# export NATS_TRUST_ENV=true
# export NATS_CONNECT_TIMEOUT_SECONDS=20

# --- marker-worker identity / behavior ---
export WORKER_ID=${WORKER_ID:-marker-$(hostname)}
export MARKER_CONCURRENCY=${MARKER_CONCURRENCY:-1}      # Marker is GPU-serial; 1 per worker
export MARKER_HTTP_TIMEOUT_SECONDS=${MARKER_HTTP_TIMEOUT_SECONDS:-6000}
export PROGRESS_INTERVAL_SECONDS=${PROGRESS_INTERVAL_SECONDS:-30}
# This script does the single readiness wait below, so the worker's own probe is
# minimal (it only confirms the server is still up before subscribing).
export MARKER_STARTUP_PROBE_ATTEMPTS=${MARKER_STARTUP_PROBE_ATTEMPTS:-3}
export MARKER_STARTUP_PROBE_INTERVAL_SECONDS=${MARKER_STARTUP_PROBE_INTERVAL_SECONDS:-2}

# --- where the local Marker server listens (run_server.sh defaults to 127.0.0.1:8000) ---
MARKER_HOST=${MARKER_HOST:-127.0.0.1}
MARKER_PORT=${MARKER_PORT:-8000}
export MARKER_BASE_URL=${MARKER_BASE_URL:-http://${MARKER_HOST}:${MARKER_PORT}}

LOG_DIR=${LOG_DIR:-$HOME/logs}
mkdir -p "$LOG_DIR"
HN=$(hostname)

# 1. start the Marker server (uses every visible GPU; loads models on startup).
HOST="$MARKER_HOST" PORT="$MARKER_PORT" \
  ./run_server.sh > "$LOG_DIR/marker_server_$HN.log" 2>&1 &
MARKER_PID=$!

cleanup() { kill "$MARKER_PID" "${WORKER_PID:-}" 2>/dev/null || true; }
trap cleanup INT TERM EXIT

# 2. single readiness gate: wait until /health answers (reaching it implies models
#    are loaded). ~30 min budget covers first-run weight downloads.
echo "[marker.sh] waiting for Marker server at $MARKER_BASE_URL/health ..."
for _ in $(seq 1 900); do
  if ! kill -0 "$MARKER_PID" 2>/dev/null; then
    echo "[marker.sh] ERROR: marker server died on startup — see $LOG_DIR/marker_server_$HN.log" >&2
    exit 1
  fi
  if curl -sf -o /dev/null --max-time 2 "$MARKER_BASE_URL/health"; then
    echo "[marker.sh] Marker server ready."
    break
  fi
  sleep 2
done

# 3. start the marker-worker (connects to NATS, joins the queue group, bridges to Marker).
marker-worker > "$LOG_DIR/marker_worker_$HN.log" 2>&1 &
WORKER_PID=$!
echo "[marker.sh] marker-worker started (WORKER_ID=$WORKER_ID, log: $LOG_DIR/marker_worker_$HN.log)"

# Exit (and clean up both) as soon as either the server or the worker exits.
wait -n "$MARKER_PID" "$WORKER_PID"
