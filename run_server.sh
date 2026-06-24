#!/usr/bin/env bash
#
# run_server.sh — launch the marker API server, using every available GPU.
#
# marker_server's own multi-worker mode (uvicorn --workers N) spawns worker
# subprocesses that crash during CUDA initialization, and a single process only
# ever uses cuda:0 anyway. So this script never passes --workers > 1. Instead it
# launches WORKERS_PER_DEVICE independent *single-worker* marker_server processes
# per device (each GPU one pinned with CUDA_VISIBLE_DEVICES) on their own internal
# ports, then runs a small least-connections proxy (marker_lb.py) in front so
# clients still hit a single endpoint (HOST:PORT). When only one backend is
# needed the proxy is skipped and that server binds HOST:PORT directly. With no
# usable GPU it falls back to CPU/MPS the same way.
#
# What it does:
#   1. Sets sane defaults for the env vars below (only if you haven't set them).
#   2. Loads overrides from a local env file (default: local.env) if present.
#   3. Detects usable GPUs and launches WORKERS_PER_DEVICE single-worker backends
#      per GPU + the front proxy.
#
# Usage:
#   ./run_server.sh                      # proxy on 127.0.0.1:7000, WORKERS_PER_DEVICE backends per GPU
#   HOST=0.0.0.0 PORT=8080 ./run_server.sh
#   NUM_DEVICES=2 ./run_server.sh        # use only the first 2 GPUs
#   CUDA_VISIBLE_DEVICES=1 ./run_server.sh   # use only physical GPU 1
#   CUDA_VISIBLE_DEVICES=2,3 ./run_server.sh # use physical GPUs 2 and 3
#   WORKERS_PER_DEVICE=2 ./run_server.sh # 2 single-worker servers per GPU (2x model VRAM/GPU)
#   TORCH_DEVICE=cpu ./run_server.sh     # force CPU
#   ENV_FILE=.env ./run_server.sh        # read a different env file
#
set -euo pipefail

cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# 1. Defaults (override by exporting before running, or via the env file).
#    `:=` only assigns when the var is unset/empty, so real env vars win.
# ---------------------------------------------------------------------------
: "${HOST:=127.0.0.1}"           # interface the public proxy binds to
: "${PORT:=7000}"                # port the public proxy listens on
: "${WORKERS_PER_DEVICE:=1}"     # single-worker server processes per device (N x model VRAM per device)
: "${NUM_DEVICES:=}"             # GPUs to use; empty = auto-detect all visible GPUs
: "${BACKEND_HOST:=127.0.0.1}"   # interface the per-backend servers bind to (kept private)
: "${BACKEND_BASE_PORT:=}"       # first backend port; empty = PORT+1, then +2, ...
: "${TORCH_DEVICE:=}"            # cpu | cuda | mps ; empty/unset = auto-detect (NOT exported when empty)
: "${LOGLEVEL:=INFO}"
: "${LOG_DIR:=./server_logs}"    # per-backend stdout/stderr logs

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
  set -a                          # auto-export every var defined while sourcing
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
else
  echo "[run_server] no $ENV_FILE found — using defaults / current environment"
fi

export HOST PORT LOGLEVEL

# TORCH_DEVICE needs special handling: an EMPTY value is not "auto-detect".
# marker/surya only auto-detect when the var is truly unset (they check
# `is not None`), so an empty string is taken as a literal device and the
# server crashes with "Device string must not be empty". Export it only when
# it holds a real device; otherwise unset it so auto-detection runs.
if [[ -n "${TORCH_DEVICE:-}" ]]; then
  export TORCH_DEVICE
else
  unset TORCH_DEVICE
fi

# CUDA_VISIBLE_DEVICES selects which physical GPU(s) to use (e.g. "1" or "2,3").
# Capture that selection, then clear the global mask: this lets torch detection
# below see and validate every GPU, and lets each backend be pinned to a real
# physical index (the per-backend CUDA_VISIBLE_DEVICES we set later is read by a
# fresh process, so it must be an absolute index, not a mask-relative one).
SELECTED_GPUS="${CUDA_VISIBLE_DEVICES:-}"
unset CUDA_VISIBLE_DEVICES

# ---------------------------------------------------------------------------
# 3. Figure out how many GPUs we can actually use. torch is authoritative here:
#    a node can have GPUs that the installed torch build can't initialize (e.g.
#    a CUDA-version/driver mismatch), in which case we must fall back to CPU.
# ---------------------------------------------------------------------------
read -r CUDA_OK TORCH_NGPU < <(python - <<'PY'
try:
    import torch
    print(int(torch.cuda.is_available()), torch.cuda.device_count())
except Exception:
    print(0, 0)
PY
)

# Honor an explicit single-device request (cpu/mps, or a specific cuda:N).
FORCE_SINGLE=0
case "${TORCH_DEVICE:-}" in
  cpu|mps|cuda:*) FORCE_SINGLE=1 ;;  # cuda:N already pins one device
esac

# Resolve the list of physical GPU indices to use. An explicit CUDA_VISIBLE_DEVICES
# selection wins; otherwise use every GPU torch can see (0..TORCH_NGPU-1).
gpu_indices=()
if [[ -n "$SELECTED_GPUS" ]]; then
  IFS=',' read -r -a requested <<< "$SELECTED_GPUS"
  for idx in "${requested[@]}"; do
    idx="${idx//[[:space:]]/}"
    if [[ "$idx" =~ ^[0-9]+$ ]] && (( idx < TORCH_NGPU )); then
      gpu_indices+=("$idx")
    else
      echo "[run_server] WARNING: ignoring requested GPU '${idx}' — torch sees ${TORCH_NGPU} GPU(s)"
    fi
  done
else
  for (( g = 0; g < TORCH_NGPU; g++ )); do
    gpu_indices+=("$g")
  done
fi

# NUM_DEVICES caps how many of the selected GPUs to use (unset = all of them).
if [[ -z "${NUM_DEVICES:-}" ]]; then
  NUM_DEVICES="${#gpu_indices[@]}"
fi
if (( NUM_DEVICES > ${#gpu_indices[@]} )); then
  NUM_DEVICES="${#gpu_indices[@]}"
fi
gpu_indices=("${gpu_indices[@]:0:NUM_DEVICES}")

# ---------------------------------------------------------------------------
# 4. Build the backend list. backend_gpu[i] is the GPU index to pin backend i
#    to, or "" to leave the device to TORCH_DEVICE / auto-detection. Each backend
#    is one single-worker marker_server process.
# ---------------------------------------------------------------------------
backend_gpu=()
if [[ "$CUDA_OK" == "1" && "$FORCE_SINGLE" != "1" && "$NUM_DEVICES" -ge 1 ]]; then
  USE_GPU=1
  for (( g = 0; g < NUM_DEVICES; g++ )); do
    for (( w = 0; w < WORKERS_PER_DEVICE; w++ )); do
      backend_gpu+=("${gpu_indices[$g]}")
    done
  done
else
  USE_GPU=0
  # Forced device (cpu/mps/cuda:N) or CPU fallback: one logical device.
  if [[ "$CUDA_OK" != "1" && -z "${TORCH_DEVICE:-}" ]]; then
    export TORCH_DEVICE=cpu       # make the fallback explicit
  fi
  for (( w = 0; w < WORKERS_PER_DEVICE; w++ )); do
    backend_gpu+=("")
  done
fi

TOTAL=${#backend_gpu[@]}

# ---------------------------------------------------------------------------
# 4a-pre. Cap CPU threads per backend. Each backend is a full process; with
# TOTAL of them, letting every one grab all cores oversubscribes the CPU and
# slows every conversion (BLAS/pdftext/torch all spin up thread pools). Mirror
# the batch path (convert.py): pin BLAS pools to 2 and split physical cores
# across backends for torch's intra-op pool. Exported here so every backend
# (single-process fast path and multi-backend path) inherits it.
PHYS_CORES=$(python - <<'PY'
try:
    import psutil
    print(psutil.cpu_count(logical=False) or 1)
except Exception:
    import os
    print(os.cpu_count() or 1)
PY
)
TORCH_THREADS=$(( PHYS_CORES / TOTAL ))
(( TORCH_THREADS < 2 )) && TORCH_THREADS=2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MARKER_TORCH_THREADS="$TORCH_THREADS"
echo "[run_server] CPU threads/backend: OMP=2 torch=${TORCH_THREADS} (phys_cores=${PHYS_CORES}, backends=${TOTAL})"

if command -v marker_server >/dev/null 2>&1; then
  SERVER_CMD=(marker_server)
else
  SERVER_CMD=(python marker_server.py)
fi

# ---------------------------------------------------------------------------
# 4a. Single-backend fast path: one process, no proxy.
# ---------------------------------------------------------------------------
if (( TOTAL <= 1 )); then
  gpu="${backend_gpu[0]:-}"
  if [[ -n "$gpu" ]]; then
    export CUDA_VISIBLE_DEVICES="$gpu"
    devdesc="GPU ${gpu}"
  else
    devdesc="${TORCH_DEVICE:-auto}"
  fi
  echo "[run_server] starting a single marker_server on http://${HOST}:${PORT} (${devdesc}, 1 worker)"
  exec "${SERVER_CMD[@]}" --host "$HOST" --port "$PORT" --workers 1
fi

# ---------------------------------------------------------------------------
# 4b. Multi-backend path: TOTAL single-worker servers + a front proxy.
# ---------------------------------------------------------------------------
: "${BACKEND_BASE_PORT:=$((PORT + 1))}"
mkdir -p "$LOG_DIR"

pids=()
backends=()

cleanup() {
  trap - INT TERM EXIT
  echo
  echo "[run_server] shutting down ${#pids[@]} process(es)..."
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

if (( USE_GPU == 1 )); then
  echo "[run_server] launching ${TOTAL} backend(s): ${NUM_DEVICES} GPU(s) x ${WORKERS_PER_DEVICE} single-worker server(s)"
else
  echo "[run_server] launching ${TOTAL} backend(s): ${WORKERS_PER_DEVICE} single-worker server(s) on ${TORCH_DEVICE:-auto}"
fi

for (( i = 0; i < TOTAL; i++ )); do
  port=$((BACKEND_BASE_PORT + i))
  gpu="${backend_gpu[$i]}"
  log="${LOG_DIR}/backend${i}.log"
  if [[ -n "$gpu" ]]; then
    echo "[run_server]   backend ${i} -> http://${BACKEND_HOST}:${port}  (GPU ${gpu}, log: ${log})"
    CUDA_VISIBLE_DEVICES="$gpu" "${SERVER_CMD[@]}" \
      --host "$BACKEND_HOST" --port "$port" --workers 1 \
      >"$log" 2>&1 &
  else
    echo "[run_server]   backend ${i} -> http://${BACKEND_HOST}:${port}  (${TORCH_DEVICE:-auto}, log: ${log})"
    "${SERVER_CMD[@]}" \
      --host "$BACKEND_HOST" --port "$port" --workers 1 \
      >"$log" 2>&1 &
  fi
  pids+=("$!")
  backends+=("http://${BACKEND_HOST}:${port}")
done

# Wait for each backend to finish loading models (first run also downloads
# weights, which can take a while). A backend answers "/" only after its
# lifespan startup — i.e. after the models are on the device.
echo "[run_server] waiting for backends to load models..."
for idx in "${!backends[@]}"; do
  b="${backends[$idx]}"
  pid="${pids[$idx]}"
  ready=0
  for _ in $(seq 1 900); do          # up to ~30 min for first-run weight download
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[run_server] ERROR: backend ${b} (pid ${pid}) died during startup — see ${LOG_DIR}/backend${idx}.log"
      exit 1
    fi
    if curl -sf -o /dev/null --max-time 2 "${b}/"; then
      ready=1
      echo "[run_server]   ready: ${b}"
      break
    fi
    sleep 2
  done
  if [[ "$ready" != "1" ]]; then
    echo "[run_server] ERROR: backend ${b} did not become ready in time — see ${LOG_DIR}/backend${idx}.log"
    exit 1
  fi
done

echo "[run_server] all ${TOTAL} backends ready"
echo "[run_server] starting load-balancing proxy on http://${HOST}:${PORT} -> ${backends[*]}"
python marker_lb.py --host "$HOST" --port "$PORT" --backends "${backends[@]}" &
pids+=("$!")

# Exit (and trigger cleanup of everything) as soon as any process exits.
wait -n
