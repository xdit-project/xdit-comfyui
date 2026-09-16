#!/bin/bash
# PID 1 supervisor: keeps ComfyUI on the default port (8188) as a child so dev restart scripts can kill it.
set -euo pipefail

_XDIT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${_XDIT_SCRIPT_DIR}/paths.sh"
# shellcheck source=/dev/null
source "${_XDIT_SCRIPT_DIR}/runtime_env.sh"

COMFY_PORT="${COMFY_PORT:-8188}"
PIDFILE=/tmp/comfyui.pid
LOG=/tmp/comfyui.log

if [ -d "${HF_CACHE_ROOT}/hub" ]; then
  export HF_HOME="${HF_CACHE_ROOT}"
  export HUGGINGFACE_HUB_CACHE="${HF_CACHE_ROOT}/hub"
  export HF_HUB_CACHE="${HF_CACHE_ROOT}/hub"
fi

cd "${COMFYUI_ROOT}"

while true; do
  "${PY}" main.py --port "${COMFY_PORT}" --listen 0.0.0.0 --disable-auto-launch \
    >>"$LOG" 2>&1 &
  comfy_pid=$!
  echo "$comfy_pid" >"$PIDFILE"
  wait "$comfy_pid" || true
  rm -f "$PIDFILE"
  sleep 1
done
