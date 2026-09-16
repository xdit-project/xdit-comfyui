#!/bin/bash
# Restart ComfyUI on the default port (8188). Works with supervisor.sh as PID 1.
set -euo pipefail

_XDIT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${_XDIT_SCRIPT_DIR}/paths.sh"
# shellcheck source=/dev/null
source "${_XDIT_SCRIPT_DIR}/runtime_env.sh"

COMFY_PORT="${COMFY_PORT:-8188}"

if [ -d "${HF_CACHE_ROOT}/hub" ]; then
  export HF_HOME="${HF_CACHE_ROOT}"
  export HUGGINGFACE_HUB_CACHE="${HF_CACHE_ROOT}/hub"
  export HF_HUB_CACHE="${HF_CACHE_ROOT}/hub"
fi

bash "${_XDIT_SCRIPT_DIR}/stop.sh"

# Supervisor (PID 1) respawns ComfyUI after stop kills the child.
if ps -p 1 -o args= 2>/dev/null | grep -q "/scripts/docker/supervisor.sh"; then
  for _ in $(seq 1 40); do
    if pgrep -f "python main.py --port ${COMFY_PORT}" >/dev/null 2>&1; then
      echo "ComfyUI restarted on ${COMFY_PORT} (supervisor respawn)."
      exit 0
    fi
    sleep 0.25
  done
  echo "Supervisor did not respawn ComfyUI; see /tmp/comfyui.log"
  exit 1
fi

# Fallback: no supervisor yet (manual session or container without startup.sh).
sleep 1
cd "${COMFYUI_ROOT}"
exec "${PY}" main.py --port "${COMFY_PORT}" --listen 0.0.0.0 --disable-auto-launch
