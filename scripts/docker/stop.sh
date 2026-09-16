#!/bin/bash
# Stop dev ComfyUI (default port 8188) and any xdit/torchrun process groups it spawned.
set -euo pipefail

_XDIT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${_XDIT_SCRIPT_DIR}/paths.sh"
COMFY_PORT="${COMFY_PORT:-8188}"
PIDFILE=/tmp/comfyui.pid
XDIT_BIN="${XDIT_BIN:-$(command -v xdit 2>/dev/null || true)}"

_kill_pgid() {
  local pid="$1"
  local sig="$2"
  local pgid
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
  if [ -n "$pgid" ] && [ "$pgid" -gt 1 ] 2>/dev/null; then
    kill "-$sig" -- "-$pgid" 2>/dev/null || true
  else
    kill "-$sig" "$pid" 2>/dev/null || true
  fi
}

_kill_tree() {
  local pid="$1"
  local sig="$2"
  local child
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    _kill_tree "$child" "$sig"
  done
  kill "-$sig" "$pid" 2>/dev/null || true
}

_stop_xdit_workers() {
  local sig="$1"
  local pid
  for pid in $(pgrep -f "xfuser\.runner|torch\.distributed\.run" 2>/dev/null || true); do
    _kill_pgid "$pid" "$sig"
  done
  if [ -n "${XDIT_BIN}" ]; then
    for pid in $(pgrep -f "${XDIT_BIN}" 2>/dev/null || true); do
      _kill_pgid "$pid" "$sig"
    done
  fi
}

_resolve_comfy_pid() {
  if [ -f "$PIDFILE" ]; then
    local pid
    pid="$(tr -d ' ' <"$PIDFILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "$pid"
      return 0
    fi
  fi
  local pid args
  while read -r pid args; do
    case "$args" in
      *python*main.py\ --port\ ${COMFY_PORT}*) echo "$pid"; return 0 ;;
    esac
  done < <(ps -eo pid=,args= 2>/dev/null | grep -F "main.py --port ${COMFY_PORT}" || true)
}

_comfy_is_pid1() {
  ps -p 1 -o args= 2>/dev/null | grep -q "main.py --port ${COMFY_PORT}"
}

COMFY_PID="$(_resolve_comfy_pid)"
if _comfy_is_pid1; then
  echo "ComfyUI is PID 1 (python was exec'd directly as container entrypoint)."
  echo "In-container restart is unavailable until the container restarts with scripts/docker/supervisor.sh."
  echo "After that, use: bash ${PLUGIN_ROOT}/scripts/docker/restart.sh"
  exit 1
fi

if [ -z "$COMFY_PID" ]; then
  _stop_xdit_workers TERM
  sleep 1
  _stop_xdit_workers KILL
  exit 0
fi

_stop_xdit_workers TERM
_kill_tree "$COMFY_PID" TERM
for _ in $(seq 1 25); do
  kill -0 "$COMFY_PID" 2>/dev/null || exit 0
  sleep 0.2
done

_stop_xdit_workers KILL
_kill_tree "$COMFY_PID" KILL
