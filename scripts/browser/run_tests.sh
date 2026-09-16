#!/usr/bin/env bash
# Run the real ComfyUI Nodes 2.0 layout suite. Pass --install once on a new machine
# to install Chromium and its required OS libraries.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-${ROOT}/.test-cache/ms-playwright}"
export PLAYWRIGHT_BROWSERS_PATH

if [ "${1:-}" = "--install" ]; then
  shift
  uv run playwright install --with-deps chromium
fi

if ! find "${PLAYWRIGHT_BROWSERS_PATH}" -maxdepth 3 -type f \
    -name 'chrome-headless-shell' -perm -u+x -print -quit 2>/dev/null | grep -q .; then
  "${ROOT}/.venv/bin/python" -m playwright install chromium
fi

server_pid=""
cleanup() {
  if [ -n "${server_pid}" ]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [ -z "${COMFYUI_URL:-}" ]; then
  port="${XDIT_BROWSER_COMFYUI_PORT:-8199}"
  comfy_root="${XDIT_BROWSER_COMFYUI_ROOT:-${ROOT}/.test-cache/ComfyUI}"
  comfy_ref="${XDIT_BROWSER_COMFYUI_REF:-v0.28.0}"
  python="${ROOT}/.venv/bin/python"
  if [ ! -x "${python}" ]; then
    uv sync --group dev
  fi
  if [ ! -d "${comfy_root}/.git" ]; then
    mkdir -p "$(dirname "${comfy_root}")"
    git clone --depth 1 --branch "${comfy_ref}" \
      https://github.com/comfyanonymous/ComfyUI.git "${comfy_root}"
  fi
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" \
    uv pip install --python "${python}" -r "${comfy_root}/requirements.txt"
  mkdir -p "${comfy_root}/custom_nodes"
  ln -sfn "${ROOT}" "${comfy_root}/custom_nodes/xdit_comfyui"
  mkdir -p "${ROOT}/.test-cache"
  "${ROOT}/.venv/bin/python" "${comfy_root}/main.py" \
    --cpu --listen 127.0.0.1 --port "${port}" --disable-auto-launch \
    >"${ROOT}/.test-cache/comfyui-browser.log" 2>&1 &
  server_pid=$!
  COMFYUI_URL="http://127.0.0.1:${port}"
  for _ in $(seq 1 120); do
    if curl -fsS "${COMFYUI_URL}/object_info/XDiTSample" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      sed -n '1,240p' "${ROOT}/.test-cache/comfyui-browser.log" >&2
      exit 1
    fi
    sleep 1
  done
  if ! curl -fsS "${COMFYUI_URL}/object_info/XDiTSample" >/dev/null; then
    sed -n '1,240p' "${ROOT}/.test-cache/comfyui-browser.log" >&2
    exit 1
  fi
fi

export COMFYUI_URL XDIT_RUN_BROWSER_TESTS=1

"${ROOT}/.venv/bin/python" -m pytest tests/browser -m browser_live -v "$@"
