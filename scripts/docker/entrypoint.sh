#!/bin/bash
# Container entrypoint: wire the pack into ComfyUI, then run the dev supervisor.
set -euo pipefail

_XDIT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${_XDIT_SCRIPT_DIR}/paths.sh"
CUSTOM_NODES="${COMFYUI_ROOT}/custom_nodes/xdit_comfyui"
mkdir -p "${COMFYUI_ROOT}/custom_nodes"
if [ ! -e "${CUSTOM_NODES}" ]; then
  ln -sfn "${PLUGIN_ROOT}" "${CUSTOM_NODES}"
fi

exec "${_XDIT_SCRIPT_DIR}/supervisor.sh"
