# Shared path defaults for dev lifecycle scripts.
# Override PLUGIN_ROOT, COMFYUI_ROOT, or HF_CACHE_ROOT before sourcing.

_xdit_paths_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PLUGIN_ROOT="${PLUGIN_ROOT:-$(cd "${_xdit_paths_script_dir}/../.." && pwd)}"
export COMFYUI_ROOT="${COMFYUI_ROOT:-/workspace/comfyui}"
export HF_CACHE_ROOT="${HF_CACHE_ROOT:-/cache/huggingface}"

unset _xdit_paths_script_dir
