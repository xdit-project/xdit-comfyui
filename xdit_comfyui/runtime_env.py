"""Process environment for xDiT child processes: PYTHONPATH, rendezvous, HF cache.

GPU-stack knobs (allocator, HIP, GPU_ARCHS, ROCM_PATH, …) belong to the base
image. Do not invent them here — NVIDIA images crash if we force HIP allocator
settings, and a ROCm image that needs them already has them.
"""

import logging
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

try:
    import folder_paths  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover - only used inside ComfyUI runtime
    folder_paths = None

_XDIT_LOG = logging.getLogger("xdit")


def _quick_run_enabled():
    return os.environ.get("XDIT_QUICK_RUN", "").strip().lower() in ("1", "true", "yes")


_POD_HF_CACHE = Path("/cache/huggingface")

_PACK_ROOT = Path(__file__).resolve().parents[1]


def _with_pack_root(pythonpath):
    """Put the pack root on the child's import path.

    ComfyUI imports a custom node by file location and never adds its directory to
    `sys.path`, so `python -m xdit_comfyui.worker_server` only
    resolves in the child if the pack root is passed down explicitly. Without this the
    pack would have to be pip-installed to run at all.
    """
    entries = [entry for entry in (pythonpath or "").split(os.pathsep) if entry]
    if str(_PACK_ROOT) in entries:
        return pythonpath
    return os.pathsep.join([str(_PACK_ROOT), *entries])


def _port_is_free(port):
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _allocate_master_port():
    existing = os.environ.get("MASTER_PORT")
    if existing and _port_is_free(int(existing)):
        return existing
    start = 12355 + (os.getpid() % 2000)
    for port in range(start, start + 300):
        if _port_is_free(port):
            port_s = str(port)
            os.environ["MASTER_PORT"] = port_s
            return port_s
    raise RuntimeError("Could not allocate a free MASTER_PORT for xDiT distributed init")


def _apply_distributed_env():
    """Pin rendezvous env for this ComfyUI process. Must not be rolled back by temp env contexts."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    _allocate_master_port()


_AITER_JIT_BUILD_ROOT = Path("/app/external/aiter/aiter/jit/build")
_AITER_JIT_ROOT = Path("/app/external/aiter/aiter/jit")
_AITER_JIT_PREWARM_MODULES = ("module_groupnorm", "module_quant")
_AITER_JIT_PREWARMED = False
_AITER_JIT_PREWARM_LOCK = threading.Lock()
_AITER_JIT_BROKEN_NINJA_MARKERS = (
    "amdgpu-coerce-illegal-types=1",
    "--offload-arch=native",
)


def _aiter_jit_build_is_stale(name, build_root):
    built_so = build_root / "build" / f"{name}.so"
    installed_so = _AITER_JIT_ROOT / f"{name}.so"
    if built_so.is_file() or installed_so.is_file():
        return False
    ninja = build_root / "build" / "build.ninja"
    if not build_root.is_dir():
        return False
    if not ninja.is_file():
        return True
    try:
        ninja_text = ninja.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True
    return any(marker in ninja_text for marker in _AITER_JIT_BROKEN_NINJA_MARKERS)


def _prune_stale_aiter_jit_build(module_name=None):
    import shutil

    if module_name is not None:
        module_names = [module_name]
    elif not _AITER_JIT_BUILD_ROOT.is_dir():
        return
    else:
        module_names = sorted(
            path.name for path in _AITER_JIT_BUILD_ROOT.iterdir() if path.is_dir()
        )
    for name in module_names:
        build_root = _AITER_JIT_BUILD_ROOT / name
        if _aiter_jit_build_is_stale(name, build_root):
            shutil.rmtree(build_root, ignore_errors=True)


def _warm_aiter_jit_modules(env_overrides=None):
    global _AITER_JIT_PREWARMED
    with _AITER_JIT_PREWARM_LOCK:
        if _AITER_JIT_PREWARMED:
            return
        try:
            import torch  # type: ignore[reportMissingImports]

            if not torch.cuda.is_available():
                return
            with _temporary_environment(_runtime_env_delta(env_overrides or {})):
                _prune_stale_aiter_jit_build()
                missing = [
                    name
                    for name in _AITER_JIT_PREWARM_MODULES
                    if not (_AITER_JIT_ROOT / f"{name}.so").is_file()
                ]
                if not missing:
                    return
                from aiter.jit.core import get_module

                for name in missing:
                    get_module(name)
        except Exception as exc:
            _XDIT_LOG.warning("aiter JIT prewarm failed (worker init may retry): %s", exc)
        finally:
            _AITER_JIT_PREWARMED = True


def _ensure_runtime_env(env_overrides=None):
    """Child env: pack on PYTHONPATH and a free torchrun rendezvous. Nothing GPU-stack."""
    merged = {**os.environ, **(env_overrides or {})}
    venv_bin = str(Path(sys.prefix) / "bin")
    path = merged.get("PATH", "")
    if venv_bin not in path.split(os.pathsep):
        merged["PATH"] = venv_bin + (os.pathsep + path if path else "")
    merged.setdefault("VIRTUAL_ENV", sys.prefix)
    merged["PYTHONPATH"] = _with_pack_root(merged.get("PYTHONPATH"))
    _apply_distributed_env()
    merged["MASTER_ADDR"] = os.environ["MASTER_ADDR"]
    merged["MASTER_PORT"] = os.environ["MASTER_PORT"]
    merged["RANK"] = os.environ["RANK"]
    merged["WORLD_SIZE"] = os.environ["WORLD_SIZE"]
    return merged


def _runtime_env_delta(env_overrides=None):
    ensured = _ensure_runtime_env(env_overrides)
    return {key: value for key, value in ensured.items() if os.environ.get(key) != value}


def _resolve_output_directory(output_directory):
    output_path = Path(output_directory)
    if not output_path.is_absolute():
        if folder_paths is not None:
            output_path = Path(folder_paths.get_output_directory()) / output_path
        else:
            output_path = Path.cwd() / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _resolve_hf_cache_root(cache_mode, hf_cache_dir):
    if cache_mode == "system_default":
        return None

    # 'auto' follows the standard HF resolution: if the environment already points at a
    # cache (e.g. a Kubernetes pod with a shared PVC), defer to it and inject nothing.
    # Otherwise fall through to the self-contained comfy_models_shared layout.
    if cache_mode == "auto":
        if (
            os.environ.get("HF_HUB_CACHE")
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
            or os.environ.get("HF_HOME")
        ):
            return None
        hub = _POD_HF_CACHE / "hub"
        if hub.is_dir():
            try:
                if any(hub.iterdir()):
                    return _POD_HF_CACHE
            except OSError:
                pass
        cache_mode = "comfy_models_shared"

    if cache_mode == "custom_path":
        if not hf_cache_dir.strip():
            raise ValueError("hf_cache_dir is required when hf_cache_mode=custom_path.")
        cache_root = Path(hf_cache_dir).expanduser()
    else:
        # comfy_models_shared
        if folder_paths is not None:
            cache_root = Path(folder_paths.models_dir) / "huggingface"
        else:
            cache_root = Path.cwd() / "models" / "huggingface"

    if not cache_root.is_absolute():
        if folder_paths is not None:
            cache_root = Path(folder_paths.models_dir) / cache_root
        else:
            cache_root = Path.cwd() / cache_root

    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def describe_hf_cache(cache_mode="auto", hf_cache_dir="huggingface"):
    """Return the effective Hugging Face cache without changing its layout."""
    if cache_mode in ("auto", "system_default"):
        for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
            value = os.environ.get(variable)
            if value:
                return {"source": variable, "path": str(Path(value).expanduser())}
        if cache_mode == "system_default":
            return {"source": "Hugging Face default", "path": "~/.cache/huggingface"}
        hub = _POD_HF_CACHE / "hub"
        if hub.is_dir():
            try:
                if any(hub.iterdir()):
                    return {"source": "container cache", "path": str(_POD_HF_CACHE)}
            except OSError:
                pass
    root = _resolve_hf_cache_root(cache_mode, hf_cache_dir)
    return {
        "source": "custom path" if cache_mode == "custom_path" else "ComfyUI models",
        "path": str(root) if root is not None else "~/.cache/huggingface",
    }


def _build_hf_cache_env(cache_root):
    if cache_root is None:
        return {}

    return {
        "HF_HOME": str(cache_root),
        "HUGGINGFACE_HUB_CACHE": str(cache_root / "hub"),
        "TRANSFORMERS_CACHE": str(cache_root / "transformers"),
        "HF_DATASETS_CACHE": str(cache_root / "datasets"),
        "DIFFUSERS_CACHE": str(cache_root / "diffusers"),
    }


@contextmanager
def _temporary_environment(overrides):
    previous = {}
    try:
        for key, value in overrides.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = str(value)
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
