"""The torchrun worker: start it, keep it warm, run on it, and take it down."""

import hashlib
import json
import os
import pickle
import re
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from .images import _diffusion_output_kind, _diffusion_output_to_comfy_image
from .log_context import (
    copy_run_context,
    residency_logger,
    run_context_from_dict,
    run_context_to_dict,
    run_logger,
    set_run_context,
    worker_logger,
)
from .progress import (
    _drain_fragments,
    _handle_subprocess_fragment,
    _is_comfy_interrupt,
    _log_worker_out_line,
    _node_id_str,
    _parse_step_progress,
    _raise_if_interrupted,
    _reset_progress_log_dedupe,
    _strip_ansi,
    _xdit_progress,
)
from .registry import REGISTRY, _progress_node_ids
from .residency import (
    _device_id_list,
    _drop_sample_run_memory_for_loader,
    _oom_attribution_text,
    record_sample_run_memory,
)
from .residency_allocator import (
    WORKER_STATE_CPU_PARKED,
    WORKER_STATE_GPU_WARM,
    demote_loader_after_run,
)
from .runner_contract import (
    RESIDENCY_KEEP_GPU,
)
from .runtime_config import (
    _build_cli_args,
    _normalize_task,
    _normalize_timeout_seconds,
    _validate_world_size,
)
from .runtime_env import (
    _allocate_master_port,
    _ensure_runtime_env,
    _prune_stale_aiter_jit_build,
    _resolve_output_directory,
    _runtime_env_delta,
    _temporary_environment,
    _warm_aiter_jit_modules,
)
from .worker_payload import loader_init_config, worker_config_payload

_WORKER_INIT_TIMEOUT_SECONDS = 900
_INTERRUPT_POLL_SECONDS = 0.05
_WORKER_WAIT_TIMEOUT_SECONDS = 5
_WORKER_WAIT_FAST_TIMEOUT_SECONDS = 0.5

_WORKER_LOG = worker_logger()
_RESIDENCY_LOG = residency_logger()
_RUN_LOG = run_logger()


def _effective_cache_key(runner_config):
    from .model_info import model_capabilities

    preferred = runner_config.get("_cache_key")
    base = preferred or _runtime_cache_key(runner_config)
    model = runner_config.get("model")
    task = _normalize_task(runner_config.get("task"))
    cached_task = _normalize_task(runner_config.get("_cache_key_task"))
    if not model or not task:
        return base
    if preferred and task == cached_task:
        return base
    valid = model_capabilities(model).get("valid_tasks") or []
    if len(valid) <= 1 or task not in valid:
        return base
    return hashlib.sha256(f"{base}\0{task}".encode("utf-8")).hexdigest()


def _become_child_subreaper():
    if sys.platform != "linux":
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_CHILD_SUBREAPER = 36
        libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    except Exception:
        pass


def _reap_children_nonblocking():
    """Sweep orphaned worker grandchildren, called only at teardown.

    Never install this on SIGCHLD: waitpid(-1) races subprocess for exit statuses,
    which silently turns a failing child into returncode 0 for anything in the
    ComfyUI process (it made aiter's hipcc flag probe report unsupported flags as
    supported, poisoning its JIT builds).
    """
    reaped = 0
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid <= 0:
            break
        reaped += 1
    return reaped


_become_child_subreaper()


def _terminate_group(proc, *, fast=False):
    """Terminate a child and its process group (start_new_session=True), escalating
    SIGTERM -> SIGKILL so a child that ignores SIGTERM cannot orphan in a K8s pod."""
    import signal

    if proc.poll() is not None:
        _reap_children_nonblocking()
        return

    pgid = None
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pass

    signals = (signal.SIGKILL,) if fast else (signal.SIGTERM, signal.SIGKILL)
    wait_timeout = _WORKER_WAIT_FAST_TIMEOUT_SECONDS if fast else _WORKER_WAIT_TIMEOUT_SECONDS
    for sig in signals:
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
            except OSError:
                pass
        try:
            proc.send_signal(sig)
        except OSError:
            _reap_children_nonblocking()
            return
        try:
            proc.wait(timeout=wait_timeout)
            _reap_children_nonblocking()
            return
        except subprocess.TimeoutExpired:
            continue
    _reap_children_nonblocking()


def _resolve_xdit_bin(xdit_bin):
    """Resolve the xdit CLI from PATH or the ComfyUI interpreter's venv (console_scripts)."""
    name = (xdit_bin or "xdit").strip() or "xdit"
    if os.path.sep in name or (os.path.altsep and os.path.altsep in name):
        path = Path(name).expanduser()
        if path.is_file():
            return str(path)
        raise RuntimeError(f"xdit_bin not found: {path}")

    resolved = shutil.which(name)
    if resolved:
        return resolved

    if sys.prefix:
        venv_candidate = Path(sys.prefix) / "bin" / name
        if venv_candidate.is_file():
            return str(venv_candidate)

    raise RuntimeError(
        f"Could not find executable '{name}'. Install xfuser in ComfyUI's Python environment "
        f"(`pip install -r requirements.txt` in this custom node pack). "
        f"ComfyUI interpreter: {sys.executable}"
    )


def _prepare_runtime(runner_config, xdit_bin):
    resolved_bin = _resolve_xdit_bin(xdit_bin)
    config = deepcopy(runner_config)
    env_overrides = deepcopy(config.get("_env", {}))
    output_directory = _resolve_output_directory(config.get("output_directory", "xdit_outputs"))
    config["output_directory"] = str(output_directory)
    cli_args = _build_cli_args(config)

    # xdit CLI itself launches torch.distributed.run and infers nproc from the degree
    # args, so it is invoked directly — never wrapped in an outer torchrun (double-launch).
    command = [resolved_bin, *cli_args]
    command_str = " ".join(shlex.quote(part) for part in command)
    return command, command_str, config, env_overrides, output_directory


def _loader_worker_token(loader_uid):
    uid = str(loader_uid or "").strip()
    digest = (
        hashlib.sha256(uid.encode("utf-8")).hexdigest() if uid else hashlib.sha256(b"").hexdigest()
    )
    return digest[:16]


def _worker_instance_namespace():
    """Stable identity for one ComfyUI endpoint, isolated from other local servers."""
    explicit = os.environ.get("XDIT_WORKER_NAMESPACE", "").strip()
    if explicit:
        material = explicit
    else:
        port = os.environ.get("COMFY_PORT", "").strip()
        if not port:
            for index, arg in enumerate(sys.argv):
                if arg == "--port" and index + 1 < len(sys.argv):
                    port = sys.argv[index + 1]
                    break
                if arg.startswith("--port="):
                    port = arg.split("=", 1)[1]
                    break
        uid = os.getuid() if hasattr(os, "getuid") else os.environ.get("USERNAME", "user")
        material = f"{uid}\0{Path.cwd().resolve()}\0{port or '8188'}"
    return hashlib.sha256(str(material).encode("utf-8")).hexdigest()[:12]


def _worker_runtime_dir():
    uid = os.getuid() if hasattr(os, "getuid") else "user"
    path = Path(tempfile.gettempdir()) / f"xdit_comfyui_{uid}_{_worker_instance_namespace()}"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass
    return path


def _evict_loader_worker(loader_uid):
    uid = str(loader_uid or "").strip()
    if not uid:
        return False
    had_pending = _abort_loader_pending(uid)
    with REGISTRY.lock():
        entry = REGISTRY.workers.pop(uid, None)
    if entry:
        _shutdown_distributed_worker(entry)
    _cleanup_worker_artifacts(_loader_worker_token(uid))
    _drop_sample_run_memory_for_loader(uid)
    return entry is not None or had_pending


def _abort_loader_pending(loader_uid):
    uid = str(loader_uid or "").strip()
    if not uid:
        return False
    with REGISTRY.lock():
        pending = REGISTRY.pending.pop(uid, None)
    if not pending:
        return False
    _abort_worker_startup(pending.get("proc"), pending.get("worker_token"))
    return True


def _register_loader_pending(loader_uid, worker_token, proc):
    uid = str(loader_uid or "").strip()
    if not uid:
        return
    with REGISTRY.lock():
        REGISTRY.pending[uid] = {"proc": proc, "worker_token": worker_token}


def _clear_loader_pending(loader_uid):
    uid = str(loader_uid or "").strip()
    if not uid:
        return
    with REGISTRY.lock():
        REGISTRY.pending.pop(uid, None)


def register_prompt_loader_consumers(counts):
    """Sample-node counts per Model node for the prompt being queued."""
    with REGISTRY.lock():
        REGISTRY.consumers.clear()
        for loader_uid, count in (counts or {}).items():
            REGISTRY.consumers[str(loader_uid)] = int(count)


def _release_loader_after_run(runtime):
    demote_loader_after_run(
        runtime,
        park_fn=_park_loader_worker,
        evict_fn=_evict_loader_worker,
    )


def _loader_worker_alive(loader_uid, cache_key=None):
    uid = str(loader_uid or "").strip()
    if not uid:
        return False
    with REGISTRY.lock():
        entry = REGISTRY.workers.get(uid)
        if entry is not None:
            proc = entry.get("proc")
            if proc is None or proc.poll() is None:
                if cache_key is None or entry.get("cache_key") == cache_key:
                    return True
    worker_token = _loader_worker_token(uid)
    if _orphan_worker_pid(worker_token) is None:
        return False
    return cache_key is None or _read_worker_identity(worker_token) == str(cache_key)


def _set_worker_residency_state(loader_uid, state, **extra):
    uid = str(loader_uid or "").strip()
    if not uid:
        return
    with REGISTRY.lock():
        entry = REGISTRY.workers.get(uid)
        if entry is None:
            return
        entry["residency_state"] = state
        entry.update(extra)


def _worker_control_response(entry, payload, *, timeout=120):
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        conn.settimeout(timeout)
        conn.connect(entry["socket_path"])
        _socket_write_json(conn, payload)
        (size,) = struct.unpack("!I", _socket_read_exact(conn, 4))
        data = _socket_read_exact(conn, size)
        return json.loads(data.decode("utf-8"))
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _park_loader_worker(loader_uid):
    uid = str(loader_uid or "").strip()
    if not uid:
        return False
    with REGISTRY.lock():
        entry = REGISTRY.workers.get(uid)
    if entry is None or entry.get("residency_state") == WORKER_STATE_CPU_PARKED:
        return entry is not None
    try:
        result = _worker_control_response(entry, {"op": "park"})
    except Exception as exc:
        _RESIDENCY_LOG.info("CPU park failed for Model node %s: %s", uid, exc)
        return False
    if not result.get("ok"):
        _RESIDENCY_LOG.info("CPU park rejected for Model node %s: %s", uid, result.get("error"))
        return False
    _set_worker_residency_state(
        uid,
        WORKER_STATE_CPU_PARKED,
        host_bytes=result.get("host_bytes"),
        gpu_bytes=result.get("gpu_bytes"),
        last_used_at=time.time(),
    )
    return True


def _restore_loader_worker(loader_uid):
    uid = str(loader_uid or "").strip()
    if not uid:
        return False
    with REGISTRY.lock():
        entry = REGISTRY.workers.get(uid)
    if entry is None:
        return False
    if entry.get("residency_state") != WORKER_STATE_CPU_PARKED:
        return True
    try:
        result = _worker_control_response(entry, {"op": "restore"})
    except Exception as exc:
        _RESIDENCY_LOG.info("CPU restore failed for Model node %s: %s", uid, exc)
        return False
    if not result.get("ok"):
        _RESIDENCY_LOG.info("CPU restore rejected for Model node %s: %s", uid, result.get("error"))
        return False
    _set_worker_residency_state(
        uid,
        WORKER_STATE_GPU_WARM,
        host_bytes=None,
        gpu_bytes=result.get("gpu_bytes"),
        last_used_at=time.time(),
    )
    return True


def _register_loader_cache(unique_id, cache_key, runtime=None):
    if not unique_id:
        return
    uid = str(unique_id)
    loader_id = _node_id_str(unique_id)
    if loader_id and isinstance(runtime, dict):
        preset = runtime.get("_preset") or {}
        REGISTRY.snapshots[loader_id] = {
            "model": runtime.get("model"),
            "preset": preset.get("name"),
            "gpu_count": runtime.get("_gpu_count"),
            "gpus": _device_id_list(runtime.get("_cuda_visible_devices")),
            "residency": runtime.get("_residency") or RESIDENCY_KEEP_GPU,
            "cache_key_short": (cache_key or "")[:12] or None,
        }
    key_changed = False
    with REGISTRY.lock():
        old_key = REGISTRY.cache_keys.get(uid)
        key_changed = old_key is not None and old_key != cache_key
        REGISTRY.cache_keys[uid] = cache_key
    if key_changed:
        _evict_loader_worker(uid)


def _clear_loader_cache(unique_id):
    uid = str(unique_id or "").strip()
    if not uid:
        return {"ok": False, "error": "missing node_id"}
    loader_id = _node_id_str(unique_id)
    with REGISTRY.lock():
        cache_key = REGISTRY.cache_keys.pop(uid, None)
    if loader_id:
        REGISTRY.snapshots.pop(loader_id, None)
    evicted = _evict_loader_worker(uid)
    if evicted or cache_key:
        return {
            "ok": True,
            "evicted": True,
            "cache_key_short": (cache_key or "")[:12] or None,
            "message": "Released GPU cache and stopped the xDiT worker.",
        }
    return {
        "ok": True,
        "evicted": False,
        "message": "No warmed cache registered for this loader.",
    }


def _registered_loader_ids():
    with REGISTRY.lock():
        return list(REGISTRY.cache_keys)


def _reap_loaders_except(live_node_ids):
    """Release workers whose Model node no longer exists in the graph.

    Deleting a node cannot free its worker on its own — the Unload button lives on
    the node that just went away — so the browser reports the surviving Model node
    ids and anything else resident here is unreachable and gets released.
    """
    live = {str(node_id) for node_id in (live_node_ids or [])}
    released = []
    for uid in _registered_loader_ids():
        if _node_id_str(uid) in live or str(uid) in live:
            continue
        result = _clear_loader_cache(uid)
        if result.get("evicted"):
            released.append(_node_id_str(uid) or str(uid))
    if released:
        _WORKER_LOG.info("Released xDiT workers for deleted Model nodes: %s", ", ".join(released))
    return {"ok": True, "released": released}


def _release_all_loaders():
    released = []
    for uid in _registered_loader_ids():
        result = _clear_loader_cache(uid)
        if result.get("evicted"):
            released.append(_node_id_str(uid) or str(uid))
    return {"ok": True, "released": released}


def _clear_all_runtime_caches():
    pending = []
    with REGISTRY.lock():
        pending = list(REGISTRY.pending.values())
        REGISTRY.cache_keys.clear()
        REGISTRY.snapshots.clear()
        REGISTRY.pending.clear()
    for item in pending:
        _abort_worker_startup(item.get("proc"), item.get("worker_token"))
    with REGISTRY.lock():
        entries = list(REGISTRY.workers.values())
        REGISTRY.workers.clear()
    for entry in entries:
        _shutdown_distributed_worker(entry)
    with REGISTRY.lock():
        REGISTRY.run_stats.clear()


_RUNTIME_KEY_IGNORE = frozenset(
    {
        "prompt",
        "negative_prompt",
        "input_images",
        "output_directory",
        "_loader_init_input_images",
        "_loader_node_id",
        "_preloaded",
        "_preset",
        "_cache_key_task",
        "_residency",
    }
)

_RUNTIME_TUNING_KEYS = frozenset(
    {
        "cache_config",
        "num_inference_steps",
    }
)


def _runtime_cache_key(runtime_config, *, include_tuning=False):
    key_config = deepcopy(runtime_config)
    for key in _RUNTIME_KEY_IGNORE:
        key_config.pop(key, None)
    if not include_tuning:
        for key in _RUNTIME_TUNING_KEYS:
            key_config.pop(key, None)
    encoded = json.dumps(key_config, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _ensure_loader_worker(runtime, loader_node_id, timeout_seconds=None):
    """Warm the distributed worker for this loader. Model init belongs to Model."""
    if not loader_node_id or runtime.get("_preloaded"):
        return runtime
    exec_cfg = runtime.get("_exec") or {}
    if not exec_cfg:
        return runtime
    timeout_seconds = _normalize_timeout_seconds(
        timeout_seconds,
        _WORKER_INIT_TIMEOUT_SECONDS,
    )
    env_overrides = runtime.get("_env") or {}
    cache_key = runtime.get("_cache_key") or _runtime_cache_key(runtime)
    loader_uid = runtime.get("_loader_node_id") or loader_node_id
    if _distributed_worker_alive(loader_uid, cache_key):
        _restore_loader_worker(loader_uid)
        runtime["_preloaded"] = True
        return runtime

    config = loader_init_config(runtime)
    nproc = int(exec_cfg.get("world_size") or _validate_world_size(config))
    gpus = _device_id_list(runtime.get("_cuda_visible_devices"))
    # An OOM here is not retried by freeing someone else's GPU: the other model is
    # resident because a Model node was told to keep it there. The error names every
    # resident worker so the choice of what to drop stays with the user.
    with _temporary_environment(_runtime_env_delta(env_overrides)):
        with _xdit_progress(
            init_node_id=loader_node_id, inference_steps=0, include_init=True
        ) as tracker:
            _get_or_create_distributed_worker(
                cache_key,
                config,
                env_overrides,
                nproc,
                init_node_id=loader_node_id,
                timeout_seconds=timeout_seconds,
                tracker=tracker,
                loader_uid=loader_uid,
                requester_gpus=gpus,
            )
    runtime["_preloaded"] = True
    return runtime


def _subprocess_child_env(env_overrides, nproc):
    child_env = _ensure_runtime_env(env_overrides)
    if nproc > 1:
        for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
            child_env.pop(key, None)
    return child_env


def _format_run_stdout(nproc, cache_status=None, cache_key=None, timings=None):
    parts = [f"worker world_size={nproc}"]
    if cache_status:
        parts.append(f"runner {cache_status}")
    if cache_key:
        parts.append(f"cache_key={cache_key[:12]}")
    if isinstance(timings, list) and timings:
        parts.append(f"inference_s={float(timings[-1]):.3f}")
    if cache_status == "created":
        parts.append(
            "note=first load includes model init/compile; subsequent runs reuse the warmed worker"
        )
    return "; ".join(parts)


def _socket_read_exact(conn, size, poll_interval=None, deadline=None):
    """Read exactly `size` bytes into one preallocated buffer.

    A video result is hundreds of MiB; collecting chunks and joining them copies the
    whole payload a second time.
    """
    if poll_interval is None:
        poll_interval = _INTERRUPT_POLL_SECONDS
    buffer = bytearray(size)
    view = memoryview(buffer)
    offset = 0
    while offset < size:
        _raise_if_interrupted()
        remaining_time = None if deadline is None else deadline - time.monotonic()
        if remaining_time is not None and remaining_time <= 0:
            raise TimeoutError("Timed out waiting for the xDiT worker response")
        try:
            conn.settimeout(
                poll_interval
                if remaining_time is None
                else min(poll_interval, max(remaining_time, 0.001))
            )
            received = conn.recv_into(view[offset:], size - offset)
        except socket.timeout:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for the xDiT worker response")
            continue
        if not received:
            raise ConnectionError("xDiT worker connection closed")
        offset += received
    # Callers (struct.unpack, pickle.loads) take any buffer, so skip a final copy.
    return buffer


def _socket_write_json(conn, payload):
    data = json.dumps(payload, default=str).encode("utf-8")
    conn.sendall(struct.pack("!I", len(data)) + data)


def _distributed_worker_paths(worker_token):
    stem = str(_worker_runtime_dir() / f"xdit_worker_{worker_token[:16]}")
    socket_path = f"{stem}.sock"
    return socket_path, f"{socket_path}.ready", f"{stem}.json"


def _worker_stats_path(worker_token):
    socket_path, _ready_path, _config_path = _distributed_worker_paths(worker_token)
    return Path(f"{socket_path}.stats")


def _worker_identity_path(worker_token):
    return _worker_runtime_dir() / f"xdit_worker_{worker_token[:16]}.key"


def _write_worker_identity(worker_token, cache_key):
    try:
        _worker_identity_path(worker_token).write_text(str(cache_key), encoding="utf-8")
    except OSError:
        pass


def _read_worker_identity(worker_token):
    try:
        return _worker_identity_path(worker_token).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _worker_log_path(worker_token):
    return _worker_runtime_dir() / f"xdit_worker_{worker_token[:16]}.log"


def _worker_log_path_for_loader(loader_uid):
    return _worker_log_path(_loader_worker_token(loader_uid))


@contextmanager
def _follow_worker_log(log_path, tracker=None, from_end=False, log_context=None):
    stop = threading.Event()
    path = Path(log_path) if log_path else None
    context_payload = run_context_to_dict(run_context_from_dict(log_context) or copy_run_context())

    def _tail():
        pos = 0
        if from_end and path is not None and path.is_file():
            pos = path.stat().st_size
        buf = ""
        idle_loops = 0
        handle = None
        while not stop.is_set():
            if path is None or not path.is_file():
                time.sleep(_INTERRUPT_POLL_SECONDS)
                continue
            try:
                if handle is None:
                    handle = open(path, encoding="utf-8", errors="replace")
                    handle.seek(pos)
                chunk = handle.read()
                pos = handle.tell()
                if chunk:
                    idle_loops = 0
                    buf += chunk
                    fragments, buf = _drain_fragments(buf)
                    for fragment in fragments:
                        if fragment.strip():
                            _log_worker_out_line(_strip_ansi(fragment), log_context=context_payload)
                            _handle_subprocess_fragment(fragment, tracker, check_interrupt=False)
                    if buf.strip() and "%|" in buf:
                        prog = _parse_step_progress(buf)
                        if prog:
                            _log_worker_out_line(_strip_ansi(buf), log_context=context_payload)
                            _handle_subprocess_fragment(buf, tracker, check_interrupt=False)
                            buf = ""
                else:
                    idle_loops += 1
                    if tracker is not None and idle_loops % 5 == 0:
                        tracker.heartbeat(check_interrupt=False)
            except Exception:
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        pass
                    handle = None
            time.sleep(_INTERRUPT_POLL_SECONDS)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        if buf.strip():
            _handle_subprocess_fragment(buf, tracker, check_interrupt=False)

    thread = None
    if path and tracker is not None:
        thread = threading.Thread(target=_tail, name="xdit-worker-progress", daemon=True)
        thread.start()
    try:
        yield
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=1)


def _start_worker_log_forwarder(proc, log_path):
    """Drain torchrun stdout so workers cannot block on a full pipe."""
    import select

    def _forward():
        stdout = proc.stdout
        if stdout is None:
            return
        fd = stdout.fileno()
        try:
            with open(log_path, "a", encoding="utf-8", buffering=1) as log:
                while True:
                    if proc.poll() is not None:
                        remaining = stdout.read()
                        if remaining:
                            text = remaining.decode("utf-8", "replace").replace("\r", "\n")
                            log.write(text)
                        break
                    rlist, _, _ = select.select([stdout], [], [], 1.0)
                    if not rlist:
                        continue
                    chunk = os.read(fd, 4096)
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", "replace").replace("\r", "\n")
                    log.write(text)
                    log.flush()
        except Exception:
            pass
        finally:
            try:
                stdout.close()
            except Exception:
                pass

    thread = threading.Thread(target=_forward, name="xdit-worker-log", daemon=True)
    thread.start()
    return thread


def _read_log_tail(log_path, *, max_lines: int = 40) -> str:
    path = Path(log_path) if log_path else None
    if path is None or not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


_OOM_MARKERS = ("outofmemoryerror", "out of memory")


def _worker_oom_line(log_path) -> str:
    """The allocator error from anywhere in the worker log, not just its tail.

    torchrun ends a failed launch with elastic's ChildFailedError summary, and with one
    traceback per rank stacked ahead of it the OOM sits hundreds of lines outside the
    tail window. Matching on the tail alone reported a bare exit code for the one
    startup failure whose cause is worth naming — and on a multi-GPU worker, which is
    where co-resident models make an OOM likely, that was every time.
    """
    path = Path(log_path) if log_path else None
    if path is None or not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for line in reversed(lines):
        stripped = _strip_ansi(line).strip()
        lowered = stripped.lower()
        if any(marker in lowered for marker in _OOM_MARKERS):
            return stripped
    return ""


def _worker_failure_detail(log_path, *, max_lines: int = 40) -> str:
    """The worker's own traceback, so a closed socket names the real failure.

    Without this the user only sees "worker connection closed", while the error that
    killed the worker sits in its log.
    """
    tail = _read_log_tail(log_path, max_lines=max_lines * 4)
    if not tail:
        return ""
    lines = [_strip_ansi(line) for line in tail.splitlines()]
    starts = [
        index
        for index, line in enumerate(lines)
        if line.lstrip().startswith("Traceback (most recent call last)")
    ]
    selected = lines[starts[-1] :] if starts else lines
    return "\n".join(selected[-max_lines:]).strip()


def _wait_for_worker_ready(ready_path, proc, timeout_seconds, started_at, log_path=None):
    deadline = time.time() + timeout_seconds
    ready = Path(ready_path)
    while time.time() < deadline:
        _raise_if_interrupted()
        if proc.poll() is not None:
            msg = f"xDiT distributed worker exited during startup (code {proc.returncode})"
            oom_line = _worker_oom_line(log_path)
            if oom_line:
                msg = f"xDiT distributed worker OOM during startup (code {proc.returncode})"
                msg = f"{msg}\n{oom_line}"
                attribution = _oom_attribution_text()
                if attribution:
                    msg = f"{msg}\n{attribution}"
            tail = _read_log_tail(log_path)
            if tail:
                msg = f"{msg}\n--- worker log tail ---\n{tail}"
            raise RuntimeError(msg)
        if ready.is_file():
            try:
                if ready.stat().st_mtime >= started_at:
                    return
            except OSError:
                pass
        time.sleep(_INTERRUPT_POLL_SECONDS)
    raise TimeoutError("Timed out waiting for xDiT distributed worker to become ready")


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


class _ProcRef:
    """Lightweight process handle for adopted torchrun parents."""

    __slots__ = ("pid",)

    def __init__(self, pid):
        self.pid = int(pid)

    def poll(self):
        return None if _pid_alive(self.pid) else 0

    def send_signal(self, sig):
        os.kill(self.pid, sig)

    def wait(self, timeout=None):
        deadline = time.time() + timeout if timeout else None
        while _pid_alive(self.pid):
            if deadline is not None and time.time() >= deadline:
                raise subprocess.TimeoutExpired(cmd=["worker"], timeout=timeout)
            time.sleep(0.2)


def _find_torchrun_parent_pid(worker_token):
    token = str(_worker_runtime_dir() / f"xdit_worker_{worker_token[:16]}")
    try:
        lines = subprocess.check_output(
            ["pgrep", "-af", token], text=True, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        return None
    for line in lines.splitlines():
        if "torch.distributed.run" in line:
            try:
                return int(line.split(None, 1)[0])
            except (ValueError, IndexError):
                continue
    return None


def _orphan_worker_pid(worker_token):
    """The torchrun pid of a ready worker for this token that we do not own."""
    socket_path, ready_path, _config_path = _distributed_worker_paths(worker_token)
    if not Path(ready_path).is_file() or not Path(socket_path).exists():
        return None
    pid = _find_torchrun_parent_pid(worker_token)
    if pid is None or not _pid_alive(pid):
        return None
    return pid


def _try_adopt_orphan_worker(worker_token, cache_key):
    """Reuse a worker left behind by a previous ComfyUI process.

    Only when it is running the config we are about to ask for: the token is derived
    from the node id alone, so an orphan whose settings changed since it started would
    otherwise answer with the old model. A mismatch is killed here rather than left to
    hold VRAM against the worker we are about to start.
    """
    socket_path, ready_path, config_path = _distributed_worker_paths(worker_token)
    pid = _orphan_worker_pid(worker_token)
    if pid is None:
        return None
    if _read_worker_identity(worker_token) != str(cache_key):
        _WORKER_LOG.info(
            "Releasing stale xDiT worker for loader token=%s (pid=%s): settings changed "
            "since it was started",
            worker_token[:12],
            pid,
        )
        _cleanup_worker_artifacts(worker_token)
        return None
    now = time.time()
    _WORKER_LOG.info(
        "Reusing live xDiT worker for loader token=%s (pid=%s)", worker_token[:12], pid
    )
    return {
        "proc": _ProcRef(pid),
        "socket_path": socket_path,
        "ready_path": ready_path,
        "config_path": config_path,
        "log_path": str(_worker_log_path(worker_token)),
        "created_at": now,
        "last_used_at": now,
        "run_count": 0,
        "nproc": None,
        "run_lock": threading.Lock(),
    }


def _live_worker_tokens():
    """Tokens of workers this ComfyUI owns, including ones it has adopted."""
    tokens = set()
    with REGISTRY.lock():
        entries = list(REGISTRY.workers.values()) + list(REGISTRY.pending.values())
    for entry in entries:
        token = entry.get("worker_token")
        if token:
            tokens.add(str(token))
    return tokens


def _orphan_worker_tokens():
    """xDiT workers on this box whose parent ComfyUI is gone.

    A worker outlives the ComfyUI that started it (own session, so a crash cannot take
    the GPUs down mid-run), which means a crashed or force-killed ComfyUI leaves the
    weights resident with nothing left to release them. Reparenting to init is the
    signal: our own children still have us as their parent.
    """
    try:
        lines = subprocess.check_output(
            ["pgrep", "-af", "xdit_comfyui.worker_server"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {}
    orphans = {}
    socket_prefix = re.escape(str(_worker_runtime_dir() / "xdit_worker_"))
    for line in lines.splitlines():
        if "torch.distributed.run" not in line:
            continue
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        match = re.search(rf"{socket_prefix}([0-9a-f]{{1,16}})\.sock", parts[1])
        if not match:
            continue
        if _parent_pid(pid) != 1:
            continue
        orphans[match.group(1)] = pid
    return orphans


def _parent_pid(pid):
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    try:
        return int(stat.rsplit(")", 1)[1].split()[1])
    except (IndexError, ValueError):
        return None


def _sweep_orphan_workers(keep_tokens=()):
    """Reclaim GPUs from workers no live ComfyUI can reach, before we need them."""
    keep = {str(token) for token in keep_tokens} | _live_worker_tokens()
    released = []
    for token, pid in _orphan_worker_tokens().items():
        if token in keep:
            continue
        try:
            _cleanup_worker_artifacts(token)
        except Exception:
            continue
        released.append(f"{token} (pid={pid})")
    if released:
        _WORKER_LOG.info(
            "Released %d orphaned xDiT worker(s) holding GPU memory: %s",
            len(released),
            ", ".join(released),
        )
    return released


def _terminate_workers_for_token(worker_token, *, fast=False):
    token = f"xdit_worker_{worker_token[:16]}"
    try:
        lines = subprocess.check_output(
            ["pgrep", "-af", token], text=True, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        return
    pids = []
    for line in lines.splitlines():
        parts = line.strip().split(None, 1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        pids.append(pid)
    if not pids:
        return
    signals = (signal.SIGKILL,) if fast else (signal.SIGTERM, signal.SIGKILL)
    settle_seconds = _INTERRUPT_POLL_SECONDS if fast else 0.2
    for sig in signals:
        for pid in pids:
            try:
                os.kill(pid, sig)
            except OSError:
                pass
        time.sleep(settle_seconds)
        if all(not _pid_alive(pid) for pid in pids):
            break
    _reap_children_nonblocking()


def _terminate_worker_proc(proc, *, fast=False):
    if proc is None or proc.poll() is not None:
        _reap_children_nonblocking()
        return
    try:
        worker_pgid = os.getpgid(proc.pid)
        comfy_pgid = os.getpgid(os.getpid())
    except OSError:
        worker_pgid = comfy_pgid = None
    if worker_pgid is not None and worker_pgid != comfy_pgid:
        _terminate_group(proc, fast=fast)
        return
    signals = (signal.SIGKILL,) if fast else (signal.SIGTERM, signal.SIGKILL)
    wait_timeout = _WORKER_WAIT_FAST_TIMEOUT_SECONDS if fast else _WORKER_WAIT_TIMEOUT_SECONDS
    for sig in signals:
        try:
            proc.send_signal(sig)
        except OSError:
            _reap_children_nonblocking()
            return
        try:
            proc.wait(timeout=wait_timeout)
            _reap_children_nonblocking()
            return
        except subprocess.TimeoutExpired:
            continue
    _reap_children_nonblocking()


def _abort_worker_startup(proc, worker_token):
    _terminate_workers_for_token(worker_token, fast=True)
    _terminate_worker_proc(proc, fast=True)
    socket_path, ready_path, config_path = _distributed_worker_paths(worker_token)
    for path in (socket_path, ready_path, config_path):
        Path(path).unlink(missing_ok=True)


def _abort_distributed_worker(entry, loader_uid=None, *, fast=False):
    if entry is None:
        return
    worker_token = entry.get("worker_token")
    if worker_token is None and loader_uid:
        worker_token = _loader_worker_token(loader_uid)
    proc = entry.get("proc")
    if isinstance(proc, _ProcRef):
        if worker_token:
            _terminate_workers_for_token(worker_token, fast=fast)
    else:
        _terminate_worker_proc(proc, fast=fast)
    for path in (entry.get("socket_path"), entry.get("ready_path"), entry.get("config_path")):
        if path:
            Path(path).unlink(missing_ok=True)
    if loader_uid:
        with REGISTRY.lock():
            cached = REGISTRY.workers.get(str(loader_uid))
            if cached is entry:
                REGISTRY.workers.pop(str(loader_uid), None)


def _interrupt_distributed_worker(entry):
    proc = entry.get("proc") if isinstance(entry, dict) else None
    if proc is None or isinstance(proc, _ProcRef) or proc.poll() is not None:
        return False
    try:
        import psutil

        workers = [
            child
            for child in psutil.Process(proc.pid).children(recursive=True)
            if "xdit_comfyui.worker_server" in " ".join(child.cmdline())
        ]
        if not workers:
            return False
        for worker in workers:
            worker.send_signal(signal.SIGUSR1)
        return True
    except Exception:
        return False


def _wait_for_cooperative_cancel(conn, timeout=1.0):
    import select

    try:
        readable, _, _ = select.select([conn], [], [], timeout)
        return bool(readable)
    except (OSError, ValueError, TypeError):
        return False


def _cleanup_worker_artifacts(worker_token):
    socket_path, ready_path, config_path = _distributed_worker_paths(worker_token)
    _terminate_workers_for_token(worker_token)
    for path in (
        socket_path,
        ready_path,
        _worker_stats_path(worker_token),
        _worker_identity_path(worker_token),
    ):
        Path(path).unlink(missing_ok=True)


def _shutdown_distributed_worker(entry):
    proc = entry.get("proc")
    socket_path = entry.get("socket_path")
    try:
        if socket_path and Path(socket_path).exists():
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.settimeout(5)
            conn.connect(socket_path)
            _socket_write_json(conn, {"op": "shutdown"})
            conn.close()
    except Exception:
        if proc is not None and proc.poll() is None:
            _terminate_worker_proc(proc)
    if proc is not None:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _terminate_worker_proc(proc)
    for path in (entry.get("socket_path"), entry.get("ready_path"), entry.get("config_path")):
        if path:
            Path(path).unlink(missing_ok=True)


def _distributed_worker_alive(loader_uid, cache_key=None):
    return _loader_worker_alive(loader_uid, cache_key)


def _wait_for_worker_with_progress(
    ready_path, proc, log_path, tracker, timeout_seconds, nproc, started_at
):
    if tracker is not None:
        tracker.on_fragment(f"Loading model on {nproc} GPUs")
    with _follow_worker_log(log_path, tracker=tracker):
        _wait_for_worker_ready(ready_path, proc, timeout_seconds, started_at, log_path=log_path)


def _get_or_create_distributed_worker(
    cache_key,
    init_config,
    env_overrides,
    nproc,
    init_node_id=None,
    timeout_seconds=900,
    tracker=None,
    loader_uid=None,
    requester_gpus=None,
):
    loader_uid = str(loader_uid or (init_config or {}).get("_loader_node_id") or "").strip()
    if not loader_uid:
        raise RuntimeError("xDiT distributed worker requires a Model node id")
    worker_token = _loader_worker_token(loader_uid)
    socket_path, ready_path, config_path = _distributed_worker_paths(worker_token)
    gpus = list(requester_gpus or _device_id_list(env_overrides.get("CUDA_VISIBLE_DEVICES")))

    stale_entry = None
    with REGISTRY.lock():
        entry = REGISTRY.workers.get(loader_uid)
        if entry is not None:
            proc = entry.get("proc")
            alive = proc is None or proc.poll() is None
            if alive and entry.get("cache_key") == cache_key:
                if not _restore_loader_worker(loader_uid):
                    stale_entry = entry
                    REGISTRY.workers.pop(loader_uid, None)
                else:
                    entry["last_used_at"] = time.time()
                    return entry, False
            elif alive:
                stale_entry = entry
                REGISTRY.workers.pop(loader_uid, None)

    if stale_entry is not None:
        _shutdown_distributed_worker(stale_entry)

    adopted = _try_adopt_orphan_worker(worker_token, cache_key)
    if adopted is not None:
        adopted["cache_key"] = cache_key
        adopted["loader_uid"] = loader_uid
        adopted["worker_token"] = worker_token
        with REGISTRY.lock():
            REGISTRY.workers[loader_uid] = adopted
        return adopted, False

    _cleanup_worker_artifacts(worker_token)
    _sweep_orphan_workers(keep_tokens=(worker_token,))
    _warm_aiter_jit_modules(env_overrides)
    _prune_stale_aiter_jit_build()
    _abort_loader_pending(loader_uid)
    Path(config_path).write_text(
        json.dumps(worker_config_payload(init_config), default=str), encoding="utf-8"
    )
    child_env = _subprocess_child_env(env_overrides, nproc)
    master_port = _allocate_master_port()
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        f"--master_port={master_port}",
        "-m",
        "xdit_comfyui.worker_server",
        socket_path,
        config_path,
    ]
    started_at = time.time()
    proc = subprocess.Popen(
        cmd,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _write_worker_identity(worker_token, cache_key)
    log_path = _worker_log_path(worker_token)
    log_path.write_text("", encoding="utf-8")
    _start_worker_log_forwarder(proc, log_path)
    _register_loader_pending(loader_uid, worker_token, proc)
    init_timeout = min(float(timeout_seconds), _WORKER_INIT_TIMEOUT_SECONDS)
    try:
        if tracker is not None:
            _wait_for_worker_with_progress(
                ready_path, proc, log_path, tracker, init_timeout, nproc, started_at
            )
        else:
            with _xdit_progress(
                init_node_id=init_node_id, inference_steps=0, include_init=True
            ) as local_tracker:
                _wait_for_worker_with_progress(
                    ready_path, proc, log_path, local_tracker, init_timeout, nproc, started_at
                )
    except BaseException as exc:
        _clear_loader_pending(loader_uid)
        _abort_worker_startup(proc, worker_token)
        raise exc
    finally:
        _clear_loader_pending(loader_uid)
    created_at = time.time()
    entry = {
        "proc": proc,
        "socket_path": socket_path,
        "ready_path": ready_path,
        "config_path": config_path,
        "log_path": str(log_path),
        "created_at": created_at,
        "last_used_at": created_at,
        "run_count": 0,
        "nproc": nproc,
        "cache_key": cache_key,
        "loader_uid": loader_uid,
        "worker_token": worker_token,
        "gpus": gpus,
        "residency_state": WORKER_STATE_GPU_WARM,
        "run_lock": threading.Lock(),
    }
    with REGISTRY.lock():
        REGISTRY.workers[loader_uid] = entry
    return entry, True


def _run_distributed_worker_locked(
    entry,
    runner_config,
    timeout_seconds,
    inference_node_id=None,
    tracker=None,
    loader_uid=None,
):
    if entry.get("residency_state") == WORKER_STATE_CPU_PARKED:
        if not _restore_loader_worker(loader_uid or entry.get("loader_uid")):
            raise RuntimeError(
                f"Model node {loader_uid or entry.get('loader_uid')} failed to restore from CPU park"
            )
    total_steps = int(runner_config.get("num_inference_steps", 0) or 0)
    log_path = entry.get("log_path")
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    deadline = time.monotonic() + float(timeout_seconds)
    try:
        conn.settimeout(timeout_seconds)
        conn.connect(entry["socket_path"])
        _socket_write_json(conn, {"op": "run", "config": worker_config_payload(runner_config)})
        _RUN_LOG.info(
            "Dispatching xDiT worker run (%s inference steps)",
            runner_config.get("num_inference_steps", 0),
        )

        def _read_response(active_tracker):
            with _follow_worker_log(log_path, tracker=active_tracker, from_end=True):
                (size,) = struct.unpack("!I", _socket_read_exact(conn, 4, deadline=deadline))
                return _socket_read_exact(conn, size, deadline=deadline)

        if tracker is not None:
            data = _read_response(tracker)
        else:
            with _xdit_progress(
                inference_node_id=inference_node_id,
                inference_steps=total_steps,
                include_init=False,
            ) as local_tracker:
                data = _read_response(local_tracker)
                local_tracker.on_decode_complete()
        if tracker is not None:
            tracker.on_decode_complete()
        result = pickle.loads(data)
        entry["run_count"] += 1
        entry["last_used_at"] = time.time()
        return result["output"], result["timings"], dict(result.get("metadata") or {})
    except BaseException as exc:
        if isinstance(exc, (TimeoutError, socket.timeout)):
            _abort_distributed_worker(
                entry,
                loader_uid=loader_uid or entry.get("loader_uid"),
                fast=True,
            )
            raise TimeoutError(
                f"xDiT worker exceeded the {timeout_seconds:g}s run timeout and was stopped"
            ) from exc
        if _is_comfy_interrupt(exc):
            cooperative = tracker is None or tracker._phase == "inference"
            cooperatively_stopped = (
                cooperative
                and _interrupt_distributed_worker(entry)
                and _wait_for_cooperative_cancel(conn)
            )
            if not cooperatively_stopped:
                _abort_distributed_worker(
                    entry,
                    loader_uid=loader_uid or entry.get("loader_uid"),
                    fast=True,
                )
            raise
        if isinstance(exc, (ConnectionError, EOFError, struct.error)):
            detail = _worker_failure_detail(log_path)
            if detail:
                raise ConnectionError(
                    f"{exc}\n--- the xDiT worker failed with ---\n{detail}"
                ) from exc
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _run_distributed_worker(
    entry,
    runner_config,
    timeout_seconds,
    inference_node_id=None,
    tracker=None,
    loader_uid=None,
):
    """Serialize commands for one worker; its rank loop accepts one job at a time."""
    lock = entry.get("run_lock")
    if lock is None:
        lock = entry["run_lock"] = threading.Lock()
    with lock:
        return _run_distributed_worker_locked(
            entry,
            runner_config,
            timeout_seconds,
            inference_node_id=inference_node_id,
            tracker=tracker,
            loader_uid=loader_uid,
        )


def _run_xdit_distributed(
    runner_config, env_overrides, nproc, preferred_cache_key, timeout_seconds, generate_node_id
):
    cache_key = _effective_cache_key(runner_config)
    loader_uid = runner_config.get("_loader_node_id")
    total_steps = int(runner_config.get("num_inference_steps", 0) or 0)
    loader_node_id, inference_node_id = _progress_node_ids(runner_config, generate_node_id)
    _reset_progress_log_dedupe()
    if not loader_uid:
        raise RuntimeError("Sample is missing Model metadata (_loader_node_id).")
    ctx_token = set_run_context(
        loader_node_id=_node_id_str(loader_node_id),
        sample_node_id=_node_id_str(inference_node_id),
        cache_key_short=(cache_key or "")[:12] or None,
    )
    try:
        with _temporary_environment(_runtime_env_delta(env_overrides)):
            entry, created = _get_or_create_distributed_worker(
                cache_key,
                runner_config,
                env_overrides,
                nproc,
                init_node_id=loader_node_id,
                timeout_seconds=timeout_seconds,
                tracker=None,
                loader_uid=loader_uid,
                requester_gpus=_device_id_list(runner_config.get("_cuda_visible_devices")),
            )
            with _xdit_progress(
                inference_node_id=inference_node_id,
                inference_steps=total_steps,
                include_init=False,
            ) as infer_tracker:
                worker_result = _run_distributed_worker(
                    entry,
                    runner_config,
                    timeout_seconds,
                    inference_node_id=inference_node_id,
                    tracker=infer_tracker,
                    loader_uid=loader_uid,
                )
                if len(worker_result) == 2:
                    output, timings = worker_result
                    metadata = {}
                else:
                    output, timings, metadata = worker_result
                output_kind = _diffusion_output_kind(output)
                frames = _diffusion_output_to_comfy_image(output)
                metadata["output_kind"] = output_kind
                metadata["actual_height"] = int(frames.shape[1])
                metadata["actual_width"] = int(frames.shape[2])
                infer_tracker.on_finalization_complete()
        stdout = _format_run_stdout(
            nproc,
            "created" if created else "reused",
            cache_key,
            timings,
        )
        timing_count = len(timings) if isinstance(timings, list) else 0
        if timing_count > 1:
            stdout += f"; timing_entries={timing_count}"
        memory_text = record_sample_run_memory(generate_node_id, metadata, loader_uid=loader_uid)
        if memory_text:
            _RUN_LOG.info("xDiT memory: %s", memory_text)
        return stdout, frames, metadata
    finally:
        from .log_context import reset_run_context

        reset_run_context(ctx_token)


def _run_xdit(
    runner_config,
    xdit_bin,
    timeout_seconds,
    dry_run=False,
    generate_node_id=None,
    return_metadata=False,
):
    nproc = _validate_world_size(runner_config)
    command, command_str, config, env_overrides, output_directory = _prepare_runtime(
        runner_config, xdit_bin
    )
    if dry_run:
        dry_run_message = (
            f"dry_run enabled; backend=subprocess; world_size={nproc}; " f"command={command_str}"
        )
        import torch  # type: ignore[reportMissingImports]

        _RUN_LOG.info(dry_run_message)
        num_frames = max(int(runner_config.get("num_frames", 1) or 1), 1)
        frames = torch.zeros((num_frames, 64, 64, 3), dtype=torch.float32)
        result = (
            frames,
            ("video" if num_frames > 1 else "image"),
            {
                "actual_height": 64,
                "actual_width": 64,
                "fps": 24,
            },
        )
        return result if return_metadata else frames

    preferred_cache_key = config.get("_cache_key")
    with _temporary_environment(_runtime_env_delta(env_overrides)):
        distributed_result = _run_xdit_distributed(
            config,
            env_overrides,
            nproc,
            preferred_cache_key,
            timeout_seconds,
            generate_node_id,
        )
        if len(distributed_result) == 2:
            stdout, frames = distributed_result
            metadata = {
                "output_kind": "video" if int(config.get("num_frames", 1) or 1) > 1 else "image",
                "actual_height": int(frames.shape[1]),
                "actual_width": int(frames.shape[2]),
                "fps": 24,
            }
        else:
            stdout, frames, metadata = distributed_result
    _RUN_LOG.info(stdout)
    result = frames, metadata["output_kind"], metadata
    return result if return_metadata else frames
