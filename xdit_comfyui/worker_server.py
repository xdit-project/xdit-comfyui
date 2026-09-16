"""Long-lived xDiT worker launched under torchrun for warm multi-GPU inference."""

import json
import pickle
import signal
import socket
import struct
import sys
from pathlib import Path


def _read_exact(conn, size):
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError("worker connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_message(conn):
    (size,) = struct.unpack("!I", _read_exact(conn, 4))
    return json.loads(_read_exact(conn, size).decode("utf-8"))


def _write_message(conn, payload):
    data = json.dumps(payload, default=str).encode("utf-8")
    conn.sendall(struct.pack("!I", len(data)) + data)


def _write_pickle(conn, obj):
    data = pickle.dumps(obj)
    conn.sendall(struct.pack("!I", len(data)) + data)


def _broadcast_job(job, src, control_group=None):
    import torch.distributed as dist

    obj = [job]
    kwargs = {"group": control_group} if control_group is not None else {}
    dist.broadcast_object_list(obj, src=src, **kwargs)
    return obj[0]


def _create_control_group(world_size):
    """Use CPU collectives for commands so idle ranks do not spin GPU kernels."""
    if world_size <= 1:
        return None
    import torch.distributed as dist

    return dist.new_group(backend="gloo")


def _physical_gpu_id(index):
    import os

    visible = [part for part in (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",") if part]
    return visible[index] if index < len(visible) else str(index)


def _alloc_retries():
    import torch

    if not torch.cuda.is_available():
        return 0
    index = torch.cuda.current_device()
    return int(torch.cuda.memory_stats(index).get("num_alloc_retries", 0))


def _memory_stats(retry_baseline=0):
    """This rank's VRAM footprint plus device totals, which include other processes."""
    import torch

    if not torch.cuda.is_available():
        return None
    index = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(index)
    return {
        "gpu": _physical_gpu_id(index),
        "held_bytes": torch.cuda.memory_reserved(index),
        "live_bytes": torch.cuda.memory_allocated(index),
        "peak_bytes": torch.cuda.max_memory_allocated(index),
        "peak_held_bytes": torch.cuda.max_memory_reserved(index),
        "device_free_bytes": free,
        "device_total_bytes": total,
        # num_alloc_retries is cumulative, so report it relative to the run start.
        "alloc_retries": max(_alloc_retries() - retry_baseline, 0),
    }


def _reset_peak_memory():
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())


def _gather_memory_stats(world, retry_baseline=0):
    snapshot = _memory_stats(retry_baseline)
    if snapshot is None:
        return []
    if world.world_size <= 1:
        return [snapshot]

    import torch.distributed as dist

    gathered = [None] * world.world_size
    dist.all_gather_object(gathered, snapshot)
    return [entry for entry in gathered if entry]


def _write_stats_file(path, payload):
    try:
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")
    except OSError:
        pass


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: distributed_worker.py <socket_path> <init_config_path>")

    socket_path = sys.argv[1]
    init_config_path = sys.argv[2]
    with open(init_config_path, encoding="utf-8") as handle:
        init_config = json.load(handle)

    from xdit_comfyui import xdit_adapter

    runner = xdit_adapter.create_initialized_runner(init_config)

    world = xdit_adapter.distributed_world()
    rank = world.rank
    output_rank = world.world_size - 1
    control_group = _create_control_group(world.world_size)
    ready_path = Path(f"{socket_path}.ready")
    stats_path = Path(f"{socket_path}.stats")
    warm_memory = _gather_memory_stats(world)
    parked = False

    def _handle_control(op, conn):
        nonlocal warm_memory, parked
        from xdit_comfyui.xdit_ext.residency_park import park_runner, restore_runner

        try:
            if op == "park":
                payload = park_runner(runner, init_config)
                parked = True
                warm_memory = _gather_memory_stats(world)
                if rank == output_rank:
                    _write_stats_file(
                        stats_path,
                        {"warm": warm_memory, "state": "cpu_parked", **payload},
                    )
            elif op == "restore":
                payload = restore_runner(runner, init_config)
                parked = False
                warm_memory = _gather_memory_stats(world)
                if rank == output_rank:
                    _write_stats_file(
                        stats_path,
                        {"warm": warm_memory, "state": "gpu_warm", **payload},
                    )
            else:
                payload = {"ok": False, "error": f"unknown control op: {op}"}
                if rank == output_rank and conn is not None:
                    _write_message(conn, payload)
                    conn.close()
                return False
            if rank == output_rank and conn is not None:
                _write_message(conn, {"ok": True, **payload})
                conn.close()
            return True
        except Exception as exc:
            if rank == output_rank and conn is not None:
                _write_message(conn, {"ok": False, "error": str(exc)})
                conn.close()
            return False

    def _interrupt_run(_signum, _frame):
        pipe = getattr(runner.model, "pipe", None)
        if pipe is not None:
            pipe._interrupt = True

    signal.signal(signal.SIGUSR1, _interrupt_run)

    if rank == output_rank:
        if Path(socket_path).exists():
            Path(socket_path).unlink()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(socket_path)
        server.listen(1)
        _write_stats_file(stats_path, {"warm": warm_memory})
        ready_path.write_text("ok", encoding="utf-8")

    try:
        while True:
            conn = None
            if rank == output_rank:
                conn, _addr = server.accept()
                job = _read_message(conn)
            else:
                job = None

            job = _broadcast_job(job, output_rank, control_group)
            op = job.get("op")

            if op == "shutdown":
                if rank == output_rank and conn is not None:
                    conn.close()
                break

            if op in ("park", "restore"):
                _handle_control(op, conn)
                continue

            if op != "run":
                if rank == output_rank and conn is not None:
                    _write_message(conn, {"ok": False, "error": f"unknown op: {op}"})
                    conn.close()
                continue

            if parked:
                from xdit_comfyui.xdit_ext.residency_park import restore_runner

                restore_runner(runner, init_config)
                parked = False
                warm_memory = _gather_memory_stats(world)

            run_config = job["config"]
            from xdit_comfyui.xdit_ext.step_cache import maybe_refresh_step_cache

            refresh_config = dict(run_config)
            if run_config.get("quick_run"):
                steps = int(run_config.get("num_inference_steps") or 4)
                refresh_config["num_inference_steps"] = max(steps, 4)

            maybe_refresh_step_cache(
                runner, refresh_config, init_cache_method=init_config.get("cache_method")
            )
            input_args = xdit_adapter.preprocess_run(runner, run_config)
            pipe = getattr(runner.model, "pipe", None)
            if pipe is not None:
                pipe._interrupt = False
            _reset_peak_memory()
            retry_baseline = _alloc_retries()
            if run_config.get("quick_run"):
                from xfuser.core.utils.runner_utils import log

                from xdit_comfyui.quick_run import quick_diffusion_output

                log("quick_run enabled; skipping inference forward")
                output = quick_diffusion_output(input_args)
                timings = [0.0]
            else:
                try:
                    output, timings = xdit_adapter.run(runner, input_args)
                finally:
                    if pipe is not None:
                        pipe._interrupt = False
            run_memory = _gather_memory_stats(world, retry_baseline)
            if rank == output_rank:
                _write_stats_file(stats_path, {"warm": warm_memory, "run": run_memory})
                settings = getattr(runner.model, "settings", None)
                metadata = {
                    "fps": getattr(settings, "fps", None),
                    "model_output_type": getattr(settings, "model_output_type", None),
                    "preprocessed_height": input_args.get("height"),
                    "preprocessed_width": input_args.get("width"),
                    "warm_memory": warm_memory,
                    "run_memory": run_memory,
                }
                try:
                    _write_pickle(
                        conn,
                        {"output": output, "timings": timings, "metadata": metadata},
                    )
                except (BrokenPipeError, ConnectionError, OSError):
                    pass
                conn.close()
    finally:
        if rank == output_rank:
            server.close()
            ready_path.unlink(missing_ok=True)
            Path(socket_path).unlink(missing_ok=True)
        if control_group is not None:
            try:
                import torch.distributed as dist

                dist.destroy_process_group(control_group)
            except Exception:
                pass
        try:
            xdit_adapter.cleanup(runner)
        except Exception:
            pass


if __name__ == "__main__":
    main()
