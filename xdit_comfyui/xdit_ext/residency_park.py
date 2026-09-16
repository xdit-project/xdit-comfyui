"""Worker-side GPU park/restore for residency policy.

Park moves diffusers pipeline components to host RAM so VRAM can be reused. This is
only safe for layouts where each rank owns ordinary modules — not FSDP-wrapped shards,
pipefusion splits, or tensor-parallel slices. Those paths must tear down instead.
"""

from __future__ import annotations

import gc
from typing import Any


def _int_degree(config: dict, key: str) -> int:
    return max(int(config.get(key) or 1), 1)


def park_feasible(config: dict | None) -> tuple[bool, str]:
    cfg = config or {}
    if _int_degree(cfg, "fully_shard_degree") > 1:
        return False, "fully_shard_degree>1 uses FSDP sharding"
    if _int_degree(cfg, "pipefusion_parallel_degree") > 1:
        return False, "pipefusion_parallel_degree>1 splits layers across GPUs"
    if _int_degree(cfg, "tensor_parallel_degree") > 1:
        return False, "tensor_parallel_degree>1 shards weights across GPUs"
    if cfg.get("memory_efficient_sharding"):
        return False, "memory_efficient_sharding uses meta/FSDP load"
    if cfg.get("enable_sequential_cpu_offload") or cfg.get("enable_model_cpu_offload"):
        return False, "inference-time CPU offload is already active"
    return True, ""


def _local_cuda_device():
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{torch.cuda.current_device()}"


def _move_pipe_components(pipe, device: str):
    moved = 0
    components = getattr(pipe, "components", {}) or {}
    for name, component in list(components.items()):
        if component is None:
            continue
        to_fn = getattr(component, "to", None)
        if not callable(to_fn):
            continue
        moved_component = to_fn(device)
        setattr(pipe, name, moved_component)
        components[name] = moved_component
        moved += 1
    if moved == 0 and hasattr(pipe, "to"):
        pipe.to(device)
    return moved


def _memory_snapshot():
    import torch

    if not torch.cuda.is_available():
        return {"gpu": "?", "held_bytes": 0, "live_bytes": 0}
    index = torch.cuda.current_device()
    visible = [
        part
        for part in (__import__("os").environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")
        if part
    ]
    gpu = visible[index] if index < len(visible) else str(index)
    return {
        "gpu": gpu,
        "held_bytes": int(torch.cuda.memory_reserved(index)),
        "live_bytes": int(torch.cuda.memory_allocated(index)),
    }


def _component_parameter_bytes(pipe, device_type: str) -> int:
    """Count component parameters without assuming the pipeline is an nn.Module."""
    total = 0
    seen: set[int] = set()
    for component in (getattr(pipe, "components", {}) or {}).values():
        if component is None or id(component) in seen:
            continue
        seen.add(id(component))
        parameters = getattr(component, "parameters", None)
        if not callable(parameters):
            continue
        for param in parameters():
            if param.device.type == device_type:
                total += param.numel() * param.element_size()
    return total


def park_runner(runner, init_config: dict | None = None) -> dict[str, Any]:
    ok, reason = park_feasible(init_config)
    if not ok:
        raise RuntimeError(f"CPU park is not supported for this layout: {reason}")

    import torch

    pipe = getattr(getattr(runner, "model", None), "pipe", None)
    if pipe is None:
        raise RuntimeError("Worker has no initialized pipeline to park")

    moved = _move_pipe_components(pipe, "cpu")
    if moved == 0:
        raise RuntimeError("Pipeline has no movable components for CPU park")

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    gpu_stats = _memory_snapshot()
    host_bytes = _component_parameter_bytes(pipe, "cpu")
    return {
        "ok": True,
        "moved_components": moved,
        "host_bytes": host_bytes,
        "gpu_bytes": gpu_stats["held_bytes"],
        "gpu_stats": gpu_stats,
    }


def restore_runner(runner, init_config: dict | None = None) -> dict[str, Any]:
    ok, reason = park_feasible(init_config)
    if not ok:
        raise RuntimeError(f"CPU restore is not supported for this layout: {reason}")

    import torch

    pipe = getattr(getattr(runner, "model", None), "pipe", None)
    if pipe is None:
        raise RuntimeError("Worker has no pipeline to restore")

    device = _local_cuda_device()
    moved = _move_pipe_components(pipe, device)
    if moved == 0:
        raise RuntimeError("Pipeline has no movable components to restore")

    torch.cuda.synchronize()
    gc.collect()
    gpu_stats = _memory_snapshot()
    return {
        "ok": True,
        "moved_components": moved,
        "device": device,
        "gpu_bytes": gpu_stats["held_bytes"],
        "gpu_stats": gpu_stats,
    }
