"""Real-GPU helpers for residency policy integration tests."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from xdit_comfyui.residency import _worker_memory_stats
from xdit_comfyui.runner_contract import RESIDENCY_KEEP_GPU
from xdit_comfyui.worker import _loader_worker_alive

from .helpers import (
    DEFAULT_GPU_TAG,
    build_preset_spec_for_tag,
    evict_all_workers_and_wait,
    gpu_min_free_mib,
)

GIB = 1024**3
DEFAULT_RESIDENCY_PRESET = os.environ.get("XDIT_RESIDENCY_GPU_PRESET", "z_image_turbo.1gpu.rdna4")


class GpuResidencyHarness:
    """Warm real workers, read device memory, and hold VRAM outside the worker."""

    def __init__(
        self,
        *,
        preset_name: str = DEFAULT_RESIDENCY_PRESET,
        gpu_tag: str = DEFAULT_GPU_TAG,
        min_free_mib: int | None = None,
    ):
        self.preset_name = preset_name
        self.gpu_tag = gpu_tag
        self.min_free_mib = int(
            min_free_mib
            if min_free_mib is not None
            else os.environ.get("XDIT_RESIDENCY_MIN_FREE_MIB", "8000")
        )

    def cleanup(self) -> None:
        evict_all_workers_and_wait(min_free_mib=self.min_free_mib)

    def device_memory(self, gpu: str = "0") -> dict[str, float]:
        from xdit_comfyui.residency import _device_memory

        row = (_device_memory([gpu]) or {}).get(str(gpu)) or {}
        return {
            "free_gib": float(row.get("free_gib") or 0.0),
            "used_gib": float(row.get("used_gib") or 0.0),
            "total_gib": float(row.get("total_gib") or 0.0),
        }

    def device_free_gib(self, gpu: str = "0") -> float:
        return self.device_memory(gpu)["free_gib"]

    @contextmanager
    def reserve_vram(self, gib: float, gpu: int = 0) -> Iterator[None]:
        """Hold *gib* on the device outside the worker."""
        if gib <= 0:
            yield
            return

        import torch

        device = torch.device(f"cuda:{gpu}")
        chunk_bytes = 256 * 1024 * 1024
        tensors = []
        remaining = int(gib * GIB)
        try:
            while remaining > 0:
                n_bytes = min(chunk_bytes, remaining)
                n_floats = max(n_bytes // 4, 1)
                tensors.append(torch.empty(n_floats, dtype=torch.float32, device=device))
                remaining -= n_bytes
            torch.cuda.synchronize(device)
            yield
        finally:
            tensors.clear()
            torch.cuda.empty_cache()

    def warm_loader(
        self,
        loader_node_id: str,
        *,
        preset_name: str | None = None,
        residency: str = RESIDENCY_KEEP_GPU,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        from xdit_comfyui.nodes import XDiTModel
        from xdit_comfyui.runtime_config import (
            _merge_loader_kwargs,
            _preset_synced_loader_kwargs,
        )

        preset = preset_name or self.preset_name
        spec = build_preset_spec_for_tag(preset, self.gpu_tag)
        merged = _merge_loader_kwargs(spec, _preset_synced_loader_kwargs(spec))
        merged["residency"] = residency
        merged["use_torch_compile"] = False
        if timeout_seconds is not None:
            merged["timeout_seconds"] = timeout_seconds
        runtime = XDiTModel.execute(
            preset=spec,
            unique_id=str(loader_node_id),
            **merged,
        )[0]
        if not runtime.get("_preloaded"):
            raise RuntimeError(
                f"Model node {loader_node_id} did not warm worker for preset {preset!r}"
            )
        return runtime

    def loader_alive(self, loader_node_id: str) -> bool:
        return _loader_worker_alive(str(loader_node_id))

    def held_gib(self, loader_node_id: str, gpu: str = "0") -> float:
        """VRAM the worker's allocator holds, as reported by the worker itself."""
        stats = _worker_memory_stats(str(loader_node_id)) or {}
        for row in stats.get("warm_memory") or []:
            if str(row.get("gpu")) == str(gpu):
                return float(row.get("held_bytes") or 0) / GIB
        return 0.0


def gpu_min_free_mib_all() -> int:
    return gpu_min_free_mib()
