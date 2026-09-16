"""GPU integration tests for the three residency policies.

Uses a real torchrun worker on GPU 0 (quick-run init, no inference forward). The point of
these tests is that the pack does what the user picked and nothing more: keep_gpu holds
VRAM, release gives it back, park_cpu gives it back without losing the worker, and a
second Model node never evicts the first.

Run:
  XDIT_RUN_GPU_TESTS=1 XDIT_RUN_GPU_RESIDENCY_TESTS=1 \\
    pytest tests/integration/gpu/test_residency.py -m gpu_residency -v

Optional:
  XDIT_RESIDENCY_GPU_PRESET=z_image_turbo.1gpu.rdna4
  XDIT_RESIDENCY_MIN_FREE_MIB=8000
"""

from __future__ import annotations

import os

import pytest

from xdit_comfyui.registry import REGISTRY
from xdit_comfyui.residency_allocator import WORKER_STATE_CPU_PARKED
from xdit_comfyui.runner_contract import (
    RESIDENCY_KEEP_GPU,
    RESIDENCY_PARK_CPU,
    RESIDENCY_RELEASE,
)
from xdit_comfyui.worker import (
    _release_loader_after_run,
    register_prompt_loader_consumers,
)

from ..gpu_residency import GpuResidencyHarness

pytestmark = [pytest.mark.gpu_live, pytest.mark.gpu_residency, pytest.mark.gpu_init]


@pytest.fixture
def require_gpu_residency(require_gpu_headroom):
    if os.environ.get("XDIT_RUN_GPU_RESIDENCY_TESTS", "").strip() != "1":
        pytest.skip("Set XDIT_RUN_GPU_RESIDENCY_TESTS=1 for GPU residency integration tests")
    return require_gpu_headroom


@pytest.fixture
def enable_quick_run(monkeypatch):
    monkeypatch.setenv("XDIT_QUICK_RUN", "1")


@pytest.fixture
def gpu_residency_harness(require_gpu_residency, enable_quick_run):
    harness = GpuResidencyHarness()
    harness.cleanup()
    register_prompt_loader_consumers({})
    yield harness
    harness.cleanup()


def test_device_memory_reads_live_gpu(gpu_residency_harness: GpuResidencyHarness):
    mem = gpu_residency_harness.device_memory("0")
    assert mem["total_gib"] > 0
    assert abs(mem["free_gib"] + mem["used_gib"] - mem["total_gib"]) < 0.5


def test_warm_takes_vram_and_the_worker_reports_it(
    gpu_residency_harness: GpuResidencyHarness,
):
    free_before = gpu_residency_harness.device_free_gib("0")
    gpu_residency_harness.warm_loader("2")
    assert gpu_residency_harness.loader_alive("2")

    held = gpu_residency_harness.held_gib("2")
    assert held > 0, "worker reported no resident VRAM after warm"
    free_after = gpu_residency_harness.device_free_gib("0")
    assert free_after < free_before
    # What the worker says it holds has to show up on the device.
    assert (free_before - free_after) >= held - 1.0


def test_keep_gpu_holds_vram_across_the_sample(gpu_residency_harness: GpuResidencyHarness):
    runtime = gpu_residency_harness.warm_loader("2", residency=RESIDENCY_KEEP_GPU)
    free_warm = gpu_residency_harness.device_free_gib("0")
    register_prompt_loader_consumers({"2": 1})
    _release_loader_after_run(runtime)

    assert gpu_residency_harness.loader_alive("2")
    assert gpu_residency_harness.device_free_gib("0") <= free_warm + 1.0


def test_release_gives_the_vram_back(gpu_residency_harness: GpuResidencyHarness):
    free_before = gpu_residency_harness.device_free_gib("0")
    runtime = gpu_residency_harness.warm_loader("2", residency=RESIDENCY_RELEASE)
    register_prompt_loader_consumers({"2": 1})
    _release_loader_after_run(runtime)

    assert not gpu_residency_harness.loader_alive("2")
    gpu_residency_harness.cleanup()
    assert gpu_residency_harness.device_free_gib("0") >= free_before - 1.0


def test_park_cpu_frees_vram_without_losing_the_worker(
    gpu_residency_harness: GpuResidencyHarness,
):
    runtime = gpu_residency_harness.warm_loader("2", residency=RESIDENCY_PARK_CPU)
    held = gpu_residency_harness.held_gib("2")
    free_warm = gpu_residency_harness.device_free_gib("0")
    register_prompt_loader_consumers({"2": 1})
    _release_loader_after_run(runtime)

    if not gpu_residency_harness.loader_alive("2"):
        pytest.skip("this layout cannot park and fell back to release")
    assert REGISTRY.workers["2"]["residency_state"] == WORKER_STATE_CPU_PARKED
    freed = gpu_residency_harness.device_free_gib("0") - free_warm
    assert freed > held / 2, f"park returned only {freed:.1f} GiB of {held:.1f} GiB"


def test_a_second_model_never_evicts_the_first(gpu_residency_harness: GpuResidencyHarness):
    """The user asked for keep_gpu on node 2; warming node 5 must not override that."""
    gpu_residency_harness.warm_loader("2", residency=RESIDENCY_KEEP_GPU)
    held = gpu_residency_harness.held_gib("2")
    free = gpu_residency_harness.device_free_gib("0")
    if free < held + 1.0:
        pytest.skip(
            f"only {free:.1f} GiB free after warm; a second copy needs ~{held:.1f} GiB "
            "and would OOM rather than test co-residency"
        )

    gpu_residency_harness.warm_loader("5", residency=RESIDENCY_KEEP_GPU)
    assert gpu_residency_harness.loader_alive("2")
    assert gpu_residency_harness.loader_alive("5")


def test_releasing_one_model_leaves_its_gpu_neighbour_running(
    gpu_residency_harness: GpuResidencyHarness,
):
    runtime = gpu_residency_harness.warm_loader("2", residency=RESIDENCY_RELEASE)
    held = gpu_residency_harness.held_gib("2")
    if gpu_residency_harness.device_free_gib("0") < held + 1.0:
        pytest.skip("not enough free VRAM for two co-resident workers")
    gpu_residency_harness.warm_loader("5", residency=RESIDENCY_KEEP_GPU)

    register_prompt_loader_consumers({"2": 1})
    _release_loader_after_run(runtime)
    assert not gpu_residency_harness.loader_alive("2")
    assert gpu_residency_harness.loader_alive("5")
