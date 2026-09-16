"""Residency contracts against a child process that speaks the worker protocol.

Residency is a user choice, so the property under test throughout is that the pack does
exactly what was asked and nothing else — in particular that warming a second Model node
never takes GPUs away from the first one.
"""

from __future__ import annotations

import pytest

from tests.support.fake_worker import REAL_POPEN, spawn_fake_worker, worker_init_config
from xdit_comfyui import worker
from xdit_comfyui.registry import REGISTRY
from xdit_comfyui.residency_allocator import (
    WORKER_STATE_CPU_PARKED,
    WORKER_STATE_GPU_WARM,
)
from xdit_comfyui.runner_contract import (
    RESIDENCY_KEEP_GPU,
    RESIDENCY_PARK_CPU,
    RESIDENCY_RELEASE,
)


@pytest.fixture(autouse=True)
def fake_worker_child(monkeypatch):
    worker._clear_all_runtime_caches()

    def _popen(cmd, **kwargs):
        if "torch.distributed.run" not in list(cmd):
            return REAL_POPEN(cmd, **kwargs)
        env = {**(kwargs.pop("env", None) or {}), "FAKE_WORKER_BEHAVIOUR": "serve"}
        return spawn_fake_worker(cmd, env=env, **kwargs)

    monkeypatch.setattr(worker.subprocess, "Popen", _popen)
    worker.register_prompt_loader_consumers({})
    yield
    worker._clear_all_runtime_caches()


def _runtime(loader_uid="2", gpus="0", residency=RESIDENCY_KEEP_GPU, **overrides):
    return {
        **worker_init_config(),
        "_loader_node_id": loader_uid,
        "_cuda_visible_devices": gpus,
        "_residency": residency,
        "fully_shard_degree": 1,
        **overrides,
    }


def _warm(loader_uid="2", gpus="0", residency=RESIDENCY_KEEP_GPU, **overrides):
    runtime = _runtime(loader_uid, gpus, residency, **overrides)
    cache_key = worker._runtime_cache_key(runtime)
    worker._register_loader_cache(loader_uid, cache_key, runtime)
    entry, _ = worker._get_or_create_distributed_worker(
        cache_key,
        runtime,
        {},
        1,
        loader_uid=loader_uid,
        requester_gpus=worker._device_id_list(gpus),
        timeout_seconds=60,
    )
    return runtime, cache_key, entry


@pytest.mark.contract
def test_two_models_on_the_same_gpu_both_stay_resident():
    _warm("2", gpus="0")
    _warm("5", gpus="0")
    assert worker._loader_worker_alive("2")
    assert worker._loader_worker_alive("5")


@pytest.mark.contract
def test_keep_gpu_survives_the_sample_that_used_it():
    runtime, _, _ = _warm(residency=RESIDENCY_KEEP_GPU)
    worker.register_prompt_loader_consumers({"2": 1})
    worker._release_loader_after_run(runtime)
    assert worker._loader_worker_alive("2")


@pytest.mark.contract
def test_release_stops_the_worker_after_the_last_sample():
    runtime, _, _ = _warm(residency=RESIDENCY_RELEASE)
    worker.register_prompt_loader_consumers({"2": 2})
    worker._release_loader_after_run(runtime)
    assert worker._loader_worker_alive("2")
    worker._release_loader_after_run(runtime)
    assert not worker._loader_worker_alive("2")


@pytest.mark.contract
def test_releasing_one_model_leaves_its_gpu_neighbour_running():
    runtime, _, _ = _warm("2", gpus="0", residency=RESIDENCY_RELEASE)
    _warm("5", gpus="0", residency=RESIDENCY_KEEP_GPU)
    worker.register_prompt_loader_consumers({"2": 1})
    worker._release_loader_after_run(runtime)
    assert not worker._loader_worker_alive("2")
    assert worker._loader_worker_alive("5")


@pytest.mark.contract
def test_park_cpu_keeps_the_worker_but_marks_it_parked():
    runtime, _, _ = _warm(residency=RESIDENCY_PARK_CPU)
    worker.register_prompt_loader_consumers({"2": 1})
    worker._release_loader_after_run(runtime)
    assert worker._loader_worker_alive("2")
    assert REGISTRY.workers["2"]["residency_state"] == WORKER_STATE_CPU_PARKED


@pytest.mark.contract
def test_a_parked_worker_is_restored_instead_of_respawned():
    runtime, cache_key, entry = _warm(residency=RESIDENCY_PARK_CPU)
    pid = entry["proc"].pid
    worker.register_prompt_loader_consumers({"2": 1})
    worker._release_loader_after_run(runtime)

    reused, created = worker._get_or_create_distributed_worker(
        cache_key,
        runtime,
        {},
        1,
        loader_uid="2",
        requester_gpus=["0"],
        timeout_seconds=60,
    )
    assert not created
    assert reused["proc"].pid == pid
    assert reused["residency_state"] == WORKER_STATE_GPU_WARM


@pytest.mark.contract
def test_park_cpu_keeps_worker_warm_when_the_layout_cannot_park():
    runtime, _, _ = _warm(residency=RESIDENCY_PARK_CPU, fully_shard_degree=8)
    worker.register_prompt_loader_consumers({"2": 1})
    worker._release_loader_after_run(runtime)
    assert worker._loader_worker_alive("2")
    assert REGISTRY.workers["2"]["residency_state"] == WORKER_STATE_GPU_WARM
