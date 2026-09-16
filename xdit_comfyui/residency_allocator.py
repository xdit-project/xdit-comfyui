"""GPU residency policy: what a Model node does with its GPUs after the last Sample.

The policy is the user's choice and nothing here second-guesses it. A Model node never
takes VRAM away from another one: co-residency is a supported setup, so two loaders that
overlap on a GPU both stay resident until one of them actually runs out of memory, and
that failure names who is holding what (`residency._oom_attribution_text`).
"""

from __future__ import annotations

from typing import Callable

from .log_context import residency_logger
from .registry import REGISTRY
from .runner_contract import (
    RESIDENCY_CHOICES,
    RESIDENCY_KEEP_GPU,
    RESIDENCY_PARK_CPU,
    RESIDENCY_RELEASE,
)
from .xdit_ext.residency_park import park_feasible

_RESIDENCY_LOG = residency_logger()

WORKER_STATE_GPU_WARM = "gpu_warm"
WORKER_STATE_CPU_PARKED = "cpu_parked"
WORKER_STATE_COLD = "cold"

EvictFn = Callable[[str], bool]
ParkFn = Callable[[str], bool]
AliveFn = Callable[[str], bool]


def normalize_residency(value) -> str:
    text = str(value or "").strip()
    return text if text in RESIDENCY_CHOICES else RESIDENCY_KEEP_GPU


def residency_pins_gpu(policy: str) -> bool:
    return normalize_residency(policy) == RESIDENCY_KEEP_GPU


def residency_choices_for_runtime(runtime: dict | None) -> tuple[list[str], str]:
    """Policies safe to offer for this resolved distributed layout."""
    can_park, reason = park_feasible(runtime)
    choices = [RESIDENCY_KEEP_GPU]
    if can_park:
        choices.append(RESIDENCY_PARK_CPU)
    choices.append(RESIDENCY_RELEASE)
    return choices, reason


def demote_policy_after_run(policy: str) -> str | None:
    """The action the policy asks for once the last Sample is done, or None to stay."""
    normalized = normalize_residency(policy)
    if normalized == RESIDENCY_PARK_CPU:
        return "park"
    if normalized == RESIDENCY_RELEASE:
        return "release"
    return None


def _last_consumer_finished(loader_uid: str) -> bool:
    """False while other Sample nodes in this prompt still need the worker."""
    with REGISTRY.lock():
        remaining = REGISTRY.consumers.get(loader_uid, 1) - 1
        if remaining > 0:
            REGISTRY.consumers[loader_uid] = remaining
            return False
        REGISTRY.consumers.pop(loader_uid, None)
    return True


def demote_loader_after_run(
    runtime: dict,
    *,
    park_fn: ParkFn,
    evict_fn: EvictFn,
) -> None:
    if not isinstance(runtime, dict):
        return
    loader_uid = str(runtime.get("_loader_node_id") or "").strip()
    if not loader_uid:
        return
    mode = demote_policy_after_run(runtime.get("_residency"))
    if mode is None or not _last_consumer_finished(loader_uid):
        return

    if mode == "park":
        can_park, reason = park_feasible(runtime)
        if can_park and park_fn(loader_uid):
            _RESIDENCY_LOG.info(
                "residency=park_cpu moved Model node %s to host RAM after the last Sample",
                loader_uid,
            )
            return
        if not can_park:
            _RESIDENCY_LOG.info(
                "residency=park_cpu cannot park Model node %s (%s); leaving it GPU-warm",
                loader_uid,
                reason,
            )
        else:
            _RESIDENCY_LOG.info(
                "residency=park_cpu was rejected for Model node %s; leaving it GPU-warm",
                loader_uid,
            )
        return

    if evict_fn(loader_uid):
        _RESIDENCY_LOG.info("residency=release stopped the worker for Model node %s", loader_uid)
