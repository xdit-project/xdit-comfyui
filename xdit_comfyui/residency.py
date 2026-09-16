"""Which Model node holds which GPU, and what the last run peaked at."""

import json
import time

from .progress import _node_id_str
from .registry import REGISTRY, node_status_snapshot


def _device_id_list(cuda_visible_devices):
    return [part.strip() for part in str(cuda_visible_devices or "").split(",") if part.strip()]


def _gib(value):
    try:
        return round(float(value) / (1024**3), 2)
    except (TypeError, ValueError):
        return None


def _device_memory(device_ids):
    """Device-level used/total, which counts every process on the GPU, not just ours."""
    try:
        import torch
    except Exception:
        return {}
    if not torch.cuda.is_available():
        return {}
    usage = {}
    for device_id in device_ids:
        try:
            index = int(device_id)
            free, total = torch.cuda.mem_get_info(index)
        except Exception:
            continue
        usage[str(device_id)] = {
            "free_gib": _gib(free),
            "total_gib": _gib(total),
            "used_gib": _gib(total - free),
        }
    return usage


def _worker_memory_stats(loader_uid):
    # Imported here because worker.py reports run peaks through this module.
    from .worker import _loader_worker_token, _worker_stats_path

    path = _worker_stats_path(_loader_worker_token(loader_uid))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _memory_rows(entries):
    rows = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        rows.append(
            {
                "gpu": str(entry.get("gpu", "")),
                "held_gib": _gib(entry.get("held_bytes")),
                "live_gib": _gib(entry.get("live_bytes")),
                "peak_gib": _gib(entry.get("peak_bytes")),
                "peak_held_gib": _gib(entry.get("peak_held_bytes")),
                "device_used_gib": _gib(
                    (entry.get("device_total_bytes") or 0) - (entry.get("device_free_bytes") or 0)
                ),
                "device_free_gib": _gib(entry.get("device_free_bytes")),
                "device_total_gib": _gib(entry.get("device_total_bytes")),
                "alloc_retries": int(entry.get("alloc_retries") or 0),
            }
        )
    return rows


def _pair_memory_rows(warm_rows, run_rows):
    """Split each run peak into weights (warm baseline) and activations.

    Everything here is reserved memory (the allocator pool), matching what the Model
    node reports as resident. Peak *allocated* would read lower than the resident
    figure whenever the pool holds free blocks, which looks like a contradiction.
    """
    warm_by_gpu = {row["gpu"]: row for row in warm_rows}
    paired = []
    for row in run_rows:
        warm = warm_by_gpu.get(row["gpu"]) or {}
        weights = warm.get("held_gib")
        peak = row.get("peak_held_gib")
        activations = None
        if peak is not None and weights is not None:
            activations = round(max(peak - weights, 0.0), 2)
        paired.append(
            {
                **row,
                "peak_gib": peak,
                "peak_allocated_gib": row.get("peak_gib"),
                "weights_gib": weights,
                "activation_gib": activations,
            }
        )
    return paired


def record_sample_run_memory(sample_node_id, metadata, loader_uid=None):
    """Keep the last run's per-GPU peaks for the Sample node's info block."""
    node_id = _node_id_str(sample_node_id)
    rows = _pair_memory_rows(
        _memory_rows((metadata or {}).get("warm_memory")),
        _memory_rows((metadata or {}).get("run_memory")),
    )
    if not rows:
        return ""
    if node_id:
        with REGISTRY.lock():
            REGISTRY.run_stats[node_id] = {
                "rows": rows,
                "at": time.time(),
                "loader_node_id": _node_id_str(loader_uid),
            }
    return _format_run_memory_rows(rows)


def _drop_sample_run_memory_for_loader(loader_uid):
    """A stopped worker invalidates its run peaks; the Sample block must not show them."""
    loader_id = _node_id_str(loader_uid)
    if not loader_id:
        return
    with REGISTRY.lock():
        for node_id, stats in list(REGISTRY.run_stats.items()):
            if (stats or {}).get("loader_node_id") == loader_id:
                REGISTRY.run_stats.pop(node_id, None)


def sample_run_memory(sample_node_id):
    node_id = _node_id_str(sample_node_id)
    with REGISTRY.lock():
        return dict(REGISTRY.run_stats.get(node_id) or {})


def _format_run_memory_rows(rows):
    parts = []
    for row in rows:
        text = f"GPU{row['gpu']} peak {row['peak_gib']}GiB"
        if row.get("weights_gib") is not None and row.get("activation_gib") is not None:
            text += f" ({row['weights_gib']} weights + {row['activation_gib']} activations)"
        if row.get("device_free_gib") is not None:
            text += f", {row['device_free_gib']}GiB free at run end"
        if row.get("alloc_retries"):
            text += f", alloc_retries={row['alloc_retries']}"
        parts.append(text)
    return "; ".join(parts)


def _footprint_rows(warm_rows, latest_rows, devices):
    """Per-GPU split where model + other + free == device total.

    `model` is this model's allocator pool; `other` is everything else on the device,
    which covers co-resident workers plus our own out-of-allocator overhead (HIP
    context, library workspaces).
    """
    weights_by_gpu = {row["gpu"]: row.get("held_gib") for row in warm_rows}
    rows = []
    for row in latest_rows or warm_rows:
        gpu = row["gpu"]
        device = devices.get(gpu) or {}
        model = row.get("held_gib")
        used = device.get("used_gib")
        other = None
        if model is not None and used is not None:
            other = round(max(used - model, 0.0), 2)
        rows.append(
            {
                "gpu": gpu,
                "model_gib": model,
                "weights_gib": weights_by_gpu.get(gpu),
                "other_gib": other,
                "free_gib": device.get("free_gib"),
                "total_gib": device.get("total_gib"),
            }
        )
    return rows


def _loader_residency_entry(loader_uid):
    from .residency_allocator import WORKER_STATE_CPU_PARKED, WORKER_STATE_GPU_WARM
    from .worker import _distributed_worker_alive

    uid = str(loader_uid or "").strip()
    loader_id = _node_id_str(uid)
    with REGISTRY.lock():
        snapshot = dict(REGISTRY.snapshots.get(loader_id) or {})
        entry = dict(REGISTRY.workers.get(uid) or {})
    warm = _distributed_worker_alive(uid)
    stats = _worker_memory_stats(uid) if warm else {}
    state = entry.get("residency_state")
    if not state and warm:
        state = WORKER_STATE_GPU_WARM
    if not warm:
        state = "cold"
    warm_rows = _memory_rows(stats.get("warm"))
    run_rows = _memory_rows(stats.get("run"))
    gpus = snapshot.get("gpus") or entry.get("gpus") or [row["gpu"] for row in warm_rows]
    devices = _device_memory(gpus)
    snapshot.update(
        {
            "node_id": loader_id,
            "warm": warm and state != WORKER_STATE_CPU_PARKED,
            "parked": warm and state == WORKER_STATE_CPU_PARKED,
            "state": state,
            "host_gib": _gib(entry.get("host_bytes")),
            "gpus": gpus,
            "footprint": _footprint_rows(warm_rows, run_rows, devices),
            "devices": devices,
        }
    )
    return snapshot


def residency_report():
    """Which loaders hold which GPUs, with our footprint and whole-device usage."""
    with REGISTRY.lock():
        loader_uids = list(REGISTRY.cache_keys)
        sample_runs = {str(node_id): dict(stats) for node_id, stats in REGISTRY.run_stats.items()}
    loaders = [_loader_residency_entry(uid) for uid in loader_uids]
    claimed = sorted({gpu for entry in loaders for gpu in entry.get("gpus") or []}, key=str)
    return {
        "loaders": loaders,
        "devices": _device_memory(claimed),
        "node_status": node_status_snapshot(),
        "sample_runs": sample_runs,
    }


def _format_footprint_rows(rows):
    parts = []
    for row in rows:
        text = f"GPU{row['gpu']} {row['model_gib']}GiB this model"
        if row.get("weights_gib") is not None:
            text += f" ({row['weights_gib']}GiB weights)"
        if row.get("other_gib") is not None:
            text += f", {row['other_gib']}GiB other"
        if row.get("free_gib") is not None:
            text += f", {row['free_gib']}GiB free"
        parts.append(text)
    return "; ".join(parts)


def _oom_attribution_text():
    """Who is holding GPU memory right now, so an OOM says what to release."""
    report = residency_report()
    lines = []
    for gpu, usage in sorted(report["devices"].items(), key=lambda item: str(item[0])):
        lines.append(
            f"  GPU {gpu}: {usage['used_gib']}/{usage['total_gib']} GiB used, "
            f"{usage['free_gib']} GiB free"
        )
    if lines:
        lines.insert(0, "GPU memory at failure:")
    resident = []
    for entry in report["loaders"]:
        if not entry.get("warm") and not entry.get("parked"):
            continue
        gpus = ",".join(entry.get("gpus") or []) or "?"
        model = entry.get("model") or "unknown model"
        state = entry.get("state") or "unknown"
        line = f"  Model node {entry.get('node_id')}: {model} on GPU {gpus} ({state})"
        held = _format_footprint_rows(entry.get("footprint") or [])
        if held:
            line += f" — {held}"
        if entry.get("parked") and entry.get("host_gib") is not None:
            line += f"; {entry['host_gib']}GiB on host"
        resident.append(line)
    if resident:
        lines.append("Resident xDiT workers:")
        lines.extend(resident)
        lines.append(
            "Nothing is evicted automatically. Free a GPU with Unload Model (Free VRAM), "
            "or set residency=release (stop the worker) or residency=park_cpu (move its "
            "weights to host RAM) on the Model node that should not stay resident."
        )
    orphans = _orphan_worker_note()
    if orphans:
        lines.append(orphans)
    return "\n".join(lines)


def _orphan_worker_note():
    """Name GPU memory held by workers left behind by a previous ComfyUI."""
    from .worker import _orphan_worker_tokens

    try:
        orphans = _orphan_worker_tokens()
    except Exception:
        return ""
    if not orphans:
        return ""
    return (
        f"{len(orphans)} orphaned xDiT worker(s) from a previous ComfyUI still hold GPU "
        "memory; the next model load releases them automatically."
    )
