"""Queue the starter graph through ComfyUI like the web UI (POST /prompt + history)."""

from __future__ import annotations

import os
import subprocess
import time

import pytest

from xdit_comfyui.presets import available_gpu_count

from ..comfy_ui_client import ComfyUiClient, comfy_reachable
from ..graph_live import (
    assert_pil_not_black,
    assert_worker_log_healthy,
    build_preset_spec_for_tag,
    build_starter_ui_prompt,
    is_video_preset,
    preset_gpu_count,
    read_comfy_log_tail,
    read_recent_worker_log,
    ui_graph_preset_names,
)
from ..helpers import clear_runtime_caches


def _gpu_free_mib(device_index: int = 0) -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        free, _total = torch.cuda.mem_get_info(device_index)
        return int(free // (1024 * 1024))
    except Exception:
        return 0


def _evict_workers_between_presets() -> None:
    clear_runtime_caches()
    subprocess.run(["pkill", "-f", "distributed_worker"], check=False, capture_output=True)
    subprocess.run(["pkill", "-f", "torchrun"], check=False, capture_output=True)
    time.sleep(3)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@pytest.fixture
def require_comfy_ui_server(comfyui_base_url):
    if os.environ.get("XDIT_RUN_UI_GRAPH_TESTS", "").strip() != "1":
        pytest.skip("Set XDIT_RUN_UI_GRAPH_TESTS=1 to queue graphs through ComfyUI")
    if not comfy_reachable(comfyui_base_url):
        pytest.skip(f"ComfyUI not reachable at {comfyui_base_url}")
    return comfyui_base_url


def _run_ui_preset(
    client: ComfyUiClient,
    preset_name: str,
    *,
    max_steps: int,
    timeout: float,
    min_mean: float,
) -> None:
    spec = build_preset_spec_for_tag(preset_name)
    if preset_gpu_count(spec) > available_gpu_count():
        pytest.skip(f"{preset_name} needs {preset_gpu_count(spec)} GPUs")

    prompt = build_starter_ui_prompt(preset_name, max_inference_steps=max_steps)
    started = time.time()

    result = client.queue_and_wait(prompt, timeout_seconds=timeout)
    if result.queue_error:
        raise AssertionError(f"queue rejected: {result.queue_error}")
    if result.node_errors:
        raise AssertionError(f"node_errors: {result.node_errors}")

    status = result.status or {}
    if status.get("status_str") == "error":
        messages = status.get("messages") or []
        raise AssertionError(f"ComfyUI execution error: {messages}")

    preview_out = result.outputs.get("4") or {}
    save_out = result.outputs.get("5") or {}

    if is_video_preset(spec):
        video_entries = save_out.get("images") or save_out.get("videos") or []
        assert video_entries, f"Save Video produced no output for {preset_name}: {result.outputs}"
        assert save_out.get("animated") in ((True,), [True], True), save_out
    else:
        assert result.outputs.get("3") or preview_out, f"no Sample/Preview output: {result.outputs}"

    preview_images = preview_out.get("images") or []
    if preview_images:
        pil = client.fetch_view_image(preview_images[0])
        assert_pil_not_black(pil, min_mean=min_mean, label=f"{preset_name} preview")

    log_after = read_comfy_log_tail()
    worker_log = read_recent_worker_log(since_mtime=started)
    assert "Prompt executed" in log_after or "got prompt" in log_after
    if worker_log:
        from xdit_comfyui.runtime_env import _quick_run_enabled

        assert_worker_log_healthy(
            worker_log,
            expected_steps=max_steps,
            quick_run=_quick_run_enabled(),
        )


@pytest.mark.comfy_live
@pytest.mark.gpu_live
@pytest.mark.long_gpu
def test_starter_graph_ui_preset_loop(require_comfy_ui_server):
    """Loop presets: queue starter graph via ComfyUI API, inspect logs and outputs."""
    if os.environ.get("XDIT_RUN_GPU_TESTS", "").strip() != "1":
        pytest.skip("Set XDIT_RUN_GPU_TESTS=1 for live inference through ComfyUI")

    max_steps = int(os.environ.get("XDIT_GRAPH_MAX_STEPS", "4"))
    timeout = float(os.environ.get("XDIT_GRAPH_TIMEOUT", "900"))
    min_mean = float(os.environ.get("XDIT_GRAPH_MIN_MEAN", "0.01"))
    presets = ui_graph_preset_names()

    client = ComfyUiClient(require_comfy_ui_server)
    failures: list[str] = []

    for preset_name in presets:
        client.interrupt()
        _evict_workers_between_presets()
        min_free = int(os.environ.get("XDIT_GRAPH_MIN_FREE_MIB", "8000"))
        free_mib = _gpu_free_mib()
        if free_mib < min_free:
            failures.append(
                f"{preset_name}: need >={min_free} MiB free on GPU 0, found {free_mib} MiB"
            )
            continue
        try:
            _run_ui_preset(
                client,
                preset_name,
                max_steps=max_steps,
                timeout=timeout,
                min_mean=min_mean,
            )
        except pytest.skip.Exception:
            raise
        except Exception as exc:
            failures.append(f"{preset_name}: {exc}")

    assert not failures, "UI graph preset failures:\n" + "\n".join(failures)
