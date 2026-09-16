"""Quick worker-init loop — full load/compile path, stub inference forward (XDIT_QUICK_RUN=1).

Exercises every gfx1201 preset that fits visible GPU count: preset → load → worker init
→ preprocess → stub output. Skips the expensive denoising forward.

Run all matching gfx1201 presets (needs GPU, no ComfyUI):
  XDIT_RUN_GPU_TESTS=1 XDIT_RUN_QUICK_GRAPH_TESTS=1 \\
    pytest tests/integration/test_starter_graph_quick_loop.py -m "gpu_init and long_gpu" -v

Subset:
  XDIT_QUICK_GRAPH_PRESETS=z_image_turbo.1gpu.rdna4,flux.1gpu.rdna4 \\
    XDIT_RUN_GPU_TESTS=1 XDIT_RUN_QUICK_GRAPH_TESTS=1 \\
    pytest tests/integration/test_starter_graph_quick_loop.py -m "gpu_init and long_gpu" -v

ComfyUI UI loop with the same stub (restart ComfyUI with XDIT_QUICK_RUN=1 first):
  XDIT_QUICK_RUN=1 bash scripts/docker/restart.sh
  XDIT_RUN_GPU_TESTS=1 XDIT_RUN_QUICK_GRAPH_TESTS=1 XDIT_RUN_UI_GRAPH_TESTS=1 \\
    XDIT_UI_GRAPH_PRESETS=z_image_turbo.1gpu.rdna4,flux.1gpu.rdna4 \\
    pytest tests/integration/test_starter_graph_quick_loop.py -m "comfy_live and gpu_init and long_gpu" -v
"""

from __future__ import annotations

import os

import pytest

from .graph_live import (
    assert_frames_not_black,
    assert_video_tensor_shape,
    assert_worker_log_healthy,
    execute_starter_graph,
    is_video_preset,
    quick_graph_preset_names,
)
from .helpers import build_preset_spec_for_tag, evict_all_workers_and_wait


@pytest.fixture
def enable_quick_run(monkeypatch):
    monkeypatch.setenv("XDIT_QUICK_RUN", "1")


@pytest.mark.gpu_init
@pytest.mark.long_gpu
def test_gfx1201_quick_worker_loop(enable_quick_run, require_gpu_headroom):
    """Loop gfx1201 presets: worker init + stub run, no real forward."""
    if os.environ.get("XDIT_RUN_QUICK_GRAPH_TESTS", "").strip() != "1":
        pytest.skip("Set XDIT_RUN_QUICK_GRAPH_TESTS=1 for quick worker-init preset loop")
    if os.environ.get("XDIT_RUN_GPU_TESTS", "").strip() != "1":
        pytest.skip("Set XDIT_RUN_GPU_TESTS=1 for GPU quick worker loop")

    presets = quick_graph_preset_names()
    assert presets, "no quick-graph presets resolved"

    timeout = int(os.environ.get("XDIT_GRAPH_TIMEOUT", "900"))
    min_free = int(os.environ.get("XDIT_GRAPH_MIN_FREE_MIB", "8000"))
    failures: list[str] = []
    total = len(presets)

    print(
        f"\nquick_run: {total} preset(s) (gpu_count={os.environ.get('XDIT_QUICK_GRAPH_GPU_COUNT', 'any')}), timeout={timeout}s",
        flush=True,
    )
    evict_all_workers_and_wait(min_free_mib=min_free)

    for index, preset_name in enumerate(presets, start=1):
        print(f"[{index}/{total}] {preset_name} ...", flush=True)
        try:
            evict_all_workers_and_wait(min_free_mib=min_free)
        except Exception as exc:
            print(f"  skip: {exc}", flush=True)
            failures.append(f"{preset_name}: {exc}")
            continue

        spec = build_preset_spec_for_tag(preset_name)
        try:
            result = execute_starter_graph(
                preset_name,
                max_inference_steps=4,
                timeout_seconds=timeout,
                save_video=is_video_preset(spec),
            )
            combined_log = "\n".join(
                part for part in (result.worker_log, result.dispatch_log) if part
            )
            assert_worker_log_healthy(combined_log, quick_run=True)
            if is_video_preset(spec):
                assert result.video is not None
                assert_video_tensor_shape(result.video, spec)
                assert_frames_not_black(result.images, label=f"{preset_name} preview")
            else:
                assert result.video is None
                assert_frames_not_black(result.images, label=preset_name)
            print("  ok", flush=True)
        except pytest.skip.Exception:
            raise
        except Exception as exc:
            print(f"  FAIL: {exc}", flush=True)
            failures.append(f"{preset_name}: {exc}")

    assert not failures, "quick worker preset failures:\n" + "\n".join(failures)


@pytest.mark.comfy_live
@pytest.mark.gpu_init
@pytest.mark.long_gpu
def test_gfx1201_quick_ui_loop(enable_quick_run, comfyui_base_url):
    """ComfyUI queue loop with XDIT_QUICK_RUN=1 (ComfyUI must inherit the env var)."""
    from xdit_comfyui.runtime_env import _quick_run_enabled

    from .comfy_ui_client import ComfyUiClient, comfy_reachable
    from .gpu.test_starter_ui import _run_ui_preset

    if os.environ.get("XDIT_RUN_QUICK_GRAPH_TESTS", "").strip() != "1":
        pytest.skip("Set XDIT_RUN_QUICK_GRAPH_TESTS=1")
    if os.environ.get("XDIT_RUN_UI_GRAPH_TESTS", "").strip() != "1":
        pytest.skip("Set XDIT_RUN_UI_GRAPH_TESTS=1")
    if os.environ.get("XDIT_RUN_GPU_TESTS", "").strip() != "1":
        pytest.skip("Set XDIT_RUN_GPU_TESTS=1")
    if not comfy_reachable(comfyui_base_url):
        pytest.skip(f"ComfyUI not reachable at {comfyui_base_url}")
    if not _quick_run_enabled():
        pytest.skip("Restart ComfyUI with XDIT_QUICK_RUN=1 for UI quick loop")

    presets = quick_graph_preset_names()
    client = ComfyUiClient(comfyui_base_url)
    failures: list[str] = []

    for preset_name in presets:
        client.interrupt()
        evict_all_workers_and_wait(
            min_free_mib=int(os.environ.get("XDIT_GRAPH_MIN_FREE_MIB", "8000"))
        )
        try:
            _run_ui_preset(
                client,
                preset_name,
                max_steps=4,
                timeout=float(os.environ.get("XDIT_GRAPH_TIMEOUT", "900")),
                min_mean=0.01,
            )
        except pytest.skip.Exception:
            raise
        except Exception as exc:
            failures.append(f"{preset_name}: {exc}")

    assert not failures, "UI quick preset failures:\n" + "\n".join(failures)
