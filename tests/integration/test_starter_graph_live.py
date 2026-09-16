"""Long integration tests — starter graph, preset values, live inference, logs, outputs.

Run (image smoke, 4 steps, default z_image_turbo — fits 32GB RDNA4):
  XDIT_RUN_GPU_TESTS=1 pytest tests/integration/test_starter_graph_live.py -m gpu_live -k live_run

Run flux explicitly (needs headroom; may OOM on 32GB during bf16 .to(cuda)):
  XDIT_RUN_GPU_TESTS=1 XDIT_GRAPH_LIVE_PRESETS=flux.1gpu.rdna4 \\
    pytest tests/integration/test_starter_graph_live.py -m gpu_live -k live_run

Run Wan full graph (long, known black-frame issue marked xfail):
  XDIT_RUN_GPU_TESTS=1 XDIT_GRAPH_LIVE_PRESETS=wan2_2_ti2v_5b.i2v.4gpu.rdna4 \\
    pytest tests/integration/test_starter_graph_live.py -m gpu_live -k wan

Stop ComfyUI or evict its worker first — live tests need ~12 GiB free on GPU 0.
"""

from __future__ import annotations

import os
import subprocess
import time

import pytest

from .graph_live import (
    assert_frames_not_black,
    assert_preset_values_in_sample_kwargs,
    assert_video_tensor_shape,
    assert_worker_log_healthy,
    execute_starter_graph,
    execute_starter_graph_via_sample_node,
    is_video_preset,
    live_graph_preset_names,
    preset_gpu_count,
    validate_hooked_starter_prompt,
)
from .helpers import build_preset_spec_for_tag, clear_runtime_caches


def _gpu_free_mib(device_index: int = 0) -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        free, _total = torch.cuda.mem_get_info(device_index)
        return int(free // (1024 * 1024))
    except Exception:
        return 0


@pytest.fixture
def require_gpu_headroom(require_gpu_live):
    """Stop ComfyUI workers and skip when VRAM is too full for a cold worker init."""
    from .helpers import stop_comfyui_dev_script

    stop_script = stop_comfyui_dev_script()
    if os.path.isfile(stop_script):
        subprocess.run(["bash", stop_script], check=False, capture_output=True, text=True)
        time.sleep(3)
    clear_runtime_caches()
    subprocess.run(["pkill", "-f", "xdit_worker_"], check=False, capture_output=True)
    time.sleep(1)

    min_free = int(os.environ.get("XDIT_GRAPH_MIN_FREE_MIB", "12000"))
    free_mib = _gpu_free_mib()
    if free_mib < min_free:
        pytest.skip(
            f"need >={min_free} MiB free on GPU 0 for live graph test, found {free_mib} MiB "
            "(stop ComfyUI workers or set XDIT_GRAPH_MIN_FREE_MIB lower)"
        )
    return free_mib


@pytest.mark.contract
@pytest.mark.parametrize("preset_name", live_graph_preset_names())
def test_starter_graph_hooked_prompt_matches_preset(preset_name):
    prompt = validate_hooked_starter_prompt(preset_name)
    spec = build_preset_spec_for_tag(preset_name)
    assert prompt["1"]["inputs"]["preset"] == preset_name
    assert prompt["2"]["inputs"]["preset"] == ["1", 0]
    assert prompt["3"]["inputs"]["preset"] == ["1", 2]
    # The starter graph wires both outputs; Sample blocks the branch it did not produce.
    assert prompt["4"]["class_type"] == "SaveImage"
    assert prompt["4"]["inputs"]["images"] == ["3", 0]
    assert prompt["5"]["class_type"] == "SaveVideo"
    assert prompt["5"]["inputs"]["video"] == ["3", 1]
    if is_video_preset(spec):
        assert prompt["3"]["inputs"]["images"] == ["1", 1]


@pytest.mark.gpu_live
@pytest.mark.parametrize("preset_name", live_graph_preset_names())
def test_starter_graph_live_run(preset_name, require_gpu_headroom, require_gpu_count):
    spec = build_preset_spec_for_tag(preset_name)
    require_gpu_count(preset_gpu_count(spec))

    max_steps = int(os.environ.get("XDIT_GRAPH_MAX_STEPS", "4"))
    timeout = int(os.environ.get("XDIT_GRAPH_TIMEOUT", "900"))
    min_mean = float(os.environ.get("XDIT_GRAPH_MIN_MEAN", "0.01"))

    result = execute_starter_graph(
        preset_name,
        max_inference_steps=max_steps,
        timeout_seconds=timeout,
        save_video=is_video_preset(spec),
    )

    assert_preset_values_in_sample_kwargs(spec, result.sample_kwargs)
    assert result.sample_kwargs["num_inference_steps"] == max_steps

    combined_log = "\n".join(part for part in (result.worker_log, result.dispatch_log) if part)
    assert_worker_log_healthy(
        combined_log,
        expected_steps=max_steps,
        allow_compile_warmup=True,
    )

    defaults = spec["generation_defaults"]
    if is_video_preset(spec):
        assert result.video is not None
        assert_video_tensor_shape(result.video, spec)
        preview = (result.save_video_ui or {}).get("images") or []
        assert preview, "Save Video should emit animated preview metadata"
        assert (result.save_video_ui or {}).get("animated") == (True,)
        assert result.saved_video_path is not None
        assert result.saved_video_path.stat().st_size > 0
    else:
        assert result.video is None
        assert result.images.shape[1] == defaults["height"]
        assert result.images.shape[2] == defaults["width"]
        assert_frames_not_black(result.images, min_mean=min_mean)


@pytest.mark.gpu_live
def test_starter_graph_via_sample_node(require_gpu_headroom, require_gpu_count):
    """Smoke the ComfyUI node entrypoint (XDiTSample.sample) on a fast image preset."""
    preset_name = "z_image_turbo.1gpu.rdna4"
    require_gpu_count(1)

    result = execute_starter_graph_via_sample_node(
        preset_name,
        max_inference_steps=4,
        timeout_seconds=900,
    )

    assert result.video is None
    assert result.images.shape[1] == result.spec["generation_defaults"]["height"]
    assert_worker_log_healthy(result.worker_log, expected_steps=4)
    assert_frames_not_black(result.images)


@pytest.mark.gpu_live
@pytest.mark.long_gpu
@pytest.mark.parametrize("preset_name", ("wan2_2_ti2v_5b.i2v.4gpu.rdna4",))
def test_starter_graph_wan_video_long(preset_name, require_gpu_headroom, require_gpu_count):
    """Full-step Wan video graph — long; opt-in via XDIT_GRAPH_LIVE_PRESETS."""
    if preset_name not in live_graph_preset_names(default=preset_name):
        pytest.skip(f"Set XDIT_GRAPH_LIVE_PRESETS to include {preset_name!r} to run Wan long test")

    spec = build_preset_spec_for_tag(preset_name)
    require_gpu_count(preset_gpu_count(spec))

    max_steps = (
        int(os.environ.get("XDIT_WAN_MAX_STEPS", "0"))
        or spec["generation_defaults"]["num_inference_steps"]
    )
    timeout = int(os.environ.get("XDIT_GRAPH_TIMEOUT", "1800"))

    result = execute_starter_graph(
        preset_name,
        max_inference_steps=max_steps,
        timeout_seconds=timeout,
        save_video=True,
    )

    assert result.video is not None
    combined_log = "\n".join(part for part in (result.worker_log, result.dispatch_log) if part)
    assert_worker_log_healthy(combined_log, expected_steps=max_steps)
    assert_video_tensor_shape(result.video, spec)
    assert_frames_not_black(result.video.get_components().images, min_mean=0.01, label="frames")
    assert result.saved_video_path is not None
