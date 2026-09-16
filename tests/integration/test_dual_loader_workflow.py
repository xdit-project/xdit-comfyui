import os

import pytest

from xdit_comfyui.presets import available_gpu_count
from xdit_comfyui.prompt_hooks import apply_preset_prompt_overrides
from xdit_comfyui.sampling import _execute_sample
from xdit_comfyui.worker import _clear_all_runtime_caches

from .comfy_ui_client import ComfyUiClient, comfy_reachable
from .helpers import (
    build_dual_loader_prompt,
    capture_loader_init_worker_payload,
    hooked_dual_loader_inputs,
    load_dual_runtimes_from_preset,
    sample_execution_kwargs_from_spec,
)

ZIMAGE_2GPU = "z_image_turbo.2gpu.rdna4"
DUPLICATE_WIDGET_BUG = {
    "model": False,
    "custom_model_id": "False",
}


@pytest.fixture(autouse=True)
def _clear_runtime_cache():
    _clear_all_runtime_caches()
    yield
    _clear_all_runtime_caches()


@pytest.mark.contract
def test_prompt_hook_repairs_duplicate_loader_widgets_on_both_loaders():
    loader_a, loader_b = hooked_dual_loader_inputs(
        ZIMAGE_2GPU,
        stale_loaders=(DUPLICATE_WIDGET_BUG, DUPLICATE_WIDGET_BUG),
    )
    for inputs in (loader_a, loader_b):
        assert inputs["model"] == "Tongyi-MAI/Z-Image-Turbo"
        assert inputs["custom_model_id"] == ""
        assert inputs["ulysses_degree"] == 2
    assert loader_a["gpu_device_ids"] == "0,1"
    assert loader_b["gpu_device_ids"] == "2,3"


@pytest.mark.contract
def test_dual_loader_prompt_keeps_distinct_gpu_device_ids():
    payload = apply_preset_prompt_overrides(build_dual_loader_prompt(ZIMAGE_2GPU))
    assert payload["prompt"]["2"]["inputs"]["gpu_device_ids"] == "0,1"
    assert payload["prompt"]["5"]["inputs"]["gpu_device_ids"] == "2,3"


@pytest.mark.contract
def test_dual_loader_init_payloads_use_zimage_on_separate_devices():
    stale_a = {**DUPLICATE_WIDGET_BUG, "gpu_device_ids": "0,1"}
    stale_b = {**DUPLICATE_WIDGET_BUG, "gpu_device_ids": "2,3"}
    _, payload_a = capture_loader_init_worker_payload(
        ZIMAGE_2GPU,
        stale_loader=stale_a,
    )
    _, payload_b = capture_loader_init_worker_payload(
        ZIMAGE_2GPU,
        stale_loader=stale_b,
    )
    assert payload_a["model"] == "Tongyi-MAI/Z-Image-Turbo"
    assert payload_b["model"] == "Tongyi-MAI/Z-Image-Turbo"
    assert payload_a["ulysses_degree"] == 2
    assert payload_b["ulysses_degree"] == 2


@pytest.mark.contract
def test_dual_loader_graph_dry_run_after_duplicate_widget_repair():
    spec, (runtime_a, runtime_b) = load_dual_runtimes_from_preset(
        ZIMAGE_2GPU,
        stale_loaders=(DUPLICATE_WIDGET_BUG, DUPLICATE_WIDGET_BUG),
    )
    kwargs = sample_execution_kwargs_from_spec(spec)
    for runtime in (runtime_a, runtime_b):
        assert runtime["model"] == "Tongyi-MAI/Z-Image-Turbo"
        assert runtime["_cuda_visible_devices"] in {"0,1", "2,3"}
        images, video = _execute_sample(
            runtime,
            preset=spec,
            output_type="pil",
            dry_run=True,
            **kwargs,
        )
        assert video is None
        assert tuple(images.shape[1:]) == (64, 64, 3)


@pytest.mark.comfy_live
@pytest.mark.gpu_live
def test_dual_loader_graph_runs_through_live_comfy(comfyui_base_url):
    if os.environ.get("XDIT_RUN_GPU_TESTS", "").strip() != "1":
        pytest.skip("Set XDIT_RUN_GPU_TESTS=1 for live dual-loader inference")
    if not comfy_reachable(comfyui_base_url):
        pytest.skip(f"ComfyUI not reachable at {comfyui_base_url}")
    if available_gpu_count() < 4:
        pytest.skip("Dual 2-GPU loader graph requires four visible GPUs")

    prompt = build_dual_loader_prompt(ZIMAGE_2GPU)["prompt"]
    result = ComfyUiClient(comfyui_base_url).queue_and_wait(prompt, timeout_seconds=900)

    assert result.queue_error is None
    assert not result.node_errors
    assert result.status.get("status_str") == "success", result.status.get("messages")
    assert {"4", "7"}.issubset(result.outputs)
    queued = result.history["prompt"][2]
    assert queued["1"]["inputs"]["preset"] == ZIMAGE_2GPU
    assert queued["2"]["inputs"]["gpu_device_ids"] == "0,1"
    assert queued["5"]["inputs"]["gpu_device_ids"] == "2,3"
