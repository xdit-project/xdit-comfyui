"""Run-path integration tests — exercise preset → loader → sample → xDiT config assembly.

These catch wiring bugs that unit tests miss (missing worker keys, bad preset merge,
xfuser preprocess failures). They do not load model weights or require a GPU.
"""

from __future__ import annotations

import os

import pytest

from .helpers import (
    RUN_LOADER_INIT_CACHE_PRESETS,
    RUN_LOADER_INIT_PREPROCESS_PRESETS,
    RUN_SMOKE_PRESETS,
    RUN_VIDEO_PRESET,
    build_preset_spec_for_tag,
    capture_loader_init_worker_payload,
    capture_runner_config,
    hooked_loader_inputs,
    loader_inputs,
    preprocess_via_xfuser,
    validate_loader_init_step_cache,
    warm_loader_from_preset,
)

_WAN_REQUIRED_KEYS = frozenset(
    {
        "guidance_scale_2",
        "num_frames",
        "num_inference_steps",
        "prompt",
        "height",
        "width",
        "guidance_scale",
        "seed",
    }
)

_IMAGE_REQUIRED_KEYS = frozenset(
    {
        "prompt",
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "seed",
        "num_frames",
    }
)


def _model_name(spec: dict) -> str:
    return str(spec.get("model_choice") or spec.get("runtime_widgets", {}).get("model") or "")


def _required_keys_for_spec(spec: dict) -> frozenset[str]:
    model = _model_name(spec).lower()
    if "wan" in model or int((spec.get("generation_defaults") or {}).get("num_frames") or 1) > 1:
        return _WAN_REQUIRED_KEYS
    return _IMAGE_REQUIRED_KEYS


@pytest.mark.contract
@pytest.mark.parametrize("preset_name", RUN_SMOKE_PRESETS)
def test_smoke_preset_worker_config_has_required_keys(preset_name):
    spec, worker_payload = capture_runner_config(preset_name)
    required = _required_keys_for_spec(spec)
    missing = required - set(worker_payload)
    assert not missing, f"{preset_name} missing worker keys: {sorted(missing)}"


@pytest.mark.contract
@pytest.mark.parametrize("preset_name", RUN_SMOKE_PRESETS)
def test_smoke_preset_preprocesses_via_xfuser(preset_name):
    spec, worker_payload = capture_runner_config(preset_name)
    init_args = preprocess_via_xfuser(worker_payload)
    required = _required_keys_for_spec(spec)
    missing = required - set(init_args)
    assert not missing, f"{preset_name} preprocess_args missing: {sorted(missing)}"


@pytest.mark.contract
@pytest.mark.parametrize("preset_name", RUN_SMOKE_PRESETS)
def test_loader_init_worker_payload_has_required_keys(preset_name):
    from xdit_comfyui.worker_payload import LOADER_INIT_REQUIRED_KEYS

    _spec, worker_payload = capture_loader_init_worker_payload(preset_name)
    missing = LOADER_INIT_REQUIRED_KEYS - set(worker_payload)
    assert not missing, f"{preset_name} loader init missing keys: {sorted(missing)}"
    assert worker_payload.get("prompt") not in (None, "")


@pytest.mark.contract
@pytest.mark.parametrize("preset_name", RUN_LOADER_INIT_PREPROCESS_PRESETS)
def test_loader_init_preprocesses_via_xfuser(preset_name):
    """Model warm config must pass xFuser preprocess_args (catches KeyError: prompt)."""
    _spec, worker_payload = capture_loader_init_worker_payload(preset_name)
    preprocess_via_xfuser(worker_payload)


@pytest.mark.contract
def test_loader_init_without_stubs_fails_xfuser_preprocess():
    from xdit_comfyui.worker_payload import worker_config_payload

    bare = worker_config_payload(
        {
            "model": "black-forest-labs/FLUX.1-dev",
            "ulysses_degree": 1,
            "attention_backend": "sdpa",
        }
    )
    with pytest.raises(KeyError, match="prompt"):
        preprocess_via_xfuser(bare)


@pytest.mark.contract
@pytest.mark.parametrize("preset_name", RUN_LOADER_INIT_CACHE_PRESETS)
def test_loader_init_step_cache_compatible(preset_name):
    """Loader warm config must pass cache_dit SCM mask build (worker initialize path)."""
    _spec, worker_payload = capture_loader_init_worker_payload(preset_name)
    assert worker_payload.get("cache_method"), f"{preset_name} expected step cache enabled"
    validate_loader_init_step_cache(worker_payload)


@pytest.mark.contract
def test_loader_init_step_cache_rejects_one_step_stub():
    from xdit_comfyui.model_info import model_generation_defaults
    from xdit_comfyui.worker_payload import loader_init_config

    runtime = {
        "model": "black-forest-labs/FLUX.1-dev",
        "cache_method": "dbcache",
        "ulysses_degree": 1,
        "attention_backend": "sdpa",
    }
    config = loader_init_config(runtime)
    assert (
        config["num_inference_steps"]
        == model_generation_defaults(runtime["model"])["num_inference_steps"]
    )
    bad = dict(config)
    bad["num_inference_steps"] = 1
    with pytest.raises(AssertionError, match="num_inference_steps=1"):
        validate_loader_init_step_cache(bad)


@pytest.mark.gpu_init
@pytest.mark.parametrize(
    "preset_name",
    (
        "z_image_turbo.1gpu.rdna4",
        pytest.param(
            "flux.1gpu.rdna4",
            marks=pytest.mark.skipif(
                os.environ.get("XDIT_LOADER_WARM_FLUX", "").strip() != "1",
                reason="Set XDIT_LOADER_WARM_FLUX=1 to run heavy flux loader warm (needs ~12GiB VRAM)",
            ),
        ),
    ),
)
def test_loader_warm_live(preset_name, require_gpu_headroom, require_gpu_count):
    """Real Model worker warm — catches init failures unit/contract tests mock away."""
    spec = build_preset_spec_for_tag(preset_name)
    require_gpu_count(int(spec.get("gpu_count") or 1))
    runtime = warm_loader_from_preset(preset_name, timeout_seconds=900)
    assert runtime.get("_preloaded")
    assert runtime.get("model")


@pytest.mark.contract
def test_a_video_task_on_flux_never_reaches_a_worker():
    """Queueing repairs a stale video task; reaching Model with one still refuses."""
    from xdit_comfyui.nodes import XDiTModel
    from xdit_comfyui.runner_contract import default_loader_widget_values

    stale = {"model": "black-forest-labs/FLUX.2-dev", "task": "i2v"}
    queued = hooked_loader_inputs("flux.1gpu.rdna4", stale_loader=stale)
    assert queued["model"] == stale["model"]
    assert queued["task"] == ""

    kwargs = {**default_loader_widget_values(), **loader_inputs(**stale)}
    kwargs.pop("preset", None)
    with pytest.raises(ValueError, match="does not support pipeline tasks"):
        XDiTModel.execute(**kwargs)


@pytest.mark.contract
def test_video_generation_overrides_reach_the_worker_contract():
    """Video-only values are carried by Sample, not assumed from benchmark tuning."""
    if RUN_VIDEO_PRESET is None:
        pytest.skip("the pinned upstream catalog contains no video preset")
    spec, worker_payload = capture_runner_config(
        RUN_VIDEO_PRESET,
        sample_overrides={"flow_shift": 5.0, "guidance_scale_2": 0.0},
    )
    assert "guidance_scale_2" in worker_payload
    assert worker_payload["guidance_scale_2"] is None
    assert worker_payload.get("flow_shift") == 5.0
    init_args = preprocess_via_xfuser(worker_payload)
    assert init_args["guidance_scale_2"] is None
    assert init_args["flow_shift"] == 5
    assert "Wan" in _model_name(spec)
