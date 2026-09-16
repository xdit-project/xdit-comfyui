"""Preset contract tests: catch UI/prompt/runtime drift without GPU inference."""

from __future__ import annotations

import pytest

from ..helpers import (
    build_preset_spec_for_tag,
    hooked_loader_inputs,
    load_runtime_from_preset,
    preview_loader,
)

PRESET_SWITCH_CASES = [
    pytest.param(
        "z_image.4gpu.rdna4",
        {"cache_method": "none", "model": "Tongyi-MAI/Z-Image"},
        {
            "cache_method": None,
            "model": "Tongyi-MAI/Z-Image",
            "gpu_count": 4,
        },
        id="valid-zimage-overrides-win",
    ),
    pytest.param(
        "flux.1gpu.rdna4",
        {"cache_method": "none", "attention_backend": "sdpa"},
        {
            "cache_method": None,
            "model": "black-forest-labs/FLUX.1-dev",
            "attention_backend": "sdpa",
            "gpu_count": 1,
        },
        id="valid-flux-overrides-win",
    ),
    pytest.param(
        "z_image_turbo.4gpu.rdna4",
        {"gpu_count": 1},
        {
            "cache_method": None,
            "model": "Tongyi-MAI/Z-Image-Turbo",
            "gpu_count": 4,
        },
        id="preset-anchors-gpu-count",
    ),
]


@pytest.mark.contract
@pytest.mark.parametrize("preset_name,stale_loader,expected", PRESET_SWITCH_CASES)
def test_loader_preview_matches_preset_contract(preset_name, stale_loader, expected):
    payload = preview_loader(preset_name, stale_loader=stale_loader)
    runtime = payload["runtime"]

    if "cache_method" in expected:
        assert runtime.get("cache_method") == expected["cache_method"]
    if "model" in expected:
        assert runtime.get("model") == expected["model"]
    if "gpu_count" in expected:
        assert payload["preset_widgets"]["gpu_count"] == expected["gpu_count"]
    if "attention_backend" in expected:
        assert runtime.get("attention_backend") == expected["attention_backend"]

    if expected.get("cache_method") == "dbcache":
        assert payload["step_cache_supported"] is True
        assert payload["display_widgets"]["cache_method"] == "dbcache"
        assert "dbcache" in payload["cache_method_choices"]


@pytest.mark.contract
@pytest.mark.parametrize("preset_name,stale_loader,expected", PRESET_SWITCH_CASES)
def test_prompt_hook_loader_matches_preview_api(preset_name, stale_loader, expected):
    loader_inputs = hooked_loader_inputs(preset_name, stale_loader=stale_loader)
    preview = preview_loader(preset_name, stale_loader=stale_loader)

    assert loader_inputs["model"] == preview["runtime"]["model"]
    if "cache_method" in expected:
        expected_cache = expected["cache_method"] or "none"
        assert loader_inputs.get("cache_method") == expected_cache
    if "gpu_count" in expected:
        from xdit_comfyui.runtime_config import _nproc_from_config

        # gpu_count is not a Model input; the parallel degrees carry the layout.
        assert "gpu_count" not in loader_inputs
        assert _nproc_from_config(loader_inputs) == expected["gpu_count"]
    if "attention_backend" in expected:
        assert loader_inputs["attention_backend"] == expected["attention_backend"]


@pytest.mark.contract
@pytest.mark.parametrize("preset_name,stale_loader,expected", PRESET_SWITCH_CASES)
def test_load_model_runtime_matches_preview_api(preset_name, stale_loader, expected):
    preview = preview_loader(preset_name, stale_loader=stale_loader)
    runtime = load_runtime_from_preset(preset_name, stale_loader=stale_loader)

    assert runtime["model"] == preview["runtime"]["model"]
    if "cache_method" in expected:
        assert runtime.get("cache_method") == expected["cache_method"]
    if "gpu_count" in expected:
        assert runtime["_gpu_count"] == expected["gpu_count"]


@pytest.mark.contract
def test_zimage_preset_preserves_valid_model_and_cache_overrides():
    turbo_spec = build_preset_spec_for_tag("z_image_turbo.4gpu.rdna4")
    assert turbo_spec["runtime_widgets"].get("cache_method") in (None, "none", "")

    zimage_preview = preview_loader(
        "z_image.4gpu.rdna4",
        stale_loader={"cache_method": "none", "model": "Tongyi-MAI/Z-Image"},
    )
    assert zimage_preview["runtime"]["model"] == "Tongyi-MAI/Z-Image"
    assert zimage_preview["runtime"]["cache_method"] is None


@pytest.mark.contract
def test_wan_preset_keeps_input_images_for_cli_not_comfy_widgets():
    from xdit_comfyui.presets import preset_by_name
    from xdit_comfyui.runner_contract import preset_to_generation_widgets

    preset = preset_by_name("wan2_2_ti2v_5b.i2v.4gpu.rdna4")
    assert preset is not None
    assert (
        preset.args.get("input_images")
        == "https://raw.githubusercontent.com/AMD-AGI/diffusion-models-inference/172fbcce2bf603216771f476fc40002b0640ce8d/assets/data/wan_input.jpg"
    )
    assert "input_images" not in preset_to_generation_widgets(preset.args)

    wan_spec = build_preset_spec_for_tag("wan2_2_ti2v_5b.i2v.4gpu.rdna4")
    assert "input_images" not in wan_spec["generation_defaults"]
    assert wan_spec["image_input_preset"]["required"] is True
    assert wan_spec["image_input_preset"]["paths"] == [preset.args["input_images"]]

    zimage_spec = build_preset_spec_for_tag("z_image.4gpu.rdna4")
    assert "input_images" not in (zimage_spec.get("generation_defaults") or {})
    assert zimage_spec["image_input_preset"]["required"] is False
    assert zimage_spec["image_input_preset"]["paths"] == []
