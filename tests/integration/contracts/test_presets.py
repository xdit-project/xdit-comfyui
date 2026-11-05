"""Preset contract tests: catch UI/prompt/runtime drift without GPU inference."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import pytest
import yaml

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
def test_preset_stays_consistent_across_preview_hook_and_runtime(
    preset_name, stale_loader, expected
):
    payload = preview_loader(preset_name, stale_loader=stale_loader)
    preview_runtime = payload["runtime"]
    loader_inputs = hooked_loader_inputs(preset_name, stale_loader=stale_loader)
    runtime = load_runtime_from_preset(preset_name, stale_loader=stale_loader)

    if "cache_method" in expected:
        assert preview_runtime.get("cache_method") == expected["cache_method"]
        assert loader_inputs.get("cache_method") == (expected["cache_method"] or "none")
        assert runtime.get("cache_method") == expected["cache_method"]
    if "model" in expected:
        assert preview_runtime.get("model") == expected["model"]
        assert loader_inputs["model"] == expected["model"]
        assert runtime["model"] == expected["model"]
    if "gpu_count" in expected:
        assert payload["preset_widgets"]["gpu_count"] == expected["gpu_count"]
        from xdit_comfyui.runtime_config import _nproc_from_config

        assert "gpu_count" not in loader_inputs
        assert _nproc_from_config(loader_inputs) == expected["gpu_count"]
        assert runtime["_gpu_count"] == expected["gpu_count"]
    if "attention_backend" in expected:
        assert preview_runtime.get("attention_backend") == expected["attention_backend"]
        assert loader_inputs["attention_backend"] == expected["attention_backend"]


@pytest.mark.contract
def test_wan_preset_keeps_input_images_for_cli_not_comfy_widgets():
    from xdit_comfyui.presets import preset_by_name
    from xdit_comfyui.runner_contract import preset_to_generation_widgets

    preset = preset_by_name("wan2_2_ti2v_5b.i2v.4gpu.rdna4")
    assert preset is not None
    pyproject = tomllib.loads(
        (Path(__file__).parents[3] / "pyproject.toml").read_text(encoding="utf-8")
    )
    commit = pyproject["tool"]["xdit-comfyui"]["preset-configs"]["commit"]
    assert preset.args.get("input_images") == (
        "https://raw.githubusercontent.com/AMD-AGI/"
        f"diffusion-models-inference/{commit}/assets/data/wan_input.jpg"
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


@pytest.mark.contract
def test_ltx25_is_not_an_rdna4_preset():
    from xdit_comfyui.presets import list_presets_for_gpu_tag, preset_by_name

    preset = preset_by_name("ltx2_5.default")
    assert preset is not None
    assert set(preset.tags) == {"gfx942", "gfx950"}
    assert "ltx2_5.default" not in list_presets_for_gpu_tag("gfx1201")


@pytest.mark.contract
def test_qwen_pipefusion_cache_preset_keeps_its_full_policy():
    from xdit_comfyui.presets import preset_by_name

    preset = preset_by_name("qwen_image.4gpu.rdna4")
    assert preset is not None
    assert preset.args["cache_method"] == "pipefusion"
    assert preset.args["cache_config"] == {
        "residual_diff_threshold": 0.1,
        "scm_policy": "alternating",
        "steps_computation_policy": "static",
    }


@pytest.mark.contract
def test_every_available_release_preset_argument_is_accounted_for():
    import xdit_comfyui.presets as presets
    from xdit_comfyui.runner_contract import (
        preset_args_to_runtime,
        preset_to_generation_widgets,
        preset_to_image_input_preset,
        runner_cli_dests,
    )

    failures = {}
    config_dir = Path(presets.__file__).with_name("preset_configs")
    cli_dests = set(runner_cli_dests())
    for path in config_dir.glob("*.yaml"):
        for entry in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
            args = dict(entry.get("args") or {})
            runtime = preset_args_to_runtime(args)
            generation = preset_to_generation_widgets(args)
            represented = set(runtime) | set(generation)
            if preset_to_image_input_preset(args)["paths"]:
                represented.add("input_images")
            available = set(args) & (cli_dests | {"no_reshard_after_forward"})
            dropped = available - represented - presets._COMFY_IGNORED_BENCHMARK_ARGS
            if dropped:
                failures[str(entry["name"])] = sorted(dropped)
    assert failures == {}
