import pytest

from xdit_comfyui.nodes import (
    XDiTModel,
    XDiTPreset,
)
from xdit_comfyui.presets import preset_by_name
from xdit_comfyui.prompt_hooks import apply_preset_prompt_overrides
from xdit_comfyui.runner_contract import default_loader_widget_values
from xdit_comfyui.worker import _clear_all_runtime_caches

ZIMAGE_1GPU = "z_image_turbo.1gpu.rdna4"
ZIMAGE_4GPU = "z_image_turbo.4gpu.rdna4"


def _loader_kwargs(**overrides):
    params = {
        "model": "black-forest-labs/FLUX.1-dev",
        "gpu_count": 1,
        "gpu_device_ids": "0",
        "custom_model_id": "",
        "use_torch_compile": False,
        "hf_cache_mode": "system_default",
        "hf_cache_dir": "huggingface",
        **default_loader_widget_values(),
    }
    params.update(overrides)
    return params


def _generate_kwargs(**overrides):
    params = {
        "prompt": "a robot in a garden",
        "negative_prompt": "",
        "num_inference_steps": 20,
        "max_sequence_length": 256,
        "guidance_scale": 3.5,
        "seed": 42,
        "timeout_seconds": 60,
        "height": 1024,
        "width": 1024,
        "Video": False,
        "num_frames": 1,
        "task": "",
        "flow_shift": 0.0,
        "guidance_scale_2": 0.0,
        "resize_input_images": False,
        "output_fps": 0,
    }
    params.update(overrides)
    return params


def _execution_kwargs(**overrides):
    kwargs = _generate_kwargs(**overrides)
    kwargs.pop("Video", None)
    return kwargs


def _build_preset_spec(preset_name: str):
    preset = preset_by_name(preset_name)
    assert preset is not None
    spec = XDiTPreset.execute("gfx1201", preset.gpu_count, preset_name, "")[0]
    assert spec["matched"], f"preset {preset_name!r} did not match gfx1201"
    return spec


def _load_runtime_from_spec(spec):
    loader_kwargs = {
        **default_loader_widget_values(),
        **(spec.get("runtime_widgets") or {}),
        "model": spec["model_choice"],
        "gpu_count": spec["gpu_count"],
        "gpu_device_ids": "0",
        "hf_cache_mode": "auto",
        "hf_cache_dir": "huggingface",
        # Worker warm-up is keyed by the Model node id, so a live run needs one.
        "unique_id": "integration-zimage-loader",
    }
    return XDiTModel.execute(preset=spec, **loader_kwargs)[0]


def _generate_from_spec(runtime, spec, *, dry_run: bool, timeout_seconds: int = 900):
    from xdit_comfyui.sampling import _execute_sample

    defaults = dict(spec.get("generation_defaults") or {})
    kwargs = _execution_kwargs(
        prompt=defaults.get("prompt", "A small cat"),
        negative_prompt=defaults.get("negative_prompt", ""),
        num_inference_steps=defaults.get("num_inference_steps", 4),
        max_sequence_length=defaults.get("max_sequence_length", 256),
        guidance_scale=defaults.get("guidance_scale", 0.0),
        seed=defaults.get("seed", 42),
        height=defaults.get("height", 512),
        width=defaults.get("width", 320),
        num_frames=max(int(defaults.get("num_frames", 1) or 1), 1),
        task=defaults.get("task", ""),
        flow_shift=float(defaults.get("flow_shift", 0.0) or 0.0),
        guidance_scale_2=float(defaults.get("guidance_scale_2", 0.0) or 0.0),
        timeout_seconds=timeout_seconds,
    )
    return _execute_sample(
        runtime,
        output_type="pil",
        dry_run=dry_run,
        preset=spec,
        **kwargs,
    )


def _comfy_graph_payload(preset_name: str, *, scrambled_generate: bool = False):
    preset = preset_by_name(preset_name)
    assert preset is not None
    preset_inputs = {
        "gpu_tag": "gfx1201",
        "gpu_count": preset.gpu_count,
        "preset": preset_name,
    }
    generate_inputs = {
        "model": ["2", 0],
        "preset": ["1", 2],
        "prompt": "wrong prompt",
        "negative_prompt": "",
        "num_inference_steps": 1,
        "max_sequence_length": 256,
        "guidance_scale": 3.5,
        "seed": 42,
        "timeout_seconds": 900,
        "height": 1024,
        "width": 1024,
        "Video": False,
        "num_frames": 1,
        "task": "",
        "flow_shift": 0.0,
        "guidance_scale_2": 0.0,
        "resize_input_images": False,
        "output_fps": 0,
    }
    if scrambled_generate:
        generate_inputs.update(
            {
                "flow_shift": "randomize",
                "guidance_scale_2": 900.0,
                "timeout_seconds": "",
                "num_frames": 0,
            }
        )
    return {
        "prompt": {
            "1": {
                "class_type": "xDiT.Preset",
                "inputs": preset_inputs,
            },
            "2": {
                "class_type": "xDiT.Model",
                "inputs": {
                    "preset": ["1", 0],
                    **_loader_kwargs(
                        model="black-forest-labs/FLUX.1-dev",
                        gpu_count=1,
                    ),
                },
            },
            "3": {
                "class_type": "xDiT.Sample",
                "inputs": generate_inputs,
            },
        }
    }


@pytest.fixture(autouse=True)
def _clear_runtime_cache():
    _clear_all_runtime_caches()
    yield
    _clear_all_runtime_caches()


@pytest.mark.contract
def test_prompt_hook_rewrites_scrambled_generate_inputs():
    payload = apply_preset_prompt_overrides(
        _comfy_graph_payload(ZIMAGE_1GPU, scrambled_generate=True)
    )
    generate_inputs = payload["prompt"]["3"]["inputs"]
    assert generate_inputs["flow_shift"] == 0.0
    assert generate_inputs["guidance_scale_2"] == 0.0
    assert generate_inputs["timeout_seconds"] == 900
    assert generate_inputs["num_frames"] == 1
    assert generate_inputs["prompt"] == "wrong prompt"


@pytest.mark.contract
def test_prompt_hook_preserves_valid_generate_overrides():
    payload = apply_preset_prompt_overrides(
        _comfy_graph_payload(ZIMAGE_1GPU, scrambled_generate=False)
    )
    generate_inputs = payload["prompt"]["3"]["inputs"]
    assert generate_inputs["prompt"] == "wrong prompt"
    assert generate_inputs["height"] == 1024


@pytest.mark.contract
def test_prompt_hook_applies_loader_preset_widgets_for_unset_fields():
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "1": {
                    "class_type": "xDiT.Preset",
                    "inputs": {
                        "gpu_tag": "gfx1201",
                        "gpu_count": 4,
                        "preset": ZIMAGE_4GPU,
                    },
                },
                "2": {
                    "class_type": "xDiT.Model",
                    "inputs": {
                        "preset": ["1", 0],
                        "gpu_device_ids": "0",
                    },
                },
            }
        }
    )
    loader_inputs = payload["prompt"]["2"]["inputs"]
    # Model derives its layout from the parallel degrees; gpu_count is not an input.
    assert "gpu_count" not in loader_inputs
    assert loader_inputs["ulysses_degree"] == 4


@pytest.mark.contract
def test_prompt_hook_preserves_loader_widget_overrides():
    """Explicit loader widgets win over the connected preset."""
    payload = apply_preset_prompt_overrides(_comfy_graph_payload(ZIMAGE_4GPU))
    loader_inputs = payload["prompt"]["2"]["inputs"]
    assert loader_inputs["model"] == "black-forest-labs/FLUX.1-dev"
    assert loader_inputs["ulysses_degree"] == 1
    assert loader_inputs.get("cache_method", "none") == "none"


@pytest.mark.gpu_live
def test_zimage_turbo_1gpu_preset_graph_live(require_gpu_live, require_gpu_count):
    require_gpu_count(1)
    spec = _build_preset_spec(ZIMAGE_1GPU)
    runtime = _load_runtime_from_spec(spec)
    images, video = _generate_from_spec(runtime, spec, dry_run=False, timeout_seconds=900)
    assert video is None
    assert images.shape[0] >= 1
    assert images.shape[1] == spec["generation_defaults"]["height"]
    assert images.shape[2] == spec["generation_defaults"]["width"]


@pytest.mark.gpu_live
def test_zimage_turbo_4gpu_preset_graph_live(require_gpu_live, require_gpu_count):
    require_gpu_count(4)
    spec = _build_preset_spec(ZIMAGE_4GPU)
    runtime = _load_runtime_from_spec(spec)
    images, video = _generate_from_spec(runtime, spec, dry_run=False, timeout_seconds=1800)
    assert video is None
    assert images.shape[0] >= 1
    assert images.shape[1] == spec["generation_defaults"]["height"]
    assert images.shape[2] == spec["generation_defaults"]["width"]
