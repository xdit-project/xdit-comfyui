from xdit_comfyui.presets import build_preset_spec
from xdit_comfyui.runner_contract import default_loader_widget_values


def _loader_kwargs(**overrides):
    params = {
        "model": "black-forest-labs/FLUX.1-dev",
        "gpu_device_ids": "0",
        "custom_model_id": "",
        "use_torch_compile": False,
        "hf_cache_mode": "system_default",
        "hf_cache_dir": "huggingface",
        **default_loader_widget_values(),
    }
    params.update(overrides)
    return params


def _preset_spec(name, gpu_tag="gfx1201"):
    from xdit_comfyui.runtime_config import _runtime_loader_model_choices

    return build_preset_spec(
        name,
        gpu_tag,
        registry_choices=_runtime_loader_model_choices(),
    )


def _generate_kwargs(**overrides):
    params = {
        "prompt": "p",
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
    }
    params.update(overrides)
    return params


def _execution_kwargs(**overrides):
    kwargs = _generate_kwargs(**overrides)
    kwargs.pop("Video", None)
    return kwargs
