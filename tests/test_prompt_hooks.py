import pytest

pytestmark = pytest.mark.usefixtures("synthetic_preset_catalog")

from xdit_comfyui.nodes import XDiTSample
from xdit_comfyui.prompt_hooks import apply_preset_prompt_overrides
from xdit_comfyui.runtime_config import (
    _merge_generation_kwargs,
    _merge_loader_kwargs,
    _normalize_task,
)


def test_merge_loader_kwargs_treats_auto_backend_as_unset():
    spec = {
        "matched": True,
        "runtime_widgets": {"attention_backend": "aiter_flydsl_fp8"},
    }
    merged = _merge_loader_kwargs(spec, {"attention_backend": "auto"})
    assert merged["attention_backend"] == "aiter_flydsl_fp8"


def test_merge_loader_kwargs_fills_unset_gpu_count_from_preset():
    spec = {
        "matched": True,
        "gpu_count": 4,
        "model": "Tongyi-MAI/Z-Image-Turbo",
        "runtime_widgets": {"ulysses_degree": 4},
    }
    merged = _merge_loader_kwargs(spec, {"gpu_count": 1, "ulysses_degree": 1})
    assert merged["gpu_count"] == 1
    assert merged["ulysses_degree"] == 1


def test_merge_loader_kwargs_keeps_explicit_widget_values():
    spec = {
        "matched": True,
        "gpu_count": 4,
        "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "runtime_widgets": {"ulysses_degree": 4, "use_cfg_parallel": False},
    }
    merged = _merge_loader_kwargs(spec, {"ulysses_degree": 4, "use_cfg_parallel": True})
    assert merged["ulysses_degree"] == 4
    assert merged["use_cfg_parallel"] is True


def test_merge_loader_kwargs_honors_widget_cache_method():
    spec = {
        "matched": True,
        "runtime_widgets": {"cache_method": "dbcache", "attention_backend": "aiter_flydsl_fp8"},
    }
    merged = _merge_loader_kwargs(spec, {"cache_method": "none", "attention_backend": "sdpa"})
    assert merged["cache_method"] == "none"
    assert merged["attention_backend"] == "sdpa"


def test_merge_loader_kwargs_honors_use_torch_compile_override():
    spec = {
        "matched": True,
        "runtime_widgets": {"use_torch_compile": True, "ulysses_degree": 4},
        "gpu_count": 4,
    }
    merged = _merge_loader_kwargs(
        spec,
        {"use_torch_compile": False, "ulysses_degree": 4},
    )
    assert merged["use_torch_compile"] is False
    assert merged["ulysses_degree"] == 4


def test_merge_loader_kwargs_applies_preset_gpu_device_ids():
    spec = {
        "matched": True,
        "gpu_count": 4,
        "gpu_device_ids": "0,1,2,3",
        "runtime_widgets": {},
    }
    merged = _merge_loader_kwargs(spec, {"gpu_device_ids": "auto"})
    assert merged["gpu_device_ids"] == "0,1,2,3"


def test_merge_loader_kwargs_keeps_pinned_gpu_device_ids():
    spec = {
        "matched": True,
        "gpu_count": 2,
        "runtime_widgets": {},
    }
    merged = _merge_loader_kwargs(spec, {"gpu_device_ids": "0,1"})
    assert merged["gpu_device_ids"] == "0,1"


def test_merge_loader_kwargs_applies_preset_use_torch_compile_when_unset():
    spec = {
        "matched": True,
        "runtime_widgets": {"use_torch_compile": True},
    }
    merged = _merge_loader_kwargs(spec, {"attention_backend": "auto"})
    assert merged["use_torch_compile"] is True


def test_merge_loader_kwargs_fills_unset_from_preset():
    spec = {
        "matched": True,
        "runtime_widgets": {"cache_method": "dbcache", "attention_backend": "aiter_flydsl_fp8"},
    }
    merged = _merge_loader_kwargs(spec, {"attention_backend": "auto"})
    assert merged["cache_method"] == "dbcache"
    assert merged["attention_backend"] == "aiter_flydsl_fp8"


def test_merge_generation_kwargs_prefers_widget_values():
    spec = {
        "matched": True,
        "generation_defaults": {"prompt": "preset prompt", "seed": 42},
    }
    merged = _merge_generation_kwargs(spec, {"prompt": "user prompt", "seed": 99})
    assert merged["prompt"] == "user prompt"
    assert merged["seed"] == 99


def test_merge_generation_kwargs_fills_unset_from_preset():
    spec = {
        "matched": True,
        "generation_defaults": {
            "prompt": "A small cat",
            "seed": 42,
            "height": 512,
            "width": 320,
            "num_inference_steps": 4,
            "guidance_scale": 0.0,
        },
    }
    merged = _merge_generation_kwargs(
        spec,
        {
            "prompt": "",
            "height": 1024,
            "width": 1024,
        },
    )
    assert merged["prompt"] == "A small cat"
    assert merged["height"] == 1024
    assert merged["width"] == 1024
    assert merged["num_inference_steps"] == 4
    assert merged["guidance_scale"] == 0.0
    assert merged["seed"] == 42


def test_prompt_hook_aligns_height_width_to_multiple_of_16():
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "3": {
                    "class_type": "xDiT.Sample",
                    "inputs": {"height": 1023, "width": 1025},
                },
            }
        }
    )
    inputs = payload["prompt"]["3"]["inputs"]
    assert inputs["height"] == 1024
    assert inputs["width"] == 1024


def test_prompt_hook_leaves_the_starter_graphs_preset_values_alone():
    """The starter graph is built for a preset, so queueing it should change nothing."""
    from xdit_comfyui.starter_workflow import build_starter_api_prompt

    queued = build_starter_api_prompt(preset_name="z_image_turbo.1gpu.rdna4")
    before = dict(queued["3"]["inputs"])
    inputs = apply_preset_prompt_overrides({"prompt": queued})["prompt"]["3"]["inputs"]
    assert inputs["prompt"] == before["prompt"]
    assert (inputs["height"], inputs["width"]) == (before["height"], before["width"])
    assert inputs["num_inference_steps"] == before["num_inference_steps"]


def test_merge_generation_kwargs_uses_preset_negative_prompt_when_widget_empty():
    spec = {
        "matched": True,
        "generation_defaults": {"negative_prompt": "preset negative"},
    }
    merged = _merge_generation_kwargs(spec, {"negative_prompt": ""})
    assert merged["negative_prompt"] == "preset negative"


def test_merge_generation_kwargs_prefers_widget_negative_prompt():
    spec = {
        "matched": True,
        "generation_defaults": {"negative_prompt": "preset negative"},
    }
    merged = _merge_generation_kwargs(spec, {"negative_prompt": "user negative"})
    assert merged["negative_prompt"] == "user negative"


def test_merge_generation_kwargs_keeps_stale_video_fields_from_widgets():
    from xdit_comfyui.presets import build_preset_spec

    spec = build_preset_spec(
        "flux.1gpu.rdna4",
        "gfx1201",
        registry_choices=["black-forest-labs/FLUX.1-dev"],
    )
    merged = _merge_generation_kwargs(
        spec,
        {
            "num_frames": 121,
            "task": "i2v",
            "flow_shift": 5.0,
            "guidance_scale_2": 3.0,
        },
    )
    assert merged["num_frames"] == 121
    assert merged["task"] == "i2v"
    assert merged["flow_shift"] == 5.0
    assert merged["guidance_scale_2"] == 3.0


def test_prompt_hook_keeps_stale_video_fields_from_widgets():
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "1": {
                    "class_type": "xDiT.Preset",
                    "inputs": {"gpu_tag": "gfx1201", "preset": "flux.1gpu.rdna4"},
                },
                "3": {
                    "class_type": "xDiT.Sample",
                    "inputs": {
                        "preset": ["1", 2],
                        "num_frames": 121,
                        "task": "i2v",
                        "flow_shift": 5.0,
                        "guidance_scale_2": 3.0,
                        "timeout_seconds": 900,
                    },
                },
            }
        }
    )
    inputs = payload["prompt"]["3"]["inputs"]
    assert inputs["num_frames"] == 121
    assert inputs["task"] == "i2v"
    assert inputs["flow_shift"] == 5.0
    assert inputs["guidance_scale_2"] == 3.0


def test_generate_is_changed_tracks_prompt():
    first = XDiTSample.fingerprint_inputs(prompt="a", seed=1)
    second = XDiTSample.fingerprint_inputs(prompt="b", seed=1)
    assert first != second


def test_normalize_task_rejects_zero():
    assert _normalize_task(0) is None
    assert _normalize_task("0") is None
    assert _normalize_task(0.0) is None
    assert _normalize_task("i2v") == "i2v"


def test_prompt_hook_normalizes_scrambled_timeout():
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "3": {
                    "class_type": "xDiT.Sample",
                    "inputs": {
                        "prompt": "hello",
                        "timeout_seconds": True,
                        "num_frames": 1,
                        "flow_shift": 0.0,
                        "guidance_scale_2": 0.0,
                    },
                },
            }
        }
    )
    assert payload["prompt"]["3"]["inputs"]["timeout_seconds"] == 900


def test_prompt_hook_clears_scrambled_model_task_zero():
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "2": {
                    "class_type": "xDiT.Model",
                    "inputs": {
                        "model": "black-forest-labs/FLUX.1-dev",
                        "gpu_device_ids": "0",
                        "task": 0,
                    },
                },
            }
        }
    )
    assert payload["prompt"]["2"]["inputs"]["task"] == ""


def test_prompt_hook_clears_task_left_over_from_a_video_model():
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "1": {
                    "class_type": "xDiT.Preset",
                    "inputs": {"gpu_tag": "gfx1201", "preset": "z_image.1gpu.rdna4"},
                },
                "2": {
                    "class_type": "xDiT.Model",
                    "inputs": {
                        "preset": ["1", 0],
                        "model": "Tongyi-MAI/Z-Image",
                        "task": "i2v",
                        "gpu_device_ids": "0",
                    },
                },
            }
        }
    )
    assert payload["prompt"]["2"]["inputs"]["task"] == ""


def test_prompt_hook_honors_use_torch_compile_override():
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "1": {
                    "class_type": "xDiT.Preset",
                    "inputs": {"gpu_tag": "gfx1201", "preset": "wan2_2_ti2v_5b.i2v.4gpu.rdna4"},
                },
                "2": {
                    "class_type": "xDiT.Model",
                    "inputs": {
                        "preset": ["1", 0],
                        "use_torch_compile": False,
                    },
                },
            }
        }
    )
    assert payload["prompt"]["2"]["inputs"]["use_torch_compile"] is False


def test_prompt_hook_rejects_preset_gpu_count_mismatch():
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "1": {
                    "class_type": "xDiT.Preset",
                    "inputs": {
                        "gpu_tag": "gfx1201",
                        "gpu_count": 1,
                        "preset": "z_image_turbo.2gpu.rdna4",
                    },
                },
                "2": {
                    "class_type": "xDiT.Model",
                    "inputs": {
                        "preset": ["1", 0],
                        "model": "black-forest-labs/FLUX.1-dev",
                    },
                },
            }
        }
    )
    assert payload["prompt"]["2"]["inputs"]["model"] == "black-forest-labs/FLUX.1-dev"


def test_prompt_hook_syncs_the_device_list_to_the_presets_gpu_count():
    """Taking the degrees from the preset but not the devices queues an invalid layout.

    A saved graph pinned to GPU 0 with a 4-GPU preset connected failed validation ten
    times over: every degree the preset set was reported against a one-device list.
    """
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "1": {
                    "class_type": "xDiT.Preset",
                    "inputs": {
                        "gpu_tag": "gfx1201",
                        "gpu_count": 4,
                        "preset": "z_image.4gpu.rdna4",
                    },
                },
                "2": {
                    "class_type": "xDiT.Model",
                    "inputs": {
                        "preset": ["1", 0],
                        "model": "Tongyi-MAI/Z-Image-Turbo",
                        "gpu_device_ids": "0",
                    },
                },
            }
        }
    )
    inputs = payload["prompt"]["2"]["inputs"]
    assert inputs["gpu_device_ids"] == "0,1,2,3"
    assert len(inputs["gpu_device_ids"].split(",")) == inputs["ulysses_degree"]


def test_prompt_hook_keeps_a_device_list_that_already_matches_the_preset():
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "1": {
                    "class_type": "xDiT.Preset",
                    "inputs": {
                        "gpu_tag": "gfx1201",
                        "gpu_count": 2,
                        "preset": "z_image_turbo.2gpu.rdna4",
                    },
                },
                "2": {
                    "class_type": "xDiT.Model",
                    "inputs": {
                        "preset": ["1", 0],
                        "model": "Tongyi-MAI/Z-Image-Turbo",
                        "gpu_device_ids": "2,3",
                    },
                },
            }
        }
    )
    assert payload["prompt"]["2"]["inputs"]["gpu_device_ids"] == "2,3"


def test_prompt_hook_normalizes_a_residency_value_the_widget_no_longer_offers():
    """A combo keeps whatever a saved graph had; ComfyUI rejects it before we normalize."""
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "2": {
                    "class_type": "xDiT.Model",
                    "inputs": {
                        "model": "black-forest-labs/FLUX.1-dev",
                        "gpu_device_ids": "0",
                        "residency": "keep_warm",
                    },
                },
            }
        }
    )
    from xdit_comfyui.runner_contract import RESIDENCY_CHOICES

    assert payload["prompt"]["2"]["inputs"]["residency"] in RESIDENCY_CHOICES


def test_prompt_hook_leaves_a_valid_residency_choice_alone():
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "2": {
                    "class_type": "xDiT.Model",
                    "inputs": {
                        "model": "black-forest-labs/FLUX.1-dev",
                        "gpu_device_ids": "0",
                        "residency": "park_cpu",
                    },
                },
            }
        }
    )
    assert payload["prompt"]["2"]["inputs"]["residency"] == "park_cpu"


def test_prompt_hook_preserves_required_native_group_toggles():
    """Native headings must survive validation even though runtime ignores them."""
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "2": {
                    "class_type": "xDiT.Model",
                    "inputs": {
                        "model": "black-forest-labs/FLUX.1-dev",
                        "gpu_device_ids": "0",
                        "STEP CACHE": True,
                        "DISTILLED WEIGHTS": False,
                    },
                },
            }
        }
    )
    inputs = payload["prompt"]["2"]["inputs"]
    assert inputs["STEP CACHE"] is True
    assert inputs["DISTILLED WEIGHTS"] is False


def test_prompt_hook_keeps_every_required_input_comfy_validates():
    """ComfyUI validates required inputs after this hook runs, so none may be dropped."""

    from xdit_comfyui.runtime_config import _generation_input_types, _runtime_loader_input_types

    def required_inputs(input_types):
        return {
            name: (entry[1] or {}).get("default", "")
            for name, entry in input_types()["required"].items()
        }

    loader_inputs = {
        **required_inputs(_runtime_loader_input_types),
        "model": "black-forest-labs/FLUX.1-dev",
    }
    sample_inputs = {**required_inputs(_generation_input_types), "model": ["2", 0]}
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "2": {"class_type": "xDiT.Model", "inputs": loader_inputs},
                "3": {"class_type": "xDiT.Sample", "inputs": sample_inputs},
            }
        }
    )
    for node_id, input_types in (
        ("2", _runtime_loader_input_types),
        ("3", _generation_input_types),
    ):
        remaining = payload["prompt"][node_id]["inputs"]
        missing = [name for name in input_types()["required"] if name not in remaining]
        assert missing == []


def test_prompt_hook_repairs_invalid_model_choice_from_preset():
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "1": {
                    "class_type": "xDiT.Preset",
                    "inputs": {"gpu_tag": "gfx1201", "preset": "flux.1gpu.rdna4"},
                },
                "2": {
                    "class_type": "xDiT.Model",
                    "inputs": {
                        "model": False,
                        "gpu_device_ids": "1",
                        "preset": ["1", 0],
                    },
                },
            }
        }
    )
    assert payload["prompt"]["2"]["inputs"]["model"] == "black-forest-labs/FLUX.1-dev"


def test_prompt_hook_clears_false_string_custom_model_id():
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "1": {
                    "class_type": "xDiT.Preset",
                    "inputs": {"gpu_tag": "gfx1201", "preset": "z_image_turbo.2gpu.rdna4"},
                },
                "2": {
                    "class_type": "xDiT.Model",
                    "inputs": {
                        "model": "Tongyi-MAI/Z-Image-Turbo",
                        "custom_model_id": "False",
                        "gpu_device_ids": "0,1",
                        "preset": ["1", 0],
                    },
                },
            }
        }
    )
    inputs = payload["prompt"]["2"]["inputs"]
    assert inputs["model"] == "Tongyi-MAI/Z-Image-Turbo"
    assert inputs["custom_model_id"] == ""


def test_prompt_hook_sanitize_only_invalid_fields():
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "1": {
                    "class_type": "xDiT.Preset",
                    "inputs": {"gpu_tag": "gfx1201", "preset": "z_image_turbo.1gpu.rdna4"},
                },
                "3": {
                    "class_type": "xDiT.Sample",
                    "inputs": {
                        "preset": ["1", 2],
                        "prompt": "custom prompt",
                        "flow_shift": "randomize",
                        "num_frames": 1,
                        "timeout_seconds": 900,
                        "guidance_scale_2": 0.0,
                    },
                },
            }
        }
    )
    inputs = payload["prompt"]["3"]["inputs"]
    assert inputs["prompt"] == "custom prompt"
    assert inputs["flow_shift"] == 0.0


def test_prompt_hook_repairs_scrambled_dimensions():
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "1": {
                    "class_type": "xDiT.Preset",
                    "inputs": {"gpu_tag": "gfx1201", "preset": "flux.1gpu.rdna4"},
                },
                "3": {
                    "class_type": "xDiT.Sample",
                    "inputs": {
                        "preset": ["1", 2],
                        "height": "/app/data/flux_cat.png",
                        "width": 1,
                        "num_frames": 1,
                        "timeout_seconds": 900,
                        "flow_shift": 0.0,
                        "guidance_scale_2": 0.0,
                    },
                },
            }
        }
    )
    inputs = payload["prompt"]["3"]["inputs"]
    assert inputs["height"] == 1024
    assert inputs["width"] == 1024


def test_prompt_hook_preserves_converted_widget_links():
    links = {
        "height": ["10", 0],
        "width": ["11", 0],
        "num_frames": ["12", 0],
        "flow_shift": ["13", 0],
        "guidance_scale_2": ["14", 0],
        "timeout_seconds": ["15", 0],
    }
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "3": {
                    "class_type": "xDiT.Sample",
                    "inputs": dict(links),
                }
            }
        }
    )

    assert payload["prompt"]["3"]["inputs"] == links


def test_prompt_hook_keeps_save_image_for_video_preset():
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "1": {
                    "class_type": "xDiT.Preset",
                    "inputs": {"gpu_tag": "gfx1201", "preset": "wan2_2_ti2v_5b.i2v.4gpu.rdna4"},
                },
                "3": {
                    "class_type": "xDiT.Sample",
                    "inputs": {"preset": ["1", 2]},
                },
                "4": {
                    "class_type": "SaveImage",
                    "inputs": {"images": ["3", 0], "filename_prefix": "xdit"},
                },
            }
        }
    )
    assert payload["prompt"]["4"]["class_type"] == "SaveImage"


def test_prompt_hook_keeps_save_image_for_image_preset():
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "1": {
                    "class_type": "xDiT.Preset",
                    "inputs": {"gpu_tag": "gfx1201", "preset": "z_image_turbo.1gpu.rdna4"},
                },
                "3": {
                    "class_type": "xDiT.Sample",
                    "inputs": {"preset": ["1", 2]},
                },
                "4": {
                    "class_type": "SaveImage",
                    "inputs": {"images": ["3", 0], "filename_prefix": "xdit"},
                },
            }
        }
    )
    assert payload["prompt"]["4"]["class_type"] == "SaveImage"


def test_prompt_hook_keeps_save_video_for_image_preset():
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "1": {
                    "class_type": "xDiT.Preset",
                    "inputs": {"gpu_tag": "gfx1201", "preset": "z_image_turbo.1gpu.rdna4"},
                },
                "3": {
                    "class_type": "xDiT.Sample",
                    "inputs": {"preset": ["1", 2]},
                },
                "5": {
                    "class_type": "SaveVideo",
                    "inputs": {
                        "video": ["3", 1],
                        "filename_prefix": "video/xdit",
                        "format": "auto",
                        "codec": "auto",
                    },
                },
            }
        }
    )
    assert payload["prompt"]["5"]["class_type"] == "SaveVideo"


def test_prompt_hook_keeps_save_video_for_video_preset():
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "1": {
                    "class_type": "xDiT.Preset",
                    "inputs": {"gpu_tag": "gfx1201", "preset": "wan2_2_ti2v_5b.i2v.4gpu.rdna4"},
                },
                "3": {
                    "class_type": "xDiT.Sample",
                    "inputs": {"preset": ["1", 2]},
                },
                "5": {
                    "class_type": "SaveVideo",
                    "inputs": {
                        "video": ["3", 1],
                        "filename_prefix": "video/xdit",
                        "format": "auto",
                        "codec": "auto",
                    },
                },
            }
        }
    )
    assert payload["prompt"]["5"]["class_type"] == "SaveVideo"
    payload = apply_preset_prompt_overrides(
        {
            "prompt": {
                "2": {
                    "class_type": "xDiT.Model",
                    "inputs": {
                        "model": "black-forest-labs/FLUX.1-dev",
                        "gpu_count": 1,
                        "gpu_device_ids": "0",
                        "hf_cache_mode": "",
                    },
                },
            }
        }
    )
    inputs = payload["prompt"]["2"]["inputs"]
    assert inputs["hf_cache_mode"] == "auto"
    assert inputs["hf_cache_dir"] == "huggingface"
    assert inputs["custom_model_id"] == ""
    assert "xdit_bin" not in inputs
    assert "raw_cli_append" not in inputs
