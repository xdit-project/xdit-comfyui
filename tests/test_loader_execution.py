"""Model execution contract: widget kwargs are canonical; preset only seeds unset fields."""

from unittest import mock

import pytest

from tests.conftest import needs_gpu
from tests.node_test_helpers import _loader_kwargs
from xdit_comfyui import presets as preset_catalog
from xdit_comfyui.nodes import XDiTModel
from xdit_comfyui.presets import build_preset_spec
from xdit_comfyui.prompt_hooks import apply_preset_prompt_overrides
from xdit_comfyui.runtime_config import (
    _merge_loader_kwargs,
    _preset_synced_loader_kwargs,
    _repair_loader_model_choice,
    _runtime_loader_model_choices,
)
from xdit_comfyui.worker import _clear_all_runtime_caches
from xdit_comfyui.worker_payload import (
    LOADER_INIT_REQUIRED_KEYS,
    loader_init_worker_payload,
)

pytestmark = pytest.mark.usefixtures("mock_loader_worker_warm", "synthetic_preset_catalog")


@pytest.fixture(autouse=True)
def _clear_caches():
    _clear_all_runtime_caches()
    yield
    _clear_all_runtime_caches()


def _flux_spec():
    return build_preset_spec(
        "flux.1gpu.rdna4",
        "gfx1201",
        registry_choices=_runtime_loader_model_choices(),
    )


def test_a_shard_degree_larger_than_the_run_is_refused_before_the_worker_starts():
    """xFuser asserts this seconds into startup; the message there names no widget."""
    from xdit_comfyui.runtime_config import _validate_world_size

    with pytest.raises(ValueError) as excinfo:
        _validate_world_size({"ulysses_degree": 1, "fully_shard_degree": 8})
    message = str(excinfo.value)
    assert "fully_shard_degree" in message
    assert "ulysses_degree" in message


def test_a_shard_degree_matching_the_run_is_allowed():
    from xdit_comfyui.runtime_config import _validate_world_size

    assert _validate_world_size({"ulysses_degree": 2, "fully_shard_degree": 2}) == 2


def test_parallel_vae_processes_do_not_count_toward_the_shard_degree():
    """xFuser splits the world into denoising and VAE ranks before checking degrees."""
    from xdit_comfyui.runtime_config import _validate_world_size

    with pytest.raises(ValueError):
        _validate_world_size(
            {
                "ulysses_degree": 2,
                "fully_shard_degree": 2,
                "use_parallel_vae": True,
                "vae_parallel_size": 1,
            }
        )


def _validate_loader(**overrides):
    kwargs = {
        "ulysses_degree": 1,
        "ring_degree": 1,
        "pipefusion_parallel_degree": 1,
        "tensor_parallel_degree": 1,
        "data_parallel_degree": 1,
        "use_cfg_parallel": False,
        "fully_shard_degree": 1,
        "use_parallel_vae": False,
        "gpu_device_ids": "0",
    }
    kwargs.update(overrides)
    return XDiTModel.validate_inputs(**kwargs)


def test_a_gpu_layout_that_cannot_run_is_refused_at_queue_time():
    """Left to execution it lands mid-run, after a warm worker has been given up."""
    result = _validate_loader(ulysses_degree=2, fully_shard_degree=8)
    assert result is not True
    assert "fully_shard_degree" in result


def test_device_ids_that_do_not_match_the_degrees_are_refused_at_queue_time():
    result = _validate_loader(gpu_device_ids="0,1")
    assert result is not True
    assert "gpu_device_ids" in result


def test_a_workable_layout_queues():
    assert _validate_loader() is True
    assert _validate_loader(gpu_device_ids="auto") is True


def test_a_degree_wired_to_another_node_is_left_for_the_run_to_judge():
    """A wired input carries no value here, and guessing one invents a failure."""
    assert XDiTModel.validate_inputs(ulysses_degree=8) is True


def test_load_model_honors_widget_attention_override_over_preset():
    spec = _flux_spec()
    runtime = XDiTModel.execute(
        preset=spec,
        **_preset_synced_loader_kwargs(spec, attention_backend="sdpa"),
    )[0]
    assert runtime["attention_backend"] == "sdpa"


def test_load_model_honors_per_loader_gpu_device_ids():
    spec = build_preset_spec(
        "z_image_turbo.2gpu.rdna4",
        "gfx1201",
        registry_choices=_runtime_loader_model_choices(),
    )
    runtime_a = XDiTModel.execute(
        preset=spec,
        unique_id="loader-a",
        **_preset_synced_loader_kwargs(spec, gpu_device_ids="0,1"),
    )[0]
    runtime_b = XDiTModel.execute(
        preset=spec,
        unique_id="loader-b",
        **_preset_synced_loader_kwargs(spec, gpu_device_ids="2,3"),
    )[0]
    assert runtime_a["_cuda_visible_devices"] == "0,1"
    assert runtime_b["_cuda_visible_devices"] == "2,3"
    assert runtime_a["model"] == runtime_b["model"] == "Tongyi-MAI/Z-Image-Turbo"


def test_invalid_model_choice_is_repaired_without_overriding_valid_widgets():
    spec = _flux_spec()
    merged = _merge_loader_kwargs(
        spec,
        {
            **_preset_synced_loader_kwargs(spec, attention_backend="sdpa"),
            "model": False,
            "custom_model_id": "False",
        },
    )
    repaired = _repair_loader_model_choice(merged, spec)
    assert repaired["model"] == spec["model_choice"]
    assert repaired["custom_model_id"] == ""
    assert repaired["attention_backend"] == "sdpa"


def test_a_task_the_chosen_model_cannot_run_follows_the_preset_to_a_model_that_can():
    """A graph left on FLUX while the preset moved to Hunyuan i2v; the run settles it."""
    spec = build_preset_spec(
        "hunyuanvideo_1_5.distilled.gfx950",
        "gfx950",
        registry_choices=_runtime_loader_model_choices(),
    )
    runtime = XDiTModel.execute(
        preset=spec,
        unique_id="loader-task-repair",
        **{
            **_preset_synced_loader_kwargs(spec, gpu_device_ids="0,1,2,3,4,5,6,7"),
            "model": "black-forest-labs/FLUX.1-dev",
            "task": "i2v",
        },
    )[0]
    assert (
        runtime["model"] == "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled"
    )
    assert runtime["task"] == "i2v"


def test_prompt_hook_repairs_only_invalid_loader_fields():
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
                        "preset": ["1", 0],
                        "model": False,
                        "custom_model_id": "False",
                        "attention_backend": "sdpa",
                        "gpu_device_ids": "0,1",
                    },
                },
            }
        }
    )
    inputs = payload["prompt"]["2"]["inputs"]
    assert inputs["model"] == "black-forest-labs/FLUX.1-dev"
    assert inputs["custom_model_id"] == ""
    assert inputs["attention_backend"] == "sdpa"
    # A second device under a 1-GPU preset is not a pin the hook may keep: every degree
    # comes out of the preset as 1, and VALIDATE_INPUTS rejects a longer device list.
    # The node-side merge already trims it, so leaving it here queued a graph that could
    # not validate.
    assert inputs["gpu_device_ids"] == "0"


def test_worker_payload_matches_widget_kwargs_not_corrupt_seed_values():
    spec = _flux_spec()
    kwargs = _repair_loader_model_choice(
        _merge_loader_kwargs(
            spec,
            _preset_synced_loader_kwargs(
                spec,
                attention_backend="sdpa",
                gemm_precision="native",
                fp8_precision_override_prefix_patterns="",
            ),
        ),
        spec,
    )
    runtime = XDiTModel.execute(preset=spec, **kwargs)[0]
    payload = loader_init_worker_payload(runtime)
    assert payload["model"] == "black-forest-labs/FLUX.1-dev"
    assert payload["attention_backend"] == "sdpa"
    assert payload.get("use_fp8_gemms") is False
    assert payload.get("fp8_precision_override_prefix_patterns") in (None, "")


def test_worker_payload_strips_unsupported_cross_attention_for_zimage():
    spec = build_preset_spec(
        "z_image_turbo.2gpu.rdna4",
        "gfx1201",
        registry_choices=_runtime_loader_model_choices(),
    )
    kwargs = _repair_loader_model_choice(
        _merge_loader_kwargs(
            spec,
            _preset_synced_loader_kwargs(
                spec,
                cross_attention_backend="aiter",
                gpu_device_ids="0,1",
            ),
        ),
        spec,
    )
    runtime = XDiTModel.execute(preset=spec, unique_id="z-copy", **kwargs)[0]
    payload = loader_init_worker_payload(runtime)
    assert payload["model"] == "Tongyi-MAI/Z-Image-Turbo"
    assert "cross_attention_backend" not in payload


def _wan_loader_payload():
    spec = build_preset_spec(
        "wan2_2_ti2v_5b.i2v.2gpu.rdna4",
        "gfx1201",
        registry_choices=_runtime_loader_model_choices(),
    )
    runtime = XDiTModel.execute(
        preset=spec,
        **_preset_synced_loader_kwargs(spec),
    )[0]
    return loader_init_worker_payload(runtime)


def test_wan_worker_payload_includes_model_and_image_from_preset():
    payload = _wan_loader_payload()
    assert payload["model"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert payload["task"] == "i2v"
    assert payload["guidance_scale_2"] is None
    assert payload["input_images"][0].startswith("https://raw.githubusercontent.com/AMD-AGI/")


@needs_gpu
def test_wan_worker_payload_preprocesses_via_xfuser():
    """xfuser's own preprocessing places tensors on the device."""
    import torch.distributed as dist
    from xfuser.runner import xFuserModelRunner

    payload = _wan_loader_payload()
    try:
        processed = xFuserModelRunner(payload).preprocess_args(payload)
        assert len(processed["input_images"]) == 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_all_shipped_presets_build_loader_init_payloads():
    choices = _runtime_loader_model_choices()
    with (
        mock.patch("xdit_comfyui.runtime_config._available_gpu_count", return_value=64),
        mock.patch("xdit_comfyui.presets.available_gpu_count", return_value=64),
        mock.patch(
            "xdit_comfyui.nodes._ensure_loader_worker",
            side_effect=lambda runtime, *_args, **_kwargs: runtime,
        ),
    ):
        for preset in preset_catalog.load_benchmark_presets():
            hardware_tags = sorted(preset.hardware_tags())
            if not hardware_tags:
                continue
            spec = build_preset_spec(
                preset.name,
                hardware_tags[0],
                registry_choices=choices,
            )
            assert spec["matched"], preset.name
            runtime = XDiTModel.execute(
                preset=spec,
                **_preset_synced_loader_kwargs(spec),
            )[0]
            payload = loader_init_worker_payload(runtime)
            assert payload["model"], preset.name
            assert LOADER_INIT_REQUIRED_KEYS.issubset(payload), preset.name
            expected_images = (spec.get("image_input_preset") or {}).get("paths") or []
            if expected_images:
                assert payload["input_images"] == expected_images, preset.name


def test_prompt_hook_strips_unsupported_cross_attention_for_zimage():
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
                        "preset": ["1", 0],
                        "model": "Tongyi-MAI/Z-Image-Turbo",
                        "cross_attention_backend": "aiter",
                        "gpu_device_ids": "2,3",
                    },
                },
            }
        }
    )
    inputs = payload["prompt"]["2"]["inputs"]
    assert inputs.get("cross_attention_backend") in (None, "", "auto")


def test_duplicate_copy_gpu_change_keeps_valid_parallelism_ints():
    """Copy Model, change gpu_device_ids only — must not blank INT widgets."""
    from xdit_comfyui.presets import build_preset_spec
    from xdit_comfyui.runtime_config import _model_preset_base

    spec = build_preset_spec(
        "z_image_turbo.2gpu.rdna4",
        "gfx1201",
        registry_choices=_runtime_loader_model_choices(),
    )
    base = _model_preset_base(spec)
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
                        "preset": ["1", 0],
                        **base,
                        "gpu_device_ids": "2,3",
                        "pipefusion_parallel_degree": "",
                        "tensor_parallel_degree": "",
                        "cross_attention_backend": "aiter",
                    },
                },
            }
        }
    )
    inputs = payload["prompt"]["2"]["inputs"]
    assert inputs["gpu_device_ids"] == "2,3"
    assert inputs["pipefusion_parallel_degree"] == 1
    assert inputs["tensor_parallel_degree"] == 1
    assert inputs["ulysses_degree"] == 2
    assert inputs.get("cross_attention_backend") in (None, "", "auto")


def test_duplicate_copy_clears_fp4_dependent_values_when_fp4_is_disabled():
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
                        "preset": ["1", 0],
                        "model": "Tongyi-MAI/Z-Image-Turbo",
                        "gpu_device_ids": "2,3",
                        "gemm_precision": "native",
                        "fp8_precision_override_prefix_patterns": "False",
                        "fp8_precision_override_suffix_patterns": "transformer.foo",
                        "use_hybrid_gemm_schedule": True,
                    },
                },
            }
        }
    )
    inputs = payload["prompt"]["2"]["inputs"]
    assert inputs["fp8_precision_override_prefix_patterns"] == ""
    assert inputs["fp8_precision_override_suffix_patterns"] == ""
    assert inputs["use_hybrid_gemm_schedule"] is False


def test_v3_model_execute_binds_hidden_unique_id(mock_loader_worker_warm):
    hidden = type("Hidden", (), {"unique_id": "model-node-17"})()
    with mock.patch.object(XDiTModel, "hidden", hidden, create=True):
        runtime = XDiTModel.execute(**_loader_kwargs())[0]
    assert runtime["_loader_node_id"] == "model-node-17"
