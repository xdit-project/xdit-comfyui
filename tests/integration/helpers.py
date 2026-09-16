"""Shared helpers for workflow-style integration tests."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from xdit_comfyui.api import _preview_payload
from xdit_comfyui.nodes import (
    XDiTModel,
    XDiTPreset,
)
from xdit_comfyui.presets import (
    build_preset_spec,
    format_gpu_detection_summary,
    load_benchmark_presets,
    preset_by_name,
)
from xdit_comfyui.prompt_hooks import apply_preset_prompt_overrides
from xdit_comfyui.runner_contract import default_loader_widget_values
from xdit_comfyui.runtime_config import (
    _merge_loader_kwargs,
    _preset_synced_loader_kwargs,
    _runtime_loader_input_types,
)
from xdit_comfyui.worker import _clear_all_runtime_caches

DEFAULT_GPU_TAG = "gfx1201"


def preset_gpu_tag(preset_name: str) -> str:
    preset = preset_by_name(preset_name)
    if preset is None:
        raise AssertionError(f"unknown integration preset {preset_name!r}")
    tags = sorted(preset.hardware_tags())
    if not tags:
        raise AssertionError(f"preset {preset_name!r} has no hardware tag")
    return tags[0]


def plugin_root() -> str:
    import os
    from pathlib import Path

    return os.environ.get("PLUGIN_ROOT", str(Path(__file__).resolve().parents[2]))


def comfyui_root() -> str:
    import os

    return os.environ.get("COMFYUI_ROOT", "/workspace/comfyui")


def stop_comfyui_dev_script() -> str:
    from pathlib import Path

    return str(Path(plugin_root()) / "scripts" / "docker" / "stop.sh")


def ensure_comfy_importable() -> None:
    """The Sample node returns ComfyUI's ExecutionBlocker, so ComfyUI must be importable."""
    import os
    import sys

    root = comfyui_root()
    if os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)


@contextmanager
def comfy_video_api():
    """Stand in for ComfyUI's VIDEO type when ComfyUI itself is not importable.

    A multi-frame run returns a VIDEO, so without this the video presets could only be
    exercised on a machine that has ComfyUI checked out — which CI does not.
    """
    import sys
    import types

    try:
        import comfy_api.latest._input_impl.video_types  # noqa: F401

        yield
        return
    except Exception:
        pass

    class VideoComponents:
        def __init__(self, images, frame_rate):
            self.images = images
            self.frame_rate = frame_rate

    class VideoFromComponents:
        def __init__(self, components):
            self.components = components

    names = (
        "comfy_api",
        "comfy_api.latest",
        "comfy_api.latest._input_impl",
        "comfy_api.latest._input_impl.video_types",
        "comfy_api.latest._util",
        "comfy_api.latest._util.video_types",
    )
    modules = {name: types.ModuleType(name) for name in names}
    modules["comfy_api.latest._input_impl.video_types"].VideoFromComponents = VideoFromComponents
    modules["comfy_api.latest._util.video_types"].VideoComponents = VideoComponents
    previous = {name: sys.modules.get(name) for name in names}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def build_preset_spec_for_tag(preset_name: str, gpu_tag: str | None = None) -> dict[str, Any]:
    gpu_tag = gpu_tag or preset_gpu_tag(preset_name)
    spec = build_preset_spec(
        preset_name,
        gpu_tag,
        registry_choices=_runtime_loader_input_types()["required"]["model"][0],
    )
    assert spec.get("matched"), f"preset {preset_name!r} did not match {gpu_tag!r}"
    return spec


def loader_inputs(**overrides) -> dict[str, Any]:
    values = {
        "model": "black-forest-labs/FLUX.1-dev",
        "task": "",
        "gpu_count": 1,
        "gpu_device_ids": "0",
        "custom_model_id": "",
        "use_torch_compile": False,
        "hf_cache_mode": "auto",
        "hf_cache_dir": "huggingface",
        **default_loader_widget_values(),
    }
    values.update(overrides)
    return values


def sample_inputs(**overrides) -> dict[str, Any]:
    values = {
        "prompt": "integration test prompt",
        "negative_prompt": "",
        "num_inference_steps": 4,
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
    values.update(overrides)
    return values


def loader_preview_body(
    preset_name: str,
    *,
    gpu_tag: str | None = None,
    stale_loader: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gpu_tag = gpu_tag or preset_gpu_tag(preset_name)
    spec = build_preset_spec_for_tag(preset_name, gpu_tag)
    body = _preset_synced_loader_kwargs(spec, **(stale_loader or {}))
    body["preset_gpu_tag"] = gpu_tag
    body["preset_gpu_count"] = spec["gpu_count"]
    body["preset_choice"] = preset_name
    return body


def preview_loader(
    preset_name: str, *, stale_loader: dict[str, Any] | None = None
) -> dict[str, Any]:
    return _preview_payload(loader_preview_body(preset_name, stale_loader=stale_loader))


def build_comfy_prompt(
    preset_name: str,
    *,
    gpu_tag: str | None = None,
    stale_loader: dict[str, Any] | None = None,
    stale_sample: dict[str, Any] | None = None,
    include_preview_output: bool = True,
    wire_image_input_preset: bool | None = None,
) -> dict[str, Any]:
    gpu_tag = gpu_tag or preset_gpu_tag(preset_name)
    spec = build_preset_spec_for_tag(preset_name, gpu_tag)
    if wire_image_input_preset is None:
        wire_image_input_preset = bool((spec.get("image_input_preset") or {}).get("required"))

    sample_node_id = "3"
    preview_node_id = "4"

    loader_kwargs = _preset_synced_loader_kwargs(spec, **(stale_loader or {}))

    prompt: dict[str, Any] = {
        "1": {
            "class_type": "xDiT.Preset",
            "inputs": {
                "gpu_tag": gpu_tag,
                "gpu_count": spec["gpu_count"],
                "preset": preset_name,
                "gpu_detection_info": format_gpu_detection_summary(),
            },
        },
        "2": {
            "class_type": "xDiT.Model",
            "inputs": {
                "preset": ["1", 0],
                **loader_kwargs,
            },
        },
        sample_node_id: {
            "class_type": "xDiT.Sample",
            "inputs": {
                "model": ["2", 0],
                "preset": ["1", 2],
                **sample_inputs(**(stale_sample or {})),
            },
        },
    }
    if wire_image_input_preset:
        prompt[sample_node_id]["inputs"]["images"] = ["1", 1]
    if include_preview_output:
        prompt[preview_node_id] = {
            "class_type": "SaveImage",
            "inputs": {"images": [sample_node_id, 0], "filename_prefix": "xdit"},
        }
    return {"prompt": prompt}


def build_dual_loader_prompt(
    preset_name: str,
    *,
    gpu_tag: str | None = None,
    device_ids: tuple[str, str] = ("0,1", "2,3"),
    stale_loaders: tuple[dict[str, Any] | None, dict[str, Any] | None] | None = None,
    include_preview_output: bool = True,
) -> dict[str, Any]:
    gpu_tag = gpu_tag or preset_gpu_tag(preset_name)
    spec = build_preset_spec_for_tag(preset_name, gpu_tag)
    stale_a, stale_b = stale_loaders or (None, None)

    loader_a_kwargs = {
        **_preset_synced_loader_kwargs(spec, **(stale_a or {})),
        "gpu_device_ids": device_ids[0],
    }
    loader_b_kwargs = {
        **_preset_synced_loader_kwargs(spec, **(stale_b or {})),
        "gpu_device_ids": device_ids[1],
    }

    prompt: dict[str, Any] = {
        "1": {
            "class_type": "xDiT.Preset",
            "inputs": {
                "gpu_tag": gpu_tag,
                "gpu_count": spec["gpu_count"],
                "preset": preset_name,
                "gpu_detection_info": format_gpu_detection_summary(),
            },
        },
        "2": {
            "class_type": "xDiT.Model",
            "inputs": {
                "preset": ["1", 0],
                **loader_a_kwargs,
            },
        },
        "3": {
            "class_type": "xDiT.Sample",
            "inputs": {
                "model": ["2", 0],
                "preset": ["1", 2],
                **sample_inputs(),
            },
        },
        "5": {
            "class_type": "xDiT.Model",
            "inputs": {
                "preset": ["1", 0],
                **loader_b_kwargs,
            },
        },
        "6": {
            "class_type": "xDiT.Sample",
            "inputs": {
                "model": ["5", 0],
                "preset": ["1", 2],
                **sample_inputs(seed=43),
            },
        },
    }
    if include_preview_output:
        prompt["4"] = {
            "class_type": "SaveImage",
            "inputs": {"images": ["3", 0], "filename_prefix": "xdit_a"},
        }
        prompt["7"] = {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": "xdit_b"},
        }
    return {"prompt": prompt}


def hooked_dual_loader_inputs(
    preset_name: str,
    *,
    stale_loaders: tuple[dict[str, Any] | None, dict[str, Any] | None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = apply_preset_prompt_overrides(
        build_dual_loader_prompt(preset_name, stale_loaders=stale_loaders)
    )
    return payload["prompt"]["2"]["inputs"], payload["prompt"]["5"]["inputs"]


def load_dual_runtimes_from_preset(
    preset_name: str,
    *,
    device_ids: tuple[str, str] = ("0,1", "2,3"),
    gpu_tag: str | None = None,
    stale_loaders: tuple[dict[str, Any] | None, dict[str, Any] | None] | None = None,
):
    gpu_tag = gpu_tag or preset_gpu_tag(preset_name)
    spec = build_preset_spec_for_tag(preset_name, gpu_tag)
    stale_a, stale_b = stale_loaders or (None, None)
    runtimes = []
    for node_id, stale, devices in (
        ("loader-a", stale_a, device_ids[0]),
        ("loader-b", stale_b, device_ids[1]),
    ):
        merged = _merge_loader_kwargs(
            spec,
            {
                **_preset_synced_loader_kwargs(spec, **(stale or {})),
                "gpu_device_ids": devices,
            },
        )
        runtime = XDiTModel.execute(
            preset=spec,
            unique_id=node_id,
            **merged,
        )[0]
        runtimes.append(runtime)
    return spec, runtimes


def hooked_loader_inputs(
    preset_name: str,
    *,
    stale_loader: dict[str, Any] | None = None,
    stale_sample: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = apply_preset_prompt_overrides(
        build_comfy_prompt(
            preset_name,
            stale_loader=stale_loader,
            stale_sample=stale_sample,
        )
    )
    return payload["prompt"]["2"]["inputs"]


def load_runtime_from_preset(
    preset_name: str,
    *,
    stale_loader: dict[str, Any] | None = None,
    gpu_tag: str | None = None,
):
    gpu_tag = gpu_tag or preset_gpu_tag(preset_name)
    spec = build_preset_spec_for_tag(preset_name, gpu_tag)
    kwargs = hooked_loader_inputs(
        preset_name,
        stale_loader=stale_loader,
    )
    kwargs.pop("preset", None)
    merged = _merge_loader_kwargs(spec, kwargs)
    merged.setdefault("unique_id", f"integration-loader-{preset_name}")
    return XDiTModel.execute(preset=spec, **merged)[0]


def build_preset_via_node(preset_name: str, gpu_tag: str | None = None):
    gpu_tag = gpu_tag or preset_gpu_tag(preset_name)
    spec = build_preset_spec_for_tag(preset_name, gpu_tag)
    raw = XDiTPreset.execute(gpu_tag, spec["gpu_count"], preset_name, "")
    if isinstance(raw, dict):
        model_spec, images, sample_spec = raw["result"]
    else:
        model_spec, images, sample_spec = raw
    assert model_spec["matched"]
    return model_spec, images, sample_spec


def clear_runtime_caches():
    _clear_all_runtime_caches()


def gpu_min_free_mib() -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        free = [
            int(torch.cuda.mem_get_info(i)[0] // (1024 * 1024))
            for i in range(torch.cuda.device_count())
        ]
        return min(free) if free else 0
    except Exception:
        return 0


def evict_all_workers_and_wait(
    *,
    min_free_mib: int = 8000,
    timeout_seconds: float = 120,
) -> None:
    import subprocess
    import time

    clear_runtime_caches()
    subprocess.run(
        ["pkill", "-f", "xdit_comfyui.worker_server"],
        check=False,
        capture_output=True,
    )
    subprocess.run(
        ["pkill", "-f", "torch.distributed.run"],
        check=False,
        capture_output=True,
    )
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        workers = subprocess.run(
            ["pgrep", "-f", "distributed_worker"],
            capture_output=True,
            text=True,
        )
        if not workers.stdout.strip() and gpu_min_free_mib() >= min_free_mib:
            return
        time.sleep(2)

    workers = subprocess.run(
        ["pgrep", "-f", "distributed_worker"],
        capture_output=True,
        text=True,
    )
    if workers.stdout.strip():
        raise RuntimeError("xDiT workers still running after eviction timeout")
    free = gpu_min_free_mib()
    if free < min_free_mib:
        raise RuntimeError(
            f"GPUs not idle after worker eviction (min free {free} MiB, need {min_free_mib})"
        )


def _first_preset(predicate, *, excluded: set[str] | None = None):
    excluded = excluded or set()
    return next(
        (
            preset
            for preset in load_benchmark_presets()
            if preset.name not in excluded and predicate(preset)
        ),
        None,
    )


def _representative_smoke_presets() -> tuple[str, ...]:
    selected = []
    predicates = (
        lambda p: p.gpu_count == 1
        and not p.args.get("input_images")
        and int(p.args.get("num_frames") or 1) == 1,
        lambda p: p.gpu_count == 1 and bool(p.args.get("input_images")),
        lambda p: p.gpu_count <= 4 and int(p.args.get("num_frames") or 1) > 1,
    )
    for predicate in predicates:
        match = _first_preset(predicate, excluded={preset.name for preset in selected})
        if match is not None:
            selected.append(match)
    return tuple(preset.name for preset in selected)


RUN_SMOKE_PRESETS = _representative_smoke_presets()
RUN_VIDEO_PRESET = next(
    (
        preset.name
        for preset in load_benchmark_presets()
        if preset.name in RUN_SMOKE_PRESETS and int(preset.args.get("num_frames") or 1) > 1
    ),
    None,
)
RUN_SMOKE_1GPU_PRESETS = tuple(
    name for name in RUN_SMOKE_PRESETS if preset_by_name(name).gpu_count == 1
)
RUN_LOADER_INIT_PREPROCESS_PRESETS = tuple(
    preset.name
    for preset in load_benchmark_presets()
    if preset.gpu_count == 1 and not preset.args.get("input_images")
)[:3]
RUN_LOADER_INIT_CACHE_PRESETS = tuple(
    preset.name
    for preset in load_benchmark_presets()
    if str(preset.args.get("cache_method") or "").lower() not in ("", "none", "false", "0")
)[:2]


def sample_execution_kwargs_from_spec(spec: dict[str, Any], **overrides) -> dict[str, Any]:
    defaults = dict(spec.get("generation_defaults") or {})
    values = {
        "prompt": defaults.get("prompt", "integration test prompt"),
        "negative_prompt": defaults.get("negative_prompt", ""),
        "num_inference_steps": min(int(defaults.get("num_inference_steps", 4)), 4),
        "max_sequence_length": defaults.get("max_sequence_length", 256),
        "guidance_scale": defaults.get("guidance_scale", 3.5),
        "seed": defaults.get("seed", 42),
        "height": defaults.get("height", 512),
        "width": defaults.get("width", 512),
        "num_frames": max(int(defaults.get("num_frames", 1) or 1), 1),
        "task": defaults.get("task", ""),
        "flow_shift": float(defaults.get("flow_shift", 0.0) or 0.0),
        "guidance_scale_2": float(defaults.get("guidance_scale_2", 0.0) or 0.0),
    }
    values.update(overrides)
    kwargs = sample_inputs(**values)
    kwargs.pop("Video", None)
    return kwargs


def capture_runner_config(
    preset_name: str,
    *,
    gpu_tag: str | None = None,
    sample_overrides: dict[str, Any] | None = None,
    stale_loader: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the config Sample would dispatch to xDiT without starting a worker."""
    from unittest import mock

    import torch

    from xdit_comfyui.sampling import _execute_sample
    from xdit_comfyui.worker_payload import worker_config_payload

    gpu_tag = gpu_tag or preset_gpu_tag(preset_name)
    spec = build_preset_spec_for_tag(preset_name, gpu_tag)
    runtime = load_runtime_from_preset(preset_name, gpu_tag=gpu_tag, stale_loader=stale_loader)
    kwargs = sample_execution_kwargs_from_spec(spec, **(sample_overrides or {}))
    captured: dict[str, Any] = {}

    def _capture(**run_kwargs):
        captured["config"] = run_kwargs["runner_config"]
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

    with (
        mock.patch("xdit_comfyui.worker._run_xdit", side_effect=_capture),
        comfy_video_api(),
    ):
        _execute_sample(
            runtime,
            preset=spec,
            output_type="pil",
            dry_run=False,
            **kwargs,
        )
    return spec, worker_config_payload(captured["config"])


def capture_loader_init_worker_payload(
    preset_name: str,
    *,
    gpu_tag: str | None = None,
    stale_loader: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the config Model sends when warming the worker (not Sample run config)."""
    from xdit_comfyui.worker_payload import loader_init_worker_payload

    gpu_tag = gpu_tag or preset_gpu_tag(preset_name)
    spec = build_preset_spec_for_tag(preset_name, gpu_tag)
    kwargs = _preset_synced_loader_kwargs(spec, **(stale_loader or {}))
    merged = _merge_loader_kwargs(spec, kwargs)
    runtime = XDiTModel.execute(preset=spec, **merged)[0]
    return spec, loader_init_worker_payload(runtime)


def skip_without_gpu() -> None:
    """xFuser's own preprocessing places tensors on the device, so CI cannot run it."""
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("xfuser preprocess_args needs a visible GPU; CI runs CPU-only")


def preprocess_via_xfuser(worker_payload: dict[str, Any]) -> dict[str, Any]:
    """Run xFuser's preprocess_args — catches missing generation keys before worker init."""
    import torch.distributed as dist
    from xfuser.runner import xFuserModelRunner

    skip_without_gpu()
    try:
        runner = xFuserModelRunner(worker_payload)
        return runner.preprocess_args(worker_payload)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def validate_loader_init_step_cache(worker_payload: dict[str, Any]) -> None:
    """Build cache_dit SCM masks for loader init — catches worker initialize() failures."""
    cache_method = str(worker_payload.get("cache_method") or "").strip().lower()
    if not cache_method or cache_method in ("none", "false", "0"):
        return

    from xfuser.model_executor.cache.adapters.cache_dit import _build_scm_mask
    from xfuser.runner import xFuserModelRunner

    skip_without_gpu()
    runner = xFuserModelRunner(worker_payload)
    init_args = runner.preprocess_args(worker_payload)
    num_steps = int(
        init_args.get("num_inference_steps") or worker_payload.get("num_inference_steps") or 0
    )
    if num_steps < 8 and num_steps not in (4, 6):
        raise AssertionError(
            f"loader init num_inference_steps={num_steps} incompatible with cache_dit "
            f"(need 4, 6, or >=8 when cache_method={cache_method!r})"
        )

    method_cfg = runner.model.settings.step_cache_config.get(cache_method)
    if method_cfg is None or method_cfg.preset is None:
        return
    scm_policy = method_cfg.preset.scm_policy
    if scm_policy:
        _build_scm_mask(scm_policy, num_steps)


def warm_loader_from_preset(
    preset_name: str,
    *,
    gpu_tag: str | None = None,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Run Model worker warm for real (requires GPU + torchrun)."""
    from xdit_comfyui.nodes import XDiTModel
    from xdit_comfyui.runtime_config import (
        _merge_loader_kwargs,
        _preset_synced_loader_kwargs,
    )

    gpu_tag = gpu_tag or preset_gpu_tag(preset_name)
    spec = build_preset_spec_for_tag(preset_name, gpu_tag)
    merged = _merge_loader_kwargs(spec, _preset_synced_loader_kwargs(spec))
    merged["timeout_seconds"] = timeout_seconds
    runtime = XDiTModel.execute(
        preset=spec,
        unique_id="integration-loader-warm",
        **merged,
    )[0]
    if not runtime.get("_preloaded"):
        raise RuntimeError(f"Model did not warm worker for preset {preset_name!r}")
    return runtime
