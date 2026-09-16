"""Live starter-graph execution helpers — preset → load → sample → save video."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xdit_comfyui.nodes import XDiTSample, _execute_sample
from xdit_comfyui.prompt_hooks import apply_preset_prompt_overrides
from xdit_comfyui.starter_workflow import build_starter_api_prompt
from xdit_comfyui.worker import _worker_log_path_for_loader

from .helpers import (
    DEFAULT_GPU_TAG,
    build_preset_spec_for_tag,
    build_preset_via_node,
    clear_runtime_caches,
    ensure_comfy_importable,
    load_runtime_from_preset,
    sample_execution_kwargs_from_spec,
)

_LOG = logging.getLogger("xdit")

_ERROR_MARKERS = (
    "Traceback (most recent call last)",
    "Exception during processing",
    "ValueError:",
    "RuntimeError:",
    "CUDA out of memory",
)

_SUCCESS_MARKERS = (
    "Iteration 1 completed",
    "inference_s=",
    "Prompt executed",
)

_QUICK_RUN_SUCCESS_MARKERS = (
    "Model initialization complete",
    "quick_run enabled; skipping inference forward",
)


def live_graph_preset_names(*, default: str = "z_image_turbo.1gpu.rdna4") -> tuple[str, ...]:
    raw = os.environ.get("XDIT_GRAPH_LIVE_PRESETS", default).strip()
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    return names or (default,)


def ui_graph_preset_names(
    *,
    default: str = "z_image_turbo.1gpu.rdna4,qwen_image.1gpu.rdna4",
) -> tuple[str, ...]:
    """Presets for ComfyUI queue loop — excludes flux (32GB OOM) unless explicitly listed."""
    from xdit_comfyui.presets import available_gpu_count

    raw = os.environ.get("XDIT_UI_GRAPH_PRESETS", default).strip()
    names = [part.strip() for part in raw.split(",") if part.strip()]
    visible = available_gpu_count()
    filtered: list[str] = []
    for name in names:
        spec = build_preset_spec_for_tag(name)
        if preset_gpu_count(spec) <= max(visible, 1):
            filtered.append(name)
    return tuple(filtered or [default.split(",")[0].strip()])


def gfx1201_preset_names(
    *,
    gpu_tag: str = DEFAULT_GPU_TAG,
    gpu_count: int | None = None,
) -> tuple[str, ...]:
    """Benchmark presets tagged for gfx1201, optionally filtered to an exact GPU count."""
    from xdit_comfyui.presets import available_gpu_count, list_presets_for_gpu_tag

    visible = max(available_gpu_count(), 1)
    target = gpu_count if gpu_count is not None else None
    names: list[str] = []
    for preset_name in list_presets_for_gpu_tag(gpu_tag):
        spec = build_preset_spec_for_tag(preset_name, gpu_tag)
        count = preset_gpu_count(spec)
        if count > visible:
            continue
        if target is not None and count != target:
            continue
        names.append(preset_name)
    return tuple(names)


def quick_graph_preset_names() -> tuple[str, ...]:
    raw = os.environ.get("XDIT_QUICK_GRAPH_PRESETS", "").strip()
    if raw:
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    gpu_count_raw = os.environ.get("XDIT_QUICK_GRAPH_GPU_COUNT", "").strip()
    gpu_count = int(gpu_count_raw) if gpu_count_raw.isdigit() else None
    return gfx1201_preset_names(gpu_count=gpu_count)


def _sync_sample_node_from_preset(
    prompt: dict[str, Any],
    spec: dict[str, Any],
    *,
    max_inference_steps: int | None,
) -> None:
    """Mirror web/xdit_sample.js preset → Sample widget sync before queue."""
    sample = prompt["3"]["inputs"]
    defaults = spec.get("generation_defaults") or {}
    for key in (
        "prompt",
        "negative_prompt",
        "height",
        "width",
        "max_sequence_length",
        "guidance_scale",
        "seed",
        "num_frames",
        "task",
        "flow_shift",
        "guidance_scale_2",
        "resize_input_images",
    ):
        if key in defaults and defaults[key] is not None:
            sample[key] = defaults[key]
    steps = int(defaults.get("num_inference_steps") or sample.get("num_inference_steps") or 4)
    if max_inference_steps is not None:
        steps = min(steps, max_inference_steps)
    sample["num_inference_steps"] = steps
    sample["Video"] = is_video_preset(spec)


def build_starter_ui_prompt(
    preset_name: str,
    *,
    gpu_tag: str = DEFAULT_GPU_TAG,
    max_inference_steps: int | None = 4,
) -> dict[str, Any]:
    """Starter graph prompt as the UI would queue it (hooks + preset widget sync)."""
    spec = build_preset_spec_for_tag(preset_name, gpu_tag)
    payload = apply_preset_prompt_overrides(
        {"prompt": build_starter_api_prompt(preset_name=preset_name, gpu_tag=gpu_tag)}
    )
    prompt = payload["prompt"]
    _sync_sample_node_from_preset(prompt, spec, max_inference_steps=max_inference_steps)
    return prompt


def read_comfy_log_tail(*, log_path: str = "/tmp/comfyui.log", max_lines: int = 80) -> str:
    path = Path(log_path)
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def read_recent_worker_log(*, since_mtime: float) -> str:
    from xdit_comfyui.worker import _worker_runtime_dir

    logs = sorted(
        _worker_runtime_dir().glob("xdit_worker_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in logs:
        if path.stat().st_mtime >= since_mtime - 1.0:
            return path.read_text(encoding="utf-8", errors="replace")
    return logs[0].read_text(encoding="utf-8", errors="replace") if logs else ""


def assert_comfy_log_healthy(log_text: str) -> None:
    if "Exception during processing" in log_text:
        raise AssertionError("ComfyUI log contains Exception during processing")
    if "Prompt executed" not in log_text and "got prompt" not in log_text:
        raise AssertionError("ComfyUI log missing queue/execution markers")


def assert_pil_not_black(image, *, min_mean: float = 0.01, label: str = "preview") -> float:
    import numpy as np

    mean = float(np.asarray(image).mean()) / 255.0
    if mean < min_mean:
        raise AssertionError(f"{label} appears black or empty (mean={mean:.6f})")
    return mean


def preset_gpu_count(spec: dict[str, Any]) -> int:
    return max(int(spec.get("gpu_count") or 1), 1)


def is_video_preset(spec: dict[str, Any]) -> bool:
    defaults = spec.get("generation_defaults") or {}
    return max(int(defaults.get("num_frames") or 1), 1) > 1


def read_worker_log(loader_uid: str) -> str:
    if not loader_uid:
        return ""
    path = _worker_log_path_for_loader(loader_uid)
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


@dataclass
class GraphRunResult:
    preset_name: str
    gpu_tag: str
    spec: dict[str, Any]
    runtime: dict[str, Any]
    sample_kwargs: dict[str, Any]
    images: Any
    video: Any
    save_video_ui: dict[str, Any] | None = None
    saved_video_path: Path | None = None
    worker_log: str = ""
    dispatch_log: str = ""
    log_records: list[str] = field(default_factory=list)


def _cap_inference_steps(kwargs: dict[str, Any], cap: int | None) -> dict[str, Any]:
    if cap is None:
        return kwargs
    merged = dict(kwargs)
    merged["num_inference_steps"] = min(int(merged.get("num_inference_steps") or cap), cap)
    return merged


def _load_reference_images(image_spec: dict[str, Any] | None):
    if not isinstance(image_spec, dict):
        return None
    if not (image_spec.get("paths") or image_spec.get("required")):
        return None
    from xdit_comfyui.images import _load_preset_reference_image

    return _load_preset_reference_image(image_spec)


def validate_hooked_starter_prompt(
    preset_name: str, *, gpu_tag: str = DEFAULT_GPU_TAG
) -> dict[str, Any]:
    """Starter API prompt after preset hooks — same shape ComfyUI queues."""
    payload = apply_preset_prompt_overrides(
        {"prompt": build_starter_api_prompt(preset_name=preset_name, gpu_tag=gpu_tag)}
    )
    prompt = payload["prompt"]
    spec = build_preset_spec_for_tag(preset_name, gpu_tag)
    defaults = spec.get("generation_defaults") or {}

    sample = prompt["3"]["inputs"]
    for key in ("height", "width", "num_inference_steps", "seed"):
        if key in defaults and key in sample:
            if key in ("height", "width"):
                assert int(sample[key]) >= 64
            continue
    if is_video_preset(spec):
        assert int(sample.get("num_frames") or 1) > 1
    return prompt


def assert_preset_values_in_sample_kwargs(
    spec: dict[str, Any], sample_kwargs: dict[str, Any]
) -> None:
    defaults = spec.get("generation_defaults") or {}
    assert sample_kwargs["prompt"] == defaults.get("prompt", sample_kwargs["prompt"])
    assert sample_kwargs["height"] == defaults["height"]
    assert sample_kwargs["width"] == defaults["width"]
    assert sample_kwargs["seed"] == defaults.get("seed", sample_kwargs["seed"])
    assert sample_kwargs["num_inference_steps"] <= defaults["num_inference_steps"]


def assert_worker_log_healthy(
    log_text: str,
    *,
    expected_steps: int | None = None,
    allow_compile_warmup: bool = True,
    quick_run: bool = False,
) -> None:
    if not log_text.strip():
        raise AssertionError("worker log is empty")

    for marker in _ERROR_MARKERS:
        if marker in log_text:
            raise AssertionError(f"worker log contains error marker: {marker!r}")

    markers = _QUICK_RUN_SUCCESS_MARKERS if quick_run else _SUCCESS_MARKERS
    if not any(marker in log_text for marker in markers):
        raise AssertionError(
            "worker log missing success marker (iteration complete / inference timing)"
        )

    if expected_steps is not None and not quick_run:
        dispatch = re.search(r"Dispatching xDiT worker run \((\d+) inference steps\)", log_text)
        if dispatch:
            assert int(dispatch.group(1)) == expected_steps

    if not allow_compile_warmup and "Warming up torch compiler" in log_text:
        raise AssertionError("unexpected torch.compile warmup in worker log")


def assert_frames_not_black(images, *, min_mean: float = 0.01, label: str = "images") -> float:
    import torch

    if not isinstance(images, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    mean = float(images.detach().cpu().mean())
    if mean < min_mean:
        raise AssertionError(
            f"{label} appears black or empty (mean={mean:.6f}, min_mean={min_mean})"
        )
    return mean


def assert_video_tensor_shape(video, spec: dict[str, Any]) -> None:
    defaults = spec.get("generation_defaults") or {}
    expected_frames = max(int(defaults.get("num_frames") or 1), 1)
    width, height = video.get_dimensions()
    assert width > 0 and height > 0
    components = video.get_components()
    frame_count = int(components.images.shape[0])
    assert frame_count == expected_frames
    assert int(components.images.shape[2]) == width
    assert int(components.images.shape[1]) == height


def _save_video_to_temp(video, *, prefix: str = "xdit_graph_test") -> tuple[dict[str, Any], Path]:
    from comfy_api.latest import Types

    out_dir = Path(tempfile.mkdtemp(prefix="xdit_graph_video_"))
    file = f"{prefix}_00001_.mp4"
    path = out_dir / file
    video.save_to(
        str(path),
        format=Types.VideoContainer("auto"),
        codec="auto",
    )
    ui = {
        "images": [{"filename": file, "subfolder": "", "type": "output"}],
        "animated": (True,),
    }
    return ui, path


def execute_starter_graph(
    preset_name: str,
    *,
    gpu_tag: str = DEFAULT_GPU_TAG,
    max_inference_steps: int | None = 4,
    timeout_seconds: int = 900,
    save_video: bool = True,
    sample_overrides: dict[str, Any] | None = None,
) -> GraphRunResult:
    """Run starter workflow nodes: preset → load model → sample → optional save video."""
    ensure_comfy_importable()
    clear_runtime_caches()
    validate_hooked_starter_prompt(preset_name, gpu_tag=gpu_tag)

    loader_spec, image_spec, sample_spec = build_preset_via_node(preset_name, gpu_tag)
    spec = build_preset_spec_for_tag(preset_name, gpu_tag)

    kwargs = _cap_inference_steps(
        sample_execution_kwargs_from_spec(spec, **(sample_overrides or {})),
        max_inference_steps,
    )
    assert_preset_values_in_sample_kwargs(spec, kwargs)

    log_records: list[str] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_records.append(record.getMessage())

    handler = _CaptureHandler()
    _LOG.addHandler(handler)

    runtime = load_runtime_from_preset(preset_name, gpu_tag=gpu_tag)
    loader_uid = str(runtime.get("_loader_node_id") or "")

    kwargs = _cap_inference_steps(
        sample_execution_kwargs_from_spec(spec, **(sample_overrides or {})),
        max_inference_steps,
    )
    kwargs["timeout_seconds"] = timeout_seconds
    images_input = _load_reference_images(image_spec)

    try:
        images, video = _execute_sample(
            runtime,
            preset=sample_spec,
            images=images_input,
            output_type="pil",
            dry_run=False,
            **kwargs,
        )
    finally:
        _LOG.removeHandler(handler)

    worker_log = read_worker_log(loader_uid)
    dispatch_log = "\n".join(log_records)
    save_ui = None
    saved_path = None
    if save_video and video is not None:
        save_ui, saved_path = _save_video_to_temp(video)

    return GraphRunResult(
        preset_name=preset_name,
        gpu_tag=gpu_tag,
        spec=spec,
        runtime=runtime,
        sample_kwargs=kwargs,
        images=images,
        video=video,
        save_video_ui=save_ui,
        saved_video_path=saved_path,
        worker_log=worker_log,
        dispatch_log=dispatch_log,
        log_records=log_records,
    )


def execute_starter_graph_via_sample_node(
    preset_name: str,
    *,
    gpu_tag: str = DEFAULT_GPU_TAG,
    max_inference_steps: int | None = 4,
    timeout_seconds: int = 900,
) -> GraphRunResult:
    """Same graph path but through XDiTSample.sample() like the ComfyUI node entrypoint."""
    ensure_comfy_importable()
    clear_runtime_caches()
    loader_spec, image_spec, sample_spec = build_preset_via_node(preset_name, gpu_tag)
    spec = build_preset_spec_for_tag(preset_name, gpu_tag)
    runtime = load_runtime_from_preset(preset_name, gpu_tag=gpu_tag)
    loader_uid = str(runtime.get("_loader_node_id") or "")

    kwargs = _cap_inference_steps(sample_execution_kwargs_from_spec(spec), max_inference_steps)
    kwargs["timeout_seconds"] = timeout_seconds
    kwargs["Video"] = is_video_preset(spec)
    images_input = _load_reference_images(image_spec)

    images, video = XDiTSample.execute(
        model=runtime,
        preset=sample_spec,
        images=images_input,
        **kwargs,
    )

    return GraphRunResult(
        preset_name=preset_name,
        gpu_tag=gpu_tag,
        spec=spec,
        runtime=runtime,
        sample_kwargs=kwargs,
        images=_unblocked(images),
        video=_unblocked(video),
        worker_log=read_worker_log(loader_uid),
    )


def _unblocked(value):
    """A blocked node output carries nothing downstream, so read it as no value."""
    from comfy_execution.graph import ExecutionBlocker

    return None if isinstance(value, ExecutionBlocker) else value
