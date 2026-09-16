"""Turn Sample widget values into one worker run and its Comfy outputs."""

from copy import deepcopy

from . import worker
from .images import (
    _generation_output_directory,
    _generation_outputs_from_frames,
    _resolve_sample_input_paths,
)
from .log_context import run_logger
from .runtime_config import (
    _merge_generation_kwargs,
    _normalize_task,
    _normalize_value,
    _optional_generation_float,
)
from .runtime_env import _quick_run_enabled

_RUN_LOG = run_logger()


def _execute_sample(
    model,
    *,
    prompt,
    negative_prompt,
    height,
    width,
    num_frames,
    num_inference_steps,
    max_sequence_length,
    guidance_scale,
    seed,
    output_type,
    task=None,
    images=None,
    flow_shift=0.0,
    guidance_scale_2=0.0,
    resize_input_images=False,
    enable_tiling=False,
    enable_slicing=False,
    vae_tile_size_height=0,
    vae_tile_size_width=0,
    vae_tile_overlap_height=0,
    vae_tile_overlap_width=0,
    output_fps=0,
    timeout_seconds,
    unique_id=None,
    preset=None,
    dry_run=False,
    return_metadata=False,
):
    if not isinstance(model, dict):
        raise ValueError("model must be a dictionary payload from XDiTModel.")

    if not dry_run and not model.get("_preloaded") and not model.get("_deferred_image_warm"):
        raise RuntimeError("Model has not warmed the xDiT worker. Queue Model before Sample.")

    generation_kwargs = _merge_generation_kwargs(
        preset,
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "max_sequence_length": max_sequence_length,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "flow_shift": flow_shift,
            "guidance_scale_2": guidance_scale_2,
            "resize_input_images": bool(resize_input_images),
        },
    )
    prompt = generation_kwargs["prompt"]
    negative_prompt = generation_kwargs["negative_prompt"]
    height = generation_kwargs["height"]
    width = generation_kwargs["width"]
    num_frames = max(int(generation_kwargs.get("num_frames") or 1), 1)
    num_inference_steps = generation_kwargs["num_inference_steps"]
    max_sequence_length = generation_kwargs["max_sequence_length"]
    guidance_scale = generation_kwargs["guidance_scale"]
    seed = generation_kwargs["seed"]
    task = _normalize_task(model.get("task"))
    _RUN_LOG.info(
        "Effective generation: seed=%s steps=%s guidance=%s size=%sx%s prompt=%r",
        seed,
        generation_kwargs["num_inference_steps"],
        generation_kwargs["guidance_scale"],
        generation_kwargs["width"],
        generation_kwargs["height"],
        (generation_kwargs["prompt"] or "")[:120],
    )
    input_paths = _resolve_sample_input_paths(images, preset)
    if model.get("_deferred_image_warm") and not input_paths:
        raise ValueError(
            "This model runs an image-conditioned task. "
            "Connect reference images to the images input on Sample."
        )
    flow_shift = generation_kwargs.get("flow_shift", flow_shift)
    guidance_scale_2 = generation_kwargs.get("guidance_scale_2", guidance_scale_2)

    runtime = deepcopy(model)
    exec_cfg = runtime.pop("_exec", {})
    if not isinstance(exec_cfg, dict):
        raise ValueError("model missing _exec configuration.")
    runtime.update(
        {
            "enable_tiling": bool(enable_tiling),
            "enable_slicing": bool(enable_slicing),
            "vae_tile_size_height": int(vae_tile_size_height or 0) or None,
            "vae_tile_size_width": int(vae_tile_size_width or 0) or None,
            "vae_tile_overlap_height": int(vae_tile_overlap_height or 0) or None,
            "vae_tile_overlap_width": int(vae_tile_overlap_width or 0) or None,
        }
    )

    from .model_info import align_generation_resolution

    height, width = align_generation_resolution(runtime.get("model") or "", height, width)

    generation = {
        "prompt": prompt,
        "negative_prompt": _normalize_value(negative_prompt),
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "max_sequence_length": max_sequence_length,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "output_type": output_type,
        "input_images": input_paths,
        "output_directory": str(_generation_output_directory()),
        "_cli_append": (runtime.get("_cli_append") or "").strip(),
        "guidance_scale_2": _optional_generation_float(guidance_scale_2),
        "resize_input_images": bool(generation_kwargs.get("resize_input_images", False)),
    }
    flow = _optional_generation_float(flow_shift)
    if flow is not None:
        generation["flow_shift"] = flow
    if task:
        generation["task"] = task
    config = {**runtime, **generation}
    if _quick_run_enabled():
        config["quick_run"] = True

    from .model_info import validate_model_task

    validate_model_task(config.get("model") or "", task)

    run_result = worker._run_xdit(
        runner_config=config,
        xdit_bin=exec_cfg.get("xdit_bin", "xdit"),
        timeout_seconds=timeout_seconds,
        dry_run=dry_run,
        generate_node_id=unique_id,
        return_metadata=True,
    )
    if isinstance(run_result, tuple) and len(run_result) == 3:
        frames, output_kind, metadata = run_result
    else:
        frames = run_result
        output_kind = "video" if num_frames > 1 else "image"
        metadata = {
            "actual_height": int(frames.shape[1]),
            "actual_width": int(frames.shape[2]),
            "fps": 24,
        }
    from .model_info import model_capabilities

    native_fps = int(
        metadata.get("fps") or model_capabilities(config.get("model") or "").get("fps") or 24
    )
    selected_fps = int(output_fps or native_fps)
    image_output, video_output = _generation_outputs_from_frames(
        frames,
        output_kind,
        selected_fps,
    )
    metadata["native_fps"] = native_fps
    metadata["selected_fps"] = selected_fps
    _RUN_LOG.info(
        "xDiT output: requested=%sx%s actual=%sx%s kind=%s fps=%s%s",
        width,
        height,
        metadata["actual_width"],
        metadata["actual_height"],
        output_kind,
        selected_fps,
        " (override)" if output_fps else " (model native)",
    )
    result = image_output, video_output, metadata
    return result if return_metadata else result[:2]
