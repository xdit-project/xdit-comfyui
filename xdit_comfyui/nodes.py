"""The xDiT node classes. Everything they call lives in a sibling module."""

from functools import lru_cache

from .identity import COMFY_CATEGORY
from .images import _load_preset_reference_image
from .log_context import worker_logger
from .presets import (
    PRESET_NONE,
    build_preset_spec,
    is_manual_preset_choice,
    list_preset_names,
    normalize_gpu_device_ids,
    normalize_gpu_tag,
    parse_gpu_device_ids,
    preset_by_name,
)
from .progress import _node_id_str
from .residency_allocator import normalize_residency
from .runner_contract import (
    loader_config_widget_names,
    preset_to_image_input_preset,
)
from .runtime_config import (
    _generation_input_types,
    _merge_loader_kwargs,
    _normalize_task,
    _normalize_timeout_seconds,
    _preset_execution_fingerprint,
    _preset_picker_input_types,
    _resolve_loader_runtime,
    _runtime_loader_input_types,
    _runtime_loader_model_choices,
    _validate_world_size,
)
from .runtime_env import _build_hf_cache_env, _resolve_hf_cache_root
from .sampling import _execute_sample
from .v3 import ComfyExtension, inputs_from_legacy, io
from .worker import (
    _ensure_loader_worker,
    _register_loader_cache,
    _release_loader_after_run,
    _runtime_cache_key,
)

_WORKER_LOG = worker_logger()


class XDiTPreset(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs, hidden = inputs_from_legacy(_preset_picker_input_types())
        return io.Schema(
            node_id="xDiT.Preset",
            display_name="xDiT Preset",
            category=COMFY_CATEGORY,
            description="Select a tested configuration for the available GPUs.",
            inputs=inputs,
            hidden=hidden,
            outputs=[
                io.AnyType.Output("model", display_name="model"),
                io.Image.Output("images", display_name="images"),
                io.AnyType.Output("sample", display_name="sample"),
            ],
        )

    @classmethod
    def validate_inputs(cls, gpu_tag, gpu_count, preset, gpu_detection_info=""):
        del gpu_detection_info
        choice = (preset or PRESET_NONE).strip()
        if is_manual_preset_choice(choice):
            return True
        if choice not in list_preset_names():
            return f"Unknown preset: {choice}"
        normalized_tag = normalize_gpu_tag(gpu_tag)
        benchmark = preset_by_name(choice)
        if benchmark is not None and normalized_tag not in benchmark.tags:
            return f"Preset {choice} does not match hardware tag {gpu_tag!r}"
        if benchmark is not None and benchmark.gpu_count != max(int(gpu_count), 1):
            return f"Preset {choice} requires {benchmark.gpu_count} GPU(s), not {gpu_count}"
        return True

    @classmethod
    def fingerprint_inputs(cls, gpu_tag, gpu_count, preset, gpu_detection_info=""):
        del gpu_detection_info
        spec = build_preset_spec(
            (preset or PRESET_NONE).strip(),
            normalize_gpu_tag(gpu_tag),
            registry_choices=_runtime_loader_model_choices(),
        )
        image_input_preset = spec.get("image_input_preset") or preset_to_image_input_preset({})
        paths = tuple(image_input_preset.get("paths") or [])
        return (gpu_tag, gpu_count, preset, paths)

    @classmethod
    def execute(cls, gpu_tag, gpu_count, preset, gpu_detection_info=""):
        del gpu_detection_info
        gpu_tag = normalize_gpu_tag(gpu_tag)
        spec = build_preset_spec(
            preset,
            gpu_tag,
            registry_choices=_runtime_loader_model_choices(),
        )
        if spec.get("matched") and int(spec.get("gpu_count") or 0) != max(int(gpu_count), 1):
            raise ValueError(
                f"Preset {preset!r} requires {spec['gpu_count']} GPU(s), not {gpu_count}."
            )
        image_input_preset = spec.get("image_input_preset") or preset_to_image_input_preset({})
        loaded = _load_preset_reference_image(image_input_preset)
        return io.NodeOutput(spec, loaded, spec)


@lru_cache(maxsize=1)
def _loader_fingerprint_keys():
    """Deferred: touching the xdit CLI at import costs ~6s of ComfyUI startup."""
    return (
        "model",
        "custom_model_id",
        "task",
        "gpu_device_ids",
        # Re-execute so a flipped policy reaches the runtime; the worker cache key
        # excludes residency, so re-running reuses the warm worker.
        "residency",
        "use_torch_compile",
        "hf_cache_mode",
        "hf_cache_dir",
        *loader_config_widget_names(),
    )


_WIRED = object()


class XDiTModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs, hidden = inputs_from_legacy(_runtime_loader_input_types())
        return io.Schema(
            node_id="xDiT.Model",
            display_name="xDiT Model",
            category=COMFY_CATEGORY,
            description="Load an xDiT model for inference.",
            inputs=inputs,
            hidden=hidden,
            outputs=[io.AnyType.Output("model", display_name="model")],
        )

    @classmethod
    def validate_inputs(
        cls,
        gpu_device_ids=_WIRED,
        ulysses_degree=_WIRED,
        ring_degree=_WIRED,
        pipefusion_parallel_degree=_WIRED,
        tensor_parallel_degree=_WIRED,
        data_parallel_degree=_WIRED,
        use_cfg_parallel=_WIRED,
        fully_shard_degree=_WIRED,
        use_parallel_vae=_WIRED,
    ):
        """Refuse a GPU layout while the graph is still on screen.

        Left to execution this lands as a red dialog after the node has started and,
        with `residency=release` in play, after a warm worker has been given up for it.
        Only the inputs the layout depends on are named: ComfyUI repeats the message
        once per argument this takes.
        """
        config = {
            "gpu_device_ids": gpu_device_ids,
            "ulysses_degree": ulysses_degree,
            "ring_degree": ring_degree,
            "pipefusion_parallel_degree": pipefusion_parallel_degree,
            "tensor_parallel_degree": tensor_parallel_degree,
            "data_parallel_degree": data_parallel_degree,
            "use_cfg_parallel": use_cfg_parallel,
            "fully_shard_degree": fully_shard_degree,
            "use_parallel_vae": use_parallel_vae,
        }
        if any(value is _WIRED for value in config.values()):
            # An input wired to another node has no value until the graph runs.
            return True
        try:
            nproc = _validate_world_size(config)
            parse_gpu_device_ids(normalize_gpu_device_ids(gpu_device_ids, nproc), nproc)
        except ValueError as exc:
            return str(exc)
        return True

    @classmethod
    def fingerprint_inputs(cls, preset=None, **kwargs):
        parts = [repr(kwargs.get(key)) for key in _loader_fingerprint_keys() if key in kwargs]
        parts.append(_preset_execution_fingerprint(preset))
        return tuple(parts)

    @classmethod
    def execute(cls, preset=None, unique_id=None, **kwargs):
        unique_id = unique_id or getattr(getattr(cls, "hidden", None), "unique_id", None)
        kwargs["unique_id"] = unique_id
        kwargs = _merge_loader_kwargs(preset, kwargs)
        runtime, resolved, preset_meta, resolved_model, gpu_count = _resolve_loader_runtime(
            preset=preset,
            **kwargs,
        )
        hf_cache_mode = kwargs.get("hf_cache_mode", "auto")
        hf_cache_dir = kwargs.get("hf_cache_dir", "huggingface")
        unique_id = kwargs.get("unique_id")
        nproc = _validate_world_size(runtime)
        if gpu_count != nproc:
            raise ValueError(
                f"Parallelism requests {nproc} GPU process(es), but the loader selected "
                f"{gpu_count} GPU(s). Make the parallel degrees match the connected preset."
            )
        gpu_device_ids = normalize_gpu_device_ids(kwargs.get("gpu_device_ids"), nproc)
        device_ids = parse_gpu_device_ids(gpu_device_ids, nproc)
        cuda_visible = ",".join(str(device_id) for device_id in device_ids)

        cache_root = _resolve_hf_cache_root(hf_cache_mode, hf_cache_dir)
        runtime["_env"] = _build_hf_cache_env(cache_root)
        runtime["_env"]["CUDA_VISIBLE_DEVICES"] = cuda_visible
        runtime["_hf_cache_root"] = str(cache_root) if cache_root is not None else ""
        runtime["_cuda_visible_devices"] = cuda_visible
        runtime["_gpu_count"] = gpu_count
        runtime["_residency"] = normalize_residency(kwargs.get("residency"))
        loader_node_id = _node_id_str(unique_id)
        runtime["_loader_node_id"] = loader_node_id
        runtime["_preset"] = preset_meta
        runtime["_preset_execution_key"] = _preset_execution_fingerprint(preset)
        if isinstance(preset, dict):
            image_preset = preset.get("image_input_preset") or {}
            runtime["_loader_init_input_images"] = list(image_preset.get("paths") or [])
        runtime_cache_key = _runtime_cache_key(runtime)
        runtime["_cache_key"] = runtime_cache_key
        runtime["_cache_key_task"] = _normalize_task(runtime.get("task"))
        runtime["_exec"] = {
            "xdit_bin": "xdit",
            "world_size": nproc,
        }
        _register_loader_cache(unique_id, runtime_cache_key, runtime)
        image_task = _normalize_task(runtime.get("task")) in {"i2v", "ti2v", "v2v"}
        if image_task and not runtime.get("_loader_init_input_images"):
            runtime["_preloaded"] = False
            runtime["_deferred_image_warm"] = True
            _WORKER_LOG.info(
                "Deferring image-conditioned worker warm-up until Sample provides a reference image."
            )
        else:
            runtime = _ensure_loader_worker(runtime, loader_node_id)
        return io.NodeOutput(runtime)


class XDiTSample(io.ComfyNode):

    _GENERATE_FINGERPRINT_KEYS = (
        "prompt",
        "negative_prompt",
        "seed",
        "num_inference_steps",
        "guidance_scale",
        "max_sequence_length",
        "timeout_seconds",
        "height",
        "width",
        "num_frames",
        "flow_shift",
        "guidance_scale_2",
        "resize_input_images",
        "output_fps",
        "enable_tiling",
        "enable_slicing",
        "vae_tile_size_height",
        "vae_tile_size_width",
        "vae_tile_overlap_height",
        "vae_tile_overlap_width",
    )

    @classmethod
    def define_schema(cls):
        inputs, hidden = inputs_from_legacy(_generation_input_types())
        return io.Schema(
            node_id="xDiT.Sample",
            display_name="xDiT Sample",
            category=COMFY_CATEGORY,
            description="Generate images or video with a loaded xDiT model.",
            inputs=inputs,
            hidden=hidden,
            outputs=[
                io.Image.Output("images", display_name="images"),
                io.Video.Output("video", display_name="video"),
            ],
            not_idempotent=True,
        )

    @classmethod
    def fingerprint_inputs(cls, images=None, preset=None, **kwargs):
        import torch  # type: ignore[reportMissingImports]

        parts = [repr(kwargs.get(key)) for key in cls._GENERATE_FINGERPRINT_KEYS if key in kwargs]
        parts.append(_preset_execution_fingerprint(preset))
        if isinstance(images, torch.Tensor):
            tensor = images.detach().cpu()
            parts.append((tuple(tensor.shape), float(tensor.mean()), float(tensor.std())))
        return tuple(parts)

    @classmethod
    def execute(
        cls,
        model,
        prompt,
        negative_prompt="",
        width=1024,
        height=1024,
        seed=0,
        num_inference_steps=28,
        guidance_scale=3.5,
        Video=False,
        num_frames=1,
        output_fps=0,
        flow_shift=0.0,
        guidance_scale_2=0.0,
        resize_input_images=False,
        max_sequence_length=256,
        timeout_seconds=900,
        VAE=False,
        enable_tiling=False,
        enable_slicing=False,
        vae_tile_size_height=0,
        vae_tile_size_width=0,
        vae_tile_overlap_height=0,
        vae_tile_overlap_width=0,
        images=None,
        preset=None,
        unique_id=None,
        **kwargs,
    ):
        unique_id = unique_id or getattr(getattr(cls, "hidden", None), "unique_id", None)
        del VAE, Video, kwargs
        timeout_seconds = _normalize_timeout_seconds(timeout_seconds)
        try:
            images, video = _execute_sample(
                model,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=max(int(num_frames or 1), 1),
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
                guidance_scale=guidance_scale,
                seed=seed,
                output_type="pil",
                images=images,
                flow_shift=flow_shift,
                guidance_scale_2=guidance_scale_2,
                resize_input_images=resize_input_images,
                enable_tiling=enable_tiling,
                enable_slicing=enable_slicing,
                vae_tile_size_height=vae_tile_size_height,
                vae_tile_size_width=vae_tile_size_width,
                vae_tile_overlap_height=vae_tile_overlap_height,
                vae_tile_overlap_width=vae_tile_overlap_width,
                output_fps=max(int(output_fps or 0), 0),
                timeout_seconds=timeout_seconds,
                unique_id=unique_id,
                preset=preset,
            )
        finally:
            _release_loader_after_run(model)
        from comfy_execution.graph import ExecutionBlocker

        return io.NodeOutput(
            images if images is not None else ExecutionBlocker(None),
            video if video is not None else ExecutionBlocker(None),
        )


class XDiTExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [XDiTPreset, XDiTModel, XDiTSample]


async def comfy_entrypoint() -> XDiTExtension:
    return XDiTExtension()
