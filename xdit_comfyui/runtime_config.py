"""Widget values in, xDiT runner config out: choices, defaults, preset merge, CLI args."""

import json
from functools import lru_cache

from .presets import (
    CUSTOM_MODEL_SENTINEL,
    PRESET_NONE,
    default_gpu_device_ids,
    default_gpu_tag,
    detect_gpu_tags,
    format_gpu_detection_summary,
    list_benchmark_hardware_tags,
    list_gpu_counts_for_gpu_tag,
    list_presets_for_gpu_tag,
    preset_backed_model_choices,
)
from .runner_contract import (
    _AUTO_BACKEND_DESTS,
    SAMPLE_VAE_DESTS,
    cache_widget_defaults_for_model,
    default_loader_widget_values,
    loader_config_input_types,
    loader_config_widget_names,
    loader_optional_input_types,
    loader_pinned_input_types,
    loader_schema,
    loader_widget_spec,
    runner_cli_dests,
    runtime_from_loader_widgets,
)


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")
INTERNAL_CONFIG_PREFIX = "_"

_DEFAULT_MODEL = "black-forest-labs/FLUX.1-dev"

_DEFAULT_TIMEOUT_SECONDS = 900


@lru_cache(maxsize=1)
def xdit_model_choices():
    """Every model xfuser can run, named the way xfuser selects it.

    The registry key is the identity: `xFuserModelRunner._select_model` looks the key up
    directly with no fallback, and `_customize_settings` reads it to pick the checkpoint
    (Qwen-Image-Edit vs -2509 vs -2511). The class-level `settings.model_name` is not that
    identity -- Wan 2.2's A14B experts report Wan 2.1's name until they are instantiated --
    so listing names instead hid every model that caches two transformers.

    Every key is offered. Which keys are true aliases is decided per runner (Qwen branches
    on the key, Hunyuan 1.5 on the task), so collapsing them would need exactly the kind of
    per-model table this pack avoids, and guessing wrong loads different weights.
    """
    from .runner_contract import _xfuser_unavailable

    try:
        import xfuser.model_executor.models.runner_models  # noqa: F401
        from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY
    except Exception as exc:
        raise _xfuser_unavailable("the model registry", exc) from exc

    choices: list[str] = []
    seen: set[str] = set()
    for key in MODEL_REGISTRY:
        if key.casefold() in seen:
            continue
        seen.add(key.casefold())
        choices.append(key)
    if not choices:
        raise RuntimeError(
            "xfuser's MODEL_REGISTRY is empty, so the model list would offer nothing."
        )
    return sorted(choices, key=str.casefold)


def _normalize_value(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _normalize_task(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if value == 0:
            return None
        value = str(int(value)) if float(value).is_integer() else str(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "0":
            return None
        return text
    return None


def _cache_config_cli_value(value):
    """`--cache_config` as xfuser parses it, without the plugin's per-denoiser section."""
    from .runner_contract import _parse_cache_config_value, broadcast_cache_config

    broadcast = broadcast_cache_config(_parse_cache_config_value(value))
    return json.dumps(broadcast, sort_keys=True) if broadcast else None


def _build_cli_args(config):
    known_dests = runner_cli_dests()
    args = []
    for key, value in config.items():
        if key.startswith(INTERNAL_CONFIG_PREFIX):
            continue
        if known_dests and key not in known_dests:
            continue
        value = _normalize_value(value)
        if value is None:
            continue
        cli_key = f"--{key}"
        if isinstance(value, bool):
            if key == "reshard_after_forward":
                if not value:
                    args.append("--no_reshard_after_forward")
            elif value:
                args.append(cli_key)
            continue
        if key == "cache_config":
            value = _cache_config_cli_value(value)
            if value is None:
                continue
        if isinstance(value, (list, tuple)):
            if value:
                args.append(cli_key)
                args.extend(str(item) for item in value)
            continue
        args.extend([cli_key, str(value)])
    return args


def _is_invalid_model_choice(value):
    if not isinstance(value, str) or not value.strip():
        return True
    normalized = value.strip().casefold()
    return normalized in {"false", "true", "none", "null"}


def _is_invalid_custom_model_id(value):
    if value is None:
        return True
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return True
        return stripped.casefold() in {"false", "true", "none", "null"}
    return True


def _normalize_custom_model_id(custom_model_id):
    if _is_invalid_custom_model_id(custom_model_id):
        return ""
    return str(custom_model_id).strip()


def _resolve_model_choice(model, custom_model_id):
    custom_id = _normalize_custom_model_id(custom_model_id)
    if custom_id:
        return custom_id
    if _is_invalid_model_choice(model):
        raise ValueError(
            "Model has an invalid model value (often from a duplicated node with "
            "misaligned widgets). Pick the model again on xDiT Model."
        )
    if model == CUSTOM_MODEL_SENTINEL:
        raise ValueError(f"custom_model_id is required when model is {CUSTOM_MODEL_SENTINEL!r}.")
    return model


def _generation_defaults_for_node_definition():
    """Widget defaults for the node definition, taken from the default model's runner.

    ComfyUI asks for one static definition, so this seeds the node with the model the
    Model node also defaults to; picking another model pushes that model's own values
    (see `/xdit/loader/preview`).
    """
    from .model_info import model_generation_defaults, model_resolution_grid

    defaults = model_generation_defaults(_DEFAULT_MODEL)
    grid = model_resolution_grid(_DEFAULT_MODEL)
    return defaults, grid["step"]


def _generation_input_types():
    model_defaults, resolution_step = _generation_defaults_for_node_definition()

    def default_for(key, fallback):
        value = model_defaults.get(key)
        return fallback if value is None else value

    return {
        "required": {
            "model": (
                any_typ,
                {"tooltip": "Warm xDiT runtime from xDiT Model."},
            ),
            "prompt": (
                "STRING",
                {
                    "default": "A cat running in a garden",
                    "multiline": True,
                    "tooltip": (
                        "Text describing the requested output. Changing it does not reload the model."
                    ),
                },
            ),
            "negative_prompt": (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Text describing content to avoid. Empty uses the model default.",
                },
            ),
            "width": (
                "INT",
                {
                    "default": default_for("width", 1024),
                    "min": resolution_step,
                    "max": 4096,
                    "step": resolution_step,
                    "tooltip": (
                        f"Output width in {resolution_step}-pixel increments. Larger values use more VRAM."
                    ),
                },
            ),
            "height": (
                "INT",
                {
                    "default": default_for("height", 1024),
                    "min": resolution_step,
                    "max": 4096,
                    "step": resolution_step,
                    "tooltip": (
                        f"Output height in {resolution_step}-pixel increments. Larger values use more VRAM."
                    ),
                },
            ),
            "seed": (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                    "tooltip": (
                        "Initial noise seed. Reuse it with the same settings for repeatable output."
                    ),
                },
            ),
            "num_inference_steps": (
                "INT",
                {
                    "default": default_for("num_inference_steps", 28),
                    "min": 1,
                    "max": 400,
                    "tooltip": (
                        "Number of denoising steps. More steps increase run time; use the preset default."
                    ),
                },
            ),
            "guidance_scale": (
                "FLOAT",
                {
                    "default": default_for("guidance_scale", 3.5),
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": (
                        "Prompt guidance strength. Higher values follow the prompt more strictly."
                    ),
                },
            ),
            "Video": (
                "BOOLEAN",
                {
                    "default": False,
                    "display_name": "VIDEO",
                    "tooltip": "Show or hide video-only generation settings.",
                },
            ),
            "num_frames": (
                "INT",
                {
                    "default": default_for("num_frames", 1),
                    "min": 1,
                    "max": 4096,
                    "tooltip": "Frame count for video models. Leave at 1 for still images.",
                },
            ),
            "output_fps": (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": 240,
                    "tooltip": "Playback FPS for VIDEO output. 0 uses the model-native FPS.",
                },
            ),
            "flow_shift": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "0 = model default.",
                },
            ),
            "guidance_scale_2": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "0 = model default.",
                },
            ),
            "resize_input_images": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": (
                        "Ask supported image-conditioned models to resize/crop references to the "
                        "requested size. When disabled, Wan i2v preserves reference aspect ratio "
                        "within the requested pixel area."
                    ),
                },
            ),
            "max_sequence_length": (
                "INT",
                {
                    "default": default_for("max_sequence_length", 256),
                    "min": 1,
                    "max": 65536,
                    "tooltip": (
                        "Maximum prompt tokens processed by the text encoder. Higher values use more memory."
                    ),
                },
            ),
            "timeout_seconds": (
                "INT",
                {
                    "default": _DEFAULT_TIMEOUT_SECONDS,
                    "min": 1,
                    "max": 43200,
                    "tooltip": (
                        "Maximum time for model loading and one generation run. "
                        "This does not unload an idle model."
                    ),
                },
            ),
            "VAE": (
                "BOOLEAN",
                {
                    "default": False,
                    "display_name": "VAE DECODE",
                    "tooltip": "Show or hide VAE decoding and tiling settings.",
                },
            ),
            **{name: loader_widget_spec(name) for name in SAMPLE_VAE_DESTS},
        },
        "optional": {
            "images": (
                "IMAGE",
                {
                    "tooltip": (
                        "Reference images for image-conditioned tasks such as edit, i2i, or i2v."
                    ),
                },
            ),
            "preset": (
                any_typ,
                {
                    "tooltip": (
                        "Optional preset defaults for generation settings. Local values can override them."
                    ),
                },
            ),
        },
        "hidden": {"unique_id": "UNIQUE_ID"},
    }


def _model_preset_base(preset):
    if not isinstance(preset, dict) or not preset.get("matched"):
        return {}
    base = dict(preset.get("runtime_widgets") or {})
    if preset.get("model_choice"):
        base["model"] = preset["model_choice"]
    preset_task = (preset.get("generation_defaults") or {}).get("task")
    base["task"] = preset_task or ""
    preset_model = preset.get("model")
    if preset_model and preset.get("model_choice") and preset_model != preset["model_choice"]:
        base["custom_model_id"] = preset_model
    preset_gpu = preset.get("gpu_count")
    if preset_gpu:
        base["gpu_count"] = preset_gpu
        base["gpu_device_ids"] = default_gpu_device_ids(preset_gpu)
    if "use_cfg_parallel" not in base:
        base["use_cfg_parallel"] = False
    return base


def _preset_synced_loader_kwargs(preset, **overrides):
    from .runner_contract import default_loader_widget_values, loader_optional_widget_defaults

    kwargs = {
        **default_loader_widget_values(),
        **loader_optional_widget_defaults(),
        "task": "",
    }
    if isinstance(preset, dict) and preset.get("matched"):
        for key, value in _model_preset_base(preset).items():
            if not _is_unset_generation_value(value):
                kwargs[key] = value
    kwargs.update(overrides)
    return kwargs


def _merge_loader_kwargs(preset, kwargs, *, preset_wins=False):
    """Combine a preset with the Model node's own widget values.

    The node normally wins, so an override survives the next widget edit. `preset_wins`
    is the moment the user picks a preset: the node still holds the previous model's
    values, and answering for those would describe a model that is on its way out.
    """
    if not isinstance(preset, dict) or not preset.get("matched"):
        return dict(kwargs)
    merged = dict(kwargs)
    preset_gpu = preset.get("gpu_count")
    base = _model_preset_base(preset)
    if preset_wins:
        # An empty task is meaningful here: it is how an image model says "no task".
        merged["task"] = base.get("task", "")
    for key, preset_value in base.items():
        if _is_unset_generation_value(preset_value):
            continue
        if key == "gpu_device_ids":
            continue
        if preset_wins or key not in merged or _is_unset_loader_value(key, merged.get(key)):
            merged[key] = preset_value
    if preset_gpu and _gpu_device_ids_need_preset_sync(merged.get("gpu_device_ids"), preset_gpu):
        merged["gpu_device_ids"] = default_gpu_device_ids(preset_gpu)
    return merged


def _is_unset_generation_value(value):
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, tuple)) and not value:
        return True
    return False


def _is_unset_loader_value(key, value):
    if key == "model" and _is_invalid_model_choice(value):
        return True
    if key == "custom_model_id" and _is_invalid_custom_model_id(value):
        return True
    if key == "gpu_device_ids":
        text = str(value or "").strip().lower()
        if text in ("", "auto"):
            return True
    if _is_unset_generation_value(value):
        return True
    if key in _AUTO_BACKEND_DESTS and str(value).strip().lower() == "auto":
        return True
    return False


def _gpu_device_ids_need_preset_sync(value, gpu_count: int) -> bool:
    text = str(value or "").strip().lower()
    if text in ("", "auto"):
        return True
    try:
        ids = [int(part.strip()) for part in str(value).replace(" ", "").split(",") if part.strip()]
    except ValueError:
        return True
    return len(ids) != max(int(gpu_count), 1)


def _merge_generation_kwargs(preset, kwargs):
    if not isinstance(preset, dict):
        return dict(kwargs)
    merged = dict(kwargs)
    base = dict(preset.get("generation_defaults") or {})
    for key, value in base.items():
        if value is None:
            continue
        if _is_unset_generation_value(merged.get(key)):
            merged[key] = value
    return merged


def _preset_meta_from_spec(preset):
    if not isinstance(preset, dict):
        return {
            "selected": PRESET_NONE,
            "matched": False,
            "name": None,
            "gpu_tags": list(detect_gpu_tags()),
        }
    return {
        "selected": preset.get("selected") or PRESET_NONE,
        "matched": bool(preset.get("matched")),
        "name": preset.get("preset_name"),
        "gpu_tag": preset.get("selected_gpu_tag"),
        "gpu_tags": list(preset.get("gpu_tags") or detect_gpu_tags()),
    }


def _preset_execution_model(preset):
    if not isinstance(preset, dict):
        return None
    return preset.get("model") or preset.get("model_choice")


def _preset_execution_fingerprint(preset):
    if not isinstance(preset, dict):
        return ""
    if not preset.get("matched"):
        return repr((preset.get("selected") or PRESET_NONE, preset.get("selected_gpu_tag")))
    return repr(
        (
            preset.get("selected") or preset.get("preset_name"),
            preset.get("selected_gpu_tag"),
            _preset_execution_model(preset),
            preset.get("gpu_count"),
        )
    )


def _optional_generation_float(value):
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if parsed == 0.0 else parsed


def _available_gpu_count():
    try:
        from .presets import available_gpu_count

        return available_gpu_count()
    except Exception:
        try:
            import torch  # type: ignore[reportMissingImports]

            return max(int(torch.cuda.device_count()), 0)
        except Exception:
            return 0


def _runtime_loader_model_choices():
    return preset_backed_model_choices(xdit_model_choices())


def _runtime_loader_gpu_choices():
    visible = _available_gpu_count()
    choices = [1, 2, 4, 8, 16, 32]
    if visible > 0:
        return [count for count in choices if count <= visible] or [1]
    return [1, 2, 4, 8]


def _default_gpu_count():
    choices = _runtime_loader_gpu_choices()
    return choices[-1] if choices else 1


def _coerce_gpu_count(raw, default=None):
    if default is None:
        default = _default_gpu_count()
    if isinstance(raw, bool):
        return default
    if isinstance(raw, int):
        return max(raw, 1)
    if isinstance(raw, float) and raw.is_integer():
        return max(int(raw), 1)
    try:
        return max(int(raw or default), 1)
    except (TypeError, ValueError):
        return default


def _normalize_timeout_seconds(value, default=_DEFAULT_TIMEOUT_SECONDS, *, min_value=60):
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed < min_value:
        return default
    return parsed


def _preset_picker_input_types():
    tag_choices = list(list_benchmark_hardware_tags())
    default_tag = default_gpu_tag()
    if tag_choices and default_tag not in tag_choices:
        default_tag = tag_choices[0]
    available_counts = list_gpu_counts_for_gpu_tag(default_tag) or [1]
    visible_count = _available_gpu_count()
    default_count = visible_count if visible_count in available_counts else available_counts[0]
    count_choices = [str(count) for count in available_counts]
    preset_choices = [PRESET_NONE, *list_presets_for_gpu_tag(default_tag, default_count)]
    gpu_summary = format_gpu_detection_summary()
    return {
        "required": {
            "gpu_detection_info": (
                "STRING",
                {
                    "default": gpu_summary,
                    "tooltip": "GPUs detected on this machine.",
                },
            ),
            "gpu_tag": (
                tag_choices or [default_tag],
                {
                    "default": default_tag,
                    "tooltip": "Hardware tag used to filter GPU counts and presets.",
                },
            ),
            "gpu_count": (
                count_choices,
                {
                    "default": str(default_count),
                    "tooltip": "GPU count required by the preset.",
                },
            ),
            "preset": (
                preset_choices,
                {
                    "default": PRESET_NONE,
                    "tooltip": "Benchmark preset for the selected hardware and GPU count.",
                },
            ),
        },
    }


def _runtime_loader_config_inputs():
    return loader_config_input_types()


def _loader_config_values(kwargs, registry_model=None):
    defaults = default_loader_widget_values()
    # An absent cache widget means "whatever the model asks for". Falling back to the
    # global DBCachePreset here would look like an override and flatten the per-transformer
    # values of models such as Wan 2.2.
    cache_defaults = cache_widget_defaults_for_model(
        registry_model or "", kwargs.get("cache_method")
    )
    return {
        name: kwargs.get(name, cache_defaults.get(name, defaults.get(name)))
        for name in loader_config_widget_names()
    }


def _resolve_loader_gpu_count(preset, runtime, kwargs):
    merged = kwargs.get("gpu_count")
    if merged not in (None, ""):
        return _coerce_gpu_count(merged)
    if isinstance(preset, dict) and preset.get("gpu_count"):
        return _coerce_gpu_count(preset["gpu_count"])
    return max(_nproc_from_config(runtime), 1)


def _repair_loader_model_choice(kwargs, preset=None):
    repaired = dict(kwargs)
    preset_base = _model_preset_base(preset) if isinstance(preset, dict) else {}

    if _is_invalid_model_choice(repaired.get("model")):
        preset_model = None
        if preset_base:
            for key in ("model_choice", "model"):
                candidate = preset_base.get(key) or (preset or {}).get(key)
                if isinstance(candidate, str) and candidate.strip():
                    preset_model = candidate.strip()
                    break
        if preset_model:
            repaired["model"] = preset_model
        else:
            choices = _runtime_loader_model_choices()
            if choices:
                fallback = _DEFAULT_MODEL if _DEFAULT_MODEL in choices else choices[0]
                repaired["model"] = fallback

    if _is_invalid_custom_model_id(repaired.get("custom_model_id")):
        preset_custom_id = preset_base.get("custom_model_id")
        if isinstance(preset_custom_id, str) and preset_custom_id.strip():
            repaired["custom_model_id"] = preset_custom_id.strip()
        else:
            repaired["custom_model_id"] = ""

    task = _normalize_task(repaired.get("task"))
    preset_task = _normalize_task(preset_base.get("task"))
    if task and preset_base.get("model"):
        from .model_info import validate_model_task

        resolved_model = _resolve_model_choice(
            repaired.get("model"),
            repaired.get("custom_model_id"),
        )
        try:
            validate_model_task(resolved_model, task)
        except ValueError:
            preset_model = _resolve_model_choice(
                preset_base["model"],
                preset_base.get("custom_model_id"),
            )
            if task == preset_task:
                repaired["model"] = preset_base["model"]
                repaired["custom_model_id"] = preset_base.get("custom_model_id", "")
            elif not preset_task and resolved_model == preset_model:
                repaired["task"] = ""

    return repaired


def _resolve_loader_runtime(preset=None, **kwargs):
    kwargs = _repair_loader_model_choice(kwargs, preset)
    model = kwargs.get("model")
    custom_model_id = kwargs.get("custom_model_id", "")

    resolved_model = _resolve_model_choice(model, custom_model_id)

    config_values = _loader_config_values(kwargs, resolved_model)
    widget_runtime = runtime_from_loader_widgets(config_values, registry_model=resolved_model)
    runtime = {"model": resolved_model, **widget_runtime}
    runtime["model"] = resolved_model
    runtime["use_torch_compile"] = bool(kwargs.get("use_torch_compile", False))
    from .model_info import resolve_model_task, validate_runtime_for_model

    preview = bool(kwargs.get("_allow_missing_task"))
    task = resolve_model_task(
        resolved_model,
        kwargs.get("task"),
        require_selection=not preview,
        drop_invalid=preview,
    )
    if task:
        runtime["task"] = task
    runtime["model"] = resolved_model
    validate_runtime_for_model(resolved_model, runtime)
    from .model_info import sanitize_runtime_for_model
    from .runner_contract import sanitize_attention_runtime

    sanitize_runtime_for_model(resolved_model, runtime)
    sanitize_attention_runtime(runtime)

    gpu_count = _resolve_loader_gpu_count(preset, runtime, kwargs)
    visible = _available_gpu_count()
    if visible and gpu_count > visible:
        raise ValueError(
            f"This configuration requests {gpu_count} GPU(s) but only {visible} are visible. "
            "Lower the parallel degrees or set gpu_device_ids explicitly."
        )

    preset_meta = _preset_meta_from_spec(preset)
    return runtime, None, preset_meta, resolved_model, gpu_count


def _runtime_loader_input_types():
    model_choices = _runtime_loader_model_choices()
    default_model = _DEFAULT_MODEL if _DEFAULT_MODEL in model_choices else model_choices[0]
    required = dict(loader_pinned_input_types(model_choices, default_model))
    config_inputs = _runtime_loader_config_inputs()
    grouped: set[str] = set()
    for group in loader_schema()["widget_groups"]:
        required[group["label"]] = (
            "BOOLEAN",
            {
                "default": not group.get("collapsed", True),
                "display_name": group["label"],
                "tooltip": group.get("description") or f"Show or hide {group['label']} settings.",
            },
        )
        for name in group["widgets"]:
            if name in config_inputs:
                required[name] = config_inputs[name]
                grouped.add(name)
    for name, spec in config_inputs.items():
        if name not in grouped:
            required[name] = spec
    return {
        "required": required,
        "optional": {
            "preset": (
                any_typ,
                {
                    "tooltip": (
                        "Optional preset defaults for model and runtime settings. "
                        "Local values can override them."
                    ),
                },
            ),
            **loader_optional_input_types(),
        },
        "hidden": {
            "unique_id": "UNIQUE_ID",
        },
    }


def _runtime_preview_dict(runtime, preset_meta):
    preview = {key: value for key, value in runtime.items() if not key.startswith("_")}
    preview["_preset"] = preset_meta
    return preview


def _nproc_from_config(config):
    """World size xdit will launch. Mirrors xfuser.cli.get_nproc_from_args:
    product of the parallel degrees, doubled when cfg-parallel is on."""
    try:
        from xfuser.cli import get_nproc_from_args  # type: ignore[reportMissingImports]

        cli_args = _build_cli_args(config)
        return int(get_nproc_from_args(cli_args))
    except Exception:
        n = 1
        for key in (
            "ulysses_degree",
            "ring_degree",
            "pipefusion_parallel_degree",
            "tensor_parallel_degree",
            "data_parallel_degree",
        ):
            n *= max(int(config.get(key, 1) or 1), 1)
        if config.get("use_cfg_parallel"):
            n *= 2
        return n


# Degrees xFuser asserts are `<= dit_parallel_size` (xfuser/config/config.py). They do
# not contribute to the world size, so nothing else catches them and the worker dies on
# a bare AssertionError seconds into startup.
_DIT_BOUNDED_DEGREES = (
    "fully_shard_degree",
    "tensor_parallel_degree",
    "pipefusion_parallel_degree",
)


def _dit_parallel_size(config, nproc):
    declared = int(config.get("dit_parallel_size", 0) or 0)
    if declared:
        return declared
    vae = int(config.get("vae_parallel_size", 0) or 0)
    if not config.get("use_parallel_vae"):
        vae = 0
    return max(nproc - vae, 1)


def _validate_dit_bounded_degrees(config, nproc):
    dit = _dit_parallel_size(config, nproc)
    for name in _DIT_BOUNDED_DEGREES:
        degree = int(config.get(name, 1) or 1)
        if degree <= dit:
            continue
        raise ValueError(
            f"{name}={degree} needs {degree} denoising process(es) but this configuration "
            f"runs {dit}. Raise ulysses_degree to {degree} (and select that many GPUs), or "
            f"lower {name} to {dit}. A preset measured on more GPUs sets this; overriding it "
            "here is fine, but the two have to agree."
        )


def _validate_world_size(config):
    nproc = _nproc_from_config(config)
    _validate_dit_bounded_degrees(config, nproc)
    gpus = _available_gpu_count()
    if nproc > 1 and gpus and nproc > gpus:
        ulysses = max(int(config.get("ulysses_degree", 1) or 1), 1)
        cfg_parallel = bool(config.get("use_cfg_parallel"))
        detail = f"ulysses_degree={ulysses}"
        if cfg_parallel:
            detail += " with use_cfg_parallel (doubles world size)"
        raise ValueError(
            f"Parallel degrees request {nproc} GPU processes but only {gpus} are visible "
            f"({detail}). For simple multi-GPU, set ulysses_degree to the GPU count, leave "
            "other parallel degrees at 1, and disable use_cfg_parallel unless the model "
            "requires it."
        )
    return nproc
