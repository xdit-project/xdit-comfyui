from __future__ import annotations

import copy
import dataclasses
import logging
from functools import lru_cache
from typing import Any

_LOG = logging.getLogger("xdit")


def _runner_registry_entry(registry_model: str):
    try:
        import xfuser.model_executor.models.runner_models  # noqa: F401
        from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY
    except Exception:
        return None, None
    if registry_model in MODEL_REGISTRY:
        return registry_model, MODEL_REGISTRY[registry_model]
    for key, cls in MODEL_REGISTRY.items():
        if getattr(getattr(cls, "settings", None), "model_name", None) == registry_model:
            return key, cls
    return None, None


def _runner_class_for_model(registry_model: str):
    return _runner_registry_entry(registry_model)[1]


@lru_cache(maxsize=64)
def runner_settings_for_model(registry_model: str):
    """Settings as the runner will actually use them, not the class defaults.

    Models customise their own settings in `_customize_settings`, which only runs on an
    instance. Wan 2.2 assigns its two-transformer cache config there, so reading the
    class attribute reports another model's warmup values as if they were Wan 2.2's.
    """
    registry_key, runner = _runner_registry_entry(registry_model)
    if runner is None:
        return None
    class_settings = getattr(runner, "settings", None)
    try:
        from xfuser.config.args import xFuserArgs

        instance = runner.__new__(runner)
        instance.settings = copy.deepcopy(class_settings)
        instance._customize_settings(xFuserArgs(model=registry_key))
        return instance.settings
    except Exception as exc:
        _LOG.debug("falling back to class settings for %s: %s", registry_model, exc, exc_info=True)
        return class_settings


def model_supports_step_cache(registry_model: str) -> bool:
    return bool(model_capabilities(registry_model).get("supported_cache_methods"))


def model_supports_distilled_weights(registry_model: str) -> bool:
    return bool(model_capabilities(registry_model).get("supports_distilled_weights"))


def supported_cache_methods(registry_model: str) -> tuple[str, ...]:
    return tuple(model_capabilities(registry_model).get("supported_cache_methods") or ())


def _supported_cache_methods(caps: Any, settings: Any) -> tuple[str, ...]:
    """Return the cache methods the xDiT runner declares for this model."""
    if not caps.supports_step_caching:
        return ()
    return tuple(settings.step_cache_config or ())


def sanitize_runtime_cache_for_model(registry_model: str, runtime: dict[str, Any]) -> None:
    from .runner_contract import sanitize_cache_config_runtime

    supported = supported_cache_methods(registry_model)
    if not supported:
        runtime["cache_method"] = None
        runtime.pop("cache_config", None)
        return
    method = runtime.get("cache_method")
    if method in (None, "") or str(method).lower() == "none":
        runtime["cache_method"] = None
        runtime.pop("cache_config", None)
        return
    if str(method) not in supported:
        runtime["cache_method"] = None
        runtime.pop("cache_config", None)
        return
    sanitize_cache_config_runtime(runtime)


def sanitize_runtime_for_model(registry_model: str, runtime: dict[str, Any]) -> None:
    from .runner_contract import loader_config_dests

    sanitize_runtime_cache_for_model(registry_model, runtime)
    caps = model_capabilities(registry_model)
    if not caps:
        return
    for dest in loader_config_dests():
        if dest in ("cache_method", "cache_config") or dest in _CAPABILITY_INDEPENDENT_WIDGETS:
            continue
        allowed = caps.get(dest)
        if allowed is None or allowed:
            continue
        value = runtime.get(dest)
        if isinstance(value, bool):
            if value:
                runtime[dest] = False
            else:
                runtime.pop(dest, None)
        elif isinstance(value, int):
            if value > 1:
                runtime[dest] = 1
            elif value <= 1:
                runtime.pop(dest, None)
        else:
            runtime.pop(dest, None)
    if not runtime.get("use_fp4_gemms"):
        runtime.pop("fp8_precision_override_prefix_patterns", None)
        runtime.pop("fp8_precision_override_suffix_patterns", None)
        runtime.pop("num_hybrid_gemm_high_precision_steps", None)
        runtime["use_hybrid_gemm_schedule"] = False


# xfuser turns VAE tiling on from the config alone (base_model._enable_options), so a
# runner leaving these off its capability list says nothing about whether the pipeline
# takes them — Qwen-Image needs tiling to decode 2k images and never declares it.
_CAPABILITY_INDEPENDENT_WIDGETS = ("enable_tiling", "enable_slicing")


def _sanitize_task_input(caps: dict[str, Any], inputs: dict[str, Any]) -> None:
    """Drop a task left over from another model instead of failing the run."""
    if "task" not in inputs:
        return
    task = str(inputs.get("task") or "").strip()
    if not task:
        return
    valid = caps.get("valid_tasks") or []
    if not valid:
        inputs["task"] = ""
    elif task not in valid:
        inputs["task"] = valid[0] if len(valid) == 1 else ""


def sanitize_loader_inputs_for_model(registry_model: str, inputs: dict[str, Any]) -> None:
    from .runner_contract import (
        _is_bool_action,
        _runner_actions,
        default_loader_widget_values,
        loader_config_dests,
        runtime_value_to_widget,
    )

    caps = model_capabilities(registry_model)
    if not caps:
        return
    _sanitize_task_input(caps, inputs)
    defaults = default_loader_widget_values()
    for dest in loader_config_dests():
        if dest in ("cache_method", "cache_config") or dest in _CAPABILITY_INDEPENDENT_WIDGETS:
            continue
        allowed = caps.get(dest)
        if allowed is None or allowed:
            continue
        if dest not in inputs:
            continue
        action = _runner_actions().get(dest)
        raw = inputs[dest]
        if action is not None and action.type is int:
            try:
                parsed = int(raw)
            except (TypeError, ValueError):
                parsed = None
            fallback = defaults.get(dest, 1)
            if parsed is None or parsed < 1:
                inputs[dest] = fallback
            elif parsed > 1:
                inputs[dest] = 1
            else:
                inputs[dest] = parsed
            continue
        if action is not None and _is_bool_action(action):
            inputs[dest] = False
            continue
        inputs[dest] = runtime_value_to_widget(dest, None)

    if inputs.get("gemm_precision") != "fp4":
        inputs["fp8_precision_override_prefix_patterns"] = ""
        inputs["fp8_precision_override_suffix_patterns"] = ""
        inputs["use_hybrid_gemm_schedule"] = False
        inputs["num_hybrid_gemm_high_precision_steps"] = 0


def sanitize_loader_cache_widgets(registry_model: str, widgets: dict[str, Any]) -> dict[str, Any]:
    from .runner_contract import (
        cache_config_widget_names,
        cache_widget_defaults_for_model,
        dbcache_only_cache_widgets,
        default_loader_widget_values,
        denoiser_cache_widget_field,
        extra_denoiser_cache_widget_names,
        normalize_cache_method,
    )

    sanitized = dict(widgets)
    supported = supported_cache_methods(registry_model)
    defaults = default_loader_widget_values()
    allowed = {"none", *supported}
    if not supported or sanitized.get("cache_method") not in allowed:
        sanitized["cache_method"] = "none"
        for name in cache_config_widget_names():
            sanitized[name] = defaults.get(name)
        for name in extra_denoiser_cache_widget_names():
            sanitized[name] = defaults.get(name)
        return sanitized

    method = normalize_cache_method(sanitized.get("cache_method"))
    if method in ("teacache", "fbcache"):
        for name in dbcache_only_cache_widgets():
            sanitized[name] = defaults.get(name)

    # A model that caches one denoiser has nothing for the later groups to say; leaving
    # stale values there would emit per_transformer overrides for a denoiser that is gone.
    denoisers = len(model_cache_transformers(registry_model, method))
    cache_defaults = cache_widget_defaults_for_model(registry_model, method)
    for name in extra_denoiser_cache_widget_names():
        index, field_name = denoiser_cache_widget_field(name)
        if denoisers >= index and (method == "dbcache" or field_name == "residual_diff_threshold"):
            continue
        sanitized[name] = cache_defaults.get(name, defaults.get(name))
    return sanitized


def model_capabilities(registry_model: str) -> dict[str, Any]:
    runner = _runner_class_for_model(registry_model)
    if runner is None:
        return {}
    caps = runner.capabilities
    settings = runner.settings
    data = {
        field.name: getattr(caps, field.name) for field in __import__("dataclasses").fields(caps)
    }
    effective_settings = runner_settings_for_model(registry_model) or settings
    data["supported_cache_methods"] = list(_supported_cache_methods(caps, effective_settings))
    data["has_fp8_modules"] = bool(settings.fp8_gemm_module_list)
    data["has_fp4_modules"] = bool(settings.fp4_gemm_module_list)
    data["has_int8_modules"] = bool(settings.int8_gemm_module_list)
    data["model_output_type"] = settings.model_output_type
    data["fps"] = getattr(settings, "fps", None)
    data["valid_tasks"] = list(settings.valid_tasks or [])
    return data


@lru_cache(maxsize=1)
def default_input_value_names() -> tuple[str, ...]:
    """The input keys xFuser fills in per model when the caller leaves them unset."""
    from .runner_contract import _xfuser_unavailable

    try:
        from xfuser.model_executor.models.runner_models.base_model import DefaultInputValues
    except Exception as exc:
        raise _xfuser_unavailable("the per-model input defaults", exc) from exc

    return tuple(DefaultInputValues.__annotations__)


@lru_cache(maxsize=64)
def model_generation_defaults(registry_model: str) -> dict[str, Any]:
    """Per-model input defaults as xFuser itself would apply them.

    `preprocess_args` fills any unset input from the runner's `DefaultInputValues`, so
    these are the values a run gets when the node sends nothing. Reading them here is
    what lets Qwen-Image open at 928x1664/50 steps without a per-model table.
    """
    runner = _runner_class_for_model(registry_model)
    if runner is None:
        return {}
    values = runner.default_input_values
    defaults: dict[str, Any] = {}
    for name in default_input_value_names():
        value = getattr(values, name, None)
        if value is not None:
            defaults[name] = value
    return defaults


# Every runner either declares mod_value/resolution_divisor or works on the VAE's
# 8-pixel grid, which is the finest granularity any of them accept.
_RESOLUTION_STEP_FALLBACK = 8


@lru_cache(maxsize=64)
def model_resolution_grid(registry_model: str) -> dict[str, int | None]:
    """The height/width grid a model accepts: `step` for the UI, `divisor` when enforced.

    `resolution_divisor` is a hard requirement (the runner raises on a mismatch);
    `mod_value` is the alignment the runner uses when it resizes reference images, so it
    makes the natural widget step.
    """
    settings = runner_settings_for_model(registry_model)
    divisor = getattr(settings, "resolution_divisor", None) if settings else None
    mod_value = getattr(settings, "mod_value", None) if settings else None
    return {
        "step": int(divisor or mod_value or _RESOLUTION_STEP_FALLBACK),
        "divisor": int(divisor) if divisor else None,
    }


def align_generation_resolution(registry_model: str, height: Any, width: Any) -> tuple[int, int]:
    """Snap a request onto the model's required grid instead of letting the run fail.

    LTX rejects any size not divisible by 64 after the worker has already loaded, so a
    2 px overshoot would cost the user the whole load.
    """
    height = int(height)
    width = int(width)
    divisor = model_resolution_grid(registry_model)["divisor"]
    if not divisor:
        return height, width
    aligned_height = max(divisor, round(height / divisor) * divisor)
    aligned_width = max(divisor, round(width / divisor) * divisor)
    if (aligned_height, aligned_width) != (height, width):
        _LOG.warning(
            "%s requires height and width divisible by %s; using %sx%s instead of %sx%s.",
            registry_model,
            divisor,
            aligned_width,
            aligned_height,
            width,
            height,
        )
    return aligned_height, aligned_width


def _is_rocm_runtime() -> bool:
    try:
        import torch  # type: ignore[reportMissingImports]

        return bool(getattr(torch.version, "hip", None))
    except Exception:
        return False


def gemm_precision_choices_for_model(registry_model: str) -> list[str]:
    del registry_model
    choices = ["native", "fp8", "fp4", "int8"]
    if _is_rocm_runtime() and "int8" in choices:
        choices.remove("int8")
    return choices


def resolve_model_task(
    registry_model: str,
    task: str | None,
    *,
    require_selection: bool = True,
    drop_invalid: bool = False,
) -> str | None:
    """Resolve the pipeline task, or raise so a run never starts on a wrong one.

    `drop_invalid` is for previews, whose job is to report the corrected state: a task
    the model cannot take is dropped and the browser rewrites the widget from
    `capabilities.valid_tasks`. A queued run keeps raising.
    """
    normalized = (task or "").strip()
    caps = model_capabilities(registry_model)
    if not caps:
        return normalized or None
    valid = caps.get("valid_tasks") or []
    if not valid:
        if not normalized or drop_invalid:
            return None
        raise ValueError(
            f"Model {registry_model!r} does not support pipeline tasks, but task {normalized!r} was "
            f"specified. Re-queue Model with a matching video preset, or clear task on Model."
        )
    if not normalized:
        if len(valid) == 1:
            return str(valid[0])
        if not require_selection:
            return None
        raise ValueError(
            f"Model {registry_model!r} requires a task. Choose one of {list(valid)} on xDiT Model."
        )
    if normalized not in valid:
        if drop_invalid:
            return str(valid[0]) if len(valid) == 1 else None
        raise ValueError(
            f"Model {registry_model!r} supports tasks {list(valid)}, not {normalized!r}."
        )
    return normalized


def validate_model_task(registry_model: str, task: str | None) -> None:
    resolve_model_task(registry_model, task)


def validate_runtime_for_model(registry_model: str, runtime: dict[str, Any]) -> None:
    offload_modes = [
        name
        for name in (
            "enable_model_cpu_offload",
            "enable_sequential_cpu_offload",
            "enable_group_cpu_offload",
        )
        if runtime.get(name)
    ]
    if len(offload_modes) > 1:
        raise ValueError("CPU offload modes are mutually exclusive; choose one CPU offload mode.")
    enabled = [
        name for name in ("use_fp8_gemms", "use_fp4_gemms", "use_int8_gemms") if runtime.get(name)
    ]
    if len(enabled) > 1:
        raise ValueError(
            "GEMM precision modes are mutually exclusive; choose exactly one of "
            "native, fp8, fp4, or int8."
        )
    precision = (
        "fp4"
        if runtime.get("use_fp4_gemms")
        else (
            "int8"
            if runtime.get("use_int8_gemms")
            else "fp8" if runtime.get("use_fp8_gemms") else "native"
        )
    )
    allowed = gemm_precision_choices_for_model(registry_model)
    if precision not in allowed:
        raise ValueError(
            f"Model {registry_model!r} does not support GEMM precision {precision!r} "
            f"on this runtime. Choose one of {allowed}."
        )


def widget_capability_gates(registry_model: str) -> dict[str, bool]:
    caps = model_capabilities(registry_model)
    if not caps:
        return {}
    supported_cache = caps.get("supported_cache_methods") or []
    step_cache_supported = bool(supported_cache)
    distilled_supported = bool(caps.get("supports_distilled_weights"))
    return {
        "STEP CACHE": step_cache_supported,
        "DISTILLED WEIGHTS": distilled_supported,
        "ulysses_degree": bool(caps.get("ulysses_degree", True)),
        "ring_degree": bool(caps.get("ring_degree", True)),
        "pipefusion_parallel_degree": bool(caps.get("pipefusion_parallel_degree")),
        "tensor_parallel_degree": bool(caps.get("tensor_parallel_degree")),
        "text_encoder_tp_degree": bool(caps.get("text_encoder_tp_degree")),
        "data_parallel_degree": bool(caps.get("data_parallel_degree", True)),
        "use_cfg_parallel": bool(caps.get("use_cfg_parallel")),
        "use_parallel_vae": bool(caps.get("use_parallel_vae")),
        "gemm_precision": bool(
            caps.get("use_fp8_gemms")
            or caps.get("has_fp8_modules")
            or caps.get("use_fp4_gemms")
            or caps.get("has_fp4_modules")
            or caps.get("use_int8_gemms")
            or caps.get("has_int8_modules")
        ),
        "use_fp8_text_encoder": bool(caps.get("use_fp8_text_encoder")),
        "fully_shard_degree": bool(caps.get("fully_shard_degree")),
        "cache_method": step_cache_supported,
        "residual_diff_threshold": step_cache_supported,
        "Fn_compute_blocks": step_cache_supported,
        "Bn_compute_blocks": step_cache_supported,
        "max_warmup_steps": step_cache_supported,
        "max_cached_steps": step_cache_supported,
        "scm_policy": step_cache_supported,
        "enable_separate_cfg": step_cache_supported,
        "enable_encoder_calibrator": step_cache_supported,
        "distilled_transformer_path": distilled_supported,
        "distilled_transformer_2_path": distilled_supported,
        "use_torch_compile": True,
        "reshard_after_forward": True,
        "memory_efficient_sharding": True,
        "cpu_offload_mode": True,
        "attention_backend": bool(caps.get("attention_backend", True)),
        "cross_attention_backend": bool(caps.get("cross_attention_backend")),
        **{name: True for name in _CAPABILITY_INDEPENDENT_WIDGETS},
    }


def loader_widget_gates(registry_model: str, cache_method: str | None = None) -> dict[str, bool]:
    from .runner_contract import cache_method_widget_gates

    gates = widget_capability_gates(registry_model)
    if gates.get("cache_method"):
        for key, allowed in cache_method_widget_gates(cache_method).items():
            gates[key] = allowed
    # A model that caches nothing still has to say so, or its denoiser groups stay on show.
    gates.update(denoiser_cache_gates(registry_model, cache_method))
    return gates


def denoiser_cache_gates(registry_model: str, cache_method: str | None) -> dict[str, bool]:
    """Show a denoiser's cache group only when the model actually caches that denoiser."""
    from .runner_contract import (
        MAX_CACHED_TRANSFORMERS,
        cache_method_widget_gates,
        denoiser_cache_group_label,
        denoiser_cache_widget_name,
    )

    denoisers = len(model_cache_transformers(registry_model, cache_method))
    field_gates = cache_method_widget_gates(cache_method)
    gates: dict[str, bool] = {}
    for index in range(2, MAX_CACHED_TRANSFORMERS + 1):
        present = denoisers >= index
        gates[denoiser_cache_group_label(index)] = present
        for field_name, allowed in field_gates.items():
            gates[denoiser_cache_widget_name(index, field_name)] = present and allowed
    return gates


def cache_method_choices_for_model(registry_model: str, all_choices: list[str]) -> list[str]:
    del registry_model
    return list(all_choices)


def _preset_field_values(preset) -> dict[str, Any]:
    try:
        from xfuser.model_executor.cache.presets import DBCachePreset
    except Exception:
        DBCachePreset = None
    if DBCachePreset is not None and isinstance(preset, DBCachePreset):
        return {
            field.name: getattr(preset, field.name)
            for field in dataclasses.fields(DBCachePreset)
            if getattr(preset, field.name, None) is not None
        }
    if isinstance(preset, dict):
        return dict(preset)
    return {}


def model_cache_transformers(registry_model: str, cache_method: str | None) -> list[dict[str, Any]]:
    """Per-transformer cache defaults, one entry per denoiser xFuser caches.

    Wan 2.2 caches a high-noise denoiser and a low-noise refiner with different warmup
    steps, so the values differ per transformer. Single-transformer models return one
    entry, which keeps callers on a single code path.
    """
    if not registry_model or not cache_method or str(cache_method).lower() in ("", "none"):
        return []
    settings = runner_settings_for_model(registry_model)
    cache_configs = settings.step_cache_config if settings else None
    if not isinstance(cache_configs, dict):
        return []
    method_cfg = cache_configs.get(str(cache_method))
    if method_cfg is None:
        return []

    adapters = getattr(method_cfg, "adapter", None)
    presets = getattr(method_cfg, "preset", None)
    if not isinstance(adapters, (list, tuple)):
        adapters = [adapters]
    if not isinstance(presets, (list, tuple)):
        presets = [presets] * len(adapters)

    fallback_names = getattr(settings, "transformer_attr_names", None) or []
    entries = []
    for index, adapter in enumerate(adapters):
        name = getattr(adapter, "transformer_attr", None)
        if not name:
            name = fallback_names[index] if index < len(fallback_names) else "transformer"
        preset = presets[index] if index < len(presets) else None
        entries.append({"transformer": str(name), "defaults": _preset_field_values(preset)})
    return [entry for entry in entries if entry["defaults"]]


def model_cache_preset_defaults(registry_model: str, cache_method: str | None) -> dict[str, Any]:
    """Defaults for the first cached transformer, which drives the single widget set."""
    entries = model_cache_transformers(registry_model, cache_method)
    return dict(entries[0]["defaults"]) if entries else {}
