"""Sanitize queued prompt inputs from connected xDiT Preset nodes before ComfyUI validation."""

from __future__ import annotations

from typing import Any

from .presets import PRESET_NONE, build_preset_spec, default_gpu_device_ids
from .runtime_config import (
    _gpu_device_ids_need_preset_sync,
    _is_invalid_custom_model_id,
    _is_unset_generation_value,
    _is_unset_loader_value,
    _model_preset_base,
    _normalize_custom_model_id,
    _normalize_task,
    _normalize_timeout_seconds,
    _runtime_loader_model_choices,
)

_LOADER_NODE_TYPES = frozenset({"xDiT.Model"})

_GENERATE_NODE_TYPES = frozenset({"xDiT.Sample"})

_GENERATION_SAFE_DEFAULTS = {
    "num_frames": 1,
    "timeout_seconds": 900,
    "flow_shift": 0.0,
    "guidance_scale_2": 0.0,
}


def _is_prompt_link(value) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def _resolve_preset_spec_from_link(prompt: dict[str, Any], link) -> dict[str, Any]:
    if not isinstance(link, list) or len(link) != 2:
        return {}
    preset_node = prompt.get(str(link[0]))
    if not isinstance(preset_node, dict) or preset_node.get("class_type") != "xDiT.Preset":
        return {}
    inputs = preset_node.get("inputs") or {}
    gpu_tag = inputs.get("gpu_tag", "")
    gpu_count = inputs.get("gpu_count")
    preset = (inputs.get("preset") or PRESET_NONE).strip() or PRESET_NONE
    spec = build_preset_spec(
        preset,
        gpu_tag,
        registry_choices=_runtime_loader_model_choices(),
    )
    if (
        spec.get("matched")
        and gpu_count not in (None, "")
        and int(spec.get("gpu_count") or 0) != _coerce_int(gpu_count, 0, min_value=1)
    ):
        return {}
    return spec


def _coerce_int(value, default, *, min_value=None, max_value=None):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if min_value is not None and parsed < min_value:
        return default
    if max_value is not None and parsed > max_value:
        return default
    return parsed


def _align_dimension(value, *, step=16, min_value=64, max_value=4096):
    aligned = int(round(value / step)) * step
    return max(min_value, min(max_value, aligned))


def _coerce_dimension(value, default, *, min_value=64, max_value=4096, step=16):
    fallback = _coerce_int(default, min_value, min_value=min_value, max_value=max_value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return _align_dimension(fallback, step=step, min_value=min_value, max_value=max_value)
        if "/" in stripped or "\\" in stripped:
            return _align_dimension(fallback, step=step, min_value=min_value, max_value=max_value)
        if stripped.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")):
            return _align_dimension(fallback, step=step, min_value=min_value, max_value=max_value)
        try:
            parsed = int(stripped)
        except ValueError:
            return _align_dimension(fallback, step=step, min_value=min_value, max_value=max_value)
    else:
        parsed = _coerce_int(value, fallback, min_value=min_value, max_value=max_value)
    return _align_dimension(parsed, step=step, min_value=min_value, max_value=max_value)


def _coerce_float(value, default, *, min_value=None, max_value=None):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if min_value is not None and parsed < min_value:
        return default
    if max_value is not None and parsed > max_value:
        return default
    return parsed


def _coerce_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off", ""):
            return False
    if value is None:
        return bool(default)
    return bool(value)


def _apply_model_preset_inputs(inputs: dict[str, Any], spec: dict[str, Any]) -> None:
    base = _model_preset_base(spec)
    for key, value in base.items():
        if _is_unset_generation_value(value):
            continue
        if (
            (key == "gpu_count" and not _is_prompt_link(inputs.get(key)))
            or key not in inputs
            or _is_unset_loader_value(key, inputs.get(key))
        ):
            inputs[key] = value

    # The preset dictates the parallel degrees, so it has to dictate how many devices
    # they run on: a kept device list of the wrong length is a layout the node refuses.
    preset_gpu_count = int(spec.get("gpu_count") or 0)
    if preset_gpu_count and not _is_prompt_link(inputs.get("gpu_device_ids")):
        if _gpu_device_ids_need_preset_sync(inputs.get("gpu_device_ids"), preset_gpu_count):
            inputs["gpu_device_ids"] = default_gpu_device_ids(preset_gpu_count)


def _coerce_loader_widget_inputs(inputs: dict[str, Any]) -> None:
    from .runner_contract import (
        _is_bool_action,
        _runner_actions,
        default_loader_widget_values,
        loader_config_widget_names,
    )

    defaults = default_loader_widget_values()
    for name in loader_config_widget_names():
        if name not in inputs:
            continue
        action = _runner_actions().get(name)
        raw = inputs.get(name)
        if _is_prompt_link(raw):
            continue
        if action is not None and action.type is int:
            fallback = defaults.get(name, 0)
            inputs[name] = _coerce_int(raw, fallback, min_value=0)
            continue
        if action is not None and _is_bool_action(action):
            inputs[name] = _coerce_bool(raw, defaults.get(name, False))
            continue
        if raw in (None, "") and name in defaults:
            inputs[name] = defaults[name]


def _sanitize_loader_inputs(inputs: dict[str, Any], _spec: dict[str, Any] | None = None) -> None:
    from .model_info import sanitize_loader_inputs_for_model
    from .residency_allocator import normalize_residency
    from .runner_contract import (
        default_loader_widget_values,
        loader_optional_widget_defaults,
        strip_undeclared_loader_keys,
    )
    from .runtime_config import _resolve_model_choice

    strip_undeclared_loader_keys(inputs)
    if not _is_prompt_link(inputs.get("residency")):
        # Combo widgets keep whatever a saved graph had, and ComfyUI rejects the prompt
        # before the runtime gets to normalize it.
        inputs["residency"] = normalize_residency(inputs.get("residency"))
    if not _is_prompt_link(inputs.get("custom_model_id")):
        if _is_invalid_custom_model_id(inputs.get("custom_model_id")):
            inputs["custom_model_id"] = ""
        else:
            inputs["custom_model_id"] = _normalize_custom_model_id(inputs.get("custom_model_id"))
    if not _is_prompt_link(inputs.get("task")):
        inputs["task"] = _normalize_task(inputs.get("task")) or ""

    defaults = {**default_loader_widget_values(), **loader_optional_widget_defaults()}
    for key, value in defaults.items():
        if key not in inputs or _is_unset_loader_value(key, inputs.get(key)):
            inputs[key] = value

    _coerce_loader_widget_inputs(inputs)

    if _is_prompt_link(inputs.get("model")):
        return
    model = _resolve_model_choice(inputs.get("model"), inputs.get("custom_model_id"))
    if model:
        sanitize_loader_inputs_for_model(model, inputs)


def _sanitize_generate_inputs(inputs: dict[str, Any], spec: dict[str, Any]) -> None:
    defaults = spec.get("generation_defaults") or {}
    fallback = {**_GENERATION_SAFE_DEFAULTS, **defaults}

    if not _is_prompt_link(inputs.get("num_frames")):
        inputs["num_frames"] = _coerce_int(
            inputs.get("num_frames"),
            fallback.get("num_frames", 1),
            min_value=1,
        )
    if not _is_prompt_link(inputs.get("timeout_seconds")):
        inputs["timeout_seconds"] = _normalize_timeout_seconds(
            inputs.get("timeout_seconds"),
            fallback.get("timeout_seconds", 900),
        )
    if not _is_prompt_link(inputs.get("flow_shift")):
        inputs["flow_shift"] = _coerce_float(
            inputs.get("flow_shift"),
            fallback.get("flow_shift", 0.0),
            min_value=0.0,
            max_value=100.0,
        )
    if not _is_prompt_link(inputs.get("guidance_scale_2")):
        inputs["guidance_scale_2"] = _coerce_float(
            inputs.get("guidance_scale_2"),
            fallback.get("guidance_scale_2", 0.0),
            min_value=0.0,
            max_value=100.0,
        )
    if (
        "resize_input_images" in inputs or "resize_input_images" in defaults
    ) and not _is_prompt_link(inputs.get("resize_input_images")):
        inputs["resize_input_images"] = _coerce_bool(
            inputs.get("resize_input_images"),
            fallback.get("resize_input_images", False),
        )
    if ("output_fps" in inputs or "output_fps" in defaults) and not _is_prompt_link(
        inputs.get("output_fps")
    ):
        inputs["output_fps"] = _coerce_int(
            inputs.get("output_fps"),
            fallback.get("output_fps", 0),
            min_value=0,
        )

    for key in ("prompt", "negative_prompt"):
        value = inputs.get(key)
        if (value is None or (isinstance(value, str) and not value.strip())) and key in fallback:
            inputs[key] = fallback[key]

    for key in ("height", "width"):
        if not _is_prompt_link(inputs.get(key)):
            inputs[key] = _coerce_dimension(
                inputs.get(key),
                fallback.get(key, 1024),
            )

    for key in ("num_inference_steps", "max_sequence_length", "guidance_scale", "seed"):
        if key not in inputs or inputs.get(key) in (None, ""):
            if key in defaults:
                inputs[key] = defaults[key]

    if spec.get("matched"):
        for key, preset_value in defaults.items():
            if preset_value is None:
                continue
            if _is_unset_generation_value(inputs.get(key)):
                inputs[key] = preset_value


def apply_preset_prompt_overrides(json_data: dict[str, Any]) -> dict[str, Any]:
    prompt = json_data.get("prompt")
    if not isinstance(prompt, dict):
        return json_data

    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        inputs = node.setdefault("inputs", {})

        if class_type in _LOADER_NODE_TYPES:
            from .runner_contract import strip_undeclared_loader_keys

            strip_undeclared_loader_keys(inputs)
            spec = _resolve_preset_spec_from_link(prompt, inputs.get("preset"))
            if spec.get("matched"):
                _apply_model_preset_inputs(inputs, spec)
            _sanitize_loader_inputs(inputs, spec if spec.get("matched") else None)
            continue

        if class_type in _GENERATE_NODE_TYPES:
            spec = _resolve_preset_spec_from_link(prompt, inputs.get("preset"))
            _sanitize_generate_inputs(inputs, spec if spec.get("matched") else {})

    _register_loader_consumers(prompt)
    return json_data


def _register_loader_consumers(prompt: dict[str, Any]) -> None:
    """Count Sample nodes per Model node so release/park_cpu act on the last one."""
    from .worker import register_prompt_loader_consumers

    counts: dict[str, int] = {}
    for node in prompt.values():
        if not isinstance(node, dict) or node.get("class_type") not in _GENERATE_NODE_TYPES:
            continue
        link = (node.get("inputs") or {}).get("model")
        if not _is_prompt_link(link):
            continue
        origin = prompt.get(str(link[0]))
        if isinstance(origin, dict) and origin.get("class_type") in _LOADER_NODE_TYPES:
            counts[str(link[0])] = counts.get(str(link[0]), 0) + 1
    register_prompt_loader_consumers(counts)
