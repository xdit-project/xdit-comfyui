from __future__ import annotations

import dataclasses
import json
from functools import lru_cache
from typing import Any

from .model_info import (
    model_cache_preset_defaults,
    model_cache_transformers,
    model_supports_step_cache,
    sanitize_loader_cache_widgets,
)

_INT_WIDGET_MAX = 4096

# ComfyUI derives float widget display precision from step and rounds the stored
# value to it, so a coarse step silently mangles values like 0.08.
_FLOAT_WIDGET_RANGE = {"min": 0.0, "max": 1000.0, "step": 0.001}

_SCM_POLICY_CHOICES = ("default", "slow", "medium", "fast", "ultra")
_OPTIONAL_BOOL_CHOICES = ("default", "true", "false")

_GENERATION_DESTS = frozenset(
    {
        "height",
        "width",
        "num_frames",
        "prompt",
        "negative_prompt",
        "num_inference_steps",
        "max_sequence_length",
        "seed",
        "guidance_scale",
        "guidance_scale_2",
        "flow_shift",
        "output_directory",
        "input_images",
        "resize_input_images",
        "task",
        "batch_size",
        "dataset_path",
        "num_iterations",
    }
)

_AUTO_BACKEND_DESTS = frozenset({"attention_backend", "cross_attention_backend"})

SAMPLE_VAE_DESTS = (
    "enable_tiling",
    "enable_slicing",
    "vae_tile_size_height",
    "vae_tile_size_width",
    "vae_tile_overlap_height",
    "vae_tile_overlap_width",
)

RESIDENCY_KEEP_GPU = "keep_gpu"
RESIDENCY_PARK_CPU = "park_cpu"
RESIDENCY_RELEASE = "release"
RESIDENCY_CHOICES = [
    RESIDENCY_KEEP_GPU,
    RESIDENCY_PARK_CPU,
    RESIDENCY_RELEASE,
]


def loader_pinned_input_types(
    model_choices: list[str],
    default_model: str | None = None,
) -> dict[str, tuple[Any, dict[str, Any]]]:
    """The widgets the Model node owns itself rather than deriving from the runner CLI."""
    if default_model is None:
        default_model = model_choices[0] if model_choices else ""
    return {
        "model": (
            list(model_choices),
            {
                "default": default_model,
                "tooltip": "Model supported by the installed xDiT version.",
            },
        ),
        "task": (
            "STRING",
            {
                "default": "",
                "tooltip": (
                    "Pipeline task for multi-task models, such as i2v or t2v. "
                    "Leave empty for single-task models."
                ),
            },
        ),
        "gpu_device_ids": (
            "STRING",
            {
                "default": "0",
                "tooltip": (
                    "Comma-separated GPU indices. Specify one index per process, such as "
                    "0,1,2,3 for four processes. Use separate indices for concurrent models."
                ),
            },
        ),
        "residency": (
            RESIDENCY_CHOICES,
            {
                "default": RESIDENCY_KEEP_GPU,
                "tooltip": (
                    "GPU memory policy after sampling. keep_gpu retains the model until "
                    "you unload it or stop ComfyUI; park_cpu moves weights to system RAM; "
                    "release stops the worker after the run."
                ),
            },
        ),
        "use_torch_compile": (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": (
                    "Compile supported model components with torch.compile. The first run "
                    "is slower; later runs with the same configuration may be faster."
                ),
            },
        ),
    }


def loader_optional_input_types() -> dict[str, tuple[Any, dict[str, Any]]]:
    """Model widgets that are optional inputs rather than runner args."""
    return {
        "hf_cache_mode": (
            ["auto", "comfy_models_shared", "system_default", "custom_path"],
            {
                "default": "auto",
                "tooltip": (
                    "Hugging Face cache location. auto uses the environment, then "
                    "ComfyUI/models/huggingface. custom_path uses the directory below."
                ),
            },
        ),
        "hf_cache_dir": (
            "STRING",
            {
                "default": "huggingface",
                "tooltip": ("Hugging Face cache root used when HF Cache Mode is custom_path."),
            },
        ),
    }


LOADER_PINNED_WIDGETS = tuple(loader_pinned_input_types([]))

LOADER_OPTIONAL_WIDGETS = tuple(loader_optional_input_types())

# Runner args the Model node does not surface as a widget of its own. Everything else
# xDiT accepts becomes a widget automatically, so a new CLI arg needs no code here.
_LOADER_EXCLUDED_DESTS = frozenset(
    {
        *_GENERATION_DESTS,
        *LOADER_PINNED_WIDGETS,
        # Folded into the cpu_offload_mode and gemm_precision combos below.
        "enable_model_cpu_offload",
        "enable_sequential_cpu_offload",
        "enable_group_cpu_offload",
        "use_fp8_gemms",
        "use_fp4_gemms",
        "use_int8_gemms",
        *SAMPLE_VAE_DESTS,
        "warmup_calls",
        "profile",
        "profile_wait",
        "profile_warmup",
        "profile_active",
        "vsa_collect_density",
        # cache_config is represented by cache_method plus the Step Cache group.
        "cache_config",
    }
)

# Combos that stand in for several runner flags at once.
_COMPOUND_LOADER_WIDGETS = ("cpu_offload_mode", "gemm_precision")

_DISTILLED_WEIGHT_TOOLTIPS = {
    "distilled_transformer_path": (
        "Wan 2.2 Distilled I2V only. Path to the high-noise LightX2V transformer file."
    ),
    "distilled_transformer_2_path": (
        "Wan 2.2 Distilled I2V only. Path to the low-noise LightX2V transformer file."
    ),
}

_LOADER_WIDGET_GROUPS = (
    {
        "id": "model_cache",
        "label": "MODEL CACHE",
        "collapsed": True,
        "widgets": ("hf_cache_mode", "hf_cache_dir"),
    },
    {
        "id": "parallelism",
        "label": "PARALLELISM",
        "collapsed": True,
        "widgets": (
            "gpu_device_ids",
            "ulysses_degree",
            "ring_degree",
            "pipefusion_parallel_degree",
            "tensor_parallel_degree",
            "text_encoder_tp_degree",
            "data_parallel_degree",
            "use_cfg_parallel",
            "fully_shard_degree",
            "reshard_after_forward",
        ),
    },
    {
        "id": "memory",
        "label": "MEMORY",
        "collapsed": True,
        "widgets": (
            "memory_efficient_sharding",
            "memory_efficient_replicated_load",
            "cpu_offload_mode",
            "group_offload_low_cpu_mem",
        ),
    },
    {
        "id": "vae",
        "label": "VAE",
        "collapsed": True,
        "widgets": (
            "use_parallel_vae",
            "use_vae_channels_last_format",
            "enable_tiling",
            "enable_slicing",
            "vae_tile_size_height",
            "vae_tile_size_width",
            "vae_tile_overlap_height",
            "vae_tile_overlap_width",
        ),
    },
    {
        "id": "attention",
        "label": "ATTENTION",
        "collapsed": True,
        "widgets": (
            "attention_backend",
            "cross_attention_backend",
            "use_hybrid_attn_schedule",
            "hybrid_attn_low_precision_backend",
            "hybrid_attn_high_precision_backend",
            "num_hybrid_attn_high_precision_steps",
            "hybrid_attn_schedule",
            "use_ssta_sparse_text_to_image",
            "spargeattn_simthreshold",
            "spargeattn_cdfthreshold",
            "spargeattn_reorder_sequence",
            "use_spargeattn_static_block_mask",
            "use_spargeattn_head_balance",
            "vsa_block_size",
            "vsa_top_k",
            "vsa_top_k_ratio",
            "vsa_drop_rates",
            "vsa_prob_threshold",
            "vsa_reorder_sequence",
            "use_vsa_static_block_mask",
            "use_vsa_first_frame_mask",
            "vsa_collect_density",
        ),
    },
    {
        "id": "quantization",
        "label": "GEMM PRECISION",
        "collapsed": True,
        "widgets": (
            "gemm_precision",
            "use_fp8_text_encoder",
            "fp8_precision_override_prefix_patterns",
            "fp8_precision_override_suffix_patterns",
            "use_hybrid_gemm_schedule",
            "num_hybrid_gemm_high_precision_steps",
        ),
    },
    {
        "id": "cache",
        "label": "STEP CACHE",
        "collapsed": True,
        "widgets": (
            "cache_method",
            "residual_diff_threshold",
            "Fn_compute_blocks",
            "Bn_compute_blocks",
            "max_warmup_steps",
            "max_cached_steps",
            "scm_policy",
            "enable_separate_cfg",
            "enable_encoder_calibrator",
            "enable_taylorseer",
        ),
    },
    {
        "id": "cache_denoiser_2",
        "label": "STEP CACHE · DENOISER 2",
        "description": (
            "Cache settings for the second transformer in dual-transformer models. "
            "The main Step Cache settings apply to the first transformer."
        ),
        "collapsed": True,
        "widgets": tuple(
            f"t2_{field_name}"
            for field_name in (
                "residual_diff_threshold",
                "Fn_compute_blocks",
                "Bn_compute_blocks",
                "max_warmup_steps",
                "max_cached_steps",
                "scm_policy",
                "enable_separate_cfg",
                "enable_encoder_calibrator",
                "enable_taylorseer",
            )
        ),
    },
    {
        "id": "distilled_weights",
        "label": "DISTILLED WEIGHTS",
        "description": (
            "Local LightX2V transformer files for Wan 2.2 Distilled I2V. "
            "Ignored by other models."
        ),
        "collapsed": True,
        "widgets": (
            "distilled_transformer_path",
            "distilled_transformer_2_path",
        ),
    },
)


def _normalize_empty(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def loader_widget_groups() -> tuple[dict[str, Any], ...]:
    return _LOADER_WIDGET_GROUPS


def loader_schema() -> dict[str, Any]:
    widget_names = (
        set(loader_config_widget_names())
        | set(LOADER_OPTIONAL_WIDGETS)
        | set(LOADER_PINNED_WIDGETS)
    )
    grouped: set[str] = set()
    groups: list[dict[str, Any]] = []
    widget_constraints: dict[str, dict[str, Any]] = {}
    for dest in loader_config_widget_names():
        typ, spec = loader_widget_spec(dest)
        if typ == "INT":
            widget_constraints[dest] = {
                "type": "int",
                "default": spec["default"],
                "min": spec.get("min", 0),
            }
        elif typ == "BOOLEAN":
            widget_constraints[dest] = {
                "type": "boolean",
                "default": spec["default"],
            }
    for group in _LOADER_WIDGET_GROUPS:
        widgets = tuple(name for name in group["widgets"] if name in widget_names)
        if not widgets:
            continue
        grouped.update(widgets)
        groups.append(
            {
                "id": group["id"],
                "label": group["label"],
                "description": group.get("description", ""),
                "collapsed": group["collapsed"],
                "expand_widget": expand_widget_name(group),
                "widgets": list(widgets),
            }
        )
    ungrouped = [name for name in loader_widget_render_order() if name not in grouped]
    if ungrouped:
        groups.append(
            {
                "id": "other",
                "label": "Other",
                "description": "",
                "collapsed": True,
                "expand_widget": "Other",
                "widgets": ungrouped,
            }
        )
    from .model_info import gemm_precision_choices_for_model

    combo_options: dict[str, list[str]] = {}
    for name in (*LOADER_PINNED_WIDGETS, *loader_widget_render_order()):
        typ = (
            loader_pinned_input_types([])[name][0]
            if name in LOADER_PINNED_WIDGETS
            else loader_widget_spec(name)[0]
        )
        if isinstance(typ, list):
            combo_options[name] = list(typ)
    # Hardware-filtered here rather than in the browser: ROCm has no int8 GEMM, and the
    # browser must not offer a value the server would reject.
    combo_options["gemm_precision"] = gemm_precision_choices_for_model("")
    # A STRING input the browser rewrites into the model's valid tasks.
    combo_widgets = sorted({*combo_options, "task"})

    return {
        "config_widgets": list(loader_config_widget_names()),
        "pinned_widgets": list(LOADER_PINNED_WIDGETS),
        "widget_groups": groups,
        "widget_defaults": default_loader_widget_values(),
        "widget_constraints": widget_constraints,
        "combo_widgets": combo_widgets,
        "combo_options": combo_options,
    }


def _xfuser_unavailable(what: str, exc: BaseException) -> RuntimeError:
    """Say which part of the node pack is broken instead of degrading silently.

    Every widget name, type, and default in this pack is read out of xfuser. Swallowing
    an import error here would render a node full of dead text boxes and drop the
    `_build_cli_args` allowlist, so nothing runs and nothing says why.
    """
    return RuntimeError(
        f"xDiT nodes cannot read {what} from xfuser. Install the xfuser this pack was "
        f"built against into ComfyUI's Python (pip install -r requirements.txt)."
    )


@lru_cache(maxsize=1)
def _runner_arg_parser():
    """The xdit CLI's own parser: the single source of runner arg names, types, and help."""
    try:
        from xfuser.config.args import FlexibleArgumentParser, xFuserArgs

        parser = FlexibleArgumentParser()
        xFuserArgs.add_runner_args(parser)
    except Exception as exc:
        raise _xfuser_unavailable("the runner CLI arguments", exc) from exc
    if not parser._actions:
        raise RuntimeError("xfuser's runner argument parser declares no arguments.")
    return parser


def _runner_cli_actions():
    return [action for action in _runner_arg_parser()._actions if action.dest != "help"]


@lru_cache(maxsize=1)
def deprecated_runner_dests() -> frozenset[str]:
    return frozenset(
        action.dest
        for action in _runner_cli_actions()
        if str(action.help or "").lstrip().startswith("[Deprecated]")
    )


@lru_cache(maxsize=1)
def runner_cli_dests() -> frozenset[str]:
    return frozenset(action.dest for action in _runner_cli_actions())


@lru_cache(maxsize=1)
def runner_cli_help() -> dict[str, str]:
    return {
        action.dest: " ".join(str(action.help).split())
        for action in _runner_cli_actions()
        if action.help
    }


def runner_nproc_from_values(values: dict[str, Any]) -> int:
    """Derive xDiT world size without requiring xDiT during plugin discovery."""
    nproc = 1
    for name in (
        "ulysses_degree",
        "ring_degree",
        "pipefusion_parallel_degree",
        "tensor_parallel_degree",
        "data_parallel_degree",
    ):
        nproc *= max(int(values.get(name, 1) or 1), 1)
    if values.get("use_cfg_parallel"):
        nproc *= 2
    return nproc


@lru_cache(maxsize=1)
def _dbcache_preset_defaults() -> dict[str, Any]:
    try:
        from xfuser.model_executor.cache.presets import DBCachePreset
    except Exception as exc:
        raise _xfuser_unavailable("the dbcache preset fields", exc) from exc
    defaults = {
        field.name: field.default
        for field in dataclasses.fields(DBCachePreset)
        if field.default is not dataclasses.MISSING
    }
    if not defaults:
        raise RuntimeError("xfuser's DBCachePreset declares no defaulted fields.")
    return defaults


@lru_cache(maxsize=1)
def cache_config_widget_names() -> tuple[str, ...]:
    return tuple(_dbcache_preset_defaults().keys())


# Wan 2.2 is the deepest stack xfuser caches today: a high-noise expert and a low-noise
# refiner. A registry scan in the tests fails if a model ever caches more, which is the
# signal to raise this and add the matching group.
MAX_CACHED_TRANSFORMERS = 2

PER_TRANSFORMER_CACHE_KEY = "per_transformer"


def denoiser_cache_group_label(index: int) -> str:
    return f"STEP CACHE · DENOISER {index}"


def denoiser_cache_widget_name(index: int, field_name: str) -> str:
    """Widget holding `field_name` for the index-th cached denoiser (1-based)."""
    if index < 1 or index > MAX_CACHED_TRANSFORMERS:
        raise ValueError(f"No cache widgets exist for denoiser {index}.")
    if index == 1:
        return field_name
    return f"t{index}_{field_name}"


@lru_cache(maxsize=1)
def extra_denoiser_cache_widget_names() -> tuple[str, ...]:
    """Cache widgets for every denoiser after the first, in group order."""
    return tuple(
        denoiser_cache_widget_name(index, field_name)
        for index in range(2, MAX_CACHED_TRANSFORMERS + 1)
        for field_name in cache_config_widget_names()
    )


def denoiser_cache_widget_field(widget_name: str) -> tuple[int, str] | None:
    """Split an extra-denoiser widget back into (index, field), or None if it is not one."""
    if not widget_name.startswith("t"):
        return None
    prefix, _, field_name = widget_name.partition("_")
    if not field_name or not prefix[1:].isdigit():
        return None
    index = int(prefix[1:])
    if index < 2 or index > MAX_CACHED_TRANSFORMERS:
        return None
    if field_name not in cache_config_widget_names():
        return None
    return index, field_name


@lru_cache(maxsize=1)
def dbcache_only_cache_widgets() -> frozenset[str]:
    return frozenset(
        name for name in cache_config_widget_names() if name != "residual_diff_threshold"
    )


def normalize_cache_method(cache_method: str | None) -> str:
    if cache_method in (None, ""):
        return "none"
    return str(cache_method).strip().lower()


def cache_config_fields_for_method(cache_method: str | None) -> frozenset[str]:
    method = normalize_cache_method(cache_method)
    if method in ("", "none"):
        return frozenset()
    if method == "dbcache":
        return frozenset(cache_config_widget_names())
    if method in ("teacache", "fbcache"):
        return frozenset({"residual_diff_threshold"})
    return frozenset()


def cache_method_widget_gates(cache_method: str | None) -> dict[str, bool]:
    allowed = cache_config_fields_for_method(cache_method)
    return {name: name in allowed for name in cache_config_widget_names()}


def filter_cache_config_for_method(
    cache_method: str | None,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    if not config:
        return {}
    allowed = cache_config_fields_for_method(cache_method)
    filtered = {key: value for key, value in config.items() if key in allowed}
    keyed = config.get(PER_TRANSFORMER_CACHE_KEY)
    if isinstance(keyed, dict):
        per_transformer = {
            name: {key: value for key, value in overrides.items() if key in allowed}
            for name, overrides in keyed.items()
            if isinstance(overrides, dict)
        }
        per_transformer = {name: o for name, o in per_transformer.items() if o}
        if per_transformer:
            filtered[PER_TRANSFORMER_CACHE_KEY] = per_transformer
    return filtered


def broadcast_cache_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """The part of a cache_config xfuser itself understands.

    `per_transformer` is read by the plugin's step-cache patch when it refreshes each
    denoiser; xfuser's own config builder takes a flat object and would reject the key.
    """
    if not config:
        return {}
    return {key: value for key, value in config.items() if key != PER_TRANSFORMER_CACHE_KEY}


def _normalize_optional_runner_str(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return value


def sanitize_attention_runtime(runtime: dict[str, Any]) -> None:
    schedule = _normalize_optional_runner_str(runtime.get("hybrid_attn_schedule"))
    low = _normalize_optional_runner_str(runtime.get("hybrid_attn_low_precision_backend"))
    high = _normalize_optional_runner_str(runtime.get("hybrid_attn_high_precision_backend"))
    runtime["hybrid_attn_schedule"] = schedule
    runtime["hybrid_attn_low_precision_backend"] = low
    runtime["hybrid_attn_high_precision_backend"] = high

    if not runtime.get("use_hybrid_attn_schedule"):
        runtime.pop("hybrid_attn_schedule", None)
        runtime.pop("hybrid_attn_low_precision_backend", None)
        runtime.pop("hybrid_attn_high_precision_backend", None)
        runtime.pop("num_hybrid_attn_high_precision_steps", None)
        return

    runtime.pop("attention_backend", None)
    if schedule is not None:
        runtime["hybrid_attn_low_precision_backend"] = None
        runtime["hybrid_attn_high_precision_backend"] = None


def sanitize_cache_config_runtime(runtime: dict[str, Any]) -> None:
    method = normalize_cache_method(runtime.get("cache_method"))
    if method in ("", "none"):
        runtime["cache_method"] = None
        runtime.pop("cache_config", None)
        return
    runtime["cache_method"] = None if method == "none" else method
    if "cache_config" not in runtime:
        return
    filtered = filter_cache_config_for_method(
        method,
        _parse_cache_config_value(runtime["cache_config"]),
    )
    if filtered:
        runtime["cache_config"] = json.dumps(filtered, sort_keys=True)
    else:
        runtime.pop("cache_config", None)


def _cache_config_help(field_name: str) -> str:
    from xfuser.model_executor.cache.presets import DBCachePreset

    documented = ""
    for field in dataclasses.fields(DBCachePreset):
        if field.name == field_name and field.metadata.get("help"):
            documented = str(field.metadata["help"])
            break
    # Fields xFuser documents nowhere else.
    hints = {
        "residual_diff_threshold": "L1 residual threshold for step reuse (all cache methods).",
        "Fn_compute_blocks": "First N blocks used to compute the L1 difference (dbcache).",
        "Bn_compute_blocks": "TaylorSeer calibrator blocks; leave at 0 for most models.",
        "max_warmup_steps": "Steps before caching activates (dbcache).",
        "max_cached_steps": "Max cached steps (-1 = unlimited).",
        "scm_policy": (
            "Step computation mask for dbcache. slow computes most steps; medium, fast, "
            "and ultra progressively skip more work. default preserves the model preset."
        ),
        "enable_taylorseer": "TaylorSeer prediction for dbcache. default = model built-in setting.",
        "enable_separate_cfg": "Separate CFG passes (Wan, Qwen-Image-Edit). default = model built-in setting.",
        "enable_encoder_calibrator": "Encoder calibrator for dbcache. default = model built-in setting.",
    }
    text = documented or hints.get(field_name, f"cache_config override: {field_name}")
    advice = _widget_advice(field_name)
    return f"{text} — {advice}" if advice else text


def cache_config_widget_spec(field_name: str) -> tuple[str, dict[str, Any]]:
    defaults = _dbcache_preset_defaults()
    default = defaults.get(field_name)
    tooltip = _cache_config_help(field_name)

    if field_name == "residual_diff_threshold":
        return ("FLOAT", {"default": float(default), **_FLOAT_WIDGET_RANGE, "tooltip": tooltip})
    if field_name in ("enable_taylorseer", "enable_separate_cfg", "enable_encoder_calibrator"):
        return (
            list(_OPTIONAL_BOOL_CHOICES),
            {"default": "default", "tooltip": tooltip},
        )
    if field_name == "scm_policy":
        return (
            list(_SCM_POLICY_CHOICES),
            {"default": "default", "tooltip": tooltip},
        )
    if isinstance(default, int):
        return ("INT", {"default": int(default), "tooltip": tooltip})
    return ("STRING", {"default": "" if default is None else str(default), "tooltip": tooltip})


def _optional_bool_widget_to_json(value: Any) -> bool | None:
    if value in (None, "", "default"):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _optional_bool_json_to_widget(value: Any) -> str:
    if value is None:
        return "default"
    return "true" if bool(value) else "false"


def _scm_policy_widget_to_json(value: Any) -> str | None:
    if value in (None, "", "default"):
        return None
    return str(value)


def _scm_policy_json_to_widget(value: Any) -> str:
    if value in (None, ""):
        return "default"
    return str(value)


def _cache_override_value(field_name: str, widget_value: Any) -> Any | None:
    if widget_value in (None, ""):
        return None
    if field_name in ("enable_taylorseer", "enable_separate_cfg", "enable_encoder_calibrator"):
        return _optional_bool_widget_to_json(widget_value)
    if field_name == "scm_policy":
        return _scm_policy_widget_to_json(widget_value)
    if field_name == "residual_diff_threshold":
        if str(widget_value).strip().lower() in ("none", ""):
            return None
        return float(widget_value)
    if field_name in (
        "Fn_compute_blocks",
        "Bn_compute_blocks",
        "max_warmup_steps",
        "max_cached_steps",
    ):
        return int(widget_value)
    return widget_value


def _merge_cache_fields(
    method: str,
    model_defaults: dict[str, Any],
    explicit: dict[str, Any] | None,
) -> dict[str, Any]:
    allowed = cache_config_fields_for_method(method)
    if not allowed:
        return {}
    if method in ("teacache", "fbcache"):
        model_defaults = {
            key: value for key, value in model_defaults.items() if key == "residual_diff_threshold"
        }
    elif method != "dbcache":
        model_defaults = {}

    global_defaults = _dbcache_preset_defaults()
    merged: dict[str, Any] = {}
    for field_name in cache_config_widget_names():
        if field_name not in allowed:
            continue
        if explicit and field_name in explicit:
            merged[field_name] = explicit[field_name]
        elif field_name in model_defaults:
            merged[field_name] = model_defaults[field_name]
        elif field_name in global_defaults:
            merged[field_name] = global_defaults[field_name]
    return merged


def resolve_effective_cache_config(
    registry_model: str,
    cache_method: str | None,
    explicit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge cache fields in order: global defaults → model preset → explicit overrides."""
    method = normalize_cache_method(cache_method)
    return _merge_cache_fields(
        method, model_cache_preset_defaults(registry_model, method), explicit
    )


def effective_cache_config_per_transformer(
    registry_model: str,
    cache_method: str | None,
    explicit: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """What each cached denoiser will actually run with.

    Wan 2.2 warms its high-noise expert for 4 steps and its low-noise refiner for 2, so
    one set of numbers cannot describe the run. Overrides from the first group reach every
    denoiser; a `per_transformer` section overrides that again for one of them.
    """
    method = normalize_cache_method(cache_method)
    broadcast = broadcast_cache_config(explicit)
    keyed = (explicit or {}).get(PER_TRANSFORMER_CACHE_KEY) or {}
    rows = []
    for index, entry in enumerate(model_cache_transformers(registry_model, method), 1):
        overrides = dict(broadcast)
        extra = keyed.get(entry["transformer"])
        if isinstance(extra, dict):
            overrides.update(extra)
        rows.append(
            {
                "transformer": entry["transformer"],
                "index": index,
                "config": _merge_cache_fields(method, entry["defaults"], overrides),
            }
        )
    return rows


def cache_config_baseline(registry_model: str, cache_method: str | None) -> dict[str, Any]:
    return resolve_effective_cache_config(registry_model, cache_method, explicit={})


def cache_widget_defaults_for_model(
    registry_model: str, cache_method: str | None
) -> dict[str, Any]:
    """Every cache widget's starting value for this model, one set per cached denoiser."""
    baseline = cache_config_baseline(registry_model, cache_method)
    if not baseline:
        return {}
    widgets = _cache_config_to_widgets(json.dumps(baseline))
    method = normalize_cache_method(cache_method)
    entries = model_cache_transformers(registry_model, method)
    for index, entry in enumerate(entries[1:MAX_CACHED_TRANSFORMERS], start=2):
        per_denoiser = _cache_config_to_widgets(
            json.dumps(_merge_cache_fields(method, entry["defaults"], None))
        )
        for field_name, value in per_denoiser.items():
            widgets[denoiser_cache_widget_name(index, field_name)] = value
    return widgets


def _cache_field_values_equal(field_name: str, left: Any, right: Any) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    if field_name == "residual_diff_threshold":
        return float(left) == float(right)
    if field_name in (
        "Fn_compute_blocks",
        "Bn_compute_blocks",
        "max_warmup_steps",
        "max_cached_steps",
    ):
        return int(left) == int(right)
    return left == right


def _cache_overrides_from_widgets(
    values: dict[str, Any],
    allowed: frozenset[str],
    baseline: dict[str, Any],
    *,
    index: int = 1,
) -> dict[str, Any]:
    """Widget values for one denoiser that differ from what it would run without them."""
    overrides: dict[str, Any] = {}
    for field_name in cache_config_widget_names():
        widget_name = denoiser_cache_widget_name(index, field_name)
        if field_name not in allowed or widget_name not in values:
            continue
        if field_name in (
            "enable_taylorseer",
            "enable_separate_cfg",
            "enable_encoder_calibrator",
            "scm_policy",
        ):
            if values[widget_name] in (None, "", "default"):
                continue
        converted = _cache_override_value(field_name, values[widget_name])
        if converted is None:
            continue
        if _cache_field_values_equal(field_name, converted, baseline.get(field_name)):
            continue
        overrides[field_name] = converted
    return overrides


def _cache_config_from_widgets(
    values: dict[str, Any],
    registry_model: str | None = None,
) -> str | None:
    cache_method = widget_value_to_runtime("cache_method", values.get("cache_method", "none"))
    method = str(cache_method) if cache_method not in (None, "") else "none"
    allowed = cache_config_fields_for_method(method)
    model = registry_model or str(values.get("model") or "")
    baseline = cache_config_baseline(model, method)
    overrides = _cache_overrides_from_widgets(values, allowed, baseline)

    # The first group's overrides reach every denoiser, so a later group only has to say
    # what it wants *differently* from that.
    keyed: dict[str, dict[str, Any]] = {}
    denoisers = model_cache_transformers(model, method)
    for index, entry in enumerate(denoisers[1:MAX_CACHED_TRANSFORMERS], start=2):
        broadcast_baseline = _merge_cache_fields(method, entry["defaults"], overrides)
        extra = _cache_overrides_from_widgets(values, allowed, broadcast_baseline, index=index)
        if extra:
            keyed[entry["transformer"]] = extra
    if keyed:
        overrides[PER_TRANSFORMER_CACHE_KEY] = keyed
    if not overrides:
        return None
    return json.dumps(overrides, sort_keys=True)


def _parse_cache_config_value(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("cache_config must be a JSON object.")
        return parsed
    raise ValueError("cache_config must be a JSON object.")


def _cache_config_to_widgets(value: Any) -> dict[str, Any]:
    parsed = _parse_cache_config_value(value)
    widgets: dict[str, Any] = {}
    for field_name in cache_config_widget_names():
        if field_name not in parsed:
            widgets[field_name] = cache_config_widget_spec(field_name)[1]["default"]
            continue
        raw = parsed[field_name]
        if field_name in ("enable_taylorseer", "enable_separate_cfg", "enable_encoder_calibrator"):
            widgets[field_name] = _optional_bool_json_to_widget(raw)
        elif field_name == "scm_policy":
            widgets[field_name] = _scm_policy_json_to_widget(raw)
        elif field_name == "residual_diff_threshold":
            widgets[field_name] = float(raw)
        elif field_name in (
            "Fn_compute_blocks",
            "Bn_compute_blocks",
            "max_warmup_steps",
            "max_cached_steps",
        ):
            widgets[field_name] = int(raw)
        else:
            widgets[field_name] = raw
    return widgets


@lru_cache(maxsize=1)
def _runner_actions() -> dict[str, Any]:
    return {action.dest: action for action in _runner_cli_actions()}


@lru_cache(maxsize=1)
def attention_backend_choices() -> list[str]:
    """Live backend list; 'auto' means None, i.e. the best backend xFuser finds."""
    try:
        from xfuser.core.distributed.attention_backend import AttentionBackendType
    except Exception as exc:
        raise _xfuser_unavailable("the attention backends", exc) from exc
    return ["auto"] + [backend.name.lower() for backend in AttentionBackendType]


@lru_cache(maxsize=1)
def cache_method_choices() -> list[str]:
    """Live cache methods from the runner's argparse choices; 'none' disables caching."""
    action = _runner_actions().get("cache_method")
    if action is None or not action.choices:
        raise RuntimeError(
            "The xdit CLI no longer offers cache_method choices; the step cache widgets "
            "would silently do nothing."
        )
    return ["none", *action.choices]


@lru_cache(maxsize=1)
def loader_config_dests() -> tuple[str, ...]:
    """Every runner arg the Model node exposes, derived from the xdit CLI itself."""
    excluded = _LOADER_EXCLUDED_DESTS | deprecated_runner_dests()
    derived = sorted(dest for dest in runner_cli_dests() if dest not in excluded)
    if not derived:
        raise RuntimeError(
            "xDiT exposed no runner arguments to build Model widgets from. "
            "Check that xfuser is importable in ComfyUI's Python environment."
        )
    return (*_COMPOUND_LOADER_WIDGETS, *derived)


def loader_widget_name(dest: str) -> str:
    return dest


def runtime_dest_from_widget(widget_name: str) -> str:
    return widget_name


@lru_cache(maxsize=1)
def loader_config_widget_names() -> tuple[str, ...]:
    return (
        tuple(loader_config_dests())
        + cache_config_widget_names()
        + extra_denoiser_cache_widget_names()
    )


# What to do with a setting, which the CLI help does not say: it documents the argument
# for someone who already knows the runner. Written to answer "should I touch this, and
# which way?" and kept next to the help text rather than replacing it. The natural home
# for these is xDiT itself, at which point this table shrinks to whatever is still local.
_WIDGET_ADVICE = {
    "ulysses_degree": (
        "The first knob to reach for on multiple GPUs: set it to your GPU count. Splits "
        "attention heads across GPUs, so it needs the model's head count to divide by it."
    ),
    "ring_degree": (
        "Splits the sequence instead of the heads. Combine with ulysses_degree when your "
        "GPU count does not divide the head count on its own; ulysses alone is faster."
    ),
    "fully_shard_degree": (
        "Raise it to save VRAM: the weights are split across this many GPUs instead of "
        "copied to each. Costs a gather per block, so use the smallest value that fits."
    ),
    "tensor_parallel_degree": (
        "Also saves VRAM by splitting weights, but communicates inside every layer. Try "
        "fully_shard_degree first."
    ),
    "pipefusion_parallel_degree": (
        "Pipeline stages across GPUs. Rarely the best use of a GPU on one machine — "
        "prefer ulysses_degree."
    ),
    "data_parallel_degree": (
        "Creates one model replica per data-parallel rank. This improves throughput but "
        "uses more VRAM."
    ),
    "use_cfg_parallel": (
        "Runs guided and unguided passes on separate GPUs. Requires an even GPU count."
    ),
    "cpu_offload_mode": (
        "Moves model components to system RAM when idle. Reduces VRAM use but slows inference."
    ),
    "gemm_precision": (
        "Lower precision reduces memory use and may improve speed, with a possible quality cost."
    ),
    "attention_backend": (
        "auto selects a supported backend. Choose another backend for testing or compatibility."
    ),
    "enable_tiling": ("Decodes in tiles to reduce peak VAE memory use."),
    "enable_slicing": ("Decodes batch items individually to reduce peak VAE memory use."),
    "use_parallel_vae": ("Distributes VAE decoding across the selected GPUs."),
    "use_hybrid_attn_schedule": (
        "Uses high-precision attention at the start and end of denoising."
    ),
    "use_hybrid_gemm_schedule": (
        "Uses FP8 at the start and end of denoising and MXFP4 between them."
    ),
    "residual_diff_threshold": (
        "Minimum residual difference required to recompute a step. Higher values reuse more steps."
    ),
    "max_warmup_steps": ("Number of initial steps to run before enabling the cache."),
    "max_cached_steps": "Maximum consecutive reused steps. -1 applies no limit.",
}


def _widget_advice(dest: str) -> str:
    # The second denoiser's widgets are the same settings for transformer_2.
    return _WIDGET_ADVICE.get(dest) or _WIDGET_ADVICE.get(
        dest[3:] if dest.startswith("t2_") else "", ""
    )


def _cli_tooltip(dest: str, fallback: str = "") -> str:
    help_text = runner_cli_help().get(dest) or fallback
    advice = _widget_advice(dest)
    if not advice:
        return help_text
    if not help_text:
        return advice
    return f"{help_text} — {advice}"


def _is_bool_action(action: Any) -> bool:
    cls = type(action).__name__
    return cls in {"_StoreTrueAction", "_StoreFalseAction", "BooleanOptionalAction"}


def _widget_default(dest: str, action: Any) -> Any:
    if dest in _AUTO_BACKEND_DESTS:
        return "auto"
    if dest == "cache_method":
        default = action.default
        return "none" if default in (None, "") else str(default)
    if _is_bool_action(action):
        return bool(action.default)
    if action.type is int:
        if action.default is None:
            return 0
        return int(action.default)
    if action.type is float:
        if action.default is None:
            return 0.0
        return float(action.default)
    if action.default is None:
        return ""
    return action.default


def loader_widget_spec(dest: str) -> tuple[str, dict[str, Any]]:
    optional = loader_optional_input_types()
    if dest in optional:
        return optional[dest]
    if dest == "cpu_offload_mode":
        return (
            ["none", "model", "sequential", "group"],
            {
                "default": "none",
                "tooltip": _cli_tooltip(dest, "CPU offload strategy for model components."),
            },
        )
    if dest == "gemm_precision":
        return (
            ["native", "fp8", "fp4", "int8"],
            {
                "default": "native",
                "tooltip": _cli_tooltip(dest, "Quantized GEMM precision used by xDiT."),
            },
        )
    if dest in cache_config_widget_names():
        return cache_config_widget_spec(dest)
    extra = denoiser_cache_widget_field(dest)
    if extra is not None:
        return cache_config_widget_spec(extra[1])
    action = _runner_actions().get(dest)
    if action is None:
        # A text box here would accept input the runner then ignores.
        raise KeyError(f"The xdit CLI has no argument {dest!r} to build a widget from.")

    tooltip = _DISTILLED_WEIGHT_TOOLTIPS.get(dest) or _cli_tooltip(dest)
    default = _widget_default(dest, action)

    if dest in _AUTO_BACKEND_DESTS:
        return (
            attention_backend_choices(),
            {"default": default, "tooltip": tooltip},
        )
    if dest == "cache_method":
        return (
            cache_method_choices(),
            {"default": default, "tooltip": tooltip},
        )
    if action.nargs in ("+", "*"):
        return (
            "STRING",
            {
                "default": "" if action.default is None else " ".join(map(str, action.default)),
                "tooltip": tooltip,
            },
        )
    if _is_bool_action(action):
        return ("BOOLEAN", {"default": default, "tooltip": tooltip})
    if action.type is int:
        spec: dict[str, Any] = {"default": default, "tooltip": tooltip, "max": _INT_WIDGET_MAX}
        if dest.endswith("_degree"):
            spec["min"] = 1
        elif action.default is None or default == 0:
            spec["min"] = 0
        else:
            spec["min"] = 1
        return ("INT", spec)
    if action.type is float:
        return ("FLOAT", {"default": default, **_FLOAT_WIDGET_RANGE, "tooltip": tooltip})
    if action.choices:
        return (list(action.choices), {"default": default, "tooltip": tooltip})
    return (
        "STRING",
        {
            "default": "" if default is None else str(default),
            "tooltip": tooltip,
            "multiline": dest.endswith("_schedule"),
        },
    )


def expand_widget_name(group: dict[str, Any]) -> str:
    return str(group["label"])


def loader_expand_widget_names() -> tuple[str, ...]:
    return tuple(expand_widget_name(group) for group in _LOADER_WIDGET_GROUPS)


def strip_undeclared_loader_keys(inputs: dict[str, Any]) -> None:
    """Drop keys a saved graph may carry that the loader does not declare as inputs."""
    inputs.pop("gpu_count", None)


def loader_widget_render_order() -> tuple[str, ...]:
    """Declared groups first, then any widget the runner grew since the groups were written."""
    known = set(loader_config_widget_names()) | set(LOADER_OPTIONAL_WIDGETS)
    grouped = [
        name for group in _LOADER_WIDGET_GROUPS for name in group["widgets"] if name in known
    ]
    return (*grouped, *sorted(known - set(grouped)))


def loader_config_input_types() -> dict[str, tuple[Any, dict[str, Any]]]:
    optional_hf = set(LOADER_OPTIONAL_WIDGETS)
    return {
        name: loader_widget_spec(name)
        for name in loader_widget_render_order()
        if name not in optional_hf
    }


def default_loader_widget_values() -> dict[str, Any]:
    """Every loader widget's default, from the same list that declares the widgets.

    A widget missing here reaches the prompt as null, which ComfyUI rejects before the
    node ever runs.
    """
    return {name: loader_widget_spec(name)[1]["default"] for name in loader_config_widget_names()}


def loader_optional_widget_defaults() -> dict[str, Any]:
    return {
        "residency": RESIDENCY_KEEP_GPU,
        "hf_cache_mode": "auto",
        "hf_cache_dir": "huggingface",
    }


def widget_value_to_runtime(dest: str, value: Any) -> Any:
    if dest in ("cpu_offload_mode", "gemm_precision"):
        return str(value or ("none" if dest == "cpu_offload_mode" else "native"))
    action = _runner_actions().get(dest)
    if dest in _AUTO_BACKEND_DESTS:
        return None if value in (None, "", "auto") else str(value)
    if dest == "cache_method":
        return None if value in (None, "", "none") else str(value)
    if action is not None and action.nargs in ("+", "*"):
        if value in (None, ""):
            return None
        values = value if isinstance(value, (list, tuple)) else str(value).replace(",", " ").split()
        converter = action.type or str
        return [converter(item) for item in values]
    if action is not None and _is_bool_action(action):
        return bool(value)
    if action is not None and action.type is int:
        if action.default is None and value in (None, "", 0):
            return None
        return int(value)
    if action is not None and action.type is float:
        if action.default is None and value in (None, "", 0, 0.0):
            return None
        return float(value)
    return _normalize_empty(value)


def runtime_value_to_widget(dest: str, value: Any) -> Any:
    if dest in _AUTO_BACKEND_DESTS:
        return "auto" if value in (None, "") else str(value)
    if dest == "cache_method":
        return "none" if value in (None, "") else str(value)
    action = _runner_actions().get(dest)
    if action is not None and action.nargs in ("+", "*"):
        if value in (None, ""):
            return ""
        values = value if isinstance(value, (list, tuple)) else [value]
        return " ".join(map(str, values))
    if isinstance(value, bool):
        return value
    if value is None:
        action = _runner_actions().get(dest)
        if action is not None and action.type is int and action.default is None:
            return 0
        if action is not None and action.type is float and action.default is None:
            return 0.0
        return ""
    return value


def _apply_compound_widgets(runtime: dict[str, Any], values: dict[str, Any]) -> None:
    offload = str(values.get("cpu_offload_mode", "none") or "none")
    runtime["enable_model_cpu_offload"] = offload == "model"
    runtime["enable_sequential_cpu_offload"] = offload == "sequential"
    runtime["enable_group_cpu_offload"] = offload == "group"

    gemm = str(values.get("gemm_precision", "native") or "native")
    runtime["use_fp8_gemms"] = gemm == "fp8"
    runtime["use_fp4_gemms"] = gemm == "fp4"
    runtime["use_int8_gemms"] = gemm == "int8"


def runtime_from_loader_widgets(
    values: dict[str, Any],
    registry_model: str | None = None,
) -> dict[str, Any]:
    runtime: dict[str, Any] = {}
    for dest in loader_config_dests():
        if dest not in values:
            continue
        if dest in ("cpu_offload_mode", "gemm_precision"):
            continue
        runtime[dest] = widget_value_to_runtime(dest, values[dest])
    _apply_compound_widgets(runtime, values)
    cache_method = runtime.get("cache_method")
    if cache_method and str(cache_method).lower() not in ("", "none"):
        cache_config = _cache_config_from_widgets(values, registry_model=registry_model)
        if cache_config is not None:
            runtime["cache_config"] = cache_config
    return runtime


def runtime_to_loader_widgets(runtime: dict[str, Any]) -> dict[str, Any]:
    widgets: dict[str, Any] = {}
    defaults = default_loader_widget_values()
    for dest in loader_config_dests():
        if dest in ("cpu_offload_mode", "gemm_precision"):
            continue
        if dest in runtime:
            widgets[dest] = runtime_value_to_widget(dest, runtime[dest])
        else:
            widgets.setdefault(dest, defaults.get(dest))
    widgets["cpu_offload_mode"] = (
        "group"
        if runtime.get("enable_group_cpu_offload")
        else (
            "sequential"
            if runtime.get("enable_sequential_cpu_offload")
            else "model" if runtime.get("enable_model_cpu_offload") else "none"
        )
    )
    widgets["gemm_precision"] = (
        "fp4"
        if runtime.get("use_fp4_gemms")
        else (
            "int8"
            if runtime.get("use_int8_gemms")
            else "fp8" if runtime.get("use_fp8_gemms") else "native"
        )
    )
    widgets["use_torch_compile"] = bool(runtime.get("use_torch_compile", False))
    if "cache_config" in runtime:
        widgets.update(_cache_config_to_widgets(runtime["cache_config"]))
    else:
        for name in cache_config_widget_names():
            widgets.setdefault(name, cache_config_widget_spec(name)[1]["default"])
    return widgets


def _apply_model_input_defaults(widgets: dict[str, Any], registry_model: str) -> None:
    """Show the model's own value for loader widgets xFuser would otherwise default.

    The hybrid high-precision step counts are runner args *and* per-model input
    defaults, so leaving the widget at 0 would pass 0 and silently override the
    model's own schedule.
    """
    from .model_info import model_generation_defaults

    model_defaults = model_generation_defaults(registry_model)
    for dest in set(loader_config_dests()) & set(model_defaults):
        if not widgets.get(dest):
            widgets[dest] = model_defaults[dest]


def loader_display_widgets(runtime: dict[str, Any], registry_model: str) -> dict[str, Any]:
    """Widget values for the Loader UI, including model cache presets when not explicitly overridden."""
    widgets = runtime_to_loader_widgets(runtime)
    widgets = sanitize_loader_cache_widgets(registry_model, widgets)
    _apply_model_input_defaults(widgets, registry_model)
    if not model_supports_step_cache(registry_model):
        return widgets

    cache_method = runtime.get("cache_method")
    if not cache_method or str(cache_method).lower() in ("", "none"):
        return widgets

    explicit = (
        _parse_cache_config_value(runtime["cache_config"]) if runtime.get("cache_config") else None
    )
    merged = resolve_effective_cache_config(registry_model, str(cache_method), explicit)
    if not merged:
        return widgets
    widgets.update(_cache_config_to_widgets(json.dumps(merged, sort_keys=True)))
    return widgets


def preset_args_to_runtime(args: dict[str, Any]) -> dict[str, Any]:
    runtime: dict[str, Any] = {}
    compound_runner_dests = {
        "enable_model_cpu_offload",
        "enable_sequential_cpu_offload",
        "enable_group_cpu_offload",
        "use_fp8_gemms",
        "use_fp4_gemms",
        "use_int8_gemms",
    }
    known_dests = (set(loader_config_dests()) - {"cpu_offload_mode", "gemm_precision"}) | {
        "cache_config",
        "use_torch_compile",
        *compound_runner_dests,
        *SAMPLE_VAE_DESTS,
    }
    cli_dests = runner_cli_dests()

    for key, value in args.items():
        if key == "no_reshard_after_forward":
            runtime["reshard_after_forward"] = not bool(value)
            continue
        dest = runtime_dest_from_widget(key) if key.endswith("_json") else key
        if dest not in cli_dests:
            continue
        if dest in known_dests:
            runtime[dest] = value
        elif dest in _GENERATION_DESTS:
            continue

    for degree_key in (
        "ulysses_degree",
        "ring_degree",
        "pipefusion_parallel_degree",
        "tensor_parallel_degree",
        "data_parallel_degree",
        "fully_shard_degree",
    ):
        runtime.setdefault(degree_key, 1)

    if "reshard_after_forward" not in runtime:
        runtime["reshard_after_forward"] = True
    if "attention_backend" not in runtime:
        runtime["attention_backend"] = None
    if "cache_method" not in runtime:
        runtime["cache_method"] = None
    return runtime


def preset_to_image_input_preset(preset_args: dict[str, Any]) -> dict[str, Any]:
    raw = preset_args.get("input_images")
    if raw is None:
        return {"paths": [], "resize_input_images": False, "required": False}
    if isinstance(raw, str):
        paths = [raw] if raw.strip() else []
    elif isinstance(raw, (list, tuple)):
        paths = [str(p) for p in raw if str(p).strip()]
    else:
        paths = []
    return {
        # Preserve immutable remote references in the graph. Preset execution resolves
        # them through the bounded atomic cache only when the image is actually needed.
        "paths": paths,
        "resize_input_images": bool(preset_args.get("resize_input_images", False)),
        "required": bool(paths),
    }


def preset_to_generation_widgets(preset_args: dict[str, Any]) -> dict[str, Any]:
    widgets: dict[str, Any] = {}
    for key, value in preset_args.items():
        if key not in _GENERATION_DESTS:
            continue
        if key in ("output_directory", "batch_size", "dataset_path", "num_iterations"):
            continue
        if key == "input_images":
            continue
        if value is None:
            continue
        widgets[key] = value
    return widgets
