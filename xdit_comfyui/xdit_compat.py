"""Compatibility for symbols that differ between supported xDiT revisions."""

from __future__ import annotations


def attention_backend_type():
    try:
        from xfuser.core.attention.spec import AttentionBackendType
    except ImportError:
        from xfuser.core.distributed.attention_backend import AttentionBackendType
    return AttentionBackendType


def cache_dit_config_helpers():
    """Private cache-dit helpers moved into a package after the Instinct pin."""
    try:
        from xfuser.model_executor.cache.adapters.cache_dit import (
            _build_config,
            _build_scm_mask,
            _import_cache_dit,
        )
    except ImportError:
        from xfuser.model_executor.cache.adapters.cache_dit.config import (
            _build_config,
            _build_scm_mask,
            _import_cache_dit,
        )
    return _build_config, _build_scm_mask, _import_cache_dit


def unwrap_cache_dit_fsdp(transformer):
    try:
        from xfuser.model_executor.cache.adapters.cache_dit import _unwrap_fsdp
    except ImportError:
        from xfuser.model_executor.cache.adapters.cache_dit.apply import _unwrap_fsdp
    return _unwrap_fsdp(transformer)


def runner_supports(dest: str) -> bool:
    from .runner_contract import runner_cli_dests

    return dest in runner_cli_dests()


def gemm_quantization_choices() -> tuple[str, ...]:
    if runner_supports("gemm_quantization"):
        from xfuser.config.gemm import _PURE_FORMATS, _TIERED_FORMATS

        pure = sorted(map(str, set(_PURE_FORMATS) - {"none"}))
        tiered = sorted(f"low={low},high={high}" for low, high in _TIERED_FORMATS)
        return ("native", *pure, *tiered)
    return ("native", "fp8", "fp4", "int8")


def gemm_precision_from_runtime(runtime: dict) -> str:
    quantization = runtime.get("gemm_quantization")
    if quantization not in (None, "", "none", "native"):
        return str(quantization)
    if runtime.get("use_fp4_gemms"):
        return "fp4"
    if runtime.get("use_int8_gemms"):
        return "int8"
    if runtime.get("use_fp8_gemms"):
        return "fp8"
    return "native"


def apply_gemm_precision(runtime: dict, precision: str) -> None:
    value = str(precision or "native")
    if runner_supports("gemm_quantization"):
        runtime["gemm_quantization"] = None if value == "native" else value
        for legacy in ("use_fp8_gemms", "use_fp4_gemms", "use_int8_gemms"):
            runtime.pop(legacy, None)
        return
    runtime["use_fp8_gemms"] = value == "fp8"
    runtime["use_fp4_gemms"] = value == "fp4"
    runtime["use_int8_gemms"] = value == "int8"
