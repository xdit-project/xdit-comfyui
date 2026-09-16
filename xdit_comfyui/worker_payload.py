"""Build xDiT worker payloads without process or socket lifecycle concerns."""

from __future__ import annotations

import json
import tempfile
from copy import deepcopy
from pathlib import Path

from .runtime_config import INTERNAL_CONFIG_PREFIX

_RUNNER_INIT_OUTPUT_DIR = Path(tempfile.gettempdir()) / "xdit_runner_init"

_LOADER_INIT_GENERATION_DEFAULTS = {
    "prompt": "xDiT worker init",
    "negative_prompt": "",
    "height": 64,
    "width": 64,
    "num_frames": 1,
    "max_sequence_length": 256,
    "guidance_scale": 1.0,
    "guidance_scale_2": None,
    "seed": 0,
    "output_type": "pil",
    "input_images": [],
}

LOADER_INIT_REQUIRED_KEYS = frozenset(
    {"model", "num_inference_steps", *_LOADER_INIT_GENERATION_DEFAULTS}
)


def worker_config_payload(config):
    payload = deepcopy(config)
    for key in list(payload):
        if key.startswith(INTERNAL_CONFIG_PREFIX):
            payload.pop(key, None)
    return payload


def loader_init_config(runtime):
    config = deepcopy(runtime)
    config.setdefault("output_directory", str(_RUNNER_INIT_OUTPUT_DIR))
    init_images = config.pop("_loader_init_input_images", None)
    model = config.get("model")
    if model:
        from .model_info import model_generation_defaults

        model_defaults = model_generation_defaults(model)
    else:
        model_defaults = {}
    for key, value in _LOADER_INIT_GENERATION_DEFAULTS.items():
        config.setdefault(key, model_defaults.get(key, value))
    if (
        "num_inference_steps" not in config
        and model_defaults.get("num_inference_steps") is not None
    ):
        config["num_inference_steps"] = model_defaults["num_inference_steps"]
    config["num_inference_steps"] = _xdit_valid_loader_init_steps(config)
    if init_images:
        config["input_images"] = init_images
    if model:
        from .model_info import sanitize_runtime_for_model
        from .runner_contract import sanitize_attention_runtime

        sanitize_runtime_for_model(model, config)
        sanitize_attention_runtime(config)
    _drop_per_denoiser_cache_config(config)
    return config


def _xdit_valid_loader_init_steps(config):
    """Smallest step count accepted by xDiT's active schedule builders."""
    steps = max(int(config.get("num_inference_steps") or 1), 1)
    explicit = str(config.get("hybrid_attn_schedule") or "").strip()
    if explicit:
        steps = max(steps, len([part for part in explicit.split(",") if part.strip()]))

    from xfuser.core.distributed.attention_backend import AttentionBackendType
    from xfuser.core.distributed.attention_schedule import (
        create_hybrid_attn_schedule,
        create_hybrid_gemm_schedule,
    )

    while True:
        try:
            if config.get("use_hybrid_attn_schedule") and not explicit:
                create_hybrid_attn_schedule(
                    num_high_precision_steps=int(
                        config.get("num_hybrid_attn_high_precision_steps") or 0
                    ),
                    low_precision_backend=AttentionBackendType.SDPA,
                    high_precision_backend=AttentionBackendType.SDPA,
                    total_steps=steps,
                )
            if config.get("use_hybrid_gemm_schedule"):
                create_hybrid_gemm_schedule(
                    num_high_precision_steps=int(
                        config.get("num_hybrid_gemm_high_precision_steps") or 0
                    ),
                    total_steps=steps,
                )
            return steps
        except ValueError:
            steps += 1


def _drop_per_denoiser_cache_config(config):
    """Remove plugin-only per-denoiser overrides from xDiT's flat init config."""
    from .runner_contract import _parse_cache_config_value, broadcast_cache_config

    if not config.get("cache_config"):
        return
    broadcast = broadcast_cache_config(_parse_cache_config_value(config["cache_config"]))
    if broadcast:
        config["cache_config"] = json.dumps(broadcast, sort_keys=True)
    else:
        config.pop("cache_config", None)


def loader_init_worker_payload(runtime):
    return worker_config_payload(loader_init_config(runtime))
