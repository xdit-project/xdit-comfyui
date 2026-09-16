"""Refresh step-cache state between runs without reloading model weights.

xDiT applies step caches once at model init. cache-dit exposes refresh_context() to
rebuild dbcache context when steps or cache_config change. teacache/fbcache thresholds
and step counts live on the patched block module and can be updated in place.
"""

import json
from typing import Any

_APPLIED_CACHE_STATE_ATTR = "_applied_cache_state"


def _applied_cache_state(runner):
    try:
        return object.__getattribute__(runner, _APPLIED_CACHE_STATE_ATTR)
    except AttributeError:
        return None


def _set_applied_cache_state(runner, state):
    object.__setattr__(runner, _APPLIED_CACHE_STATE_ATTR, state)


def _parse_cache_config(run_config):
    if isinstance(run_config, dict):
        raw = run_config.get("cache_config")
    else:
        raw = run_config
    if raw in (None, ""):
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("cache_config must be a JSON object.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("cache_config must be a JSON object.")
    return parsed


def _resolve_transformer(runner, attr):
    pipe = getattr(getattr(runner, "model", None), "pipe", None)
    if pipe is None:
        return None
    transformer = getattr(pipe, attr, None)
    if transformer is None:
        return None
    try:
        from xfuser.model_executor.cache.adapters.cache_dit import _unwrap_fsdp
    except ImportError:
        return transformer
    return _unwrap_fsdp(transformer)


def _tree_cache_blocks(runner):
    """Patched block per transformer; teacache/fbcache wrap each one separately."""
    model = getattr(runner, "model", None)
    names = getattr(getattr(model, "settings", None), "transformer_attr_names", None) or []
    found = []
    for name in names:
        transformer = getattr(getattr(model, "pipe", None), name, None)
        blocks = getattr(transformer, "transformer_blocks", None) if transformer else None
        if blocks is None or len(blocks) != 1:
            continue
        if hasattr(blocks[0], "rel_l1_thresh"):
            found.append(blocks[0])
    return found


def _resolved_cache_method(runner, run_config, init_cache_method=None):
    if init_cache_method:
        return init_cache_method
    config = getattr(runner, "config", None)
    if config is not None:
        method = getattr(config, "cache_method", None)
        if method:
            return method
    return run_config.get("cache_method")


def _seed_applied_state(runner, method):
    state = _applied_cache_state(runner)
    if state is not None:
        return state
    config = getattr(runner, "config", None)
    init_steps = getattr(config, "num_inference_steps", None) if config is not None else None
    init_config = {}
    if config is not None and getattr(config, "cache_config", None):
        init_config = _parse_cache_config(config.cache_config)
    state = {
        "method": method,
        "steps": int(init_steps) if init_steps is not None else None,
        "cache_config": init_config,
    }
    if method == "dbcache" and init_steps is not None:
        try:
            # xFuser broadcasts one cache_config to every transformer at init, so the
            # baseline must ignore per-transformer overrides or the first run would
            # skip the refresh that actually applies them.
            state["refresh_signature"] = _plan_signature(
                _dbcache_refresh_plan(runner, int(init_steps), init_config, per_transformer=False)
            )
        except Exception:
            pass
    if method in ("teacache", "fbcache"):
        threshold = init_config.get("residual_diff_threshold")
        state["threshold"] = float(threshold) if threshold is not None else None
    _set_applied_cache_state(runner, state)
    return state


def _resolve_tree_threshold(run_config, cache_config):
    if "residual_diff_threshold" in cache_config:
        return float(cache_config["residual_diff_threshold"])
    raw = run_config.get("residual_diff_threshold")
    if raw not in (None, ""):
        return float(raw)
    return None


def _dbcache_method_cfg(runner):
    settings = getattr(getattr(runner, "model", None), "settings", None)
    cache_configs = settings.step_cache_config if settings else None
    if not isinstance(cache_configs, dict):
        return None
    return cache_configs.get("dbcache")


PER_TRANSFORMER_KEY = "per_transformer"


def _dbcache_targets(runner):
    """One (name, preset, enable_separate_cfg) per cached transformer.

    Wan 2.2 pairs a high-noise denoiser with a low-noise refiner that wants a shorter
    warmup, so xFuser turns `adapter` and `preset` into aligned lists. Treating that
    list as a single preset raises inside cache-dit's config builder.
    """
    method_cfg = _dbcache_method_cfg(runner)
    adapters = getattr(method_cfg, "adapter", None) if method_cfg else None
    presets = getattr(method_cfg, "preset", None) if method_cfg else None
    if not isinstance(adapters, (list, tuple)):
        adapters = [adapters]
    if not isinstance(presets, (list, tuple)):
        presets = [presets] * len(adapters)

    fallback_names = (
        getattr(
            getattr(getattr(runner, "model", None), "settings", None),
            "transformer_attr_names",
            None,
        )
        or []
    )
    targets = []
    for index, adapter in enumerate(adapters):
        name = getattr(adapter, "transformer_attr", None)
        if not name:
            name = fallback_names[index] if index < len(fallback_names) else "transformer"
        targets.append(
            {
                "name": str(name),
                "preset": presets[index] if index < len(presets) else None,
                "enable_separate_cfg": bool(getattr(adapter, "enable_separate_cfg", False)),
            }
        )
    return targets


def _split_per_transformer_overrides(sparse_overrides: dict[str, Any]):
    """Broadcast overrides plus any keyed per-transformer overrides."""
    broadcast = dict(sparse_overrides or {})
    keyed = broadcast.pop(PER_TRANSFORMER_KEY, None)
    if not isinstance(keyed, dict):
        keyed = {}
    return broadcast, keyed


def _overrides_for_target(name, broadcast, keyed, *, per_transformer: bool):
    merged = dict(broadcast)
    if per_transformer:
        extra = keyed.get(name)
        if isinstance(extra, dict):
            merged.update(extra)
    return merged


def _build_dbcache_refresh_config(runner, steps, target, overrides: dict[str, Any]):
    from xfuser.model_executor.cache.adapters.cache_dit import _build_config, _import_cache_dit

    cache_config_json = json.dumps(overrides, sort_keys=True) if overrides else None
    _, DBCacheConfig, _, _ = _import_cache_dit()
    return _build_config(
        num_steps=steps,
        preset_kwargs=target["preset"],
        cache_config_json=cache_config_json,
        enable_separate_cfg=target["enable_separate_cfg"],
        DBCacheConfig=DBCacheConfig,
    )


def _dbcache_refresh_plan(
    runner, steps: int, sparse_overrides: dict[str, Any], *, per_transformer=True
):
    """Per-transformer configs plus the signature that decides whether to refresh."""
    broadcast, keyed = _split_per_transformer_overrides(sparse_overrides)
    plan = []
    for target in _dbcache_targets(runner):
        overrides = _overrides_for_target(
            target["name"], broadcast, keyed, per_transformer=per_transformer
        )
        db_config, calibrator_config = _build_dbcache_refresh_config(
            runner, steps, target, overrides
        )
        plan.append(
            {
                "name": target["name"],
                "cache_config": db_config,
                "calibrator_config": calibrator_config,
                "signature": _dbcache_refresh_signature(steps, db_config, calibrator_config),
            }
        )
    return plan


def _plan_signature(plan):
    return [{"name": entry["name"], **entry["signature"]} for entry in plan]


def _dbcache_refresh_signature(steps: int, db_config, calibrator_config) -> dict[str, Any]:
    signature: dict[str, Any] = {
        "steps": steps,
        "Fn_compute_blocks": db_config.Fn_compute_blocks,
        "Bn_compute_blocks": db_config.Bn_compute_blocks,
        "residual_diff_threshold": db_config.residual_diff_threshold,
        "max_warmup_steps": db_config.max_warmup_steps,
        "max_cached_steps": db_config.max_cached_steps,
    }
    if getattr(db_config, "steps_computation_mask", None) is not None:
        signature["steps_computation_mask"] = db_config.steps_computation_mask
    if calibrator_config is not None:
        signature["enable_encoder_calibrator"] = getattr(
            calibrator_config,
            "enable_encoder_calibrator",
            None,
        )
    return signature


def _refresh_dbcache(runner, run_config):
    raw_steps = run_config.get("num_inference_steps")
    if raw_steps is None:
        return
    steps = int(raw_steps)
    if steps < 1:
        return

    sparse_overrides = _parse_cache_config(run_config)
    plan = _dbcache_refresh_plan(runner, steps, sparse_overrides)
    signature = _plan_signature(plan)
    state = _seed_applied_state(runner, "dbcache")
    if state.get("method") == "dbcache" and state.get("refresh_signature") == signature:
        return

    import cache_dit

    refreshed = 0
    for entry in plan:
        transformer = _resolve_transformer(runner, entry["name"])
        if transformer is None:
            continue
        refresh_kwargs: dict[str, Any] = {"cache_config": entry["cache_config"]}
        if entry["calibrator_config"] is not None:
            refresh_kwargs["calibrator_config"] = entry["calibrator_config"]
        cache_dit.refresh_context(transformer, **refresh_kwargs)
        refreshed += 1

    if not refreshed:
        raise RuntimeError(
            "dbcache is enabled but no transformer module is available for step-cache refresh"
        )
    _set_applied_cache_state(
        runner,
        {"method": "dbcache", "steps": steps, "refresh_signature": signature},
    )


def _refresh_tree_cache(runner, run_config, method):
    raw_steps = run_config.get("num_inference_steps")
    steps = int(raw_steps) if raw_steps is not None else None
    cache_config = _parse_cache_config(run_config)
    threshold = _resolve_tree_threshold(run_config, cache_config)
    if threshold is None:
        return

    state = _seed_applied_state(runner, method)
    if (
        state.get("method") == method
        and state.get("steps") == steps
        and state.get("threshold") == threshold
    ):
        return

    cached_blocks = _tree_cache_blocks(runner)
    if not cached_blocks:
        return

    for cached in cached_blocks:
        cached.rel_l1_thresh.fill_(threshold)
        if steps is not None and steps >= 1:
            cached.num_steps = steps
    _set_applied_cache_state(
        runner,
        {"method": method, "steps": steps, "threshold": threshold},
    )


def maybe_refresh_step_cache(runner, run_config, *, init_cache_method=None):
    method = _resolved_cache_method(runner, run_config, init_cache_method)
    if not method or str(method).lower() in ("", "none"):
        return
    method = str(method).lower()
    if method == "dbcache":
        _refresh_dbcache(runner, run_config)
        return
    if method in ("teacache", "fbcache"):
        _refresh_tree_cache(runner, run_config, method)
