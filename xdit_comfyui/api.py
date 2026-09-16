import asyncio
import json
import logging
from pathlib import Path

from aiohttp import web

from .benchmark_data import REMOTE_CACHE_DIR, benchmark_image_preview_entries
from .model_info import (
    cache_method_choices_for_model,
    gemm_precision_choices_for_model,
    loader_widget_gates,
    model_capabilities,
    model_generation_defaults,
    model_resolution_grid,
    model_supports_step_cache,
)
from .presets import (
    PRESET_NONE,
    build_preset_spec,
    default_gpu_tag,
    detect_gpu_tags,
    format_gpu_detection_summary,
    is_manual_preset_choice,
    list_applicable_presets,
    list_benchmark_hardware_tags,
    list_gpu_counts_for_gpu_tag,
    list_presets_for_gpu_tag,
    normalize_gpu_tag,
    preset_by_name,
    preset_loader_seed,
)
from .prompt_hooks import apply_preset_prompt_overrides
from .runner_contract import (
    cache_method_choices,
    cache_widget_defaults_for_model,
    default_loader_widget_values,
    effective_cache_config_per_transformer,
    loader_config_widget_names,
    loader_display_widgets,
    loader_schema,
    loader_widget_spec,
)
from .runtime_config import (
    _available_gpu_count,
    _coerce_gpu_count,
    _default_gpu_count,
    _generation_input_types,
    _merge_loader_kwargs,
    _preset_meta_from_spec,
    _resolve_loader_runtime,
    _runtime_loader_gpu_choices,
    _runtime_loader_input_types,
    _runtime_loader_model_choices,
    _runtime_preview_dict,
)
from .runtime_env import describe_hf_cache
from .starter_workflow import apply_starter_template_revision
from .worker import _clear_loader_cache


def _sanitize_loader_preview_body(body):
    body = dict(body or {})
    defaults = default_loader_widget_values()

    for name in loader_config_widget_names():
        if name not in body:
            continue
        typ, spec = loader_widget_spec(name)
        value = body[name]
        fallback = defaults.get(name, spec.get("default"))
        if typ == "INT":
            try:
                parsed = int(value)
                if parsed < spec.get("min", 0):
                    body[name] = fallback
                else:
                    body[name] = parsed
            except (TypeError, ValueError):
                body[name] = fallback
            continue
        if typ == "FLOAT":
            try:
                if isinstance(value, str) and value.strip().lower() in ("none", ""):
                    body[name] = fallback
                else:
                    body[name] = float(value)
            except (TypeError, ValueError):
                body[name] = fallback
            continue
        if isinstance(typ, list):
            if value not in typ:
                body[name] = fallback
            continue
        if typ == "BOOLEAN" and not isinstance(value, bool):
            if str(value).strip().lower() in ("true", "1"):
                body[name] = True
            elif str(value).strip().lower() in ("false", "0"):
                body[name] = False
            else:
                body[name] = fallback
    return body


def _build_preset_from_body(body):
    if isinstance(body.get("preset"), dict):
        return body["preset"]
    gpu_tag = body.get("preset_gpu_tag") or body.get("gpu_tag")
    preset = body.get("preset_choice") or body.get("preset")
    if gpu_tag is None and preset is None:
        return None
    spec = build_preset_spec(
        preset or PRESET_NONE,
        normalize_gpu_tag(gpu_tag or default_gpu_tag()),
        registry_choices=_runtime_loader_model_choices(),
    )
    raw_count = body.get("preset_gpu_count", body.get("gpu_count"))
    if (
        spec.get("matched")
        and raw_count not in (None, "")
        and spec.get("gpu_count") != _coerce_gpu_count(raw_count)
    ):
        return None
    return spec


def _preset_widgets_from_spec(preset):
    if not isinstance(preset, dict) or not preset.get("matched"):
        return {}
    from .runtime_config import _model_preset_base

    return dict(_model_preset_base(preset))


def _loader_preview_display_widgets(runtime, model, preset, body):
    return loader_display_widgets(runtime, model)


def _preview_payload(body):
    body = _sanitize_loader_preview_body(body)
    preset_applied = bool(body.pop("preset_applied", False))
    preset = _build_preset_from_body(body)
    visible = _available_gpu_count()
    suggested = _default_gpu_count()
    gpu_count = _coerce_gpu_count(body.get("gpu_count"), suggested)
    body["gpu_count"] = gpu_count
    gpu_count_suggested = suggested
    if visible and gpu_count > visible:
        gpu_count_suggested = visible
        body["gpu_count"] = visible
        gpu_count = visible
    if preset:
        body = _merge_loader_kwargs(preset, body, preset_wins=preset_applied)
        gpu_count = _coerce_gpu_count(body.get("gpu_count"), suggested)
    runtime, _, preset_meta, _, _ = _resolve_loader_runtime(
        preset=preset,
        _allow_missing_task=True,
        **body,
    )
    preview = _runtime_preview_dict(runtime, preset_meta)
    model = runtime.get("model", "")
    display_widgets = _loader_preview_display_widgets(runtime, model, preset, body)
    from .residency_allocator import residency_choices_for_runtime

    residency_choices, residency_unavailable_reason = residency_choices_for_runtime(runtime)
    preset_key = None
    if isinstance(preset, dict) and preset.get("matched"):
        preset_key = f"{preset.get('selected_gpu_tag')}:{preset.get('selected')}"
    # Static lists (config_widgets, attention backends) belong to /xdit/loader/schema,
    # which the browser fetches once; this runs on every widget change.
    return {
        "runtime": preview,
        "capabilities": model_capabilities(model),
        # Which options this model can actually use, keyed by widget name and by group
        # label: the queue sanitizer resets the rest, so the UI must not offer them.
        "widget_gates": loader_widget_gates(model, runtime.get("cache_method")),
        "cache_method_choices": cache_method_choices_for_model(model, cache_method_choices()),
        "gemm_precision_choices": gemm_precision_choices_for_model(model),
        "residency_choices": residency_choices,
        "residency_unavailable_reason": residency_unavailable_reason,
        "visible_gpus": _available_gpu_count(),
        "gpu_tags": list(detect_gpu_tags()),
        "gpu_count_choices": _runtime_loader_gpu_choices(),
        "gpu_count_suggested": gpu_count_suggested,
        "preset_meta": preset_meta,
        "preset_widgets": _preset_widgets_from_spec(preset),
        "display_widgets": display_widgets,
        "preset_key": preset_key,
        "step_cache_supported": model_supports_step_cache(model),
        # What this model implies for the Sample node, so the browser needs no per-model
        # table of its own.
        "generation": _generation_constraints(model),
        # The model's own cache values, independent of what the widgets currently hold:
        # hydrating from `display_widgets` would only echo the widget values back and the
        # model's numbers would never appear.
        "cache_defaults": cache_widget_defaults_for_model(model, runtime.get("cache_method")),
        "cache_transformers": _cache_transformer_rows(runtime, model),
        "model_cache": describe_hf_cache(
            body.get("hf_cache_mode", "auto"), body.get("hf_cache_dir", "huggingface")
        ),
    }


def _cache_transformer_rows(runtime, model):
    """The cache config each denoiser will run with, one entry per cached transformer."""
    from .runner_contract import _parse_cache_config_value

    explicit = (
        _parse_cache_config_value(runtime["cache_config"]) if runtime.get("cache_config") else None
    )
    return effective_cache_config_per_transformer(model, runtime.get("cache_method"), explicit)


def _generation_constraints(model):
    grid = model_resolution_grid(model)
    defaults = {
        key: value
        for key, value in model_generation_defaults(model).items()
        # An empty negative prompt already means "use the model's", and pasting the
        # model's paragraph into the widget only makes it harder to edit.
        if key != "negative_prompt"
    }
    return {
        "model": model,
        "defaults": defaults,
        "resolution_step": grid["step"],
        "resolution_divisor": grid["divisor"],
        # A still-image model ignores the frame count and the video-only guidance.
        "output_kind": model_capabilities(model).get("model_output_type") or "image",
    }


def _preset_preview_payload(body):
    body = dict(body or {})
    gpu_tag = normalize_gpu_tag(body.get("gpu_tag"))
    tag_choices = list(list_benchmark_hardware_tags())
    gpu_count_choices = list_gpu_counts_for_gpu_tag(gpu_tag) or [1]
    requested_count = _coerce_gpu_count(body.get("gpu_count"), _default_gpu_count())
    gpu_count = requested_count if requested_count in gpu_count_choices else gpu_count_choices[0]
    preset = (body.get("preset") or PRESET_NONE).strip()
    selected_preset = preset_by_name(preset)
    if not is_manual_preset_choice(preset) and (
        selected_preset is None
        or gpu_tag not in selected_preset.tags
        or selected_preset.gpu_count != gpu_count
    ):
        preset = PRESET_NONE
    spec = build_preset_spec(
        preset,
        gpu_tag,
        registry_choices=_runtime_loader_model_choices(),
    )
    hardware_presets = list_presets_for_gpu_tag(gpu_tag, gpu_count)
    model = spec.get("model") or ""
    applicable = (
        list_applicable_presets(model, spec.get("gpu_count") or 1, gpu_tags=(gpu_tag,))
        if model
        else []
    )
    return {
        "preset": spec,
        # The resolved selection: the browser applies these to its combos rather than
        # deriving them from its own copy of the benchmark tables.
        "gpu_tag": gpu_tag,
        "gpu_tag_choices": tag_choices,
        "gpu_tag_suggested": default_gpu_tag(),
        "gpu_count": gpu_count,
        "gpu_count_choices": gpu_count_choices,
        "gpu_detection_summary": format_gpu_detection_summary(),
        "choices": [PRESET_NONE] + hardware_presets,
        "hardware_presets": hardware_presets,
        "applicable_presets": [entry["name"] for entry in applicable],
        "preset_meta": _preset_meta_from_spec(spec),
        "image_previews": benchmark_image_preview_entries(
            (spec.get("image_input_preset") or {}).get("paths") or []
        ),
    }


def _preset_filter_schema():
    tags = list(list_benchmark_hardware_tags())
    counts_by_tag = {tag: list_gpu_counts_for_gpu_tag(tag) for tag in tags}
    return {
        "gpu_counts_by_tag": counts_by_tag,
        "presets_by_tag_and_count": {
            tag: {str(count): list_presets_for_gpu_tag(tag, count) for count in counts_by_tag[tag]}
            for tag in tags
        },
    }


_EXAMPLE_WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / "example_workflows"
from .identity import TEMPLATE_MODULE as _TEMPLATE_MODULE

_TEMPLATE_FILENAMES = frozenset({"xDiT-Starter.json", "xDiT-Starter.jpg"})


def register_routes():
    try:
        from server import PromptServer  # type: ignore[reportMissingImports]
    except Exception:
        return

    routes = PromptServer.instance.routes

    @routes.get(f"/api/workflow_templates/{_TEMPLATE_MODULE}/{{filename}}")
    async def serve_example_workflow(request):
        raw = request.match_info["filename"]
        if "." not in Path(raw).name:
            raw = f"{raw}.json"
        path = _EXAMPLE_WORKFLOWS_DIR / Path(raw).name
        if path.name not in _TEMPLATE_FILENAMES or not path.is_file():
            raise web.HTTPNotFound(text=f"Unknown template: {path.name}")
        if path.suffix.lower() == ".jpg":
            return web.FileResponse(
                path,
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate",
                    "Pragma": "no-cache",
                },
            )
        workflow = json.loads(path.read_text(encoding="utf-8"))
        apply_starter_template_revision(workflow)
        return web.json_response(
            workflow,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
            },
        )

    @routes.get("/xdit/loader/schema")
    async def loader_schema_route(_request):
        return web.json_response(await asyncio.to_thread(loader_schema))

    @routes.get("/xdit/benchmark_cache/{filename}")
    async def benchmark_cache_file(request):
        filename = Path(request.match_info["filename"]).name
        path = REMOTE_CACHE_DIR / filename
        if not path.is_file():
            raise web.HTTPNotFound(text=f"Unknown cached benchmark image: {filename}")
        return web.FileResponse(path)

    @routes.post("/xdit/loader/preview")
    async def loader_preview(request):
        body = await request.json()
        # Off the event loop: preset previews can fetch a remote reference image, and a
        # cache miss would otherwise stall every request and the progress websocket.
        return web.json_response(await asyncio.to_thread(_preview_payload, body))

    @routes.post("/xdit/loader/clear")
    async def loader_clear(request):
        body = await request.json()
        node_id = body.get("node_id") or body.get("unique_id")
        result = await asyncio.to_thread(_clear_loader_cache, node_id)
        return web.json_response(result)

    @routes.post("/xdit/loader/reap")
    async def loader_reap(request):
        """Release workers whose Model node is gone from the graph."""
        from .worker import _reap_loaders_except, _release_all_loaders

        body = await request.json()
        if body.get("all"):
            result = await asyncio.to_thread(_release_all_loaders)
        else:
            result = await asyncio.to_thread(_reap_loaders_except, body.get("live_node_ids") or [])
        return web.json_response(result)

    @routes.get("/xdit/residency")
    async def residency(request):
        from .residency import residency_report, sample_run_memory

        payload = await asyncio.to_thread(residency_report)
        sample_id = request.query.get("sample_node_id")
        if sample_id:
            payload["sample_run"] = sample_run_memory(sample_id)
        return web.json_response(payload)

    @routes.post("/xdit/preset/preview")
    async def preset_preview(request):
        body = await request.json()
        return web.json_response(await asyncio.to_thread(_preset_preview_payload, body))

    @routes.get("/xdit/preset/schema")
    async def preset_schema(_request):
        return web.json_response(await asyncio.to_thread(_preset_filter_schema))

    @routes.get("/xdit/presets/{name}")
    async def preset_detail(request):
        name = request.match_info["name"]
        if is_manual_preset_choice(name):
            return web.json_response(
                {"name": name, "widgets": {}, "generation_defaults": {}, "image_input_preset": {}}
            )
        preset = preset_by_name(name)
        if preset is None:
            raise web.HTTPNotFound(text=f"Unknown preset: {name}")
        model_choices = _runtime_loader_model_choices()
        seed = preset_loader_seed(preset, model_choices)
        spec = await asyncio.to_thread(
            build_preset_spec, name, default_gpu_tag(), registry_choices=model_choices
        )
        return web.json_response(
            {
                "name": preset.name,
                "model": seed["model_choice"],
                "preset_model": preset.model,
                "gpu_count": seed["gpu_count"],
                "widgets": {
                    key: value
                    for key, value in seed.items()
                    if key not in ("model_choice", "gpu_count")
                },
                "generation_defaults": spec.get("generation_defaults") or {},
                "image_input_preset": spec.get("image_input_preset") or {},
            }
        )

    async def warm_schema_cache(_app):
        """Introspect the xdit CLI off the event loop, before the browser asks.

        The introspection is deliberately not done at import time (it cost seconds of
        ComfyUI startup), so warm it here or the first /object_info would pay for it
        while holding the event loop.
        """

        async def _warm():
            try:
                await asyncio.to_thread(_runtime_loader_input_types)
                await asyncio.to_thread(_generation_input_types)
                await asyncio.to_thread(loader_schema)
            except Exception:
                logging.getLogger("xdit").debug("schema warm failed", exc_info=True)

        asyncio.create_task(_warm())

    async def shutdown_workers(_app):
        """A normal ComfyUI shutdown must not leave torchrun holding the GPUs."""
        from .worker import _clear_all_runtime_caches

        await asyncio.to_thread(_clear_all_runtime_caches)

    try:
        PromptServer.instance.app.on_startup.append(warm_schema_cache)
    except Exception:
        pass

    try:
        PromptServer.instance.app.on_cleanup.append(shutdown_workers)
    except Exception:
        pass

    try:
        PromptServer.instance.add_on_prompt_handler(apply_preset_prompt_overrides)
    except Exception:
        pass


register_routes()
