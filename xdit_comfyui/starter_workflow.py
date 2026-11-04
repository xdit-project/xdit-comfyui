"""Build ComfyUI workflow JSON for the xDiT benchmark starter template."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from .identity import (
    MODEL_NODE_ID,
    NODE_OUTPUTS,
    PRESET_NODE_ID,
    SAMPLE_NODE_ID,
    SAVE_IMAGE_NODE_ID,
    SAVE_VIDEO_NODE_ID,
)
from .presets import PRESET_NONE, format_gpu_detection_summary
from .runner_contract import default_loader_widget_values, loader_pinned_input_types
from .runtime_config import (
    _generation_input_types,
    _preset_picker_input_types,
    _runtime_loader_input_types,
    _runtime_loader_model_choices,
    any_typ,
)
from .save_outputs import (
    IMAGE_FILENAME_PREFIX,
    VIDEO_FILENAME_PREFIX,
    save_image_input_types,
    save_video_input_types,
)

# Must match NAMED_VALUES_KEY in web/xdit_widget_serialization.js.
NAMED_WIDGET_VALUES_KEY = "xdit_widget_values"

_COMFY_INPUT_SLOTS: dict[str, dict[str, int]] = {}


_XDIT_INPUT_TYPES = {
    PRESET_NODE_ID: _preset_picker_input_types,
    MODEL_NODE_ID: _runtime_loader_input_types,
    SAMPLE_NODE_ID: _generation_input_types,
    SAVE_IMAGE_NODE_ID: save_image_input_types,
    SAVE_VIDEO_NODE_ID: save_video_input_types,
}

_SOCKET_INPUT_TYPES = frozenset(
    {
        "*",
        "IMAGE",
        "VIDEO",
        "MASK",
        "LATENT",
        "CONDITIONING",
        "MODEL",
        "CLIP",
        "VAE",
        "CONTROL_NET",
    }
)


def _input_slot(class_type: str, input_name: str) -> int:
    known = _COMFY_INPUT_SLOTS.get(class_type, {}).get(input_name)
    if known is not None:
        return known
    inputs = _XDIT_INPUT_TYPES[class_type]()
    slot = 0
    for section in ("required", "optional"):
        for name in inputs.get(section, {}):
            if name == input_name:
                return slot
            slot += 1
    raise KeyError(f"{class_type} has no input {input_name!r}")


def _output_slot(class_type: str, output_name: str) -> int:
    for index, (name, _type_name) in enumerate(NODE_OUTPUTS[class_type]):
        if name == output_name:
            return index
    raise KeyError(f"{class_type} has no output {output_name!r}")


def _spec_type_name(spec: tuple[Any, dict[str, Any]]) -> str:
    type_spec = spec[0]
    if type_spec is any_typ:
        return "*"
    if isinstance(type_spec, list):
        return "COMBO"
    if isinstance(type_spec, str):
        return type_spec
    return "*"


def _spec_default(spec: tuple[Any, dict[str, Any]]) -> Any:
    meta = spec[1] if len(spec) > 1 else {}
    default = meta.get("default")
    type_spec = spec[0]
    if isinstance(type_spec, list):
        if default in type_spec:
            return default
        return type_spec[0] if type_spec else ""
    if default is not None:
        return default
    if type_spec == "STRING":
        return ""
    if type_spec == "BOOLEAN":
        return False
    if type_spec in ("INT", "FLOAT"):
        return 0
    return None


def _is_widget_input(spec: tuple[Any, dict[str, Any]]) -> bool:
    type_spec = spec[0]
    if type_spec is any_typ:
        return False
    if isinstance(type_spec, str):
        return type_spec not in _SOCKET_INPUT_TYPES
    if isinstance(type_spec, list):
        return True
    return type_spec is not any_typ


def _build_workflow_node(
    node_id: int,
    class_type: str,
    pos: list[float],
    size: list[float],
    order: int,
    *,
    linked: dict[str, int] | None = None,
    widget_overrides: dict[str, Any] | None = None,
    output_links: dict[str, list[int] | None] | None = None,
    properties: dict[str, Any] | None = None,
    named_widgets: bool = True,
) -> dict[str, Any]:
    linked = linked or {}
    widget_overrides = widget_overrides or {}
    output_links = output_links or {}

    inputs_json: list[dict[str, Any]] = []
    widget_values: dict[str, Any] = {}
    inputs_spec = _XDIT_INPUT_TYPES[class_type]()

    for section in ("required", "optional"):
        for name, spec in inputs_spec.get(section, {}).items():
            entry: dict[str, Any] = {
                "name": name,
                "type": _spec_type_name(spec),
            }
            if name in linked:
                entry["link"] = linked[name]
                inputs_json.append(entry)
                continue
            if _is_widget_input(spec):
                entry["widget"] = {"name": name}
                entry["link"] = None
                widget_values[name] = widget_overrides.get(name, _spec_default(spec))
                meta = spec[1] if len(spec) > 1 else {}
                if name == "seed" and meta.get("control_after_generate"):
                    widget_values["control_after_generate"] = widget_overrides.get(
                        "control_after_generate", "randomize"
                    )
            else:
                entry["link"] = None
                if section == "optional":
                    entry["shape"] = 7
            inputs_json.append(entry)

    outputs_json: list[dict[str, Any]] = []
    for name, type_name in NODE_OUTPUTS[class_type]:
        if type_name is any_typ:
            type_name = "*"
        links = output_links.get(name)
        outputs_json.append(
            {
                "name": name,
                "type": type_name,
                "links": links if links else None,
            }
        )

    node = {
        "id": node_id,
        "type": class_type,
        "pos": pos,
        "size": size,
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": inputs_json,
        "outputs": outputs_json,
        "properties": {"Node name for S&R": class_type, **(properties or {})},
        "widgets_values": [] if named_widgets else list(widget_values.values()),
    }
    if named_widgets:
        # Model, Preset, and Sample reorder and group their widgets, so values are
        # keyed by name (see web/xdit_widget_serialization.js).
        node[NAMED_WIDGET_VALUES_KEY] = widget_values
    return node


def _workflow_link(
    link_id: int,
    origin_id: int,
    origin_type: str,
    origin_name: str,
    target_id: int,
    target_type: str,
    target_name: str,
    *,
    type_name: str = "*",
) -> list[Any]:
    return [
        link_id,
        origin_id,
        _output_slot(origin_type, origin_name),
        target_id,
        _input_slot(target_type, target_name),
        type_name,
    ]


def _loader_widget_defaults(
    *,
    preset_name: str = PRESET_NONE,
    gpu_tag: str = "gfx1201",
) -> dict[str, Any]:
    from .presets import build_preset_spec
    from .runtime_config import _model_preset_base

    model_choices = _runtime_loader_model_choices()
    values = {
        name: spec.get("default")
        for name, (_typ, spec) in loader_pinned_input_types(
            model_choices,
            "black-forest-labs/FLUX.1-dev",
        ).items()
    }
    values.update(
        {
            "hf_cache_mode": "auto",
            "hf_cache_dir": "huggingface",
            **default_loader_widget_values(),
        }
    )
    spec = build_preset_spec(
        preset_name,
        gpu_tag,
        registry_choices=_runtime_loader_model_choices(),
    )
    for name, value in _model_preset_base(spec).items():
        if name in values:
            values[name] = value
    return values


_REVISION_EXCLUDE_KEYS = frozenset({"id", "revision"})


def starter_template_revision(workflow: dict[str, Any]) -> str:
    """Stable content hash for template change detection (excludes volatile id/revision)."""
    payload = {key: value for key, value in workflow.items() if key not in _REVISION_EXCLUDE_KEYS}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def apply_starter_template_revision(workflow: dict[str, Any]) -> dict[str, Any]:
    workflow["revision"] = starter_template_revision(workflow)
    return workflow


def _sample_starter_widget_overrides(
    *,
    preset_name: str = PRESET_NONE,
    gpu_tag: str = "gfx1201",
) -> dict[str, Any]:
    from .presets import build_preset_spec

    spec = build_preset_spec(
        preset_name,
        gpu_tag,
        registry_choices=_runtime_loader_model_choices(),
    )
    overrides: dict[str, Any] = {"control_after_generate": "randomize"}
    # The graph is built for a named preset, so it carries that preset's generation
    # values the way picking the preset in the UI does — a video preset that fell back
    # to the node defaults would render a single frame.
    defaults = spec.get("generation_defaults") or {}
    overrides.update({key: value for key, value in defaults.items() if value is not None})
    return overrides


def _sample_widget_values(**kwargs: Any) -> dict[str, Any]:
    """Sample widget values for the starter graph, taken from the node definition."""
    overrides = _sample_starter_widget_overrides(**kwargs)
    values: dict[str, Any] = {}
    for name, spec in _generation_input_types().get("required", {}).items():
        if not _is_widget_input(spec):
            continue
        values[name] = overrides.get(name, _spec_default(spec))
    return values


def _starter_graph(
    *,
    preset_name: str = PRESET_NONE,
    gpu_tag: str = "gfx1201",
) -> tuple[list[dict[str, Any]], list[tuple[int, str, int, str, str]]]:
    from .presets import preset_by_name

    selected_preset = preset_by_name(preset_name)
    preset_gpu_count = selected_preset.gpu_count if selected_preset is not None else 1
    nodes = [
        {
            "id": 1,
            "class_type": PRESET_NODE_ID,
            "pos": [40, 40],
            "size": [420, 180],
            "widgets": {
                "gpu_tag": gpu_tag,
                "gpu_count": str(preset_gpu_count),
                "preset": preset_name,
                "gpu_detection_info": "",
            },
        },
        {
            "id": 2,
            "class_type": MODEL_NODE_ID,
            "pos": [520, 40],
            "size": [420, 420],
            "widgets": _loader_widget_defaults(preset_name=preset_name, gpu_tag=gpu_tag),
            "properties": {
                "_xdit_model_preset_trigger": (f"{gpu_tag}:{preset_gpu_count}:{preset_name}"),
            },
        },
        {
            "id": 3,
            "class_type": SAMPLE_NODE_ID,
            "pos": [1000, 40],
            "size": [420, 520],
            "widgets": _sample_widget_values(preset_name=preset_name, gpu_tag=gpu_tag),
        },
        {
            "id": 4,
            "class_type": SAVE_IMAGE_NODE_ID,
            "pos": [1480, 40],
            "size": [320, 220],
            "widgets": {"filename_prefix": IMAGE_FILENAME_PREFIX},
            "named_widgets": False,
        },
        {
            "id": 5,
            "class_type": SAVE_VIDEO_NODE_ID,
            "pos": [1480, 420],
            "size": [320, 220],
            "widgets": {
                "filename_prefix": VIDEO_FILENAME_PREFIX,
                "format": "auto",
                "codec": "auto",
            },
            "named_widgets": False,
        },
    ]
    edges = [
        (1, "model", 2, "preset", "*"),
        (1, "sample", 3, "preset", "*"),
        (2, "model", 3, "model", "*"),
        (3, "images", 4, "images", "IMAGE"),
        (1, "images", 3, "images", "IMAGE"),
        (3, "video", 5, "video", "VIDEO"),
    ]
    return nodes, edges


def build_starter_workflow_dict(
    *,
    preset_name: str = PRESET_NONE,
    gpu_tag: str = "gfx1201",
) -> dict[str, Any]:
    graph_nodes, graph_edges = _starter_graph(preset_name=preset_name, gpu_tag=gpu_tag)
    by_id = {node["id"]: node for node in graph_nodes}
    linked: dict[int, dict[str, int]] = {node["id"]: {} for node in graph_nodes}
    output_links: dict[int, dict[str, list[int]]] = {node["id"]: {} for node in graph_nodes}
    links = []
    for link_id, (origin_id, origin_name, target_id, target_name, type_name) in enumerate(
        graph_edges, 1
    ):
        origin = by_id[origin_id]
        target = by_id[target_id]
        links.append(
            _workflow_link(
                link_id,
                origin_id,
                origin["class_type"],
                origin_name,
                target_id,
                target["class_type"],
                target_name,
                type_name=type_name,
            )
        )
        linked[target_id][target_name] = link_id
        output_links[origin_id].setdefault(origin_name, []).append(link_id)

    nodes = [
        _build_workflow_node(
            node["id"],
            node["class_type"],
            node["pos"],
            node["size"],
            order,
            linked=linked[node["id"]],
            widget_overrides=node["widgets"],
            output_links=output_links[node["id"]],
            properties=node.get("properties"),
            named_widgets=node.get("named_widgets", True),
        )
        for order, node in enumerate(graph_nodes)
    ]
    last_node_id = max(by_id)

    workflow = {
        "last_node_id": last_node_id,
        "last_link_id": len(links),
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {
            "ds": {
                "scale": 0.8,
                "offset": [120, 120],
            }
        },
        "version": 0.4,
    }
    workflow["revision"] = starter_template_revision(workflow)
    workflow["id"] = str(uuid.uuid4())
    return workflow


def build_starter_api_prompt(
    *,
    preset_name: str = PRESET_NONE,
    gpu_tag: str = "gfx1201",
) -> dict[str, Any]:
    graph_nodes, graph_edges = _starter_graph(preset_name=preset_name, gpu_tag=gpu_tag)
    by_id = {node["id"]: node for node in graph_nodes}
    prompt = {
        str(node["id"]): {
            "class_type": node["class_type"],
            "inputs": dict(node["widgets"]),
        }
        for node in graph_nodes
    }
    prompt["1"]["inputs"]["gpu_detection_info"] = format_gpu_detection_summary()
    for origin_id, origin_name, target_id, target_name, _type_name in graph_edges:
        origin_type = by_id[origin_id]["class_type"]
        prompt[str(target_id)]["inputs"][target_name] = [
            str(origin_id),
            _output_slot(origin_type, origin_name),
        ]
    return prompt


def write_starter_workflow(
    path: Path, *, preset_name: str = PRESET_NONE, gpu_tag: str = "gfx1201"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_starter_workflow_dict(preset_name=preset_name, gpu_tag=gpu_tag), indent=2)
        + "\n",
        encoding="utf-8",
    )
