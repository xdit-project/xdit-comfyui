"""Build ComfyUI workflow JSON for the xDiT benchmark starter template."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from .presets import format_gpu_detection_summary
from .runner_contract import default_loader_widget_values, loader_pinned_input_types
from .runtime_config import (
    _generation_input_types,
    _preset_picker_input_types,
    _runtime_loader_input_types,
    _runtime_loader_model_choices,
    any_typ,
)

# Must match NAMED_VALUES_KEY in web/xdit_widget_serialization.js.
NAMED_WIDGET_VALUES_KEY = "xdit_widget_values"

_COMFY_INPUT_SLOTS = {
    "SaveImage": {"images": 0},
    "SaveVideo": {"video": 0},
}

_XDIT_INPUT_TYPES = {
    "xDiT.Preset": _preset_picker_input_types,
    "xDiT.Model": _runtime_loader_input_types,
    "xDiT.Sample": _generation_input_types,
}

_XDIT_OUTPUT_NAMES = {
    "xDiT.Preset": ("model", "images", "sample"),
    "xDiT.Model": ("model",),
    "xDiT.Sample": ("images", "video"),
}

_XDIT_OUTPUT_TYPES = {
    "xDiT.Preset": ("*", "IMAGE", "*"),
    "xDiT.Model": ("*",),
    "xDiT.Sample": ("IMAGE", "VIDEO"),
}

_SOCKET_INPUT_TYPES = frozenset(
    {
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
    names = _XDIT_OUTPUT_NAMES[class_type]
    for index, name in enumerate(names):
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
    return_names = _XDIT_OUTPUT_NAMES[class_type]
    return_types = _XDIT_OUTPUT_TYPES[class_type]
    for index, name in enumerate(return_names):
        type_name = return_types[index] if index < len(return_types) else "*"
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

    return {
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
        # xDiT nodes reorder and group their widgets, so values are keyed by name
        # (see web/xdit_widget_serialization.js). The positional array is unused.
        "widgets_values": [],
        NAMED_WIDGET_VALUES_KEY: widget_values,
    }


def _save_image_node(
    node_id: int,
    pos: list[float],
    size: list[float],
    order: int,
    *,
    images_link: int,
    filename_prefix: str = "xdit",
) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "SaveImage",
        "pos": pos,
        "size": size,
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": [
            {
                "localized_name": "images",
                "name": "images",
                "type": "IMAGE",
                "link": images_link,
            },
            {
                "localized_name": "filename_prefix",
                "name": "filename_prefix",
                "type": "STRING",
                "widget": {"name": "filename_prefix"},
                "link": None,
            },
        ],
        "outputs": [
            {
                "localized_name": "images",
                "name": "images",
                "type": "IMAGE",
                "links": None,
            }
        ],
        "properties": {"Node name for S&R": "SaveImage"},
        "widgets_values": [filename_prefix],
    }


def _save_video_node(
    node_id: int,
    pos: list[float],
    size: list[float],
    order: int,
    *,
    video_link: int,
    filename_prefix: str = "video/xdit",
) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "SaveVideo",
        "pos": pos,
        "size": size,
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": [
            {
                "localized_name": "video",
                "name": "video",
                "type": "VIDEO",
                "link": video_link,
            },
            {
                "localized_name": "filename_prefix",
                "name": "filename_prefix",
                "type": "STRING",
                "widget": {"name": "filename_prefix"},
                "link": None,
            },
            {
                "localized_name": "format",
                "name": "format",
                "type": "COMBO",
                "widget": {"name": "format"},
                "link": None,
            },
            {
                "localized_name": "codec",
                "name": "codec",
                "type": "COMBO",
                "widget": {"name": "codec"},
                "link": None,
            },
        ],
        "outputs": [
            {
                "localized_name": "video",
                "name": "video",
                "type": "VIDEO",
                "links": None,
            }
        ],
        "properties": {"Node name for S&R": "SaveVideo"},
        "widgets_values": [filename_prefix, "auto", "auto"],
    }


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
    preset_name: str = "flux.1gpu.rdna4",
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
    preset_name: str = "flux.1gpu.rdna4",
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


def build_starter_workflow_dict(
    *,
    preset_name: str = "flux.1gpu.rdna4",
    gpu_tag: str = "gfx1201",
) -> dict[str, Any]:
    from .presets import preset_by_name

    selected_preset = preset_by_name(preset_name)
    preset_gpu_count = selected_preset.gpu_count if selected_preset is not None else 1
    preset_id = 1
    load_model_id = 2
    sample_id = 3
    save_image_id = 4
    save_video_id = 5

    links = [
        _workflow_link(1, preset_id, "xDiT.Preset", "model", load_model_id, "xDiT.Model", "preset"),
        _workflow_link(2, preset_id, "xDiT.Preset", "sample", sample_id, "xDiT.Sample", "preset"),
        _workflow_link(3, load_model_id, "xDiT.Model", "model", sample_id, "xDiT.Sample", "model"),
        _workflow_link(
            4,
            sample_id,
            "xDiT.Sample",
            "images",
            save_image_id,
            "SaveImage",
            "images",
            type_name="IMAGE",
        ),
        _workflow_link(
            5,
            preset_id,
            "xDiT.Preset",
            "images",
            sample_id,
            "xDiT.Sample",
            "images",
            type_name="IMAGE",
        ),
        _workflow_link(
            6,
            sample_id,
            "xDiT.Sample",
            "video",
            save_video_id,
            "SaveVideo",
            "video",
            type_name="VIDEO",
        ),
    ]

    nodes = [
        _build_workflow_node(
            preset_id,
            "xDiT.Preset",
            [40, 40],
            [420, 180],
            0,
            widget_overrides={
                "gpu_tag": gpu_tag,
                "gpu_count": str(preset_gpu_count),
                "preset": preset_name,
                "gpu_detection_info": "",
            },
            output_links={
                "model": [1],
                "images": [5],
                "sample": [2],
            },
        ),
        _build_workflow_node(
            load_model_id,
            "xDiT.Model",
            [520, 40],
            [420, 420],
            1,
            linked={"preset": 1},
            widget_overrides=_loader_widget_defaults(
                preset_name=preset_name,
                gpu_tag=gpu_tag,
            ),
            output_links={"model": [3]},
            properties={
                "_xdit_model_preset_trigger": (f"{gpu_tag}:{preset_gpu_count}:{preset_name}"),
            },
        ),
        _build_workflow_node(
            sample_id,
            "xDiT.Sample",
            [1000, 40],
            [420, 520],
            2,
            linked={"model": 3, "preset": 2, "images": 5},
            widget_overrides=_sample_starter_widget_overrides(
                preset_name=preset_name,
                gpu_tag=gpu_tag,
            ),
            output_links={"images": [4], "video": [6]},
        ),
        _save_image_node(save_image_id, [1480, 40], [320, 220], 3, images_link=4),
        _save_video_node(save_video_id, [1480, 420], [320, 220], 4, video_link=6),
    ]

    workflow = {
        "last_node_id": save_video_id,
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
    preset_name: str = "flux.1gpu.rdna4",
    gpu_tag: str = "gfx1201",
) -> dict[str, Any]:
    from .presets import preset_by_name

    selected_preset = preset_by_name(preset_name)
    preset_gpu_count = selected_preset.gpu_count if selected_preset is not None else 1
    loader_values = _loader_widget_defaults(preset_name=preset_name, gpu_tag=gpu_tag)
    return {
        "1": {
            "class_type": "xDiT.Preset",
            "inputs": {
                "gpu_tag": gpu_tag,
                "gpu_count": str(preset_gpu_count),
                "preset": preset_name,
                "gpu_detection_info": format_gpu_detection_summary(),
            },
        },
        "2": {
            "class_type": "xDiT.Model",
            "inputs": {"preset": ["1", 0], **loader_values},
        },
        "3": {
            "class_type": "xDiT.Sample",
            "inputs": {
                "model": ["2", 0],
                "images": ["1", 1],
                "preset": ["1", 2],
                **_sample_widget_values(preset_name=preset_name, gpu_tag=gpu_tag),
            },
        },
        "4": {
            "class_type": "SaveImage",
            "inputs": {"images": ["3", 0], "filename_prefix": "xdit"},
        },
        "5": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["3", 1],
                "filename_prefix": "video/xdit",
                "format": "auto",
                "codec": "auto",
            },
        },
    }


def write_starter_workflow(
    path: Path, *, preset_name: str = "flux.1gpu.rdna4", gpu_tag: str = "gfx1201"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_starter_workflow_dict(preset_name=preset_name, gpu_tag=gpu_tag), indent=2)
        + "\n",
        encoding="utf-8",
    )
