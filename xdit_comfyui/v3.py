"""Small V3 schema helpers, with a test-only fallback outside ComfyUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:  # ComfyUI owns this API; it is present whenever the node pack is loaded.
    from comfy_api.v0_0_2 import ComfyExtension, io
except ImportError:  # Keep pure unit tests and package metadata imports lightweight.

    class _Hidden(Enum):
        unique_id = "UNIQUE_ID"
        prompt = "PROMPT"
        extra_pnginfo = "EXTRA_PNGINFO"

    @dataclass
    class _Input:
        id: str
        display_name: str | None = None
        optional: bool = False
        tooltip: str | None = None
        type: str = "*"
        extra_dict: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class _Output:
        id: str | None = None
        display_name: str | None = None
        type: str = "*"

    class _Type:
        type = "*"

        @classmethod
        def Input(
            cls, id, display_name=None, optional=False, tooltip=None, extra_dict=None, **kwargs
        ):
            return _Input(
                id,
                display_name,
                optional,
                tooltip,
                cls.type,
                {**(extra_dict or {}), **kwargs},
            )

        @classmethod
        def Output(cls, id=None, display_name=None, **_kwargs):
            return _Output(id, display_name, cls.type)

    def _typed(name):
        return type(name.title(), (_Type,), {"type": name})

    class _Combo(_Type):
        type = "COMBO"

        @classmethod
        def Input(cls, id, options=None, **kwargs):
            result = super().Input(id, **kwargs)
            result.extra_dict["options"] = options or []
            return result

    @dataclass
    class _Schema:
        node_id: str
        display_name: str | None = None
        category: str = "sd"
        inputs: list[Any] = field(default_factory=list)
        outputs: list[Any] = field(default_factory=list)
        hidden: list[Any] = field(default_factory=list)
        description: str = ""
        not_idempotent: bool = False

    class _NodeOutput(tuple):
        def __new__(cls, *values):
            return super().__new__(cls, values)

    class _ComfyNode:
        pass

    class _IO:
        ComfyNode = _ComfyNode
        NodeOutput = _NodeOutput
        Schema = _Schema
        Hidden = _Hidden
        AnyType = _typed("*")
        String = _typed("STRING")
        Int = _typed("INT")
        Float = _typed("FLOAT")
        Boolean = _typed("BOOLEAN")
        Image = _typed("IMAGE")
        Video = _typed("VIDEO")
        Combo = _Combo

    io = _IO()

    class ComfyExtension:
        pass


_INPUT_TYPES = {
    "STRING": io.String,
    "INT": io.Int,
    "FLOAT": io.Float,
    "BOOLEAN": io.Boolean,
    "IMAGE": io.Image,
    "VIDEO": io.Video,
    "*": io.AnyType,
}

_LABEL_WORDS = {
    "cfg": "CFG",
    "cpu": "CPU",
    "fp4": "FP4",
    "fp8": "FP8",
    "fps": "FPS",
    "gpu": "GPU",
    "hf": "HF",
    "id": "ID",
    "ids": "IDs",
    "gemm": "GEMM",
    "scm": "SCM",
    "ssta": "SSTA",
    "t2": "T2",
    "tp": "TP",
    "vae": "VAE",
    "vram": "VRAM",
    "vsa": "VSA",
    "xdit": "xDiT",
}


def _display_label(input_id: str) -> str:
    if input_id.startswith("["):
        return input_id
    words = input_id.split("_")
    return " ".join(_LABEL_WORDS.get(word.casefold(), word.capitalize()) for word in words)


def inputs_from_legacy(spec: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    """Translate the runner-derived widget schema into V3 input declarations."""
    inputs = []
    for section in ("required", "optional"):
        for input_id, definition in spec.get(section, {}).items():
            type_spec, options = definition
            options = dict(options or {})
            display_name = options.pop("display_name", None) or _display_label(input_id)
            tooltip = options.pop("tooltip", None)
            optional = section == "optional"
            if isinstance(type_spec, (list, tuple)):
                inputs.append(
                    io.Combo.Input(
                        input_id,
                        options=list(type_spec),
                        display_name=display_name,
                        optional=optional,
                        tooltip=tooltip,
                        **options,
                    )
                )
            else:
                input_type = _INPUT_TYPES.get(str(type_spec), io.AnyType)
                inputs.append(
                    input_type.Input(
                        input_id,
                        display_name=display_name,
                        optional=optional,
                        tooltip=tooltip,
                        **options,
                    )
                )
    hidden = [io.Hidden.unique_id] if "unique_id" in spec.get("hidden", {}) else []
    return inputs, hidden
