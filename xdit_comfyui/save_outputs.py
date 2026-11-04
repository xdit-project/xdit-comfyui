"""Output nodes that explicitly clear the media type a Sample run did not produce."""

from __future__ import annotations

import os
from typing import Any

from .identity import (
    COMFY_CATEGORY,
    NODE_DISPLAY_NAMES,
    SAVE_IMAGE_NODE_ID,
    SAVE_VIDEO_NODE_ID,
)
from .v3 import Types, inputs_from_legacy, io, ui

IMAGE_FILENAME_PREFIX = "xdit"
VIDEO_FILENAME_PREFIX = "video/xdit"


def _save_input_types(input_name: str, input_type: str, filename_prefix: str) -> dict[str, Any]:
    return {
        input_name: (
            input_type,
            {"tooltip": f"{input_name.title()} from xDiT Sample."},
        ),
        "filename_prefix": (
            "STRING",
            {
                "default": filename_prefix,
                "tooltip": "Output filename prefix. Date and node-value substitutions are supported.",
            },
        ),
    }


def save_image_input_types() -> dict[str, Any]:
    return {"required": _save_input_types("images", "IMAGE", IMAGE_FILENAME_PREFIX)}


def save_video_input_types() -> dict[str, Any]:
    return {
        "required": {
            **_save_input_types("video", "VIDEO", VIDEO_FILENAME_PREFIX),
            "format": (
                Types.VideoContainer.as_input(),
                {
                    "default": "auto",
                    "tooltip": "Container format for the saved video.",
                },
            ),
            "codec": (
                Types.VideoCodec.as_input(),
                {
                    "default": "auto",
                    "tooltip": "Codec for the saved video.",
                },
            ),
        }
    }


def _save_schema(node_id: str, description: str, input_types: dict[str, Any]):
    inputs, _hidden = inputs_from_legacy(input_types)
    return io.Schema(
        node_id=node_id,
        display_name=NODE_DISPLAY_NAMES[node_id],
        category=COMFY_CATEGORY,
        description=description,
        inputs=inputs,
        hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        is_output_node=True,
    )


def _empty_media_output():
    return io.NodeOutput(ui=ui.SavedImages([]))


def _output_metadata(node_type: type) -> dict[str, Any] | None:
    from comfy.cli_args import args  # type: ignore[reportMissingImports]

    if args.disable_metadata:
        return None
    metadata = dict(node_type.hidden.extra_pnginfo or {})
    if node_type.hidden.prompt is not None:
        metadata["prompt"] = node_type.hidden.prompt
    return metadata or None


class XDiTSaveImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _save_schema(
            SAVE_IMAGE_NODE_ID,
            "Save image results and clear this output after a video run.",
            save_image_input_types(),
        )

    @classmethod
    def execute(cls, images, filename_prefix=IMAGE_FILENAME_PREFIX):
        if images is None:
            return _empty_media_output()
        saved = ui.ImageSaveHelper.get_save_images_ui(images, filename_prefix, cls)
        return io.NodeOutput(ui=saved)


class XDiTSaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _save_schema(
            SAVE_VIDEO_NODE_ID,
            "Save video results and clear this output after an image run.",
            save_video_input_types(),
        )

    @classmethod
    def execute(cls, video, filename_prefix=VIDEO_FILENAME_PREFIX, format="auto", codec="auto"):
        if video is None:
            return _empty_media_output()

        import folder_paths  # type: ignore[reportMissingImports]

        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height,
        )
        metadata = _output_metadata(cls)
        file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension(format)}"
        video.save_to(
            os.path.join(full_output_folder, file),
            format=Types.VideoContainer(format),
            codec=codec,
            metadata=metadata,
        )
        saved = ui.SavedResult(file, subfolder, io.FolderType.output)
        return io.NodeOutput(ui=ui.PreviewVideo([saved]))
