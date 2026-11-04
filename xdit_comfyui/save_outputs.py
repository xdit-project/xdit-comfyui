"""Output nodes that explicitly clear the media type a Sample run did not produce."""

from __future__ import annotations

import os
from typing import Any

from .identity import COMFY_CATEGORY
from .v3 import Types, inputs_from_legacy, io, ui


def save_image_input_types() -> dict[str, Any]:
    return {
        "required": {
            "result": (
                "*",
                {"tooltip": "Image or video result from xDiT Sample."},
            ),
            "filename_prefix": (
                "STRING",
                {
                    "default": "xdit",
                    "tooltip": (
                        "Output filename prefix. Date and node-value substitutions are supported."
                    ),
                },
            ),
        }
    }


def save_video_input_types() -> dict[str, Any]:
    return {
        "required": {
            "result": (
                "*",
                {"tooltip": "Image or video result from xDiT Sample."},
            ),
            "filename_prefix": (
                "STRING",
                {
                    "default": "video/xdit",
                    "tooltip": (
                        "Output filename prefix. Date and node-value substitutions are supported."
                    ),
                },
            ),
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


class XDiTSaveImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs, _hidden = inputs_from_legacy(save_image_input_types())
        return io.Schema(
            node_id="xDiT.SaveImage",
            display_name="Save xDiT Image",
            category=COMFY_CATEGORY,
            description="Save image results and clear this output after a video run.",
            inputs=inputs,
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, result, filename_prefix="xdit"):
        images = result.get("images") if isinstance(result, dict) else None
        if images is None:
            return io.NodeOutput(ui=ui.SavedImages([]))
        saved = ui.ImageSaveHelper.get_save_images_ui(images, filename_prefix, cls)
        return io.NodeOutput(ui=saved)


class XDiTSaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs, _hidden = inputs_from_legacy(save_video_input_types())
        return io.Schema(
            node_id="xDiT.SaveVideo",
            display_name="Save xDiT Video",
            category=COMFY_CATEGORY,
            description="Save video results and clear this output after an image run.",
            inputs=inputs,
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, result, filename_prefix="video/xdit", format="auto", codec="auto"):
        video = result.get("video") if isinstance(result, dict) else None
        if video is None:
            return io.NodeOutput(ui=ui.SavedImages([]))

        import folder_paths  # type: ignore[reportMissingImports]
        from comfy.cli_args import args  # type: ignore[reportMissingImports]

        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height,
        )
        metadata = None
        if not args.disable_metadata:
            metadata = {}
            if cls.hidden.extra_pnginfo is not None:
                metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
            if not metadata:
                metadata = None
        file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension(format)}"
        video.save_to(
            os.path.join(full_output_folder, file),
            format=Types.VideoContainer(format),
            codec=codec,
            metadata=metadata,
        )
        saved = ui.SavedResult(file, subfolder, io.FolderType.output)
        return io.NodeOutput(ui=ui.PreviewVideo([saved]))
