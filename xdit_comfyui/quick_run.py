"""Stub diffusion outputs when quick_run is enabled (worker init only, no forward)."""

from __future__ import annotations

from typing import Any


def quick_diffusion_output(input_args: dict[str, Any]):
    import numpy as np
    from PIL import Image
    from xfuser.model_executor.models.runner_models.base_model import DiffusionOutput

    height = max(int(input_args.get("height") or 512), 64)
    width = max(int(input_args.get("width") or 512), 64)
    num_frames = max(int(input_args.get("num_frames") or 1), 1)
    seed = int(input_args.get("seed") or 42)
    color = (
        (seed * 17) % 200 + 55,
        (seed * 31) % 200 + 55,
        (seed * 47) % 200 + 55,
    )

    if num_frames > 1:
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        videos = [np.stack([frame] * num_frames)]
        return DiffusionOutput(videos=videos, pipe_args=dict(input_args))

    image = Image.new("RGB", (width, height), color)
    return DiffusionOutput(images=[image], pipe_args=dict(input_args))
