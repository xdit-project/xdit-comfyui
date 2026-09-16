#!/usr/bin/env python3
"""Write the xDiT benchmark starter workflow into example_workflows/."""

import os
from pathlib import Path

from xdit_comfyui.starter_workflow import write_starter_workflow

TEMPLATE_NAMES = ("xDiT-Starter.json",)


def _comfyui_workflows_dir() -> Path:
    comfy_root = Path(os.environ.get("COMFYUI_ROOT", "/workspace/comfyui"))
    return comfy_root / "user" / "default" / "workflows"


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    obsolete_names = (
        "xDiT Benchmark Starter.json",
        "xDiT_Benchmark_Starter.json",
        "xDiT-Benchmark-Starter.json",
        "xDiT_Starter.json",
        "xDiT_Model_Starter.json",
    )
    for name in obsolete_names:
        obsolete_starter = root / "example_workflows" / name
        if obsolete_starter.is_file():
            obsolete_starter.unlink()
        obsolete_user_starter = _comfyui_workflows_dir() / name
        if obsolete_user_starter.is_file():
            obsolete_user_starter.unlink()
    for name in TEMPLATE_NAMES:
        write_starter_workflow(root / "example_workflows" / name)
        write_starter_workflow(_comfyui_workflows_dir() / name)


if __name__ == "__main__":
    main()
