"""The narrow boundary between the plugin runtime and xDiT's Python API.

Keeping these calls here makes version drift explicit and keeps the worker transport,
ComfyUI nodes, and lifecycle code independent of xDiT's module layout.
"""

from __future__ import annotations


def create_initialized_runner(config: dict):
    from xfuser.runner import xFuserModelRunner

    runner = xFuserModelRunner(config)
    runner.initialize(runner.preprocess_args(config))
    return runner


def distributed_world():
    from xfuser.core.distributed import get_world_group

    return get_world_group()


def preprocess_run(runner, config: dict):
    return runner.preprocess_args(config)


def run(runner, args):
    return runner.run(args)


def cleanup(runner) -> None:
    runner.cleanup()
