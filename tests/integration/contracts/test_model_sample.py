"""Model-to-sample execution contracts with a mocked worker."""

from unittest import mock

import pytest

from tests.node_test_helpers import _execution_kwargs, _loader_kwargs
from xdit_comfyui.nodes import XDiTModel
from xdit_comfyui.sampling import _execute_sample
from xdit_comfyui.worker import _clear_all_runtime_caches


@pytest.fixture(autouse=True)
def _clear_runtime_cache():
    _clear_all_runtime_caches()
    yield
    _clear_all_runtime_caches()


@pytest.mark.contract
def test_loader_generate_subprocess_dry_run_workflow():
    runtime = XDiTModel.execute(
        **_loader_kwargs(hf_cache_mode="comfy_models_shared"),
    )[0]
    assert runtime["model"] == "black-forest-labs/FLUX.1-dev"

    images, video = _execute_sample(
        runtime,
        dry_run=True,
        output_type="pil",
        **_execution_kwargs(),
    )
    assert video is None
    assert tuple(images.shape[1:]) == (64, 64, 3)


@pytest.mark.contract
def test_loader_generate_worker_dispatch():
    lifecycle = {"calls": 0}

    def fake_distributed(
        runner_config,
        env_overrides,
        nproc,
        preferred_cache_key,
        timeout_seconds,
        generate_node_id,
    ):
        lifecycle["calls"] += 1
        import torch

        return "worker runner created", torch.zeros((1, 8, 8, 3), dtype=torch.float32)

    with mock.patch(
        "xdit_comfyui.worker._run_xdit_distributed",
        side_effect=fake_distributed,
    ):
        runtime = XDiTModel.execute(**_loader_kwargs())[0]

        images1, _ = _execute_sample(
            runtime,
            dry_run=False,
            output_type="pil",
            **_execution_kwargs(prompt="p1"),
        )

        images2, _ = _execute_sample(
            runtime,
            dry_run=False,
            output_type="pil",
            **_execution_kwargs(prompt="p2", seed=43),
        )

    assert lifecycle["calls"] == 2
    assert tuple(images1.shape[1:]) == (8, 8, 3)
    assert tuple(images2.shape[1:]) == (8, 8, 3)
