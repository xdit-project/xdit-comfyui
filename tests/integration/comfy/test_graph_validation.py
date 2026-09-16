"""ComfyUI validate_prompt integration — catches missing widgets and bad graph wiring."""

from __future__ import annotations

import asyncio

import pytest

from ..helpers import build_comfy_prompt, build_preset_spec_for_tag


@pytest.mark.comfy_import
def test_comfy_validate_linked_preset_graph(comfy_nodes_ready):
    from execution import validate_prompt

    payload = build_comfy_prompt("z_image.4gpu.rdna4")
    ok, error, good_outputs, node_errors = asyncio.run(
        validate_prompt("integration-test", payload["prompt"], None)
    )
    assert ok, (error, node_errors, good_outputs)


@pytest.mark.comfy_import
def test_comfy_validate_graph_with_stale_loader_widgets(comfy_nodes_ready):
    from execution import validate_prompt

    payload = build_comfy_prompt(
        "z_image.4gpu.rdna4",
        stale_loader={"cache_method": "none", "model": "Tongyi-MAI/Z-Image-Turbo"},
    )
    ok, error, good_outputs, node_errors = asyncio.run(
        validate_prompt("integration-test-stale", payload["prompt"], None)
    )
    assert ok, (error, node_errors, good_outputs)


@pytest.mark.comfy_import
def test_comfy_validate_starter_workflow(comfy_nodes_ready):
    from execution import validate_prompt

    from xdit_comfyui.prompt_hooks import apply_preset_prompt_overrides
    from xdit_comfyui.starter_workflow import build_starter_api_prompt

    payload = apply_preset_prompt_overrides({"prompt": build_starter_api_prompt()})
    ok, error, good_outputs, node_errors = asyncio.run(
        validate_prompt("integration-starter", payload["prompt"], None)
    )
    assert ok, (error, node_errors, good_outputs)


@pytest.mark.comfy_import
def test_comfy_refuses_a_gpu_layout_that_cannot_run(comfy_nodes_ready):
    """The layout is decided here, not minutes later with a worker already up."""
    from execution import validate_prompt

    payload = build_comfy_prompt(
        "z_image.4gpu.rdna4",
        stale_loader={"fully_shard_degree": 64},
    )
    ok, _error, _good_outputs, node_errors = asyncio.run(
        validate_prompt("integration-bad-layout", payload["prompt"], None)
    )
    assert not ok
    assert "fully_shard_degree" in str(node_errors)


@pytest.mark.comfy_import
def test_comfy_validate_wan_i2v_graph(comfy_nodes_ready):
    from execution import validate_prompt

    preset_name = "wan2_2_ti2v_5b.i2v.4gpu.rdna4"
    build_preset_spec_for_tag(preset_name)
    payload = build_comfy_prompt(
        preset_name,
        stale_sample={"num_frames": 81, "task": "i2v"},
    )
    ok, error, good_outputs, node_errors = asyncio.run(
        validate_prompt("integration-test-wan", payload["prompt"], None)
    )
    assert ok, (error, node_errors, good_outputs)
