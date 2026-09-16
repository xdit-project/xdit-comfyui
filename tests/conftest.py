import os
import sys
import types
from unittest import mock

import pytest


@pytest.fixture
def synthetic_preset_catalog():
    """Inject stable adapter inputs without reading or maintaining config files."""
    from xdit_comfyui import presets

    def preset(name, tags, model, gpu_count, **args):
        return presets.BenchmarkPreset(
            name=name,
            tags=tuple(tags),
            model=model,
            gpu_count=gpu_count,
            args=args,
            source_file="<in-memory-test-data>",
        )

    catalog = (
        preset(
            "flux.1gpu.rdna4",
            ["gfx1201"],
            "black-forest-labs/FLUX.1-dev",
            1,
            prompt="synthetic text prompt",
            height=1024,
            width=1024,
            num_inference_steps=25,
            max_sequence_length=256,
            guidance_scale=0.0,
            attention_backend="aiter_flydsl",
            use_torch_compile=True,
            use_fp8_gemms=True,
        ),
        preset(
            "flux.usp_1k.4gpu.rdna4",
            ["gfx1201"],
            "black-forest-labs/FLUX.1-dev",
            4,
            ulysses_degree=4,
            attention_backend="aiter_flydsl",
            use_fp8_gemms=True,
            enable_tiling=True,
            use_parallel_vae=True,
        ),
        preset("flux.1gpu.hopper", ["h100"], "black-forest-labs/FLUX.1-dev", 1, ulysses_degree=1),
        preset("flux.usp.hopper", ["h100"], "black-forest-labs/FLUX.1-dev", 4, ulysses_degree=4),
        preset(
            "flux.1gpu.blackwell", ["b200"], "black-forest-labs/FLUX.1-dev", 1, ulysses_degree=1
        ),
        preset(
            "flux2.i2i_2k.1gpu.rdna4",
            ["gfx1201"],
            "black-forest-labs/FLUX.2-dev",
            1,
            input_images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png",
            resize_input_images=True,
            enable_tiling=True,
            enable_sequential_cpu_offload=True,
        ),
        preset(
            "flux2.t-multi-i2i_1k",
            ["gfx942"],
            "black-forest-labs/FLUX.2-dev",
            8,
            ulysses_degree=8,
            input_images=[
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png",
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/grumpy.jpg",
            ],
        ),
        preset(
            "flux_kontext.1gpu.rdna4",
            ["gfx1201"],
            "black-forest-labs/FLUX.1-Kontext-dev",
            1,
            input_images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png",
            resize_input_images=True,
        ),
        preset(
            "qwen_image.1gpu.rdna4",
            ["gfx1201"],
            "Qwen/Qwen-Image-2512",
            1,
            enable_tiling=True,
            enable_slicing=True,
            enable_model_cpu_offload=True,
            use_fp8_gemms=True,
        ),
        preset(
            "wan2_2_ti2v_5b.i2v.2gpu.rdna4",
            ["gfx1201"],
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            2,
            input_images="https://raw.githubusercontent.com/AMD-AGI/diffusion-models-inference/172fbcce2bf603216771f476fc40002b0640ce8d/assets/data/wan_input.jpg",
            num_frames=81,
            height=720,
            width=1280,
            task="i2v",
            ulysses_degree=2,
            fully_shard_degree=2,
            memory_efficient_sharding=True,
        ),
        preset(
            "wan2_2_ti2v_5b.i2v.4gpu.rdna4",
            ["gfx1201"],
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            4,
            input_images="https://raw.githubusercontent.com/AMD-AGI/diffusion-models-inference/172fbcce2bf603216771f476fc40002b0640ce8d/assets/data/wan_input.jpg",
            num_frames=81,
            height=720,
            width=1280,
            task="i2v",
            ulysses_degree=4,
            fully_shard_degree=4,
            memory_efficient_sharding=True,
        ),
        preset(
            "z_image.1gpu.rdna4",
            ["gfx1201"],
            "Tongyi-MAI/Z-Image",
            1,
            enable_tiling=True,
            enable_slicing=True,
            use_fp8_gemms=True,
        ),
        preset(
            "z_image.4gpu.rdna4",
            ["gfx1201"],
            "Tongyi-MAI/Z-Image",
            4,
            ulysses_degree=4,
            enable_tiling=True,
            enable_slicing=True,
            use_fp8_gemms=True,
        ),
        preset(
            "z_image_turbo.1gpu.rdna4",
            ["gfx1201"],
            "Tongyi-MAI/Z-Image-Turbo",
            1,
            ulysses_degree=1,
            num_inference_steps=4,
        ),
        preset(
            "z_image_turbo.2gpu.rdna4",
            ["gfx1201"],
            "Tongyi-MAI/Z-Image-Turbo",
            2,
            ulysses_degree=2,
            enable_tiling=True,
            enable_slicing=True,
        ),
        preset(
            "z_image_turbo.4gpu.rdna4",
            ["gfx1201"],
            "Tongyi-MAI/Z-Image-Turbo",
            4,
            ulysses_degree=4,
            enable_tiling=True,
            enable_slicing=True,
        ),
        preset(
            "hunyuanvideo_1_5.distilled.gfx950",
            ["gfx950"],
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled",
            8,
            task="i2v",
            ulysses_degree=8,
            num_frames=61,
            height=720,
            width=1280,
        ),
    )
    with mock.patch.object(presets, "load_benchmark_presets", return_value=catalog):
        yield


def _gpu_present():
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(not _gpu_present(), reason="needs a visible GPU; CI runs CPU-only")


@pytest.fixture(scope="session", autouse=True)
def isolate_worker_tokens():
    """Keep the tests from ever addressing a worker that belongs to a real ComfyUI.

    A worker is found by a token hashed from the Model node id, and the release path
    reaches processes by that token rather than by parentage — that is how it reclaims
    GPUs from a ComfyUI that died. So a test warming node "7" on a machine where a
    ComfyUI is serving node "7" kills that user's worker, mid-run. Salting the token for
    the whole session puts the tests in a namespace no graph can collide with.
    """
    from xdit_comfyui import worker

    original = worker._loader_worker_token
    salt = f"pytest-{os.getpid()}-"
    with mock.patch.object(worker, "_loader_worker_token", lambda uid: original(f"{salt}{uid}")):
        yield


@pytest.fixture
def mock_loader_worker_warm():
    def _fake_ensure(runtime, loader_node_id, timeout_seconds=None):
        runtime["_preloaded"] = True
        return runtime

    with mock.patch(
        "xdit_comfyui.nodes._ensure_loader_worker",
        side_effect=_fake_ensure,
    ):
        yield


@pytest.fixture
def mock_comfy_execution(monkeypatch):
    class ExecutionBlocker:
        def __init__(self, message):
            self.message = message

    package = types.ModuleType("comfy_execution")
    graph = types.ModuleType("comfy_execution.graph")
    graph.ExecutionBlocker = ExecutionBlocker
    package.graph = graph
    monkeypatch.setitem(sys.modules, "comfy_execution", package)
    monkeypatch.setitem(sys.modules, "comfy_execution.graph", graph)
    return ExecutionBlocker


@pytest.fixture
def mock_comfy_video_api(monkeypatch):
    """Stand in for ComfyUI's VIDEO type so multi-frame runs can return an output."""

    class VideoComponents:
        def __init__(self, images, frame_rate):
            self.images = images
            self.frame_rate = frame_rate

    class VideoFromComponents:
        def __init__(self, components):
            self.components = components

    modules = {
        "comfy_api": types.ModuleType("comfy_api"),
        "comfy_api.latest": types.ModuleType("comfy_api.latest"),
        "comfy_api.latest._input_impl": types.ModuleType("comfy_api.latest._input_impl"),
        "comfy_api.latest._input_impl.video_types": types.ModuleType(
            "comfy_api.latest._input_impl.video_types"
        ),
        "comfy_api.latest._util": types.ModuleType("comfy_api.latest._util"),
        "comfy_api.latest._util.video_types": types.ModuleType(
            "comfy_api.latest._util.video_types"
        ),
    }
    modules["comfy_api.latest._input_impl.video_types"].VideoFromComponents = VideoFromComponents
    modules["comfy_api.latest._util.video_types"].VideoComponents = VideoComponents
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return VideoFromComponents


@pytest.fixture
def require_gpu_count():
    def _require(count: int):
        from xdit_comfyui.presets import available_gpu_count

        visible = available_gpu_count()
        if visible < count:
            pytest.skip(f"needs {count} visible GPU(s), found {visible}")

    return _require


@pytest.fixture
def require_gpu_live():
    if os.environ.get("XDIT_RUN_GPU_TESTS", "").strip() != "1":
        pytest.skip("Set XDIT_RUN_GPU_TESTS=1 to run live GPU inference tests")
