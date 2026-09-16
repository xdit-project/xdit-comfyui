import asyncio
import os
from unittest import mock

import pytest

from .helpers import clear_runtime_caches


@pytest.fixture(autouse=True)
def _mock_worker_warm_for_non_gpu_tests(request):
    if (
        request.node.get_closest_marker("gpu_live")
        or request.node.get_closest_marker("gpu_init")
        or request.node.get_closest_marker("gpu_residency")
    ):
        yield
        return

    def _fake_ensure(runtime, loader_node_id, timeout_seconds=None):
        runtime["_preloaded"] = True
        return runtime

    with mock.patch(
        "xdit_comfyui.nodes._ensure_loader_worker",
        side_effect=_fake_ensure,
    ):
        yield


@pytest.fixture(autouse=True)
def _reset_runtime_caches():
    clear_runtime_caches()
    yield
    clear_runtime_caches()


@pytest.fixture
def require_gpu_headroom(require_gpu_live):
    """Stop ComfyUI workers and skip when VRAM is too full for a cold worker init."""
    import subprocess
    import time

    from .helpers import clear_runtime_caches, gpu_min_free_mib, stop_comfyui_dev_script

    stop_script = stop_comfyui_dev_script()
    if os.path.isfile(stop_script):
        subprocess.run(["bash", stop_script], check=False, capture_output=True, text=True)
        time.sleep(3)
    clear_runtime_caches()
    subprocess.run(["pkill", "-f", "distributed_worker"], check=False, capture_output=True)
    time.sleep(1)

    min_free = int(os.environ.get("XDIT_GRAPH_MIN_FREE_MIB", "12000"))
    free_mib = gpu_min_free_mib()
    if free_mib < min_free:
        pytest.skip(
            f"need >={min_free} MiB free on GPU 0 for live loader warm, found {free_mib} MiB"
        )
    return free_mib


@pytest.fixture(scope="session")
def comfy_nodes_ready():
    if os.environ.get("XDIT_SKIP_COMFY_INIT", "").strip() == "1":
        pytest.skip("Set XDIT_SKIP_COMFY_INIT=0 to run ComfyUI graph validation tests")
    from .helpers import comfyui_root

    comfy_root = comfyui_root()
    if not os.path.isdir(comfy_root):
        pytest.skip(f"ComfyUI root not found: {comfy_root}")

    import sys

    if comfy_root not in sys.path:
        sys.path.insert(0, comfy_root)

    import nodes
    from nodes import init_extra_nodes

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(init_extra_nodes(init_custom_nodes=True))
    finally:
        loop.close()

    required = {"xDiT.Preset", "xDiT.Model", "xDiT.Sample", "SaveImage", "SaveVideo"}
    missing = required - set(nodes.NODE_CLASS_MAPPINGS)
    if missing:
        pytest.skip(f"ComfyUI missing node classes: {sorted(missing)}")
    return nodes.NODE_CLASS_MAPPINGS


@pytest.fixture
def comfyui_base_url():
    return os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
