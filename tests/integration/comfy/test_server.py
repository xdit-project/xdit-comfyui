"""Optional live ComfyUI server checks — same HTTP surface the UI uses."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from ..helpers import loader_preview_body


def _post_json(url: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(payload)
        except json.JSONDecodeError:
            return exc.code, {"raw": payload}


def _comfy_reachable(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/system_stats", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


@pytest.fixture
def require_comfy_server(comfyui_base_url):
    if not _comfy_reachable(comfyui_base_url):
        pytest.skip(f"ComfyUI not reachable at {comfyui_base_url}; " "start it or set COMFYUI_URL")
    return comfyui_base_url


@pytest.mark.comfy_live
def test_live_loader_preview_matches_python_api(require_comfy_server):
    from ..helpers import preview_loader

    stale_loader = {"cache_method": "none", "model": "Tongyi-MAI/Z-Image-Turbo"}
    body = loader_preview_body("z_image.4gpu.rdna4", stale_loader=stale_loader)
    status, payload = _post_json(f"{require_comfy_server}/xdit/loader/preview", body)
    assert status == 200
    expected = preview_loader("z_image.4gpu.rdna4", stale_loader=stale_loader)
    assert payload["runtime"] == expected["runtime"]
    assert payload["display_widgets"] == expected["display_widgets"]


@pytest.mark.comfy_live
def test_live_preset_preview(require_comfy_server):
    status, payload = _post_json(
        f"{require_comfy_server}/xdit/preset/preview",
        {
            "gpu_tag": "gfx1201",
            "gpu_count": 4,
            "preset": "z_image.4gpu.rdna4",
        },
    )
    assert status == 200
    assert payload["preset"]["matched"] is True
    assert payload["preset"]["generation_defaults"]["height"] == 1088
