"""Rendered Nodes 2.0 layout checks against a live ComfyUI server."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.browser_live,
    pytest.mark.skipif(
        os.environ.get("XDIT_RUN_BROWSER_TESTS") != "1",
        reason="Set XDIT_RUN_BROWSER_TESTS=1 to run live Playwright layout tests.",
    ),
]

_WORKFLOW = Path(__file__).resolve().parents[2] / "example_workflows" / "xDiT-Starter.json"


def _row(node, label: str):
    row = node.get_by_text(re.compile(rf"^{re.escape(label)}$", re.I)).first
    row.wait_for(state="visible")
    box = row.bounding_box()
    assert box is not None, f"{label!r} has no rendered box"
    return box


@pytest.mark.parametrize("vue_nodes", [False, True], ids=["nodes-1-canvas", "nodes-2-vue"])
def test_sample_vram_and_groups_render_in_order(vue_nodes):
    playwright = pytest.importorskip("playwright.sync_api")
    url = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")

    with playwright.sync_playwright() as driver:
        try:
            browser = driver.chromium.launch(headless=True)
        except playwright.Error as exc:
            pytest.fail(
                "Playwright Chromium is unavailable. Run "
                "`bash scripts/browser/run_tests.sh --install` first.\n"
                f"{exc}"
            )
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        browser_errors = []
        page.on("pageerror", lambda error: browser_errors.append(f"pageerror: {error}"))
        page.on(
            "console",
            lambda message: (
                browser_errors.append(f"console.{message.type}: {message.text}")
                if message.type == "error"
                else None
            ),
        )
        page.goto(url, wait_until="networkidle")

        settings_response = page.request.post(
            f"{url}/settings/Comfy.VueNodes.Enabled",
            data=vue_nodes,
        )
        assert settings_response.ok, (
            f"could not enable Nodes 2.0: {settings_response.status} " f"{settings_response.text()}"
        )
        page.reload(wait_until="networkidle")

        workflow_text = _WORKFLOW.read_text(encoding="utf-8")
        page.evaluate(
            """({ name, content }) => {
                const transfer = new DataTransfer();
                transfer.items.add(new File([content], name, { type: 'application/json' }));
                const target = document.querySelector('canvas') || document.body;
                target.dispatchEvent(new DragEvent('dragover', {
                    bubbles: true, cancelable: true, dataTransfer: transfer,
                }));
                target.dispatchEvent(new DragEvent('drop', {
                    bubbles: true, cancelable: true, dataTransfer: transfer,
                }));
            }""",
            {"name": _WORKFLOW.name, "content": workflow_text},
        )

        page.wait_for_timeout(2000)
        state = page.evaluate("""() => ({
                vueNodesMode: globalThis.LiteGraph?.vueNodesMode,
                renderedNodes: [...document.querySelectorAll('[data-node-id]')]
                    .map((element) => ({
                        id: element.getAttribute('data-node-id'),
                        text: element.innerText?.slice(0, 500),
                        classes: element.className,
                    })),
                sampleState: (() => {
                    const graph = globalThis.graph ?? globalThis.app?.graph;
                    const sample = graph?._nodes?.find((node) => node.type === 'xDiT.Sample');
                    return {
                        widgets: sample?.widgets?.filter((widget) =>
                            ['width', 'height', 'seed', 'Video', 'VAE'].includes(widget.name)
                        ).map((widget) => ({
                            name: widget.name,
                            hidden: widget.hidden,
                            optionsHidden: widget.options?.hidden,
                            value: widget.value,
                        })),
                        inputs: sample?.inputs?.map((input, index) => ({
                            index, name: input.name, widget: input.widget?.name,
                        })),
                    };
                })(),
            })""")
        assert state["vueNodesMode"] is vue_nodes, f"wrong node renderer: {state}"

        if not vue_nodes:
            canvas_state = page.evaluate("""() => {
                        const graph = globalThis.graph
                            ?? globalThis.app?.graph
                            ?? globalThis.LGraphCanvas?.active_canvas?.graph;
                    const sample = graph?._nodes?.find(
                        (node) => node.type === 'xDiT.Sample' || node.comfyClass === 'xDiT.Sample'
                    );
                        return {
                            nodeTypes: graph?._nodes?.map((node) => node.type),
                        rows: sample?.widgets?.map((widget) => ({
                            name: widget.name,
                            y: widget.y,
                            last_y: widget.last_y,
                            hidden: widget.hidden,
                        })),
                        };
                }""")
            assert canvas_state["rows"], f"Sample did not load in canvas graph: {canvas_state}"
            names = [row["name"] for row in canvas_state["rows"]]
            assert "xdit_info" not in names
            assert names.index("Video") < names.index("VAE")
            rows = {row["name"]: row for row in canvas_state["rows"]}
            assert rows["enable_tiling"]["hidden"] is True
            browser.close()
            return

        assert state["renderedNodes"], f"Vue nodes did not render: {state}"
        assert any(
            "xDiT Sample" in (node["text"] or "") for node in state["renderedNodes"]
        ), f"starter workflow did not render the Sample node: {state}; errors={browser_errors}"
        sample = (
            page.locator("[data-node-id]").filter(has_text=re.compile("xDiT Sample", re.I)).first
        )
        sample.wait_for(state="visible")

        widget_state = {item["name"]: item for item in state["sampleState"]["widgets"]}
        assert widget_state.get("width") and not widget_state["width"]["hidden"], state
        assert widget_state.get("height") and not widget_state["height"]["hidden"], state

        prompt = _row(sample, "Prompt")
        negative = _row(sample, "Negative Prompt")
        width = _row(sample, "Width")
        height = _row(sample, "Height")
        seed = _row(sample, "Seed")
        assert prompt["y"] < negative["y"] < width["y"] < height["y"] < seed["y"], (
            "rendered Sample rows are out of order: "
            f"prompt={prompt}, negative={negative}, width={width}, "
            f"height={height}, seed={seed}"
        )
        assert widget_state["VAE"]["value"] is False
        assert widget_state["VAE"]["hidden"] is not True
        assert not sample.get_by_text(re.compile(r"^Enable Tiling$", re.I)).is_visible()

        browser.close()


def test_preset_gpu_count_refreshes_choices_through_real_nodes2_controls():
    playwright = pytest.importorskip("playwright.sync_api")
    url = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(url, wait_until="networkidle")
        page.request.post(f"{url}/settings/Comfy.VueNodes.Enabled", data=True)
        page.reload(wait_until="networkidle")
        workflow_text = _WORKFLOW.read_text(encoding="utf-8")
        page.evaluate(
            """({ name, content }) => {
                const transfer = new DataTransfer();
                transfer.items.add(new File([content], name, { type: 'application/json' }));
                const target = document.querySelector('canvas') || document.body;
                target.dispatchEvent(new DragEvent('drop', { bubbles: true, cancelable: true, dataTransfer: transfer }));
            }""",
            {"name": _WORKFLOW.name, "content": workflow_text},
        )
        page.wait_for_timeout(2000)
        preset = (
            page.locator("[data-node-id]").filter(has_text=re.compile("xDiT Preset", re.I)).first
        )
        preset.get_by_role("combobox", name="GPU Tag").click()
        page.get_by_role("option", name="gfx950", exact=True).click()
        page.wait_for_function("""() => {
                const graph = globalThis.graph ?? globalThis.app?.graph;
                const node = graph?._nodes?.find((item) => item.type === 'xDiT.Preset');
                return node?.widgets?.find((widget) => widget.name === 'gpu_tag')?.value === 'gfx950';
            }""")

        expected = page.request.post(
            f"{url}/xdit/preset/preview",
            data={"gpu_tag": "gfx950", "gpu_count": 8, "preset": "none"},
        ).json()["choices"]
        page.evaluate("""() => {
                const originalFetch = globalThis.fetch;
                globalThis.fetch = (input, init) => {
                    if (String(input) !== '/xdit/preset/preview') return originalFetch(input, init);
                    return new Promise((resolve, reject) => setTimeout(
                        () => originalFetch(input, init).then(resolve, reject), 1500
                    ));
                };
            }""")

        preset.get_by_role("combobox", name="GPU Count").click()
        page.get_by_role("option", name="8", exact=True).click()
        preset.get_by_role("combobox", name="Preset").click()
        rendered_choices = page.get_by_role("option").all_inner_texts()
        assert rendered_choices == expected
        page.keyboard.press("Escape")
        state = page.evaluate("""() => {
                const graph = globalThis.graph ?? globalThis.app?.graph;
                const node = graph?._nodes?.find((item) => item.type === 'xDiT.Preset');
                const value = (name) => node?.widgets?.find((widget) => widget.name === name)?.value;
                return { gpuTag: value('gpu_tag'), gpuCount: value('gpu_count') };
            }""")
        assert state == {"gpuTag": "gfx950", "gpuCount": "8"}
        assert len(expected) > 2, "fixture no longer exercises a populated 8-GPU list"
        browser.close()


def test_starter_template_renders_its_thumbnail():
    playwright = pytest.importorskip("playwright.sync_api")
    url = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        responses = []
        page.on(
            "response",
            lambda response: (
                responses.append((response.status, response.url))
                if "xDiT-Starter.jpg" in response.url
                else None
            ),
        )
        page.goto(url, wait_until="networkidle")
        page.get_by_role("button", name="Templates", exact=True).click()
        page.get_by_role("button", name="xdit_comfyui", exact=True).click()
        thumbnail = page.get_by_role("img", name="xDiT-Starter")
        thumbnail.wait_for(state="visible")
        dimensions = thumbnail.evaluate(
            "image => ({ width: image.naturalWidth, height: image.naturalHeight })"
        )
        assert dimensions == {"width": 400, "height": 400}
        assert responses and responses[-1][0] == 200
        browser.close()


def test_xdit_sidebar_owns_runtime_status_and_unload_controls():
    playwright = pytest.importorskip("playwright.sync_api")
    url = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(url, wait_until="networkidle")
        workflow_text = _WORKFLOW.read_text(encoding="utf-8")
        page.evaluate(
            """({ name, content }) => {
                const transfer = new DataTransfer();
                transfer.items.add(new File([content], name, { type: 'application/json' }));
                const target = document.querySelector('canvas') || document.body;
                target.dispatchEvent(new DragEvent('drop', {
                    bubbles: true, cancelable: true, dataTransfer: transfer,
                }));
            }""",
            {"name": _WORKFLOW.name, "content": workflow_text},
        )
        page.wait_for_timeout(2000)

        page.get_by_text("xDiT", exact=True).first.click()
        sidebar = page.get_by_text("Models", exact=True).last.locator("..")
        page.get_by_text("Presets", exact=True).last.wait_for(state="visible")
        page.get_by_text("Samples", exact=True).last.wait_for(state="visible")
        page.get_by_role("button", name="Unload all models", exact=True).wait_for(state="visible")
        assert page.get_by_text("Not loaded.", exact=True).count() >= 1
        assert "0.0 GiB model" not in sidebar.inner_text()
        assert page.get_by_text("Run VRAM", exact=True).count() == 0
        assert page.get_by_text("GPU residency", exact=True).count() == 0
        browser.close()
