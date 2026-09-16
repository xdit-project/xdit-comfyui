"""HTTP routes themselves, not just the payload builders behind them.

Every route here is a closure inside `register_routes()`, so it only runs when ComfyUI's
`server` module imports. Nothing else in the suite ever calls one, which left the whole
browser-facing surface — request parsing, 404s, filename handling — untested: a wrong
key or a missing `await` would only show up as a broken node in a browser.
"""

import asyncio
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import pytest

pytestmark = pytest.mark.usefixtures("mock_loader_worker_warm")

from aiohttp import web

from xdit_comfyui.presets import PRESET_NONE, list_presets_for_gpu_tag


class _RouteTable:
    """Stands in for aiohttp's RouteTableDef, keeping what was registered."""

    def __init__(self):
        self.handlers = {}

    def _register(self, method, path):
        def decorate(handler):
            self.handlers[(method, path)] = handler
            return handler

        return decorate

    def get(self, path):
        return self._register("GET", path)

    def post(self, path):
        return self._register("POST", path)


class _Request:
    def __init__(self, *, body=None, match_info=None, query=None):
        self._body = body
        self.match_info = match_info or {}
        self.query = query or {}

    async def json(self):
        return self._body


def _registered_routes():
    """Run `register_routes()` against a stand-in PromptServer and collect the handlers."""
    from xdit_comfyui import api

    table = _RouteTable()
    instance = types.SimpleNamespace(
        routes=table,
        app=types.SimpleNamespace(on_startup=[]),
        add_on_prompt_handler=mock.Mock(),
    )
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=instance)
    with mock.patch.dict(sys.modules, {"server": server}):
        api.register_routes()
    return table.handlers, instance


def _call(handler, **kwargs):
    return asyncio.run(handler(_Request(**kwargs)))


def _json_body(response):
    return json.loads(response.text)


class RouteRegistrationTest(unittest.TestCase):
    def test_every_route_the_browser_calls_is_registered(self):
        handlers, _ = _registered_routes()
        expected = {
            ("GET", "/xdit/loader/schema"),
            ("POST", "/xdit/loader/preview"),
            ("POST", "/xdit/loader/clear"),
            ("POST", "/xdit/loader/reap"),
            ("POST", "/xdit/preset/preview"),
            ("GET", "/xdit/residency"),
            ("GET", "/xdit/presets/{name}"),
            ("GET", "/xdit/benchmark_cache/{filename}"),
        }
        self.assertTrue(expected <= set(handlers), sorted(expected - set(handlers)))

    def test_registration_hooks_the_prompt_and_warms_the_schema(self):
        _, instance = _registered_routes()
        self.assertEqual(1, len(instance.app.on_startup))
        instance.add_on_prompt_handler.assert_called_once()

    def test_registration_is_a_no_op_without_comfyui(self):
        """Imported by the tests and by tooling that has no ComfyUI server."""
        from xdit_comfyui import api

        with mock.patch.dict(sys.modules, {"server": None}):
            api.register_routes()


class LoaderRouteTest(unittest.TestCase):
    def setUp(self):
        self.handlers, _ = _registered_routes()

    def test_schema_route_serves_the_widget_schema(self):
        payload = _json_body(_call(self.handlers[("GET", "/xdit/loader/schema")]))
        self.assertIn("config_widgets", payload)
        self.assertIn("widget_groups", payload)

    def test_preview_route_answers_a_model_change(self):
        payload = _json_body(
            _call(
                self.handlers[("POST", "/xdit/loader/preview")],
                body={"model": "Qwen/Qwen-Image", "gpu_count": 1, "gpu_device_ids": "0"},
            )
        )
        self.assertEqual("Qwen/Qwen-Image", payload["generation"]["model"])
        self.assertIn("widget_gates", payload)

    def test_preview_route_survives_an_empty_body(self):
        """The node fires a preview before the user has touched anything."""
        payload = _json_body(_call(self.handlers[("POST", "/xdit/loader/preview")], body={}))
        self.assertIn("generation", payload)

    def test_clear_route_reports_nothing_to_evict(self):
        payload = _json_body(
            _call(self.handlers[("POST", "/xdit/loader/clear")], body={"node_id": "42"})
        )
        self.assertFalse(payload["evicted"])

    def test_reap_route_releases_everything_when_asked(self):
        with mock.patch(
            "xdit_comfyui.worker._release_all_loaders",
            return_value={"released": ["7"]},
        ) as release:
            payload = _json_body(
                _call(self.handlers[("POST", "/xdit/loader/reap")], body={"all": True})
            )
        release.assert_called_once()
        self.assertEqual(["7"], payload["released"])

    def test_reap_route_keeps_the_nodes_the_graph_still_has(self):
        with mock.patch(
            "xdit_comfyui.worker._reap_loaders_except",
            return_value={"released": []},
        ) as reap:
            _call(
                self.handlers[("POST", "/xdit/loader/reap")],
                body={"live_node_ids": ["3", "4"]},
            )
        reap.assert_called_once_with(["3", "4"])


class PresetRouteTest(unittest.TestCase):
    def setUp(self):
        self.handlers, _ = _registered_routes()

    def test_preset_preview_route_lists_the_hardware_presets(self):
        payload = _json_body(
            _call(
                self.handlers[("POST", "/xdit/preset/preview")],
                body={"preset": PRESET_NONE, "gpu_tag": "gfx1201"},
            )
        )
        self.assertTrue(payload["hardware_presets"])
        self.assertIn("preset", payload)

    def test_preset_detail_route_seeds_the_model_widgets(self):
        name = list_presets_for_gpu_tag("gfx1201", 1)[0]
        payload = _json_body(
            _call(self.handlers[("GET", "/xdit/presets/{name}")], match_info={"name": name})
        )
        self.assertEqual(name, payload["name"])
        self.assertTrue(payload["model"])
        self.assertIn("generation_defaults", payload)

    def test_preset_detail_route_answers_the_manual_choice_with_no_widgets(self):
        payload = _json_body(
            _call(
                self.handlers[("GET", "/xdit/presets/{name}")],
                match_info={"name": PRESET_NONE},
            )
        )
        self.assertEqual({}, payload["widgets"])

    def test_preset_detail_route_404s_on_an_unknown_preset(self):
        with self.assertRaises(web.HTTPNotFound):
            _call(
                self.handlers[("GET", "/xdit/presets/{name}")],
                match_info={"name": "no.such.preset"},
            )


class StaticFileRouteTest(unittest.TestCase):
    """These take a filename straight off the URL."""

    def setUp(self):
        self.handlers, _ = _registered_routes()

    def test_asset_routes_cannot_be_walked_out_of_their_directory(self):
        for route in ("/xdit/benchmark_cache/{filename}",):
            with self.subTest(route=route):
                with self.assertRaises(web.HTTPNotFound):
                    _call(
                        self.handlers[("GET", route)],
                        match_info={"filename": "../../../etc/passwd"},
                    )

    def test_workflow_template_route_serves_a_starter_graph(self):
        from xdit_comfyui.api import _EXAMPLE_WORKFLOWS_DIR

        route = next(
            handler
            for (method, path), handler in self.handlers.items()
            if method == "GET" and path.startswith("/api/workflow_templates/")
        )
        template = next(_EXAMPLE_WORKFLOWS_DIR.glob("*.json")).stem
        payload = _json_body(_call(route, match_info={"filename": template}))
        self.assertIn("nodes", payload)

    def test_workflow_template_route_serves_the_matching_jpeg_thumbnail(self):
        from xdit_comfyui.api import _EXAMPLE_WORKFLOWS_DIR

        route = next(
            handler
            for (method, path), handler in self.handlers.items()
            if method == "GET" and path.startswith("/api/workflow_templates/")
        )
        response = _call(route, match_info={"filename": "xDiT-Starter.jpg"})
        self.assertIsInstance(response, web.FileResponse)
        self.assertEqual(
            _EXAMPLE_WORKFLOWS_DIR / "xDiT-Starter.jpg",
            Path(response._path),
        )

        from PIL import Image

        with Image.open(response._path) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.size, (400, 400))

    def test_workflow_template_route_404s_on_an_unknown_template(self):
        route = next(
            handler
            for (method, path), handler in self.handlers.items()
            if method == "GET" and path.startswith("/api/workflow_templates/")
        )
        with self.assertRaises(web.HTTPNotFound):
            _call(route, match_info={"filename": "no_such_template"})


class ResidencyRouteTest(unittest.TestCase):
    def setUp(self):
        self.handlers, _ = _registered_routes()

    def test_residency_route_reports_resident_models(self):
        payload = _json_body(_call(self.handlers[("GET", "/xdit/residency")]))
        self.assertIn("loaders", payload)

    def test_residency_route_adds_the_run_memory_for_a_sample_node(self):
        with mock.patch(
            "xdit_comfyui.residency.sample_run_memory",
            return_value={"peak_mib": 1024},
        ):
            payload = _json_body(
                _call(
                    self.handlers[("GET", "/xdit/residency")],
                    query={"sample_node_id": "9"},
                )
            )
        self.assertEqual({"peak_mib": 1024}, payload["sample_run"])


if __name__ == "__main__":
    unittest.main()
