import unittest
from pathlib import Path
from unittest import mock

import pytest

pytestmark = pytest.mark.usefixtures("synthetic_preset_catalog")

from xdit_comfyui.benchmark_data import (
    REMOTE_CACHE_DIR,
    benchmark_image_preview_entries,
    fetch_remote_image,
    is_remote_image_ref,
    resolve_benchmark_data_path,
)
from xdit_comfyui.presets import build_preset_spec, preset_by_name
from xdit_comfyui.runner_contract import preset_to_image_input_preset


class BenchmarkDataTest(unittest.TestCase):
    def setUp(self):
        if REMOTE_CACHE_DIR.exists():
            for path in REMOTE_CACHE_DIR.glob("*"):
                path.unlink()

    def test_is_remote_image_ref(self):
        self.assertTrue(is_remote_image_ref("https://example.com/cat.jpg"))
        self.assertFalse(is_remote_image_ref("/app/data/flux_cat.png"))

    @mock.patch("urllib.request.urlopen")
    def test_fetch_remote_image_downloads_and_caches(self, urlopen):
        url = "https://example.com/grumpy.jpg"
        response = mock.Mock()
        response.read.return_value = b"\xff\xd8\xfffakejpeg"
        urlopen.return_value.__enter__.return_value = response

        first = fetch_remote_image(url)
        second = fetch_remote_image(url)

        self.assertTrue(first.is_file())
        self.assertEqual(first, second)
        urlopen.assert_called_once()

    @mock.patch("urllib.request.urlopen")
    def test_resolve_benchmark_data_path_downloads_remote_urls(self, urlopen):
        url = "https://example.com/grumpy.jpg"
        response = mock.Mock()
        response.read.return_value = b"\xff\xd8\xfffakejpeg"
        urlopen.return_value.__enter__.return_value = response

        resolved = Path(resolve_benchmark_data_path(url))
        self.assertTrue(resolved.is_file())
        self.assertEqual(resolved.parent, REMOTE_CACHE_DIR.resolve())

    @mock.patch("urllib.request.urlopen")
    def test_benchmark_image_preview_entries_remote_url(self, urlopen):
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/grumpy.jpg"
        response = mock.Mock()
        response.read.return_value = b"\xff\xd8\xfffakejpeg"
        urlopen.return_value.__enter__.return_value = response

        entries = benchmark_image_preview_entries([url])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["name"], "grumpy.jpg")
        self.assertTrue(entries[0]["url"].startswith("/xdit/benchmark_cache/"))

    @mock.patch("urllib.request.urlopen")
    def test_preset_to_image_input_preset_resolves_remote_images(self, urlopen):
        response = mock.Mock()
        response.read.return_value = b"\xff\xd8\xfffakejpeg"
        urlopen.return_value.__enter__.return_value = response

        spec = preset_to_image_input_preset(
            {
                "input_images": [
                    "https://example.com/wan_input.jpg",
                    "https://example.com/grumpy.jpg",
                ]
            }
        )
        self.assertTrue(spec["required"])
        self.assertEqual(len(spec["paths"]), 2)
        self.assertEqual(
            spec["paths"],
            ["https://example.com/wan_input.jpg", "https://example.com/grumpy.jpg"],
        )
        urlopen.assert_not_called()

    @mock.patch("urllib.request.urlopen")
    def test_flux2_multi_image_preset_builds_two_previews(self, urlopen):
        preset = preset_by_name("flux2.t-multi-i2i_1k")
        self.assertIsNotNone(preset)
        response = mock.Mock()
        response.read.return_value = b"\xff\xd8\xfffakejpeg"
        urlopen.return_value.__enter__.return_value = response

        spec = build_preset_spec(
            preset.name,
            "gfx942",
            registry_choices=["black-forest-labs/FLUX.2-dev"],
        )
        self.assertTrue(spec["matched"])
        paths = spec["image_input_preset"]["paths"]
        self.assertEqual(len(paths), 2)
        self.assertTrue(str(paths[0]).endswith("cat.png"))

        entries = benchmark_image_preview_entries(paths)
        self.assertEqual(len(entries), 2)
        names = {entry["name"] for entry in entries}
        self.assertTrue(any(name.endswith("cat.png") for name in names))
        self.assertTrue(any(name.endswith("grumpy.jpg") for name in names))


if __name__ == "__main__":
    unittest.main()
