import unittest

import pytest

pytestmark = pytest.mark.usefixtures("synthetic_preset_catalog")

from xdit_comfyui import presets as preset_catalog
from xdit_comfyui.presets import (
    available_gpu_count,
    build_preset_spec,
    default_gpu_device_ids,
    default_gpu_tag,
    detect_gpu_tags,
    format_gpu_detection_summary,
    is_manual_preset_choice,
    list_applicable_presets,
    list_benchmark_hardware_tags,
    list_gpu_counts_for_gpu_tag,
    list_hardware_matching_preset_names,
    list_preset_names,
    list_presets_for_gpu_tag,
    normalize_gpu_device_ids,
    parse_gpu_device_ids,
    preset_by_name,
    preset_loader_seed,
    preset_to_loader_widgets,
    resolve_loader_preset,
    resolve_preset_choice,
)
from xdit_comfyui.runtime_config import (
    _generation_input_types,
    _preset_picker_input_types,
    _runtime_loader_input_types,
)


class PresetTest(unittest.TestCase):
    def test_benchmark_only_warmup_and_compile_are_not_applied(self):
        from xdit_comfyui.presets import _COMFY_IGNORED_BENCHMARK_ARGS

        self.assertTrue(
            {"warmup_calls", "num_iterations", "use_torch_compile"} <= _COMFY_IGNORED_BENCHMARK_ARGS
        )

    def test_preset_configs_are_available(self):
        self.assertGreaterEqual(len(preset_catalog.load_benchmark_presets()), 10)

    def test_benchmark_hardware_tags(self):
        tags = list_benchmark_hardware_tags()
        self.assertIn("gfx1201", tags)
        self.assertIn("gfx942", tags)

    def test_default_gpu_tag_prefers_detected_arch(self):
        tag = default_gpu_tag()
        self.assertIn(tag, list_benchmark_hardware_tags())

    def test_detect_gpu_tags_normalizes_rocm_feature_suffixes(self):
        from types import SimpleNamespace
        from unittest import mock

        torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                get_device_properties=lambda _index: SimpleNamespace(
                    gcnArchName="gfx950:sramecc+:xnack-",
                    name="AMD Instinct MI355X",
                ),
            )
        )
        with mock.patch.dict("sys.modules", {"torch": torch}):
            self.assertEqual(detect_gpu_tags(), ("gfx950",))

    def test_list_presets_for_gpu_tag(self):
        gfx1201 = list_presets_for_gpu_tag("gfx1201")
        h100 = list_presets_for_gpu_tag("h100")
        self.assertTrue(any(name == "flux.1gpu.rdna4" for name in gfx1201))
        self.assertTrue(any(name == "flux.usp.hopper" for name in h100))
        self.assertFalse(any(name == "flux.1gpu.rdna4" for name in h100))
        self.assertEqual(list_presets_for_gpu_tag(""), [])
        self.assertFalse(set(gfx1201) & set(h100))

    def test_list_gpu_counts_and_presets_for_tag_and_count(self):
        counts = list_gpu_counts_for_gpu_tag("gfx1201")
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts, [1, 2, 4])
        two_gpu = list_presets_for_gpu_tag("gfx1201", 2)
        self.assertTrue(two_gpu)
        self.assertTrue(all(preset_by_name(name).gpu_count == 2 for name in two_gpu))
        self.assertEqual(list_gpu_counts_for_gpu_tag(""), [])

    def test_hardware_tags_are_direct(self):
        tags = set(list_benchmark_hardware_tags())
        self.assertIn("gfx1201", tags)
        self.assertIn("h100", tags)
        self.assertNotIn("hopper", tags)
        self.assertNotIn("blackwell", tags)

    def test_preset_picker_seeds_default_tag_presets(self):

        preset_spec = _preset_picker_input_types()["required"]["preset"]
        choices = preset_spec[0]
        required = _preset_picker_input_types()["required"]
        default_count = required["gpu_count"][1]["default"]
        self.assertIn("none", choices)
        self.assertTrue(any("rdna4" in name for name in choices))
        self.assertEqual(
            len(choices),
            len(list_presets_for_gpu_tag(default_gpu_tag(), default_count)) + 1,
        )
        self.assertFalse(any(name.endswith(".hopper") for name in choices if name != "none"))

    def test_normalize_gpu_tag_uses_canonical_tags_only(self):
        from xdit_comfyui.presets import normalize_gpu_tag

        self.assertEqual(normalize_gpu_tag("gfx1201"), "gfx1201")
        self.assertEqual(normalize_gpu_tag("rdna4"), default_gpu_tag())

    def test_preset_validate_inputs_accepts_known_preset(self):
        from xdit_comfyui.nodes import XDiTPreset

        self.assertTrue(XDiTPreset.validate_inputs("gfx1201", 4, "z_image_turbo.4gpu.rdna4", ""))

    def test_preset_validate_inputs_rejects_gpu_count_mismatch(self):
        from xdit_comfyui.nodes import XDiTPreset

        result = XDiTPreset.validate_inputs("gfx1201", 2, "z_image_turbo.4gpu.rdna4", "")
        self.assertIn("requires 4 GPU(s)", result)

    def test_default_gpu_device_ids(self):
        self.assertEqual(default_gpu_device_ids(1), "0")
        self.assertEqual(default_gpu_device_ids(4), "0,1,2,3")

    def test_normalize_gpu_device_ids_replaces_auto(self):
        self.assertEqual(normalize_gpu_device_ids("auto", 4), "0,1,2,3")
        self.assertEqual(normalize_gpu_device_ids("", 2), "0,1")

    def test_parse_gpu_device_ids_explicit(self):
        self.assertEqual(parse_gpu_device_ids("2,3", 2), [2, 3])
        self.assertEqual(parse_gpu_device_ids("0,1,2,3", 4), [0, 1, 2, 3])

    def test_parse_gpu_device_ids_rejects_wrong_count(self):
        with self.assertRaisesRegex(ValueError, "exactly 4"):
            parse_gpu_device_ids("0", 4)

    def test_normalize_gpu_device_ids_preserves_explicit_selection(self):
        self.assertEqual(normalize_gpu_device_ids("0", 4), "0")

    def test_resolve_flux_rdna4_single_gpu(self):
        resolved = resolve_loader_preset(
            "FLUX.1-dev",
            1,
            gpu_tags=("gfx1201", "rdna4"),
        )
        self.assertTrue(resolved.matched)
        self.assertEqual(resolved.preset_name, "flux.1gpu.rdna4")
        self.assertEqual(resolved.runtime["ulysses_degree"], 1)
        self.assertTrue(resolved.runtime["use_fp8_gemms"])

    def test_resolve_z_image_turbo_two_gpu(self):
        resolved = resolve_loader_preset(
            "Tongyi-MAI/Z-Image-Turbo",
            2,
            gpu_tags=("gfx1201", "rdna4"),
        )
        self.assertTrue(resolved.matched)
        self.assertEqual(resolved.preset_name, "z_image_turbo.2gpu.rdna4")
        self.assertEqual(resolved.runtime["ulysses_degree"], 2)

    def test_list_applicable_presets_flux_rdna4(self):
        presets = list_applicable_presets(
            "FLUX.1-dev",
            1,
            gpu_tags=("gfx1201", "rdna4"),
        )
        self.assertTrue(any(item["name"] == "flux.1gpu.rdna4" for item in presets))

    def test_preset_to_loader_widgets(self):
        preset = preset_by_name("flux.1gpu.rdna4")
        self.assertIsNotNone(preset)
        widgets = preset_to_loader_widgets(preset)
        self.assertEqual(widgets["ulysses_degree"], 1)
        self.assertEqual(widgets["gemm_precision"], "fp8")

    def test_format_gpu_detection_summary_is_compact(self):
        summary = format_gpu_detection_summary(("rdna4", "gfx1201"))
        self.assertNotIn("presets:", summary)
        self.assertNotIn("AMD Radeon", summary)
        if available_gpu_count() > 1:
            self.assertRegex(summary, r"^\[\d+\] ")
        elif available_gpu_count() == 1:
            self.assertTrue(summary.startswith("[0] "))

    def test_preset_node_exposes_gpu_tag_and_detection(self):
        from xdit_comfyui.nodes import XDiTPreset

        required = _preset_picker_input_types()["required"]
        optional = _preset_picker_input_types().get("optional", {})
        self.assertIn("gpu_tag", required)
        self.assertIn("gpu_detection_info", required)
        self.assertEqual(
            list(required.keys())[:4],
            ["gpu_detection_info", "gpu_tag", "gpu_count", "preset"],
        )
        self.assertEqual(required["gpu_count"][0], ["1", "2", "4"])
        self.assertNotIn("multiline", required["gpu_detection_info"][1])
        self.assertEqual(
            [output.id for output in XDiTPreset.define_schema().outputs],
            ["model", "images", "sample"],
        )
        self.assertNotIn("gpu_detection_info", optional)

    def test_loader_and_sample_preset_input_names(self):

        loader_required = list(_runtime_loader_input_types()["required"])
        loader_optional = list(_runtime_loader_input_types()["optional"])
        generate_optional = list(_generation_input_types()["optional"])
        self.assertNotIn("gpu_count", loader_required)
        self.assertIn("model", loader_required)
        self.assertIn("task", loader_required)
        self.assertNotIn("model_choice", loader_required)
        self.assertIn("gpu_device_ids", loader_required)
        self.assertEqual(loader_optional[0], "preset")
        self.assertEqual(generate_optional[0], "images")
        self.assertEqual(generate_optional[1], "preset")
        self.assertNotIn("task", _generation_input_types()["required"])
        sample_required = _generation_input_types()["required"]
        self.assertEqual(list(sample_required)[:2], ["model", "prompt"])
        sample_order = list(sample_required)
        # Declaration order is the node's layout: each collapsible section has to be a
        # contiguous run, or its heading would fold widgets belonging to another section.
        self.assertEqual(
            sample_order[sample_order.index("num_frames") : sample_order.index("num_frames") + 4],
            ["num_frames", "output_fps", "flow_shift", "guidance_scale_2"],
        )
        self.assertEqual(
            sample_order[sample_order.index("resize_input_images") : sample_order.index("VAE")],
            ["resize_input_images", "max_sequence_length", "timeout_seconds"],
        )
        self.assertIn("Video", sample_order)
        self.assertEqual(
            sample_order[sample_order.index("VAE") + 1 :],
            [
                "enable_tiling",
                "enable_slicing",
                "vae_tile_size_height",
                "vae_tile_size_width",
                "vae_tile_overlap_height",
                "vae_tile_overlap_width",
            ],
        )
        self.assertNotIn("Advanced", sample_order)
        self.assertFalse(sample_required["resize_input_images"][1]["default"])
        self.assertEqual(sample_required["output_fps"][1]["default"], 0)

    def test_vae_defaults_are_owned_by_the_sample_preset_payload(self):
        from xdit_comfyui.presets import build_preset_spec

        spec = build_preset_spec("flux2.i2i_2k.1gpu.rdna4", "gfx1201")
        self.assertTrue(spec["vae_defaults"]["enable_tiling"])
        self.assertNotIn("enable_tiling", spec["runtime_widgets"])

    def test_preset_spec_reads_explicit_model_gpu_count(self):
        preset = preset_by_name("flux.usp_1k.4gpu.rdna4")
        self.assertIsNotNone(preset)
        self.assertEqual(preset.gpu_count, 4)
        spec = build_preset_spec(
            "flux.usp_1k.4gpu.rdna4", "gfx1201", registry_choices=[preset.model]
        )
        self.assertEqual(spec["gpu_count"], 4)

    def test_parse_benchmark_model_uses_explicit_or_xdit_derived_gpu_count(self):
        from xdit_comfyui.presets import _parse_benchmark_model

        self.assertEqual(
            _parse_benchmark_model(
                {
                    "name": "derived",
                    "model": "black-forest-labs/FLUX.1-dev",
                    "args": {"ulysses_degree": 2},
                }
            ),
            ("black-forest-labs/FLUX.1-dev", 2),
        )
        model, count = _parse_benchmark_model(
            {
                "name": "ok",
                "model": "black-forest-labs/FLUX.1-dev",
                "gpu_count": 2,
            }
        )
        self.assertEqual(model, "black-forest-labs/FLUX.1-dev")
        self.assertEqual(count, 2)

    def test_preset_spec_includes_generation_defaults(self):
        preset = preset_by_name("flux.1gpu.rdna4")
        self.assertIsNotNone(preset)
        spec = build_preset_spec("flux.1gpu.rdna4", "gfx1201", registry_choices=[preset.model])
        self.assertTrue(spec["matched"])
        self.assertEqual(spec["gpu_count"], 1)
        self.assertEqual(spec["selected_gpu_tag"], "gfx1201")
        self.assertIn("prompt", spec["generation_defaults"])
        self.assertIn("num_inference_steps", spec["generation_defaults"])
        self.assertFalse(spec["image_input_preset"]["required"])

    def test_preset_spec_rejects_tag_mismatch(self):
        preset = preset_by_name("flux.1gpu.rdna4")
        self.assertIsNotNone(preset)
        spec = build_preset_spec("flux.1gpu.rdna4", "gfx942", registry_choices=[preset.model])
        self.assertFalse(spec["matched"])

    def test_list_hardware_matching_preset_names(self):
        names = list_hardware_matching_preset_names(gpu_tag="gfx1201")
        self.assertTrue(any("rdna4" in name for name in names))
        self.assertFalse(any("gfx942" in name and "rdna4" not in name for name in names))

    def test_list_preset_names_includes_all_benchmarks(self):
        self.assertGreaterEqual(
            len(list_preset_names()), len(preset_catalog.load_benchmark_presets())
        )

    def test_preset_loader_seed_maps_model_and_gpu(self):
        preset = preset_by_name("flux.1gpu.rdna4")
        self.assertIsNotNone(preset)
        seed = preset_loader_seed(preset, ["black-forest-labs/FLUX.1-dev"])
        self.assertEqual(seed["model_choice"], "black-forest-labs/FLUX.1-dev")
        self.assertEqual(seed["gpu_count"], 1)
        self.assertEqual(seed["attention_backend"], "aiter_flydsl")

    def test_is_manual_preset_choice(self):
        self.assertTrue(is_manual_preset_choice("none"))
        self.assertTrue(is_manual_preset_choice("custom"))
        self.assertTrue(is_manual_preset_choice("auto (best for hardware)"))
        self.assertFalse(is_manual_preset_choice("flux.1gpu.rdna4"))

    def test_resolve_preset_choice_none_returns_none(self):
        self.assertIsNone(
            resolve_preset_choice("none", "FLUX.1-dev", 1, gpu_tags=("gfx1201", "rdna4"))
        )

    def test_resolve_preset_choice_unknown_returns_none(self):
        self.assertIsNone(resolve_preset_choice("not.a.real.preset", "FLUX.1-dev", 1))

    def test_unmatched_falls_back_to_ulysses(self):
        resolved = resolve_loader_preset(
            "FLUX.1-dev",
            3,
            gpu_tags=("gfx1201", "rdna4"),
        )
        self.assertFalse(resolved.matched)
        self.assertEqual(resolved.runtime["ulysses_degree"], 3)

    def test_qwen_image_2512_maps_to_registry_model_choice(self):
        spec = build_preset_spec(
            "qwen_image.1gpu.rdna4",
            "gfx1201",
            registry_choices=["Qwen/Qwen-Image", "Custom (HF repo id)"],
        )
        self.assertTrue(spec["matched"])
        self.assertEqual(spec["model_choice"], "Qwen/Qwen-Image")
        self.assertEqual(spec["model"], "Qwen/Qwen-Image-2512")

    def test_model_preset_base_sets_custom_model_id_for_versioned_model(self):
        from xdit_comfyui.runtime_config import _model_preset_base

        base = _model_preset_base(
            {
                "matched": True,
                "model": "Qwen/Qwen-Image-2512",
                "model_choice": "Qwen/Qwen-Image",
                "gpu_count": 1,
                "runtime_widgets": {},
            }
        )
        self.assertEqual(base["model"], "Qwen/Qwen-Image")
        self.assertEqual(base["custom_model_id"], "Qwen/Qwen-Image-2512")


if __name__ == "__main__":
    unittest.main()
