"""Loader and preset preview behavior."""

import unittest
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.usefixtures("mock_loader_worker_warm", "synthetic_preset_catalog")

from xdit_comfyui.api import (
    _preset_filter_schema,
    _preset_preview_payload,
    _preview_payload,
    _sanitize_loader_preview_body,
)
from xdit_comfyui.nodes import XDiTModel
from xdit_comfyui.presets import PRESET_NONE, list_presets_for_gpu_tag
from xdit_comfyui.runner_contract import default_loader_widget_values
from xdit_comfyui.runtime_config import _default_gpu_count, _runtime_loader_input_types
from xdit_comfyui.worker import _clear_loader_cache


class LoaderPreviewTest(unittest.TestCase):
    def test_preset_filter_schema_indexes_every_tag_and_gpu_count(self):
        schema = _preset_filter_schema()
        for tag, counts in schema["gpu_counts_by_tag"].items():
            for count in counts:
                self.assertIn(str(count), schema["presets_by_tag_and_count"][tag])

    def test_default_gpu_count_uses_visible_hardware(self):
        with patch("xdit_comfyui.runtime_config._available_gpu_count", return_value=4):
            self.assertEqual(_default_gpu_count(), 4)

    def test_loader_input_types_exclude_gpu_count_widget(self):
        required = _runtime_loader_input_types()["required"]
        self.assertNotIn("gpu_count", required)
        self.assertIn("gpu_device_ids", required)

    def test_preset_preview_disjoint_tag_presets(self):
        gfx1201 = _preset_preview_payload({"preset": PRESET_NONE, "gpu_tag": "gfx1201"})
        b200 = _preset_preview_payload({"preset": PRESET_NONE, "gpu_tag": "b200"})
        self.assertTrue(gfx1201["hardware_presets"])
        self.assertTrue(b200["hardware_presets"])
        self.assertFalse(set(gfx1201["hardware_presets"]) & set(b200["hardware_presets"]))

    def test_preset_preview_reports_the_resolved_selection(self):
        """The browser applies these to its combos instead of re-deriving them."""
        payload = _preset_preview_payload(
            {
                "preset": PRESET_NONE,
                "gpu_tag": "gfx1201",
                "gpu_count": 2,
            }
        )
        names = payload["hardware_presets"]
        self.assertTrue(names)
        self.assertTrue(any("rdna4" in name for name in names))
        self.assertEqual(payload["gpu_tag"], "gfx1201")
        self.assertIn("gfx1201", payload["gpu_tag_choices"])
        self.assertIn(payload["gpu_tag_suggested"], payload["gpu_tag_choices"])
        self.assertEqual(payload["gpu_count"], 2)
        self.assertEqual(payload["gpu_count_choices"], [1, 2, 4])
        self.assertEqual(names, list_presets_for_gpu_tag("gfx1201", 2))
        self.assertEqual(payload["choices"], [PRESET_NONE, *names])

    def test_preset_preview_falls_back_to_a_valid_gpu_count(self):
        payload = _preset_preview_payload(
            {"preset": PRESET_NONE, "gpu_tag": "gfx1201", "gpu_count": 3}
        )
        self.assertIn(payload["gpu_count"], payload["gpu_count_choices"])

    def test_preset_preview_filters_gfx1201_presets_by_count(self):
        payload = _preset_preview_payload(
            {"preset": PRESET_NONE, "gpu_tag": "gfx1201", "gpu_count": 1}
        )
        self.assertEqual(
            payload["hardware_presets"],
            list_presets_for_gpu_tag("gfx1201", 1),
        )

    def test_loader_preview_applies_connected_preset_for_unset_widgets(self):
        payload = _preview_payload(
            {
                "model": "black-forest-labs/FLUX.1-dev",
                "gpu_device_ids": "0",
                "preset_gpu_tag": "gfx1201",
                "preset_gpu_count": 1,
                "preset_choice": "flux.1gpu.rdna4",
            }
        )
        self.assertEqual(payload["preset_meta"]["name"], "flux.1gpu.rdna4")
        self.assertEqual(payload["preset_widgets"]["gpu_count"], 1)
        self.assertEqual(payload["preset_widgets"]["attention_backend"], "aiter_flydsl")
        self.assertEqual(payload["runtime"]["attention_backend"], "aiter_flydsl")
        self.assertEqual(payload["runtime"]["ulysses_degree"], 1)
        self.assertIsNone(payload["runtime"]["cache_method"])

    def test_loader_preview_uses_widget_cache_method(self):
        payload = _preview_payload(
            {
                **default_loader_widget_values(),
                "model": "black-forest-labs/FLUX.1-dev",
                "gpu_count": 4,
                "gpu_device_ids": "0",
                "cache_method": "none",
                "preset_gpu_tag": "gfx1201",
                "preset_gpu_count": 1,
                "preset_choice": "flux.1gpu.rdna4",
            }
        )
        self.assertEqual(payload["preset_meta"]["name"], "flux.1gpu.rdna4")
        self.assertIsNone(payload["runtime"].get("cache_method"))
        self.assertEqual(payload["display_widgets"]["cache_method"], "none")

    def test_loader_preview_uses_widget_values_when_switching_from_turbo_preset(self):
        payload = _preview_payload(
            {
                **default_loader_widget_values(),
                "model": "Tongyi-MAI/Z-Image-Turbo",
                "gpu_count": 4,
                "gpu_device_ids": "0",
                "cache_method": "none",
                "preset_gpu_tag": "gfx1201",
                "preset_choice": "z_image.4gpu.rdna4",
            }
        )
        self.assertEqual(payload["preset_meta"]["name"], "z_image.4gpu.rdna4")
        self.assertEqual(payload["runtime"].get("model"), "Tongyi-MAI/Z-Image-Turbo")
        self.assertIsNone(payload["runtime"].get("cache_method"))
        self.assertEqual(payload["display_widgets"]["cache_method"], "none")

    def test_loader_preview_reports_step_cache_support(self):
        supported = _preview_payload(
            {
                "model": "black-forest-labs/FLUX.1-dev",
                "gpu_count": 1,
                "gpu_device_ids": "0",
            }
        )
        self.assertTrue(supported["step_cache_supported"])

        unsupported = _preview_payload(
            {
                "model": "Tongyi-MAI/Z-Image-Turbo",
                "gpu_count": 1,
                "gpu_device_ids": "0",
            }
        )
        self.assertFalse(unsupported["step_cache_supported"])
        self.assertEqual(
            unsupported["cache_method_choices"],
            ["none", "teacache", "fbcache", "dbcache"],
        )
        self.assertEqual(unsupported["display_widgets"]["cache_method"], "none")

    def test_loader_preview_only_offers_park_for_compatible_layouts(self):
        replicated = _preview_payload(
            {"model": "FLUX.1-dev", "gpu_count": 1, "gpu_device_ids": "0"}
        )
        self.assertIn("park_cpu", replicated["residency_choices"])
        self.assertEqual(replicated["residency_unavailable_reason"], "")

        sharded = _preview_payload(
            {
                "model": "FLUX.1-dev",
                "gpu_count": 4,
                "gpu_device_ids": "0,1,2,3",
                "fully_shard_degree": 4,
            }
        )
        self.assertNotIn("park_cpu", sharded["residency_choices"])
        self.assertIn("FSDP", sharded["residency_unavailable_reason"])

    def test_loader_preview_says_which_options_the_model_cannot_use(self):
        """Whatever the queue sanitizer would reset, the UI has to stop offering."""
        payload = _preview_payload(
            {"model": "Tongyi-MAI/Z-Image-Turbo", "gpu_count": 1, "gpu_device_ids": "0"}
        )
        gates = payload["widget_gates"]
        self.assertFalse(gates["use_cfg_parallel"])
        self.assertFalse(gates["tensor_parallel_degree"])
        self.assertTrue(gates["ulysses_degree"])
        # Groups are keyed by their label so a whole inapplicable section can go away.
        self.assertFalse(gates["STEP CACHE"])

        supported = _preview_payload(
            {
                "model": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "gpu_count": 1,
                "gpu_device_ids": "0",
                "cache_method": "dbcache",
            }
        )
        self.assertTrue(supported["widget_gates"]["use_cfg_parallel"])
        self.assertTrue(supported["widget_gates"]["STEP CACHE · DENOISER 2"])

    def test_loader_preview_tells_sample_whether_the_model_makes_video(self):
        image = _preview_payload(
            {"model": "Tongyi-MAI/Z-Image-Turbo", "gpu_count": 1, "gpu_device_ids": "0"}
        )
        video = _preview_payload(
            {"model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "gpu_count": 1, "gpu_device_ids": "0"}
        )
        self.assertEqual(image["generation"]["output_kind"], "image")
        self.assertEqual(video["generation"]["output_kind"], "video")

    def test_loader_preview_carries_the_models_generation_defaults(self):
        """The Sample node gets the selected model's numbers, not the definition's."""
        payload = _preview_payload(
            {"model": "Qwen/Qwen-Image", "gpu_count": 1, "gpu_device_ids": "0"}
        )
        generation = payload["generation"]
        self.assertEqual(generation["defaults"]["height"], 928)
        self.assertEqual(generation["defaults"]["width"], 1664)
        self.assertEqual(generation["defaults"]["num_inference_steps"], 50)
        self.assertEqual(generation["resolution_step"], 8)
        self.assertIsNone(generation["resolution_divisor"])
        # The model's paragraph-long negative prompt would only get in the way.
        self.assertNotIn("negative_prompt", generation["defaults"])

    def test_loader_preview_reports_an_enforced_resolution_divisor(self):
        payload = _preview_payload(
            {"model": "Lightricks/LTX-2", "gpu_count": 1, "gpu_device_ids": "0"}
        )
        self.assertEqual(payload["generation"]["resolution_divisor"], 64)
        self.assertEqual(payload["generation"]["resolution_step"], 64)

    def test_loader_preview_shows_the_models_hybrid_step_counts(self):
        """Left at 0 the widget would pass 0 and override the model's schedule."""
        payload = _preview_payload(
            {
                **default_loader_widget_values(),
                "model": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                "gpu_count": 1,
                "gpu_device_ids": "0",
            }
        )
        self.assertEqual(payload["display_widgets"]["num_hybrid_attn_high_precision_steps"], 5)

    def test_loader_preview_survives_a_task_the_model_cannot_take(self):
        """A stale graph can hand the task widget anything; a preview must still answer."""
        payload = _preview_payload(
            {
                "model": "black-forest-labs/FLUX.1-dev",
                "task": "Tongyi-MAI/Z-Image-Turbo",
                "gpu_count": 1,
                "gpu_device_ids": "0",
            }
        )
        self.assertNotIn("task", payload["runtime"])
        self.assertEqual(payload["capabilities"]["valid_tasks"], [])

    def test_loader_preview_replaces_a_wrong_task_with_the_models_own(self):
        payload = _preview_payload(
            {
                "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "task": "not-a-task",
                "gpu_count": 1,
                "gpu_device_ids": "0",
            }
        )
        # Two valid tasks, so the browser picks from `valid_tasks` rather than the
        # server guessing which one the user meant.
        self.assertNotIn("task", payload["runtime"])
        self.assertEqual(payload["capabilities"]["valid_tasks"], ["i2v", "t2v"])

    def test_a_queued_run_still_refuses_a_task_the_model_cannot_take(self):
        from xdit_comfyui.model_info import resolve_model_task

        with self.assertRaises(ValueError):
            resolve_model_task("black-forest-labs/FLUX.1-dev", "i2v")
        with self.assertRaises(ValueError):
            resolve_model_task("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "not-a-task")

    def test_loader_preview_reports_each_transformers_cache_schedule(self):
        payload = _preview_payload(
            {
                "model": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "cache_method": "dbcache",
                "gpu_count": 1,
                "gpu_device_ids": "0",
            }
        )
        rows = payload["cache_transformers"]
        self.assertEqual([row["transformer"] for row in rows], ["transformer", "transformer_2"])
        self.assertEqual([row["config"]["max_warmup_steps"] for row in rows], [4, 2])
        # No override was asked for, so xFuser keeps applying its own per-transformer
        # presets instead of one broadcast value.
        self.assertIsNone(payload["runtime"].get("cache_config"))

    def test_an_unset_cache_widget_means_the_models_value(self):
        """The global DBCachePreset default would read as an override and flatten Wan 2.2."""
        payload = _preview_payload(
            {
                "model": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "cache_method": "dbcache",
                "gpu_count": 1,
                "gpu_device_ids": "0",
            }
        )
        self.assertEqual(payload["cache_defaults"]["max_warmup_steps"], 4)
        self.assertNotEqual(
            payload["cache_defaults"]["max_warmup_steps"],
            default_loader_widget_values()["max_warmup_steps"],
        )

    def test_loader_preview_applies_preset_model_and_task_atomically(self):
        payload = _preview_payload(
            {
                **default_loader_widget_values(),
                "model": "black-forest-labs/FLUX.1-dev",
                "task": "i2v",
                "gpu_device_ids": "0,1,2,3,4,5,6,7",
                "preset_gpu_tag": "gfx950",
                "preset_gpu_count": 8,
                "preset_choice": "hunyuanvideo_1_5.distilled.gfx950",
            }
        )
        self.assertEqual(
            payload["runtime"]["model"],
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled",
        )
        self.assertEqual(payload["runtime"]["task"], "i2v")

    def test_loader_preview_clears_stale_task_for_image_model_preset(self):
        payload = _preview_payload(
            {
                **default_loader_widget_values(),
                "model": "Tongyi-MAI/Z-Image-Turbo",
                "task": "i2v",
                "gpu_device_ids": "0",
                "preset_gpu_tag": "gfx1201",
                "preset_gpu_count": 1,
                "preset_choice": "z_image_turbo.1gpu.rdna4",
            }
        )
        self.assertNotIn("task", payload["runtime"])

    def test_switching_off_a_video_preset_answers_for_the_incoming_model(self):
        """The widgets still hold Wan when the user picks an image preset.

        Answering for Wan hands back its tasks, and the browser puts `task` straight
        back on the node it was just cleared from.
        """
        payload = _preview_payload(
            {
                **default_loader_widget_values(),
                "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "task": "i2v",
                "gpu_device_ids": "0",
                "preset_gpu_tag": "gfx1201",
                "preset_gpu_count": 1,
                "preset_choice": "z_image_turbo.1gpu.rdna4",
                "preset_applied": True,
            }
        )
        self.assertEqual(payload["runtime"]["model"], "Tongyi-MAI/Z-Image-Turbo")
        self.assertNotIn("task", payload["runtime"])
        self.assertEqual(payload["capabilities"]["valid_tasks"], [])
        self.assertEqual(payload["preset_widgets"]["task"], "")

    def test_a_preset_that_is_merely_connected_does_not_overwrite_the_model(self):
        """Only picking a preset applies it; otherwise an override would not survive."""
        payload = _preview_payload(
            {
                **default_loader_widget_values(),
                "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "task": "i2v",
                "gpu_device_ids": "0",
                "preset_gpu_tag": "gfx1201",
                "preset_gpu_count": 1,
                "preset_choice": "z_image_turbo.1gpu.rdna4",
            }
        )
        self.assertEqual(payload["runtime"]["model"], "Wan-AI/Wan2.2-TI2V-5B-Diffusers")

    def test_clear_loader_cache_api_helper(self):
        from xdit_comfyui.runner_contract import default_loader_widget_values

        XDiTModel.execute(
            **{
                **default_loader_widget_values(),
                "model": "black-forest-labs/FLUX.1-dev",
                "gpu_count": 1,
                "gpu_device_ids": "0",
                "custom_model_id": "",
                "use_torch_compile": False,
                "hf_cache_mode": "system_default",
                "hf_cache_dir": "huggingface",
                "unique_id": "42",
            }
        )
        with patch("xdit_comfyui.worker._evict_loader_worker") as evict:
            result = _clear_loader_cache("42")
        self.assertTrue(result["ok"])
        self.assertTrue(result["evicted"])
        evict.assert_called_once()

        with patch("xdit_comfyui.api._available_gpu_count", return_value=4):
            payload = _preview_payload(
                {
                    "model": "black-forest-labs/FLUX.1-dev",
                    "gpu_count": "4",
                    "gpu_device_ids": "0",
                    "cache_method": False,
                    "attention_backend": False,
                    "residual_diff_threshold": "none",
                    "scm_policy": -1,
                    "num_hybrid_attn_high_precision_steps": "",
                }
            )
        self.assertIn("runtime", payload)
        self.assertEqual(payload["preset_meta"]["selected"], PRESET_NONE)

    def test_sanitize_resets_invalid_cache_fields(self):
        cleaned = _sanitize_loader_preview_body(
            {
                "cache_method": False,
                "residual_diff_threshold": "none",
                "scm_policy": -1,
            }
        )
        self.assertNotEqual(cleaned["cache_method"], False)
        self.assertNotEqual(cleaned["residual_diff_threshold"], "none")
        self.assertNotEqual(cleaned["scm_policy"], -1)


if __name__ == "__main__":
    unittest.main()
