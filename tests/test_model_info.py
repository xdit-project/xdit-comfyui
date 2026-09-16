import unittest
from types import SimpleNamespace
from unittest import mock

import pytest

pytestmark = pytest.mark.usefixtures("synthetic_preset_catalog")

from xdit_comfyui.model_info import (
    align_generation_resolution,
    gemm_precision_choices_for_model,
    loader_widget_gates,
    model_cache_preset_defaults,
    model_cache_transformers,
    model_capabilities,
    model_generation_defaults,
    model_resolution_grid,
    model_supports_distilled_weights,
    model_supports_step_cache,
    resolve_model_task,
    sanitize_loader_cache_widgets,
    sanitize_runtime_cache_for_model,
    validate_model_task,
    validate_runtime_for_model,
    widget_capability_gates,
)
from xdit_comfyui.runtime_config import _runtime_loader_gpu_choices


class ModelInfoTest(unittest.TestCase):
    def test_gpu_defaults_when_visible(self):
        choices = _runtime_loader_gpu_choices()
        self.assertEqual(choices[0], 1)
        self.assertGreaterEqual(len(choices), 1)

    def test_flux_capabilities(self):
        caps = model_capabilities("FLUX.1-dev")
        self.assertTrue(caps.get("use_fp8_gemms"))
        gates = widget_capability_gates("FLUX.1-dev")
        self.assertTrue(gates.get("gemm_precision"))
        self.assertTrue(gates.get("use_parallel_vae"))

    def test_cache_methods_come_from_runner_settings(self):
        from xdit_comfyui.model_info import _supported_cache_methods

        caps = SimpleNamespace(supports_step_caching=True)
        settings = SimpleNamespace(step_cache_config={"teacache": None, "dbcache": object()})
        self.assertEqual(
            _supported_cache_methods(caps, settings),
            ("teacache", "dbcache"),
        )

    def test_cache_methods_stay_disabled_when_capability_is_false(self):
        from xdit_comfyui.model_info import _supported_cache_methods

        caps = SimpleNamespace(supports_step_caching=False)
        settings = SimpleNamespace(step_cache_config={"dbcache": object()})
        self.assertEqual(_supported_cache_methods(caps, settings), ())

    def test_z_image_model_cache_defaults(self):
        defaults = model_cache_preset_defaults("Tongyi-MAI/Z-Image", "dbcache")
        self.assertEqual(defaults.get("Fn_compute_blocks"), 3)
        self.assertEqual(defaults.get("residual_diff_threshold"), 0.12)
        self.assertEqual(defaults.get("scm_policy"), "ultra")

    def test_every_runner_in_the_registry_is_selectable(self):
        """Grouping by the class-level model_name hid Wan 2.2's A14B experts entirely."""
        import xfuser.model_executor.models.runner_models  # noqa: F401
        from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

        from xdit_comfyui.runtime_config import xdit_model_choices

        choices = set(xdit_model_choices())
        runners = {MODEL_REGISTRY[key] for key in choices if key in MODEL_REGISTRY}
        self.assertEqual(runners, set(MODEL_REGISTRY.values()))

    def test_offered_models_are_names_xfuser_can_resolve(self):
        """`--model` is looked up in the registry by exact key, with no name fallback."""
        import xfuser.model_executor.models.runner_models  # noqa: F401
        from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

        from xdit_comfyui.runtime_config import CUSTOM_MODEL_SENTINEL, xdit_model_choices

        for choice in xdit_model_choices():
            if choice == CUSTOM_MODEL_SENTINEL:
                continue
            self.assertIn(choice, MODEL_REGISTRY)

    def test_the_models_a_preset_names_can_be_picked_by_hand(self):
        from xdit_comfyui.presets import load_benchmark_presets
        from xdit_comfyui.runtime_config import xdit_model_choices

        choices = set(xdit_model_choices())
        named = {preset.model for preset in load_benchmark_presets() if preset.model}
        self.assertTrue(named)
        self.assertEqual(named - choices, set())

    def test_switching_to_a_single_denoiser_model_clears_the_second_groups_values(self):
        """Left behind, they would override a denoiser the new model does not have."""
        from xdit_comfyui.model_info import sanitize_loader_cache_widgets

        sanitized = sanitize_loader_cache_widgets(
            "Tongyi-MAI/Z-Image",
            {"cache_method": "dbcache", "t2_max_warmup_steps": 6},
        )
        self.assertNotEqual(sanitized["t2_max_warmup_steps"], 6)

    def test_a_two_denoiser_model_keeps_the_second_groups_values(self):
        from xdit_comfyui.model_info import sanitize_loader_cache_widgets

        sanitized = sanitize_loader_cache_widgets(
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            {"cache_method": "dbcache", "t2_max_warmup_steps": 6},
        )
        self.assertEqual(sanitized["t2_max_warmup_steps"], 6)

    def test_no_model_caches_more_denoisers_than_there_are_widget_groups(self):
        """Raise MAX_CACHED_TRANSFORMERS and add the group when this fails."""
        from xdit_comfyui.runner_contract import MAX_CACHED_TRANSFORMERS
        from xdit_comfyui.runtime_config import (
            CUSTOM_MODEL_SENTINEL,
            xdit_model_choices,
        )

        for model in xdit_model_choices():
            if model == CUSTOM_MODEL_SENTINEL:
                continue
            for method in ("dbcache", "teacache", "fbcache"):
                self.assertLessEqual(
                    len(model_cache_transformers(model, method)),
                    MAX_CACHED_TRANSFORMERS,
                    f"{model} caches more denoisers than the node can show",
                )

    def test_single_transformer_model_reports_one_cache_entry(self):
        entries = model_cache_transformers("black-forest-labs/FLUX.1-dev", "dbcache")
        self.assertEqual([entry["transformer"] for entry in entries], ["transformer"])

    def test_wan22_reports_per_transformer_cache_defaults(self):
        """Wan 2.2's refiner warms for fewer steps than its high-noise denoiser."""
        entries = model_cache_transformers("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "dbcache")
        self.assertEqual(
            [entry["transformer"] for entry in entries], ["transformer", "transformer_2"]
        )
        warmups = [entry["defaults"]["max_warmup_steps"] for entry in entries]
        self.assertEqual(warmups, [4, 2])
        # Read from class settings this collapses to one wrong shared value.
        self.assertNotEqual(warmups[0], warmups[1])

    def test_generation_defaults_come_from_the_runner(self):
        """Each model's own numbers, not one hardcoded set for every model."""
        qwen = model_generation_defaults("Qwen/Qwen-Image")
        self.assertEqual((qwen["height"], qwen["width"]), (928, 1664))
        self.assertEqual(qwen["num_inference_steps"], 50)

        flux = model_generation_defaults("black-forest-labs/FLUX.1-dev")
        self.assertEqual((flux["height"], flux["width"]), (1024, 1024))
        self.assertEqual(flux["max_sequence_length"], 512)

        wan = model_generation_defaults("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
        self.assertEqual(wan["num_frames"], 81)
        self.assertEqual(wan["flow_shift"], 12)

    def test_generation_defaults_empty_for_unknown_model(self):
        self.assertEqual(model_generation_defaults("someone/not-in-the-registry"), {})

    def test_resolution_grid_prefers_divisor_then_mod_value(self):
        self.assertEqual(model_resolution_grid("Lightricks/LTX-2")["divisor"], 64)
        self.assertEqual(model_resolution_grid("Lightricks/LTX-2")["step"], 64)
        self.assertEqual(model_resolution_grid("Wan-AI/Wan2.2-TI2V-5B-Diffusers")["step"], 32)
        # No declared grid: the VAE's 8-pixel granularity.
        grid = model_resolution_grid("black-forest-labs/FLUX.1-dev")
        self.assertEqual((grid["step"], grid["divisor"]), (8, None))

    def test_enforced_divisor_snaps_the_request(self):
        """LTX raises after the weights are loaded, so align before the run."""
        self.assertEqual(align_generation_resolution("Lightricks/LTX-2", 1000, 1536), (1024, 1536))
        self.assertEqual(align_generation_resolution("Lightricks/LTX-2", 1, 1), (64, 64))
        # A model without an enforced divisor keeps the exact request.
        self.assertEqual(
            align_generation_resolution("black-forest-labs/FLUX.1-dev", 1000, 1000),
            (1000, 1000),
        )

    def test_cache_preset_defaults_track_the_first_transformer(self):
        entries = model_cache_transformers("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "dbcache")
        defaults = model_cache_preset_defaults("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "dbcache")
        self.assertEqual(defaults, entries[0]["defaults"])

    def test_cache_transformers_empty_without_a_cache_method(self):
        self.assertEqual(model_cache_transformers("black-forest-labs/FLUX.1-dev", "none"), [])
        self.assertEqual(model_cache_transformers("black-forest-labs/FLUX.1-dev", None), [])

    def test_z_image_turbo_disables_step_cache(self):
        self.assertFalse(model_supports_step_cache("Tongyi-MAI/Z-Image-Turbo"))
        gates = widget_capability_gates("Tongyi-MAI/Z-Image-Turbo")
        self.assertFalse(gates.get("cache_method"))
        self.assertFalse(gates.get("STEP CACHE"))
        self.assertFalse(gates.get("cross_attention_backend"))

    def test_sanitize_runtime_for_model_drops_unsupported_null_cross_attention(self):
        from xdit_comfyui.model_info import sanitize_runtime_for_model

        runtime = {
            "model": "Tongyi-MAI/Z-Image-Turbo",
            "cross_attention_backend": None,
            "attention_backend": "aiter_flydsl",
        }
        sanitize_runtime_for_model("Tongyi-MAI/Z-Image-Turbo", runtime)
        self.assertNotIn("cross_attention_backend", runtime)

    def test_z_image_turbo_cannot_smuggle_cfg_parallel_into_the_worker(self):
        from xdit_comfyui.model_info import sanitize_runtime_for_model

        runtime = {"model": "Tongyi-MAI/Z-Image-Turbo", "use_cfg_parallel": True}
        sanitize_runtime_for_model("Tongyi-MAI/Z-Image-Turbo", runtime)
        self.assertFalse(runtime["use_cfg_parallel"])

    def test_vae_tiling_survives_for_a_runner_that_declares_it(self):
        """Qwen-Image keeps tiling enabled when its runner advertises the capability."""
        from xdit_comfyui.model_info import (
            model_capabilities,
            sanitize_loader_inputs_for_model,
            sanitize_runtime_for_model,
        )

        model = "Qwen/Qwen-Image-2512"
        self.assertTrue(model_capabilities(model).get("enable_tiling"))

        inputs = {"model": model, "enable_tiling": True, "enable_slicing": True}
        sanitize_loader_inputs_for_model(model, inputs)
        self.assertTrue(inputs["enable_tiling"])
        self.assertTrue(inputs["enable_slicing"])

        runtime = {"model": model, "enable_tiling": True}
        sanitize_runtime_for_model(model, runtime)
        self.assertTrue(runtime["enable_tiling"])

    def test_distilled_weights_only_enabled_for_wan_distilled_model(self):
        self.assertTrue(model_supports_distilled_weights("Wan2.2-Distilled-I2V"))
        distilled_gates = widget_capability_gates("Wan2.2-Distilled-I2V")
        self.assertTrue(distilled_gates.get("DISTILLED WEIGHTS"))
        self.assertTrue(distilled_gates.get("distilled_transformer_path"))

        self.assertFalse(model_supports_distilled_weights("black-forest-labs/FLUX.1-dev"))
        flux_gates = widget_capability_gates("black-forest-labs/FLUX.1-dev")
        self.assertFalse(flux_gates.get("DISTILLED WEIGHTS"))
        self.assertFalse(flux_gates.get("distilled_transformer_path"))

    def test_a_model_that_cannot_cache_hides_every_denoiser_group(self):
        """Without a gate of its own the second group would stay on show for it."""
        gates = loader_widget_gates("Tongyi-MAI/Z-Image-Turbo")
        self.assertFalse(gates["STEP CACHE"])
        self.assertFalse(gates["STEP CACHE · DENOISER 2"])

    def test_loader_widget_gates_disable_dbcache_fields_for_teacache(self):
        gates = loader_widget_gates("FLUX.1-dev", "teacache")
        self.assertTrue(gates["cache_method"])
        self.assertTrue(gates["residual_diff_threshold"])
        self.assertFalse(gates["Fn_compute_blocks"])
        self.assertFalse(gates["scm_policy"])

    def test_sanitize_runtime_cache_for_unsupported_model(self):
        runtime = {
            "model": "Tongyi-MAI/Z-Image-Turbo",
            "cache_method": "dbcache",
            "cache_config": '{"Fn_compute_blocks": 3}',
        }
        sanitize_runtime_cache_for_model("Tongyi-MAI/Z-Image-Turbo", runtime)
        self.assertIsNone(runtime.get("cache_method"))
        self.assertNotIn("cache_config", runtime)

    def test_sanitize_loader_cache_widgets_for_unsupported_model(self):
        from xdit_comfyui.runner_contract import default_loader_widget_values

        widgets = sanitize_loader_cache_widgets(
            "Tongyi-MAI/Z-Image-Turbo",
            {**default_loader_widget_values(), "cache_method": "dbcache", "Fn_compute_blocks": 8},
        )
        self.assertEqual(widgets["cache_method"], "none")
        self.assertEqual(
            widgets["Fn_compute_blocks"],
            default_loader_widget_values()["Fn_compute_blocks"],
        )

    def test_validate_model_task_rejects_task_on_flux(self):
        with self.assertRaises(ValueError) as ctx:
            validate_model_task("black-forest-labs/FLUX.2-dev", "i2v")
        self.assertIn("does not support pipeline tasks", str(ctx.exception))

    def test_validate_model_task_accepts_wan_i2v(self):
        validate_model_task("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "i2v")

    def test_resolve_model_task_selects_only_valid_task(self):
        with mock.patch(
            "xdit_comfyui.model_info.model_capabilities",
            return_value={"valid_tasks": ["i2v"]},
        ):
            self.assertEqual(resolve_model_task("model", ""), "i2v")

    def test_resolve_model_task_requires_multitask_choice(self):
        with mock.patch(
            "xdit_comfyui.model_info.model_capabilities",
            return_value={"valid_tasks": ["i2v", "t2v"]},
        ):
            with self.assertRaisesRegex(ValueError, "requires a task"):
                resolve_model_task("model", "")

    def test_custom_model_task_is_not_rejected_without_capabilities(self):
        with mock.patch(
            "xdit_comfyui.model_info.model_capabilities",
            return_value={},
        ):
            self.assertEqual(resolve_model_task("custom/model", "custom-task"), "custom-task")

    def test_rocm_filters_int8_gemm(self):
        with (
            mock.patch(
                "xdit_comfyui.model_info.model_capabilities",
                return_value={"has_int8_modules": True},
            ),
            mock.patch(
                "xdit_comfyui.model_info._is_rocm_runtime",
                return_value=True,
            ),
        ):
            self.assertEqual(gemm_precision_choices_for_model("model"), ["native", "fp8", "fp4"])

    def test_runtime_rejects_multiple_gemm_modes(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_runtime_for_model(
                "model",
                {"use_fp8_gemms": True, "use_fp4_gemms": True},
            )

    def test_runtime_rejects_multiple_cpu_offload_modes(self):
        with self.assertRaisesRegex(ValueError, "CPU offload modes are mutually exclusive"):
            validate_runtime_for_model(
                "model",
                {
                    "enable_model_cpu_offload": True,
                    "enable_group_cpu_offload": True,
                },
            )


if __name__ == "__main__":
    unittest.main()
