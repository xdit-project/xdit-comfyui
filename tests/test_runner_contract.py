import json
import unittest

import pytest

pytestmark = pytest.mark.usefixtures("synthetic_preset_catalog")

from xdit_comfyui.presets import (
    BenchmarkPreset,
    preset_by_name,
    preset_to_loader_widgets,
    resolve_preset_choice,
)
from xdit_comfyui.runner_contract import (
    SAMPLE_VAE_DESTS,
    cache_config_baseline,
    cache_config_fields_for_method,
    cache_config_widget_names,
    cache_config_widget_spec,
    cache_method_widget_gates,
    default_loader_widget_values,
    effective_cache_config_per_transformer,
    loader_config_dests,
    loader_config_widget_names,
    loader_display_widgets,
    loader_schema,
    loader_widget_groups,
    loader_widget_spec,
    preset_args_to_runtime,
    resolve_effective_cache_config,
    runner_cli_dests,
    runtime_from_loader_widgets,
    sanitize_attention_runtime,
    sanitize_cache_config_runtime,
)
from xdit_comfyui.runtime_config import _runtime_loader_input_types

_WAN22 = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"


class RunnerContractTest(unittest.TestCase):
    def test_loader_keeps_loader_owned_vae_widgets(self):
        dests = set(loader_config_dests())
        for key in (
            "cpu_offload_mode",
            "use_parallel_vae",
            "use_vae_channels_last_format",
            "cross_attention_backend",
        ):
            self.assertIn(key, dests)
        for key in SAMPLE_VAE_DESTS:
            self.assertNotIn(key, dests)

    def test_every_runner_arg_is_either_a_widget_or_explicitly_excluded(self):
        """A new xdit CLI arg must reach the UI without a code change here."""
        from xdit_comfyui.runner_contract import (
            _LOADER_EXCLUDED_DESTS,
            deprecated_runner_dests,
        )

        unaccounted = (
            set(runner_cli_dests())
            - set(loader_config_dests())
            - _LOADER_EXCLUDED_DESTS
            - deprecated_runner_dests()
        )
        self.assertEqual(unaccounted, set())

    def test_a_new_runner_arg_becomes_a_widget_and_an_input(self):
        from unittest import mock

        from xdit_comfyui import runner_contract

        action = mock.Mock(dest="brand_new_knob", default=3, choices=None, help="A new runner arg.")
        action.type = int
        with (
            mock.patch.object(
                runner_contract,
                "runner_cli_dests",
                return_value=frozenset({*runner_cli_dests(), "brand_new_knob"}),
            ),
            mock.patch.object(
                runner_contract,
                "_runner_actions",
                return_value={**runner_contract._runner_actions(), "brand_new_knob": action},
            ),
        ):
            runner_contract.loader_config_dests.cache_clear()
            runner_contract.loader_config_widget_names.cache_clear()
            try:
                self.assertIn("brand_new_knob", runner_contract.loader_config_dests())
                self.assertIn("brand_new_knob", runner_contract.loader_config_input_types())
                other = next(
                    group
                    for group in runner_contract.loader_schema()["widget_groups"]
                    if group["id"] == "other"
                )
                self.assertIn("brand_new_knob", other["widgets"])
            finally:
                runner_contract.loader_config_dests.cache_clear()
                runner_contract.loader_config_widget_names.cache_clear()

    def test_a_missing_xfuser_fails_loudly_instead_of_emptying_the_nodes(self):
        """An empty introspection would render dead text boxes and pass no CLI args."""
        from unittest import mock

        from xdit_comfyui import runner_contract

        broken = {"xfuser.config.args": None}
        with mock.patch.dict("sys.modules", broken):
            runner_contract._runner_arg_parser.cache_clear()
            try:
                with self.assertRaises(RuntimeError) as caught:
                    runner_contract._runner_arg_parser()
            finally:
                runner_contract._runner_arg_parser.cache_clear()
        self.assertIn("xfuser", str(caught.exception))

    def test_a_renamed_runner_arg_fails_instead_of_becoming_a_text_box(self):
        with self.assertRaises(KeyError):
            loader_widget_spec("an_arg_the_cli_no_longer_has")

    def test_loader_widget_names_include_cache_fields(self):
        names = loader_config_widget_names()
        self.assertIn("residual_diff_threshold", names)
        self.assertIn("Fn_compute_blocks", names)
        self.assertNotIn("cache_config_json", names)
        self.assertIn("cpu_offload_mode", names)
        for key in ("profile", "profile_wait", "profile_warmup", "profile_active"):
            self.assertNotIn(key, names)

    def test_profile_fields_are_not_exposed_in_comfyui(self):
        schema = loader_schema()
        self.assertNotIn("profiling", {group["id"] for group in schema["widget_groups"]})

    def test_new_runner_flags_live_in_their_named_groups(self):
        groups = {group["id"]: set(group["widgets"]) for group in loader_schema()["widget_groups"]}
        self.assertIn("text_encoder_tp_degree", groups["parallelism"])
        self.assertIn("memory_efficient_replicated_load", groups["memory"])
        self.assertIn("group_offload_low_cpu_mem", groups["memory"])
        self.assertEqual(groups["vae"], {"use_parallel_vae", "use_vae_channels_last_format"})
        self.assertIn("use_fp8_text_encoder", groups["quantization"])
        self.assertIn("vsa_drop_rates", groups["attention"])

    def test_model_groups_match_comfyui_user_tasks(self):
        groups = {group["id"]: set(group["widgets"]) for group in loader_schema()["widget_groups"]}
        self.assertEqual(groups["model_cache"], {"hf_cache_mode", "hf_cache_dir"})
        self.assertNotIn("residency", set().union(*groups.values()))
        self.assertNotIn("use_torch_compile", set().union(*groups.values()))
        self.assertEqual(
            next(
                group["label"]
                for group in loader_schema()["widget_groups"]
                if group["id"] == "quantization"
            ),
            "GEMM PRECISION",
        )

    def test_vsa_drop_rates_round_trip_as_a_float_list(self):
        from xdit_comfyui.runner_contract import (
            runtime_value_to_widget,
            widget_value_to_runtime,
        )

        self.assertEqual(widget_value_to_runtime("vsa_drop_rates", "0.1, 0.3"), [0.1, 0.3])
        self.assertEqual(runtime_value_to_widget("vsa_drop_rates", [0.1, 0.3]), "0.1 0.3")
        self.assertEqual(loader_widget_spec("vsa_drop_rates")[0], "STRING")

    def test_taylorseer_is_an_optional_boolean_cache_override(self):
        from xdit_comfyui.runner_contract import (
            _cache_config_to_widgets,
            _cache_override_value,
        )

        self.assertEqual(loader_widget_spec("enable_taylorseer")[0], ["default", "true", "false"])
        self.assertIsNone(_cache_override_value("enable_taylorseer", "default"))
        self.assertTrue(_cache_override_value("enable_taylorseer", "true"))
        self.assertEqual(
            _cache_config_to_widgets('{"enable_taylorseer": false}')["enable_taylorseer"],
            "false",
        )

    def test_preset_preserves_upstream_tiling(self):
        preset = preset_by_name("flux2.i2i_2k.1gpu.rdna4")
        self.assertIsNotNone(preset)
        runtime = preset_args_to_runtime(preset.args)
        self.assertTrue(runtime["enable_tiling"])
        self.assertNotIn("warmup_calls", runtime)

    def test_distilled_weights_group_has_wan_tooltip(self):
        schema = loader_schema()
        self.assertIn("distilled_weights", {group["id"] for group in schema["widget_groups"]})
        distilled = next(
            group for group in schema["widget_groups"] if group["id"] == "distilled_weights"
        )
        self.assertIn("LightX2V", distilled["description"])

    def test_every_model_setting_says_what_it_is_for(self):
        """A widget with no tooltip is a knob the user can only learn by breaking a run."""
        from xdit_comfyui.runner_contract import (
            loader_config_widget_names,
            loader_widget_spec,
        )

        undocumented = [
            name
            for name in loader_config_widget_names()
            if not (loader_widget_spec(name)[1].get("tooltip") or "").strip()
        ]
        self.assertEqual([], undocumented)

    def test_the_settings_worth_advice_carry_it_alongside_the_cli_help(self):
        """The runner's help names the argument; this says which way to turn it."""
        from xdit_comfyui.runner_contract import loader_widget_spec

        tooltip = loader_widget_spec("fully_shard_degree")[1]["tooltip"]
        self.assertIn("Fully sharding", tooltip, "the runner's own help was dropped")
        self.assertIn("save VRAM", tooltip)
        # The second denoiser's copies describe the same setting.
        self.assertEqual(
            loader_widget_spec("residual_diff_threshold")[1]["tooltip"],
            loader_widget_spec("t2_residual_diff_threshold")[1]["tooltip"],
        )

    def test_named_preset_runtime_includes_upstream_tiling(self):
        resolved = resolve_preset_choice(
            "flux2.i2i_2k.1gpu.rdna4",
            "black-forest-labs/FLUX.2-dev",
            1,
            gpu_tags=("gfx1201", "rdna4"),
        )
        self.assertIsNotNone(resolved)
        self.assertTrue(resolved.runtime["enable_tiling"])

    def test_optional_step_widgets_allow_zero(self):

        required = _runtime_loader_input_types()["required"]
        for name in (
            "num_hybrid_attn_high_precision_steps",
            "num_hybrid_gemm_high_precision_steps",
        ):
            spec = required[name][1]
            self.assertEqual(spec["default"], 0)
            self.assertEqual(spec["min"], 0)

    def test_loader_input_types_include_native_group_toggles(self):
        """Native schema toggles keep layout, theme, and tooltips owned by ComfyUI."""
        from xdit_comfyui.runner_contract import loader_expand_widget_names

        required = _runtime_loader_input_types()["required"]
        for name in loader_expand_widget_names():
            self.assertIn(name, required)
            self.assertEqual(required[name][0], "BOOLEAN")
        self.assertIn("ulysses_degree", required)

    def test_loader_schema_includes_expand_widget_metadata(self):
        schema = loader_schema()
        parallelism = next(
            group for group in schema["widget_groups"] if group["id"] == "parallelism"
        )
        self.assertEqual(parallelism["expand_widget"], "PARALLELISM")

    def test_loader_widget_groups_cover_runner_config(self):
        grouped = set()
        for group in loader_widget_groups():
            grouped.update(group["widgets"])
        for widget in loader_config_widget_names():
            self.assertIn(widget, grouped)
        self.assertNotIn("warmup_calls", loader_config_widget_names())
        schema = loader_schema()
        self.assertIn("parallelism", {group["id"] for group in schema["widget_groups"]})
        self.assertNotIn("execution_widget_order", schema)

    def test_every_group_names_the_toggle_the_browser_should_build(self):
        from xdit_comfyui.runner_contract import loader_expand_widget_names

        schema = loader_schema()
        for group in schema["widget_groups"]:
            self.assertEqual(group["expand_widget"], group["label"])
            self.assertIn("collapsed", group)
        self.assertEqual(
            set(loader_expand_widget_names()) - {"Other"},
            {group["label"] for group in schema["widget_groups"]} - {"Other"},
        )

    def test_compound_widgets_translate_to_runner_flags(self):
        runtime = runtime_from_loader_widgets(
            {
                **default_loader_widget_values(),
                "cpu_offload_mode": "group",
                "gemm_precision": "fp4",
            }
        )
        self.assertTrue(runtime["enable_group_cpu_offload"])
        self.assertFalse(runtime["enable_model_cpu_offload"])
        self.assertTrue(runtime["use_fp4_gemms"])
        self.assertFalse(runtime["use_fp8_gemms"])

    def test_schema_exposes_combo_options(self):
        schema = loader_schema()
        self.assertEqual(
            schema["combo_options"]["cpu_offload_mode"],
            ["none", "model", "sequential", "group"],
        )

    def test_strip_undeclared_loader_keys_preserves_native_group_toggles(self):
        from xdit_comfyui.runner_contract import strip_undeclared_loader_keys

        inputs = {
            "model": "FLUX.1-dev",
            "PARALLELISM": True,
            "gpu_count": 4,
            "ulysses_degree": 2,
        }
        strip_undeclared_loader_keys(inputs)
        self.assertEqual(
            inputs,
            {"model": "FLUX.1-dev", "PARALLELISM": True, "ulysses_degree": 2},
        )

    def test_preset_cache_config_round_trips_to_widgets(self):
        preset = BenchmarkPreset(
            name="synthetic-cache-preset",
            tags=("gfx950",),
            model="Qwen/Qwen-Image",
            gpu_count=1,
            args={
                "cache_method": "dbcache",
                "cache_config": '{"residual_diff_threshold": 0.123}',
            },
            source_file="synthetic.yaml",
        )
        runtime = preset_args_to_runtime(preset.args)
        self.assertEqual(
            json.loads(runtime["cache_config"]),
            {"residual_diff_threshold": 0.123},
        )
        widgets = preset_to_loader_widgets(preset)
        self.assertEqual(widgets["residual_diff_threshold"], 0.123)
        self.assertNotIn("warmup_calls", widgets)

    def test_runtime_from_loader_widgets_builds_cache_config_json(self):
        from xdit_comfyui.runner_contract import runtime_from_loader_widgets

        runtime = runtime_from_loader_widgets(
            {
                **default_loader_widget_values(),
                "cache_method": "dbcache",
                "residual_diff_threshold": 0.1,
            }
        )
        self.assertEqual(runtime["cache_method"], "dbcache")
        self.assertEqual(json.loads(runtime["cache_config"]), {"residual_diff_threshold": 0.1})

    def test_every_loader_widget_has_a_usable_default(self):
        """A widget with no default reaches the prompt as null and fails validation."""
        defaults = default_loader_widget_values()
        for name in loader_config_widget_names():
            self.assertIn(name, defaults, f"{name} has no default")
            self.assertIsNotNone(defaults[name], f"{name} defaults to null")

    def test_each_denoiser_group_offers_the_same_fields_as_the_first(self):
        from xdit_comfyui.runner_contract import denoiser_cache_widget_field

        groups = {group["id"]: group for group in loader_widget_groups()}
        first = [name for name in groups["cache"]["widgets"] if name != "cache_method"]
        for group_id, group in groups.items():
            if not group_id.startswith("cache_denoiser_"):
                continue
            fields = [denoiser_cache_widget_field(name)[1] for name in group["widgets"]]
            self.assertEqual(fields, first, f"{group_id} drifted from the first group")

    def test_a_second_denoiser_starts_from_its_own_cache_preset(self):
        from xdit_comfyui.runner_contract import cache_widget_defaults_for_model

        widgets = cache_widget_defaults_for_model(_WAN22, "dbcache")
        self.assertEqual(widgets["max_warmup_steps"], 4)
        self.assertEqual(widgets["t2_max_warmup_steps"], 2)

    def test_tuning_the_second_denoiser_only_moves_that_one(self):
        from xdit_comfyui.runner_contract import (
            cache_widget_defaults_for_model,
            runtime_from_loader_widgets,
        )

        widgets = {
            **default_loader_widget_values(),
            **cache_widget_defaults_for_model(_WAN22, "dbcache"),
            "model": _WAN22,
            "cache_method": "dbcache",
        }
        self.assertIsNone(
            json.loads(runtime_from_loader_widgets(widgets).get("cache_config", "null")),
            "untouched widgets must not override the model's own presets",
        )

        widgets["t2_max_warmup_steps"] = 6
        runtime = runtime_from_loader_widgets(widgets)
        self.assertEqual(
            json.loads(runtime["cache_config"]),
            {"per_transformer": {"transformer_2": {"max_warmup_steps": 6}}},
        )

    def test_a_broadcast_edit_leaves_the_second_denoiser_where_the_widget_shows_it(self):
        """The first group reaches every denoiser, so the second has to pin its own value."""
        from xdit_comfyui.runner_contract import (
            cache_widget_defaults_for_model,
            runtime_from_loader_widgets,
        )

        widgets = {
            **default_loader_widget_values(),
            **cache_widget_defaults_for_model(_WAN22, "dbcache"),
            "model": _WAN22,
            "cache_method": "dbcache",
            "max_warmup_steps": 10,
        }
        self.assertEqual(
            json.loads(runtime_from_loader_widgets(widgets)["cache_config"]),
            {
                "max_warmup_steps": 10,
                "per_transformer": {"transformer_2": {"max_warmup_steps": 2}},
            },
        )

    def test_a_single_denoiser_model_emits_no_per_transformer_section(self):
        from xdit_comfyui.runner_contract import runtime_from_loader_widgets

        runtime = runtime_from_loader_widgets(
            {
                **default_loader_widget_values(),
                "model": "Tongyi-MAI/Z-Image",
                "cache_method": "dbcache",
                "t2_max_warmup_steps": 6,
            }
        )
        self.assertNotIn("per_transformer", runtime.get("cache_config") or "")

    def test_xfuser_never_sees_the_per_denoiser_section(self):
        """xfuser builds one flat cache config at init and would reject the extra key."""
        from xdit_comfyui.runtime_config import _build_cli_args
        from xdit_comfyui.worker_payload import loader_init_config

        cache_config = json.dumps(
            {
                "max_warmup_steps": 10,
                "per_transformer": {"transformer_2": {"max_warmup_steps": 2}},
            }
        )
        args = _build_cli_args({"model": _WAN22, "cache_config": cache_config})
        self.assertEqual(
            json.loads(args[args.index("--cache_config") + 1]), {"max_warmup_steps": 10}
        )

        init = loader_init_config(
            {"model": _WAN22, "cache_method": "dbcache", "cache_config": cache_config}
        )
        self.assertEqual(json.loads(init["cache_config"]), {"max_warmup_steps": 10})

    def test_the_run_payload_keeps_the_per_denoiser_section(self):
        """It is applied by the step-cache patch once the transformers exist."""
        from xdit_comfyui.worker_payload import worker_config_payload

        cache_config = json.dumps({"per_transformer": {"transformer_2": {"max_warmup_steps": 6}}})
        payload = worker_config_payload({"model": _WAN22, "cache_config": cache_config})
        self.assertEqual(json.loads(payload["cache_config"]), json.loads(cache_config))

    def test_loader_display_widgets_uses_model_cache_defaults(self):
        from xdit_comfyui.runner_contract import loader_display_widgets

        widgets = loader_display_widgets(
            {
                "model": "Tongyi-MAI/Z-Image",
                "cache_method": "dbcache",
            },
            "Tongyi-MAI/Z-Image",
        )
        self.assertEqual(widgets["Fn_compute_blocks"], 3)
        self.assertEqual(widgets["residual_diff_threshold"], 0.12)
        self.assertEqual(widgets["scm_policy"], "ultra")

    def test_loader_display_widgets_keeps_explicit_preset_overrides(self):
        baseline = loader_display_widgets(
            {
                "model": "Tongyi-MAI/Z-Image",
                "cache_method": "dbcache",
            },
            "Tongyi-MAI/Z-Image",
        )
        widgets = loader_display_widgets(
            {
                "model": "Tongyi-MAI/Z-Image",
                "cache_method": "dbcache",
                "cache_config": {"residual_diff_threshold": 0.1},
            },
            "Tongyi-MAI/Z-Image",
        )
        self.assertEqual(widgets["Fn_compute_blocks"], 3)
        self.assertEqual(widgets["residual_diff_threshold"], 0.1)
        self.assertEqual(widgets["scm_policy"], baseline["scm_policy"])

    def test_float_widget_specs_carry_step_fine_enough_for_preset_values(self):
        """ComfyUI rounds float widgets to the precision implied by step."""
        for spec in (
            cache_config_widget_spec("residual_diff_threshold"),
            loader_widget_spec("spargeattn_cdfthreshold"),
        ):
            self.assertEqual(spec[0], "FLOAT")
            self.assertLessEqual(spec[1]["step"], 0.001)

    def test_preset_to_loader_widgets_merges_model_cache_defaults(self):
        preset = preset_by_name("z_image.1gpu.rdna4")
        self.assertIsNotNone(preset)
        widgets = preset_to_loader_widgets(preset)
        self.assertEqual(widgets["cache_method"], "none")
        self.assertEqual(widgets["residual_diff_threshold"], 0.08)
        self.assertEqual(widgets["scm_policy"], "default")

    def test_resolve_effective_cache_config_merge_order(self):
        merged = resolve_effective_cache_config(
            "Tongyi-MAI/Z-Image",
            "dbcache",
            {"residual_diff_threshold": 0.1},
        )
        self.assertEqual(merged["Fn_compute_blocks"], 3)
        self.assertEqual(merged["residual_diff_threshold"], 0.1)
        self.assertEqual(merged["scm_policy"], "ultra")

    def test_per_transformer_config_keeps_each_denoisers_schedule(self):
        rows = effective_cache_config_per_transformer("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "dbcache")
        self.assertEqual([row["transformer"] for row in rows], ["transformer", "transformer_2"])
        self.assertEqual([row["config"]["max_warmup_steps"] for row in rows], [4, 2])

    def test_an_override_lands_on_every_transformer(self):
        """`--cache_config` is one JSON object xFuser broadcasts, so the rows say so."""
        rows = effective_cache_config_per_transformer(
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "dbcache",
            {"max_warmup_steps": 6},
        )
        self.assertEqual([row["config"]["max_warmup_steps"] for row in rows], [6, 6])

    def test_a_single_transformer_model_reports_one_row(self):
        rows = effective_cache_config_per_transformer("Tongyi-MAI/Z-Image", "dbcache")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["config"], cache_config_baseline("Tongyi-MAI/Z-Image", "dbcache"))

    def test_cache_config_baseline_uses_model_defaults_not_global(self):
        baseline = cache_config_baseline("Tongyi-MAI/Z-Image", "dbcache")
        self.assertEqual(baseline["Fn_compute_blocks"], 3)
        self.assertNotEqual(baseline["Fn_compute_blocks"], 8)

    def test_runtime_from_loader_widgets_omits_model_default_cache_fields(self):
        runtime = runtime_from_loader_widgets(
            {
                **default_loader_widget_values(),
                "model": "Tongyi-MAI/Z-Image",
                "cache_method": "dbcache",
                "Fn_compute_blocks": 3,
                "Bn_compute_blocks": 0,
                "residual_diff_threshold": 0.12,
                "max_warmup_steps": 8,
                "max_cached_steps": -1,
                "scm_policy": "ultra",
            },
            registry_model="Tongyi-MAI/Z-Image",
        )
        self.assertEqual(runtime["cache_method"], "dbcache")
        self.assertNotIn("cache_config", runtime)

    def test_runtime_from_loader_widgets_keeps_widget_value_matching_global_default(self):
        """A widget value is an override even when it equals the DBCachePreset default."""
        runtime = runtime_from_loader_widgets(
            {
                **default_loader_widget_values(),
                "model": "Tongyi-MAI/Z-Image",
                "cache_method": "dbcache",
                "residual_diff_threshold": 0.08,
                "Fn_compute_blocks": 6,
                "scm_policy": "ultra",
            },
            registry_model="Tongyi-MAI/Z-Image",
        )
        self.assertEqual(runtime["cache_method"], "dbcache")
        self.assertEqual(json.loads(runtime["cache_config"])["residual_diff_threshold"], 0.08)
        widgets = loader_display_widgets(runtime, "Tongyi-MAI/Z-Image")
        self.assertEqual(widgets["residual_diff_threshold"], 0.08)
        self.assertEqual(widgets["Fn_compute_blocks"], 6)

    def test_cache_method_widget_gates_disable_dbcache_fields_for_teacache(self):
        gates = cache_method_widget_gates("teacache")
        self.assertTrue(gates["residual_diff_threshold"])
        self.assertFalse(gates["Fn_compute_blocks"])
        self.assertFalse(gates["scm_policy"])

    def test_runtime_from_loader_widgets_teacache_ignores_dbcache_fields(self):
        runtime = runtime_from_loader_widgets(
            {
                **default_loader_widget_values(),
                "cache_method": "teacache",
                "residual_diff_threshold": 0.15,
                "Fn_compute_blocks": 8,
                "scm_policy": "ultra",
            }
        )
        self.assertEqual(runtime["cache_method"], "teacache")
        self.assertEqual(json.loads(runtime["cache_config"]), {"residual_diff_threshold": 0.15})

    def test_sanitize_cache_config_runtime_strips_dbcache_fields_for_fbcache(self):
        runtime = {
            "cache_method": "fbcache",
            "cache_config": json.dumps(
                {"residual_diff_threshold": 0.2, "Fn_compute_blocks": 8, "scm_policy": "fast"}
            ),
        }
        sanitize_cache_config_runtime(runtime)
        self.assertEqual(
            json.loads(runtime["cache_config"]),
            {"residual_diff_threshold": 0.2},
        )

    def test_cache_config_fields_for_method(self):
        self.assertEqual(
            cache_config_fields_for_method("dbcache"), frozenset(cache_config_widget_names())
        )
        self.assertEqual(
            cache_config_fields_for_method("teacache"), frozenset({"residual_diff_threshold"})
        )
        self.assertEqual(cache_config_fields_for_method("none"), frozenset())

    def test_generation_dests_not_on_loader(self):
        loader = set(loader_config_dests())
        for key in ("prompt", "height", "width", "num_inference_steps", "seed"):
            self.assertNotIn(key, loader)
            self.assertIn(key, runner_cli_dests())

    def test_sanitize_attention_runtime_clears_backend_for_hybrid_schedule(self):
        runtime = {
            "use_hybrid_attn_schedule": True,
            "attention_backend": "aiter_flydsl_fp8",
            "hybrid_attn_low_precision_backend": "aiter_sage_v2",
            "hybrid_attn_high_precision_backend": "aiter",
        }
        sanitize_attention_runtime(runtime)
        self.assertNotIn("attention_backend", runtime)
        self.assertEqual(runtime["hybrid_attn_low_precision_backend"], "aiter_sage_v2")
        self.assertEqual(runtime["hybrid_attn_high_precision_backend"], "aiter")

    def test_sanitize_attention_runtime_clears_backends_for_explicit_schedule(self):
        runtime = {
            "use_hybrid_attn_schedule": True,
            "attention_backend": "aiter_flydsl_fp8",
            "hybrid_attn_schedule": "0,1,0,1",
            "hybrid_attn_low_precision_backend": "aiter_sage_v2",
            "hybrid_attn_high_precision_backend": "aiter",
        }
        sanitize_attention_runtime(runtime)
        self.assertNotIn("attention_backend", runtime)
        self.assertIsNone(runtime["hybrid_attn_low_precision_backend"])
        self.assertIsNone(runtime["hybrid_attn_high_precision_backend"])
        self.assertEqual(runtime["hybrid_attn_schedule"], "0,1,0,1")

    def test_sanitize_attention_runtime_keeps_backend_without_hybrid(self):
        runtime = {"attention_backend": "sdpa"}
        sanitize_attention_runtime(runtime)
        self.assertEqual(runtime["attention_backend"], "sdpa")


if __name__ == "__main__":
    unittest.main()
