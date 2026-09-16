"""Loader behavior, cache paths, and child runtime environment."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

pytestmark = pytest.mark.usefixtures("mock_loader_worker_warm", "synthetic_preset_catalog")

from tests.node_test_helpers import (
    _loader_kwargs,
    _preset_spec,
)
from xdit_comfyui.nodes import XDiTModel, XDiTPreset, XDiTSample
from xdit_comfyui.presets import build_preset_spec, preset_by_name
from xdit_comfyui.runtime_config import (
    _generation_input_types,
    _preset_synced_loader_kwargs,
    _runtime_loader_input_types,
)
from xdit_comfyui.runtime_env import (
    _ensure_runtime_env,
    _resolve_hf_cache_root,
    _runtime_env_delta,
)
from xdit_comfyui.worker import (
    _clear_all_runtime_caches,
    _clear_loader_cache,
    _effective_cache_key,
    _resolve_xdit_bin,
    _runtime_cache_key,
)


class HFCacheModeTest(unittest.TestCase):
    _HF_VARS = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME")

    def test_system_default_returns_none(self):
        self.assertIsNone(_resolve_hf_cache_root("system_default", "huggingface"))

    def test_auto_defers_to_env_when_set(self):
        for var in self._HF_VARS:
            with mock.patch.dict(os.environ, {v: "" for v in self._HF_VARS}, clear=False):
                for v in self._HF_VARS:
                    os.environ.pop(v, None)
                os.environ[var] = "/cache/huggingface"
                self.assertIsNone(
                    _resolve_hf_cache_root("auto", "huggingface"),
                    msg=f"auto should defer to existing {var}",
                )

    def test_auto_uses_pod_cache_when_present(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for v in self._HF_VARS:
                os.environ.pop(v, None)
            with mock.patch("xdit_comfyui.runtime_env._POD_HF_CACHE", Path("/cache/huggingface")):
                with mock.patch.object(Path, "is_dir", return_value=True):
                    with mock.patch.object(Path, "iterdir", return_value=iter([Path("models--x")])):
                        root = _resolve_hf_cache_root("auto", "huggingface")
        self.assertEqual(root, Path("/cache/huggingface"))

    def test_auto_falls_back_to_comfy_shared_when_env_absent(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for v in self._HF_VARS:
                os.environ.pop(v, None)
            with mock.patch(
                "xdit_comfyui.runtime_env._POD_HF_CACHE", Path("/nonexistent/pod/cache")
            ):
                root = _resolve_hf_cache_root("auto", "huggingface")
        self.assertIsNotNone(root)
        self.assertEqual(root.name, "huggingface")


class XDiTRuntimeEnvTest(unittest.TestCase):
    _STACK_KEYS = (
        "ROCM_PATH",
        "ROCM_HOME",
        "TRITON_HIP_LLD_PATH",
        "AITER_USE_SYSTEM_TRITON",
        "PYTORCH_MIOPEN_SUGGEST_NHWC",
        "GPU_ARCHS",
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_HIP_ALLOC_CONF",
        "HIP_FORCE_DEV_KERNARG",
        "SAFETENSORS_FAST_GPU",
        "TORCHINDUCTOR_LAYOUT_OPTIMIZATION",
    )

    def test_ensure_does_not_invent_gpu_stack_env(self):
        clean = {k: v for k, v in os.environ.items() if k not in self._STACK_KEYS and k != "PATH"}
        with mock.patch.dict(os.environ, clean, clear=True):
            env = _ensure_runtime_env({"HF_HOME": "/tmp/hf"})
        self.assertEqual(env["HF_HOME"], "/tmp/hf")
        for key in self._STACK_KEYS:
            self.assertNotIn(key, env)
        self.assertIn(f"{sys.prefix}/bin", env["PATH"].split(os.pathsep))

    def test_ensure_does_not_override_existing(self):
        with mock.patch.dict(os.environ, {"ROCM_PATH": "/custom/rocm"}, clear=False):
            env = _ensure_runtime_env()
        self.assertEqual(env["ROCM_PATH"], "/custom/rocm")

    def test_runtime_env_delta_does_not_add_stack_knobs(self):
        clean = {k: v for k, v in os.environ.items() if k not in self._STACK_KEYS}
        with mock.patch.dict(os.environ, clean, clear=True):
            delta = _runtime_env_delta()
        for key in self._STACK_KEYS:
            self.assertNotIn(key, delta)

    def test_prune_stale_aiter_jit_build_removes_broken_ninja(self):
        from xdit_comfyui.runtime_env import _prune_stale_aiter_jit_build

        build_root = Path("/tmp/xdit_test_aiter_build/module_quant")
        ninja_dir = build_root / "build"
        ninja_dir.mkdir(parents=True, exist_ok=True)
        (ninja_dir / "build.ninja").write_text(
            "cuda_cflags = --offload-arch=native -mllvm -amdgpu-coerce-illegal-types=1",
            encoding="utf-8",
        )
        with mock.patch("xdit_comfyui.runtime_env._AITER_JIT_BUILD_ROOT", build_root.parent):
            with mock.patch(
                "xdit_comfyui.runtime_env._AITER_JIT_ROOT",
                Path("/tmp/xdit_test_aiter_jit"),
            ):
                _prune_stale_aiter_jit_build()
        self.assertFalse(build_root.exists())


class NodeRebrandTest(unittest.TestCase):
    def test_v3_identity(self):
        schemas = [node.define_schema() for node in (XDiTPreset, XDiTModel, XDiTSample)]
        self.assertEqual(
            [schema.node_id for schema in schemas], ["xDiT.Preset", "xDiT.Model", "xDiT.Sample"]
        )
        self.assertEqual(
            [schema.display_name for schema in schemas],
            ["xDiT Preset", "xDiT Model", "xDiT Sample"],
        )
        self.assertTrue(all(schema.category == "xDiT" for schema in schemas))


class RunnerNodesTest(unittest.TestCase):
    def test_loader_init_uses_model_steps_that_cover_hybrid_boundaries(self):
        from xdit_comfyui.worker_payload import loader_init_config

        config = loader_init_config(
            {
                "model": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                "use_hybrid_attn_schedule": True,
                "num_hybrid_attn_high_precision_steps": 5,
                "use_hybrid_gemm_schedule": True,
                "num_hybrid_gemm_high_precision_steps": 7,
            }
        )
        # Wan's xDiT DefaultInputValues supplies 40; the plugin does not replace it
        # with a synthetic universal warmup count.
        self.assertEqual(config["num_inference_steps"], 40)

    def setUp(self):
        _clear_all_runtime_caches()

    def test_resolve_xdit_bin_from_path(self):
        with mock.patch("shutil.which", return_value="/usr/bin/xdit"):
            self.assertEqual(_resolve_xdit_bin("xdit"), "/usr/bin/xdit")

    def test_resolve_xdit_bin_from_comfy_interpreter_venv(self):
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp)
            bindir = venv / "bin"
            bindir.mkdir()
            xdit = bindir / "xdit"
            xdit.write_text("#!/bin/sh\n", encoding="utf-8")
            xdit.chmod(0o755)
            with mock.patch("shutil.which", return_value=None):
                with mock.patch("xdit_comfyui.runtime_env.sys.prefix", str(venv)):
                    self.assertEqual(_resolve_xdit_bin("xdit"), str(xdit))

    def test_runtime_loader_applies_named_preset(self):

        preset = preset_by_name("flux.1gpu.rdna4")
        self.assertIsNotNone(preset)
        runtime = XDiTModel.execute(
            **_loader_kwargs(),
            preset=_preset_spec("flux.1gpu.rdna4"),
        )[0]
        self.assertTrue(runtime["_preset"]["matched"])
        self.assertEqual(runtime["_preset"]["name"], "flux.1gpu.rdna4")
        self.assertEqual(runtime["attention_backend"], "aiter_flydsl")

    def test_runtime_loader_named_preset_applies_gpu_count(self):
        spec = _preset_spec("z_image_turbo.4gpu.rdna4")
        with self.assertRaisesRegex(ValueError, "gpu_device_ids must list exactly 1"):
            XDiTModel.execute(
                **_loader_kwargs(gpu_count=1),
                preset=spec,
            )

    def test_runtime_loader_named_preset_fills_unset_gpu_count(self):
        spec = _preset_spec("z_image_turbo.4gpu.rdna4")
        runtime = XDiTModel.execute(
            **_preset_synced_loader_kwargs(spec),
            preset=spec,
        )[0]
        self.assertEqual(runtime["_gpu_count"], 4)

    def test_runtime_loader_named_preset_fills_unset_loader_widgets(self):
        runtime = XDiTModel.execute(
            **_loader_kwargs(
                attention_backend="auto",
            ),
            preset=_preset_spec("flux.1gpu.rdna4"),
        )[0]
        self.assertEqual(runtime["attention_backend"], "aiter_flydsl")
        self.assertEqual(runtime["_preset"]["name"], "flux.1gpu.rdna4")

    def test_runtime_loader_widget_overrides_preset(self):
        runtime = XDiTModel.execute(
            **_loader_kwargs(
                attention_backend="sdpa",
            ),
            preset=_preset_spec("flux.1gpu.rdna4"),
        )[0]
        self.assertEqual(runtime["attention_backend"], "sdpa")
        self.assertEqual(runtime["_preset"]["name"], "flux.1gpu.rdna4")

    def test_runtime_loader_preset_syncs_cache_method(self):
        runtime = XDiTModel.execute(
            **_loader_kwargs(
                cache_method="none",
            ),
            preset=_preset_spec("flux.1gpu.rdna4"),
        )[0]
        self.assertIsNone(runtime.get("cache_method"))
        self.assertEqual(runtime["_preset"]["name"], "flux.1gpu.rdna4")

    def test_runtime_loader_honors_use_torch_compile_override(self):
        spec = _preset_spec("wan2_2_ti2v_5b.i2v.4gpu.rdna4")
        runtime = XDiTModel.execute(
            **_preset_synced_loader_kwargs(
                spec,
                use_torch_compile=False,
            ),
            preset=spec,
        )[0]
        self.assertFalse(runtime["use_torch_compile"])
        self.assertEqual(runtime["_preset"]["name"], "wan2_2_ti2v_5b.i2v.4gpu.rdna4")

    def test_runtime_loader_preset_syncs_cache_when_switching_from_turbo(self):
        spec = _preset_spec("z_image.4gpu.rdna4")
        runtime = XDiTModel.execute(
            **_preset_synced_loader_kwargs(
                spec,
                model="Tongyi-MAI/Z-Image-Turbo",
                cache_method="none",
            ),
            preset=spec,
        )[0]
        self.assertEqual(runtime.get("model"), "Tongyi-MAI/Z-Image-Turbo")
        self.assertIsNone(runtime.get("cache_method"))
        self.assertEqual(runtime["_preset"]["name"], "z_image.4gpu.rdna4")

    def test_runtime_loader_strips_cache_for_unsupported_model(self):
        runtime = XDiTModel.execute(
            **_loader_kwargs(
                model="Tongyi-MAI/Z-Image-Turbo",
                cache_method="dbcache",
                Fn_compute_blocks=8,
            ),
        )[0]
        self.assertIsNone(runtime.get("cache_method"))
        self.assertNotIn("cache_config", runtime)

    def test_runtime_loader_none_uses_config_widgets(self):
        runtime = XDiTModel.execute(
            **_loader_kwargs(
                ulysses_degree=2,
                gpu_device_ids="0,1",
                use_torch_compile=True,
            )
        )[0]
        self.assertEqual(runtime["_preset"]["selected"], "none")
        self.assertEqual(runtime["ulysses_degree"], 2)
        self.assertTrue(runtime["use_torch_compile"])

    def test_runtime_loader_manual_without_preset(self):
        runtime = XDiTModel.execute(
            **_loader_kwargs(
                ulysses_degree=2,
                gpu_device_ids="0,1",
            )
        )[0]
        self.assertEqual(runtime["ulysses_degree"], 2)

    def test_runtime_loader_none_ignores_stale_auto_selection(self):
        runtime = XDiTModel.execute(
            **_loader_kwargs(
                gpu_count=1,
                ulysses_degree=1,
                use_torch_compile=True,
                attention_backend="sdpa",
            ),
            preset=_preset_spec("auto (best for hardware)"),
        )[0]
        self.assertTrue(runtime["use_torch_compile"])
        self.assertEqual(runtime["attention_backend"], "sdpa")
        self.assertFalse(runtime["_preset"]["matched"])

    def test_runtime_loader_config_widgets_have_cli_tooltips(self):
        required = _runtime_loader_input_types()["required"]
        self.assertIn("ulysses", required["ulysses_degree"][1]["tooltip"].lower())
        self.assertIn("attention", required["attention_backend"][1]["tooltip"].lower())
        self.assertIn("residual", required["residual_diff_threshold"][1]["tooltip"].lower())
        self.assertIn("cpu_offload_mode", required)
        self.assertIn("enable_tiling", _generation_input_types()["required"])
        self.assertIn("cpu offload", required["cpu_offload_mode"][1]["tooltip"].lower())

    def test_runtime_loader_group_cpu_offload(self):
        runtime = XDiTModel.execute(**_loader_kwargs(cpu_offload_mode="group"))[0]
        self.assertTrue(runtime["enable_group_cpu_offload"])

    def test_resolve_model_choice_ignores_non_string_override(self):
        from xdit_comfyui.runtime_config import _resolve_model_choice

        self.assertEqual(
            _resolve_model_choice("black-forest-labs/FLUX.1-dev", False),
            "black-forest-labs/FLUX.1-dev",
        )
        self.assertEqual(
            _resolve_model_choice("black-forest-labs/FLUX.1-dev", True),
            "black-forest-labs/FLUX.1-dev",
        )

    def test_merge_loader_kwargs_replaces_invalid_model_choice_from_preset(self):
        from xdit_comfyui.runtime_config import _merge_loader_kwargs, _runtime_loader_model_choices

        spec = build_preset_spec(
            "flux.1gpu.rdna4",
            "gfx1201",
            registry_choices=_runtime_loader_model_choices(),
        )
        merged = _merge_loader_kwargs(spec, {"model": False, "gpu_device_ids": "1"})
        self.assertEqual(merged["model"], spec["model_choice"])

    def test_resolve_model_choice_rejects_bool_model_choice(self):
        from xdit_comfyui.runtime_config import _resolve_model_choice

        with self.assertRaisesRegex(ValueError, "invalid model"):
            _resolve_model_choice(False, "")

    def test_repair_loader_model_choice_uses_preset(self):
        from xdit_comfyui.runtime_config import (
            _repair_loader_model_choice,
            _runtime_loader_model_choices,
        )

        spec = build_preset_spec(
            "flux.1gpu.rdna4",
            "gfx1201",
            registry_choices=_runtime_loader_model_choices(),
        )
        repaired = _repair_loader_model_choice({"model": False}, spec)
        self.assertEqual(repaired["model"], spec["model_choice"])

    def test_repair_loader_model_choice_clears_false_string_override(self):
        from xdit_comfyui.runtime_config import (
            _repair_loader_model_choice,
            _runtime_loader_model_choices,
        )

        spec = build_preset_spec(
            "z_image_turbo.4gpu.rdna4",
            "gfx1201",
            registry_choices=_runtime_loader_model_choices(),
        )
        repaired = _repair_loader_model_choice(
            {"model": spec["model_choice"], "custom_model_id": "False"},
            spec,
        )
        self.assertEqual(repaired["model"], spec["model_choice"])
        self.assertEqual(repaired["custom_model_id"], "")

    def test_resolve_model_choice_ignores_false_string_override(self):
        from xdit_comfyui.runtime_config import _resolve_model_choice

        self.assertEqual(
            _resolve_model_choice("Tongyi-MAI/Z-Image-Turbo", "False"),
            "Tongyi-MAI/Z-Image-Turbo",
        )
        self.assertEqual(
            _resolve_model_choice("Tongyi-MAI/Z-Image-Turbo", False),
            "Tongyi-MAI/Z-Image-Turbo",
        )

    def test_effective_cache_key_dbcache_ignores_steps(self):
        base_runtime = {
            "model": "black-forest-labs/FLUX.1-dev",
            "cache_method": "dbcache",
            "ulysses_degree": 4,
            "_cache_key": "abc123",
        }
        key_25 = _effective_cache_key({**base_runtime, "num_inference_steps": 25})
        key_20 = _effective_cache_key({**base_runtime, "num_inference_steps": 20})
        self.assertEqual(key_25, key_20)

    def test_effective_cache_key_includes_task_for_multi_task_models(self):
        base = {
            "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "_cache_key": "wan-base",
            "_cache_key_task": "i2v",
        }
        i2v = _effective_cache_key({**base, "task": "i2v"})
        t2v = _effective_cache_key({**base, "task": "t2v"})
        self.assertEqual(i2v, "wan-base")
        self.assertNotEqual(i2v, t2v)

    def test_runtime_cache_key_ignores_cache_tuning(self):
        base = {
            "model": "black-forest-labs/FLUX.1-dev",
            "cache_method": "dbcache",
            "ulysses_degree": 1,
        }
        key_a = _runtime_cache_key({**base, "cache_config": '{"residual_diff_threshold": 0.08}'})
        key_b = _runtime_cache_key({**base, "cache_config": '{"residual_diff_threshold": 0.12}'})
        self.assertEqual(key_a, key_b)
        key_none = _runtime_cache_key({**base, "cache_method": None})
        self.assertNotEqual(key_a, key_none)

    def test_maybe_refresh_step_cache_skips_unchanged_steps(self):
        from xdit_comfyui.xdit_ext.step_cache import maybe_refresh_step_cache

        runner = mock.MagicMock()
        runner.config.cache_method = "dbcache"
        runner.config.num_inference_steps = 25
        runner.config.cache_config = None
        with mock.patch("cache_dit.refresh_context") as refresh:
            maybe_refresh_step_cache(runner, {"num_inference_steps": 25})
            refresh.assert_not_called()

    def test_maybe_refresh_step_cache_refreshes_on_step_change(self):
        from xdit_comfyui.xdit_ext.step_cache import (
            _APPLIED_CACHE_STATE_ATTR,
            maybe_refresh_step_cache,
        )

        transformer = mock.MagicMock()
        runner = mock.MagicMock()
        runner.config.cache_method = "dbcache"
        runner.config.num_inference_steps = 25
        runner.config.cache_config = None
        runner.model.settings.transformer_attr_names = ["transformer"]
        runner.model.pipe.transformer = transformer
        with (
            mock.patch(
                "xfuser.model_executor.cache.adapters.cache_dit._unwrap_fsdp",
                side_effect=lambda module: module,
            ),
            mock.patch("cache_dit.refresh_context") as refresh,
        ):
            maybe_refresh_step_cache(runner, {"num_inference_steps": 20})
            refresh.assert_called_once()
            self.assertIs(refresh.call_args.args[0], transformer)
            self.assertIn("cache_config", refresh.call_args.kwargs)
            state = object.__getattribute__(runner, _APPLIED_CACHE_STATE_ATTR)
            self.assertEqual(state["steps"], 20)

    def _wan22_style_runner(self):
        """A runner whose dbcache config declares two transformers, as Wan 2.2 does."""
        from xfuser.model_executor.cache.presets import DBCachePreset

        runner = mock.MagicMock()
        runner.config.cache_method = "dbcache"
        runner.config.num_inference_steps = 25
        runner.config.cache_config = None
        runner.model.settings.transformer_attr_names = ["transformer", "transformer_2"]

        high = mock.MagicMock(transformer_attr="transformer", enable_separate_cfg=True)
        low = mock.MagicMock(transformer_attr="transformer_2", enable_separate_cfg=True)
        runner.model.settings.step_cache_config = {
            "dbcache": mock.MagicMock(
                adapter=[high, low],
                preset=[
                    DBCachePreset(Fn_compute_blocks=4, max_warmup_steps=4),
                    DBCachePreset(Fn_compute_blocks=4, max_warmup_steps=2),
                ],
            )
        }
        return runner

    def test_step_cache_refreshes_every_transformer_of_a_multi_denoiser_model(self):
        """A list preset used to reach cache-dit unwrapped and raise TypeError."""
        from xdit_comfyui.xdit_ext.step_cache import maybe_refresh_step_cache

        runner = self._wan22_style_runner()
        first = mock.MagicMock()
        second = mock.MagicMock()
        runner.model.pipe.transformer = first
        runner.model.pipe.transformer_2 = second

        with (
            mock.patch(
                "xfuser.model_executor.cache.adapters.cache_dit._unwrap_fsdp",
                side_effect=lambda module: module,
            ),
            mock.patch("cache_dit.refresh_context") as refresh,
        ):
            maybe_refresh_step_cache(runner, {"num_inference_steps": 20})

        self.assertEqual(refresh.call_count, 2)
        refreshed = [call.args[0] for call in refresh.call_args_list]
        self.assertEqual(refreshed, [first, second])
        warmups = [call.kwargs["cache_config"].max_warmup_steps for call in refresh.call_args_list]
        self.assertEqual(warmups, [4, 2])

    def test_per_transformer_overrides_reach_only_their_transformer(self):
        from xdit_comfyui.xdit_ext.step_cache import maybe_refresh_step_cache

        runner = self._wan22_style_runner()
        runner.model.pipe.transformer = mock.MagicMock()
        runner.model.pipe.transformer_2 = mock.MagicMock()

        with (
            mock.patch(
                "xfuser.model_executor.cache.adapters.cache_dit._unwrap_fsdp",
                side_effect=lambda module: module,
            ),
            mock.patch("cache_dit.refresh_context") as refresh,
        ):
            maybe_refresh_step_cache(
                runner,
                {
                    "num_inference_steps": 20,
                    "cache_config": {
                        "residual_diff_threshold": 0.2,
                        "per_transformer": {"transformer_2": {"max_warmup_steps": 6}},
                    },
                },
            )

        configs = [call.kwargs["cache_config"] for call in refresh.call_args_list]
        self.assertEqual([config.max_warmup_steps for config in configs], [4, 6])
        # The broadcast key still lands on both.
        self.assertEqual([config.residual_diff_threshold for config in configs], [0.2, 0.2])

    def test_maybe_refresh_step_cache_parses_string_init_cache_config(self):
        from xdit_comfyui.xdit_ext.step_cache import (
            _parse_cache_config,
            maybe_refresh_step_cache,
        )

        self.assertEqual(
            _parse_cache_config('{"residual_diff_threshold": 0.1}'),
            {"residual_diff_threshold": 0.1},
        )

        transformer = mock.MagicMock()
        runner = mock.MagicMock()
        runner.config.cache_method = "dbcache"
        runner.config.num_inference_steps = 25
        runner.config.cache_config = '{"residual_diff_threshold": 0.1}'
        runner.model.settings.transformer_attr_names = ["transformer"]
        runner.model.pipe.transformer = transformer
        with (
            mock.patch(
                "xfuser.model_executor.cache.adapters.cache_dit._unwrap_fsdp",
                side_effect=lambda module: module,
            ),
            mock.patch("cache_dit.refresh_context") as refresh,
        ):
            maybe_refresh_step_cache(
                runner,
                {
                    "num_inference_steps": 20,
                    "cache_config": '{"residual_diff_threshold": 0.1}',
                },
            )
            refresh.assert_called_once()
            self.assertIs(refresh.call_args.args[0], transformer)
            self.assertIn("cache_config", refresh.call_args.kwargs)

    def _teacache_style_runner(self, threshold=0.1, steps=25):
        """teacache and fbcache patch one block per transformer and tune it in place."""
        import torch

        block = mock.MagicMock()
        block.rel_l1_thresh = torch.tensor(threshold)
        block.num_steps = steps
        runner = mock.MagicMock()
        runner.config.cache_method = "teacache"
        runner.config.num_inference_steps = steps
        runner.config.cache_config = json.dumps({"residual_diff_threshold": threshold})
        runner.model.settings.transformer_attr_names = ["transformer"]
        runner.model.pipe.transformer.transformer_blocks = [block]
        return runner, block

    def test_a_new_threshold_reaches_the_patched_block_without_a_reload(self):
        from xdit_comfyui.xdit_ext.step_cache import maybe_refresh_step_cache

        runner, block = self._teacache_style_runner()
        maybe_refresh_step_cache(
            runner,
            {"num_inference_steps": 12, "cache_config": '{"residual_diff_threshold": 0.4}'},
        )
        self.assertAlmostEqual(0.4, float(block.rel_l1_thresh), places=6)
        self.assertEqual(12, block.num_steps)

    def test_rerunning_with_the_same_tuning_leaves_the_block_as_it_was(self):
        from xdit_comfyui.xdit_ext.step_cache import (
            _APPLIED_CACHE_STATE_ATTR,
            maybe_refresh_step_cache,
        )

        runner, block = self._teacache_style_runner(threshold=0.1, steps=25)
        maybe_refresh_step_cache(
            runner,
            {"num_inference_steps": 25, "cache_config": '{"residual_diff_threshold": 0.1}'},
        )
        self.assertAlmostEqual(0.1, float(block.rel_l1_thresh), places=6)
        self.assertEqual(25, block.num_steps)
        state = object.__getattribute__(runner, _APPLIED_CACHE_STATE_ATTR)
        self.assertEqual(0.1, state["threshold"])

    def test_a_run_that_names_no_threshold_leaves_the_warm_setting_alone(self):
        from xdit_comfyui.xdit_ext.step_cache import maybe_refresh_step_cache

        runner, block = self._teacache_style_runner(threshold=0.1)
        maybe_refresh_step_cache(runner, {"num_inference_steps": 12})
        self.assertAlmostEqual(0.1, float(block.rel_l1_thresh), places=6)
        self.assertEqual(25, block.num_steps, "the step count moved without a threshold")

    def test_the_threshold_can_also_come_from_the_run_config_itself(self):
        from xdit_comfyui.xdit_ext.step_cache import maybe_refresh_step_cache

        runner, block = self._teacache_style_runner(threshold=0.1)
        maybe_refresh_step_cache(
            runner, {"num_inference_steps": 20, "residual_diff_threshold": 0.25}
        )
        self.assertAlmostEqual(0.25, float(block.rel_l1_thresh), places=6)

    def test_fbcache_is_tuned_the_same_way(self):
        from xdit_comfyui.xdit_ext.step_cache import maybe_refresh_step_cache

        runner, block = self._teacache_style_runner()
        runner.config.cache_method = "fbcache"
        maybe_refresh_step_cache(
            runner,
            {"num_inference_steps": 8, "cache_config": '{"residual_diff_threshold": 0.3}'},
        )
        self.assertAlmostEqual(0.3, float(block.rel_l1_thresh), places=6)
        self.assertEqual(8, block.num_steps)

    def test_loader_auto_evicts_when_config_changes(self):
        with mock.patch("xdit_comfyui.worker._evict_loader_worker") as evict:
            XDiTModel.execute(**_loader_kwargs(unique_id="loader-1"))
            XDiTModel.execute(
                **_loader_kwargs(
                    model="Tongyi-MAI/Z-Image-Turbo",
                    unique_id="loader-1",
                )
            )
        evict.assert_called_once()

    def test_loader_auto_evicts_when_preset_changes(self):
        flux = _preset_spec("flux.1gpu.rdna4")
        wan = _preset_spec("wan2_2_ti2v_5b.i2v.4gpu.rdna4")
        with mock.patch("xdit_comfyui.worker._evict_loader_worker") as evict:
            XDiTModel.execute(
                **_loader_kwargs(unique_id="loader-1"),
                preset=flux,
            )
            XDiTModel.execute(
                **_preset_synced_loader_kwargs(wan, unique_id="loader-1"),
                preset=wan,
            )
        evict.assert_called_once()

    def test_load_model_is_changed_includes_preset(self):
        flux = _preset_spec("flux.1gpu.rdna4")
        wan = _preset_spec("wan2_2_ti2v_5b.i2v.4gpu.rdna4")
        kwargs = _loader_kwargs()
        flux_fp = XDiTModel.fingerprint_inputs(preset=flux, **kwargs)
        wan_fp = XDiTModel.fingerprint_inputs(preset=wan, **kwargs)
        self.assertNotEqual(flux_fp, wan_fp)

    def test_runtime_cache_key_includes_preset_execution_key(self):
        flux = _preset_spec("flux.1gpu.rdna4")
        wan = _preset_spec("wan2_2_ti2v_5b.i2v.4gpu.rdna4")
        flux_runtime = XDiTModel.execute(
            **_loader_kwargs(),
            preset=flux,
        )[0]
        wan_runtime = XDiTModel.execute(
            **_preset_synced_loader_kwargs(wan),
            preset=wan,
        )[0]
        self.assertNotEqual(
            flux_runtime["_cache_key"],
            wan_runtime["_cache_key"],
        )

    def test_two_loaders_keep_both_caches(self):
        with mock.patch("xdit_comfyui.worker._evict_loader_worker") as evict:
            XDiTModel.execute(
                **_loader_kwargs(model="black-forest-labs/FLUX.1-dev", unique_id="loader-a")
            )
            XDiTModel.execute(
                **_loader_kwargs(model="Tongyi-MAI/Z-Image-Turbo", unique_id="loader-b")
            )
        evict.assert_not_called()

    def test_clear_loader_cache_evicts_when_last_reference(self):
        with mock.patch("xdit_comfyui.worker._evict_loader_worker") as evict:
            XDiTModel.execute(**_loader_kwargs(unique_id="loader-1"))
            result = _clear_loader_cache("loader-1")
        self.assertTrue(result["ok"])
        self.assertTrue(result["evicted"])
        evict.assert_called_once()

    def test_clear_loader_cache_always_evicts_worker(self):
        with mock.patch("xdit_comfyui.worker._evict_loader_worker") as evict:
            XDiTModel.execute(**_loader_kwargs(unique_id="loader-a"))
            XDiTModel.execute(**_loader_kwargs(unique_id="loader-b"))
            result = _clear_loader_cache("loader-a")
            self.assertTrue(result["ok"])
            self.assertTrue(result["evicted"])
            evict.assert_called_once()
            result = _clear_loader_cache("loader-b")
            self.assertTrue(result["evicted"])
        self.assertEqual(evict.call_count, 2)

    def test_clear_loader_cache_without_registration(self):
        result = _clear_loader_cache("missing-loader")
        self.assertTrue(result["ok"])
        self.assertFalse(result["evicted"])

    def test_clear_loader_cache_aborts_pending_worker(self):
        from xdit_comfyui.worker import _loader_worker_token, _register_loader_pending

        proc = mock.Mock()
        proc.poll.return_value = None
        worker_token = _loader_worker_token("loader-pending")
        with mock.patch("xdit_comfyui.worker._abort_worker_startup") as abort:
            _register_loader_pending("loader-pending", worker_token, proc)
            result = _clear_loader_cache("loader-pending")
        self.assertTrue(result["ok"])
        self.assertTrue(result["evicted"])
        abort.assert_called_once_with(proc, worker_token)

    def test_loader_preset_change_evicts_previous_worker(self):
        with mock.patch("xdit_comfyui.worker._evict_loader_worker") as evict:
            XDiTModel.execute(
                **_loader_kwargs(
                    model="black-forest-labs/FLUX.1-dev",
                    unique_id="loader-1",
                )
            )
            XDiTModel.execute(
                **_loader_kwargs(
                    model="Tongyi-MAI/Z-Image-Turbo",
                    unique_id="loader-1",
                )
            )
        evict.assert_called_once()
