import unittest
from unittest import mock

import pytest

pytestmark = pytest.mark.usefixtures(
    "mock_loader_worker_warm",
    "mock_comfy_execution",
    "mock_comfy_video_api",
    "synthetic_preset_catalog",
)

from tests.node_test_helpers import (
    _execution_kwargs,
    _generate_kwargs,
    _loader_kwargs,
    _preset_spec,
)
from xdit_comfyui.nodes import (
    XDiTModel,
    XDiTSample,
)
from xdit_comfyui.presets import build_preset_spec
from xdit_comfyui.sampling import _execute_sample
from xdit_comfyui.worker import _clear_all_runtime_caches


class RunnerNodesTest(unittest.TestCase):
    def setUp(self):
        _clear_all_runtime_caches()

    def test_execute_sample_omits_zero_wan_video_keys(self):
        runtime = XDiTModel.execute(**_loader_kwargs(hf_cache_mode="comfy_models_shared"))[0]
        captured = {}

        def _capture_config(**kwargs):
            captured.update(kwargs.get("runner_config") or {})
            import torch

            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        with mock.patch("xdit_comfyui.worker._run_xdit", side_effect=_capture_config):
            _execute_sample(
                runtime,
                dry_run=False,
                output_type="pil",
                **_execution_kwargs(num_frames=81, flow_shift=0.0, guidance_scale_2=0.0),
            )
        self.assertNotIn("flow_shift", captured)
        self.assertIn("guidance_scale_2", captured)
        self.assertIsNone(captured["guidance_scale_2"])

    def test_execute_sample_includes_nonzero_wan_video_keys(self):
        runtime = XDiTModel.execute(**_loader_kwargs(hf_cache_mode="comfy_models_shared"))[0]
        captured = {}

        def _capture_config(**kwargs):
            captured.update(kwargs.get("runner_config") or {})
            import torch

            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        with mock.patch("xdit_comfyui.worker._run_xdit", side_effect=_capture_config):
            _execute_sample(
                runtime,
                dry_run=False,
                output_type="pil",
                **_execution_kwargs(num_frames=81, flow_shift=5.0, guidance_scale_2=3.0),
            )
        self.assertEqual(captured["flow_shift"], 5.0)
        self.assertEqual(captured["guidance_scale_2"], 3.0)

    def test_execute_sample_aligns_size_to_an_enforced_divisor(self):
        """LTX raises on an unaligned size, and only after the weights are loaded."""
        runtime = XDiTModel.execute(
            **_loader_kwargs(model="Lightricks/LTX-2", hf_cache_mode="comfy_models_shared")
        )[0]
        captured = {}

        def _capture_config(**kwargs):
            captured.update(kwargs.get("runner_config") or {})
            import torch

            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        with mock.patch("xdit_comfyui.worker._run_xdit", side_effect=_capture_config):
            _execute_sample(
                runtime,
                dry_run=False,
                output_type="pil",
                **_execution_kwargs(height=1000, width=1500),
            )
        self.assertEqual((captured["height"], captured["width"]), (1024, 1472))

    def test_execute_sample_keeps_the_exact_size_without_a_divisor(self):
        runtime = XDiTModel.execute(**_loader_kwargs(hf_cache_mode="comfy_models_shared"))[0]
        captured = {}

        def _capture_config(**kwargs):
            captured.update(kwargs.get("runner_config") or {})
            import torch

            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        with mock.patch("xdit_comfyui.worker._run_xdit", side_effect=_capture_config):
            _execute_sample(
                runtime,
                dry_run=False,
                output_type="pil",
                **_execution_kwargs(height=1000, width=1500),
            )
        self.assertEqual((captured["height"], captured["width"]), (1000, 1500))

    def test_dry_run_returns_synthetic_image_output_contract(self):
        runtime = XDiTModel.execute(**_loader_kwargs(hf_cache_mode="comfy_models_shared"))[0]
        self.assertIn("_exec", runtime)
        self.assertEqual(runtime["model"], "black-forest-labs/FLUX.1-dev")

        images, video = _execute_sample(
            runtime,
            dry_run=True,
            output_type="pil",
            **_execution_kwargs(),
        )
        self.assertIsNone(video)
        self.assertEqual(tuple(images.shape[1:]), (64, 64, 3))

    def test_sample_blocks_inactive_output_branch(self):
        from comfy_execution.graph import ExecutionBlocker

        image = object()
        video = object()
        with mock.patch(
            "xdit_comfyui.nodes._execute_sample",
            side_effect=(
                (image, None),
                (None, video),
            ),
        ):
            image_result = XDiTSample.execute(model={}, **_generate_kwargs())
            video_result = XDiTSample.execute(model={}, **_generate_kwargs(num_frames=81))

        self.assertIs(image_result[0], image)
        self.assertIsInstance(image_result[1], ExecutionBlocker)
        self.assertIsInstance(video_result[0], ExecutionBlocker)
        self.assertIs(video_result[1], video)

    def test_loader_init_config_includes_generation_stubs(self):
        from xdit_comfyui.model_info import model_generation_defaults
        from xdit_comfyui.worker_payload import (
            LOADER_INIT_REQUIRED_KEYS,
            loader_init_config,
            loader_init_worker_payload,
        )

        runtime = {"model": "black-forest-labs/FLUX.1-dev", "ulysses_degree": 1}
        config = loader_init_config(runtime)
        payload = loader_init_worker_payload(runtime)
        self.assertLessEqual(LOADER_INIT_REQUIRED_KEYS, set(payload))
        self.assertEqual(config["prompt"], "xDiT worker init")
        self.assertEqual(
            config["num_inference_steps"],
            model_generation_defaults(runtime["model"])["num_inference_steps"],
        )
        self.assertEqual(config["input_images"], [])

    def test_runtime_loader_warms_worker(self):
        with mock.patch("xdit_comfyui.nodes._ensure_loader_worker") as warm:
            warm.side_effect = lambda runtime, loader_node_id, timeout_seconds=None: runtime
            XDiTModel.execute(**_loader_kwargs())
            warm.assert_called_once()

    def test_manual_image_model_defers_warm_until_sample_image(self):
        import torch

        with (
            mock.patch("xdit_comfyui.nodes._ensure_loader_worker") as warm,
            mock.patch("xdit_comfyui.model_info.resolve_model_task", return_value="i2v"),
        ):
            runtime = XDiTModel.execute(
                **_loader_kwargs(
                    model="black-forest-labs/FLUX.1-dev",
                    task="i2v",
                    gpu_count=4,
                    gpu_device_ids="0,1,2,3",
                    ulysses_degree=4,
                    unique_id="manual-wan",
                )
            )[0]
        warm.assert_not_called()
        self.assertTrue(runtime["_deferred_image_warm"])

        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs["runner_config"])
            return (
                torch.zeros((1, 8, 8, 3)),
                "image",
                {"fps": 16, "actual_height": 8, "actual_width": 8},
            )

        with (
            mock.patch("xdit_comfyui.worker._run_xdit", side_effect=fake_run),
            mock.patch("xdit_comfyui.model_info.resolve_model_task", return_value="i2v"),
        ):
            images, video = _execute_sample(
                runtime,
                dry_run=False,
                output_type="pil",
                images=torch.zeros((1, 8, 8, 3)),
                **_execution_kwargs(num_frames=1),
            )
        self.assertIsNone(video)
        self.assertEqual(tuple(images.shape), (1, 8, 8, 3))
        self.assertEqual(len(captured["input_images"]), 1)

    def test_sample_recreates_missing_worker(self):
        from contextlib import nullcontext

        import torch

        from xdit_comfyui.worker import _run_xdit_distributed

        runtime = {
            "model": "black-forest-labs/FLUX.1-dev",
            "_loader_node_id": "loader-1",
            "_cache_key": "abc",
            "num_inference_steps": 4,
        }
        entry = {"loader_uid": "loader-1"}
        with (
            mock.patch(
                "xdit_comfyui.worker._get_or_create_distributed_worker",
                return_value=(entry, True),
            ) as create,
            mock.patch(
                "xdit_comfyui.worker._run_distributed_worker",
                return_value=(mock.Mock(images=[object()], videos=[]), [], {}),
            ),
            mock.patch(
                "xdit_comfyui.worker._diffusion_output_to_comfy_image",
                return_value=torch.zeros((1, 8, 8, 3)),
            ),
            mock.patch(
                "xdit_comfyui.worker._xdit_progress",
                return_value=nullcontext(mock.Mock()),
            ),
        ):
            stdout, output, metadata = _run_xdit_distributed(
                runtime,
                {},
                1,
                "abc",
                60,
                "sample-1",
            )

        create.assert_called_once()
        self.assertEqual(tuple(output.shape), (1, 8, 8, 3))
        self.assertEqual(metadata["output_kind"], "image")
        self.assertIn("runner created", stdout)

    def test_sample_is_changed_includes_preset(self):
        flux = _preset_spec("flux.1gpu.rdna4")
        wan = _preset_spec("wan2_2_ti2v_5b.i2v.4gpu.rdna4")
        kwargs = _generate_kwargs()
        flux_fp = XDiTSample.fingerprint_inputs(preset=flux, **kwargs)
        wan_fp = XDiTSample.fingerprint_inputs(preset=wan, **kwargs)
        self.assertNotEqual(flux_fp, wan_fp)

    def test_execute_sample_allows_overriding_preset_parallelism(self):
        """A preset is a source of defaults, not a runtime contract.

        Changing a loader widget that shifts the derived world size used to raise
        "warmed for a different preset" and left the graph unrunnable, because the
        Model node it told the user to re-run was already up to date.
        """
        flux = _preset_spec("flux.1gpu.rdna4")
        runtime = XDiTModel.execute(
            **_loader_kwargs(unique_id="loader-override"),
            preset=flux,
        )[0]
        runtime["_gpu_count"] = (runtime.get("_gpu_count") or 1) + 1

        with mock.patch("xdit_comfyui.worker._evict_loader_worker") as evict:
            images, _video = _execute_sample(
                runtime,
                dry_run=True,
                output_type="pil",
                preset=flux,
                **_execution_kwargs(),
            )
        evict.assert_not_called()
        self.assertIsNotNone(images)

    def test_execute_sample_accepts_versioned_preset_model_choice(self):
        from xdit_comfyui.runtime_config import _runtime_loader_model_choices

        spec = build_preset_spec(
            "qwen_image.1gpu.rdna4",
            "gfx1201",
            registry_choices=_runtime_loader_model_choices(),
        )
        runtime = XDiTModel.execute(
            **_loader_kwargs(),
            preset=spec,
        )[0]
        _execute_sample(
            runtime,
            dry_run=True,
            output_type="pil",
            preset=spec,
            **_execution_kwargs(),
        )

    def test_two_loaders_same_config_use_separate_workers(self):
        seen_loaders = []

        def fake_get_or_create(cache_key, init_config, env_overrides, nproc, **kwargs):
            loader_uid = kwargs.get("loader_uid") or init_config.get("_loader_node_id")
            seen_loaders.append(loader_uid)
            return {
                "proc": mock.Mock(poll=mock.Mock(return_value=None)),
                "socket_path": f"/tmp/x-{loader_uid}.sock",
                "cache_key": cache_key,
                "loader_uid": loader_uid,
            }, True

        runtime_a = XDiTModel.execute(**_loader_kwargs(unique_id="loader-a"))[0]
        runtime_b = XDiTModel.execute(**_loader_kwargs(unique_id="loader-b"))[0]
        self.assertEqual(runtime_a["_cache_key"], runtime_b["_cache_key"])

        import torch

        with mock.patch(
            "xdit_comfyui.worker._distributed_worker_alive",
            return_value=True,
        ):
            with mock.patch(
                "xdit_comfyui.worker._get_or_create_distributed_worker",
                side_effect=fake_get_or_create,
            ):
                with mock.patch(
                    "xdit_comfyui.worker._run_distributed_worker",
                    return_value=(mock.Mock(images=[object()], videos=[]), [], {}),
                ):
                    with mock.patch(
                        "xdit_comfyui.worker._diffusion_output_to_comfy_image",
                        return_value=torch.zeros((1, 8, 8, 3), dtype=torch.float32),
                    ):
                        XDiTSample.execute(model=runtime_a, **_generate_kwargs())
                        XDiTSample.execute(model=runtime_b, **_generate_kwargs())

        self.assertEqual(seen_loaders, ["loader-a", "loader-b"])

    def test_worker_runtime_reuse(self):
        lifecycle = {"calls": 0}

        def fake_distributed(
            runner_config,
            env_overrides,
            nproc,
            preferred_cache_key,
            timeout_seconds,
            generate_node_id,
        ):
            lifecycle["calls"] += 1
            import torch

            status = "created" if lifecycle["calls"] == 1 else "reused"
            return f"worker runner {status}", torch.zeros((1, 8, 8, 3), dtype=torch.float32)

        with mock.patch(
            "xdit_comfyui.worker._run_xdit_distributed",
            side_effect=fake_distributed,
        ):
            runtime = XDiTModel.execute(**_loader_kwargs())[0]

            result1 = XDiTSample.execute(
                model=runtime,
                **_generate_kwargs(prompt="p1"),
            )

            result2 = XDiTSample.execute(
                model=runtime,
                **_generate_kwargs(prompt="p2", seed=43),
            )

            self.assertEqual(tuple(result1[0].shape[-3:]), (8, 8, 3))
            self.assertEqual(tuple(result2[0].shape[-3:]), (8, 8, 3))
            self.assertEqual(lifecycle["calls"], 2)

    def test_execute_sample_uses_actual_kind_native_fps_and_exact_size(self):
        import torch

        runtime = XDiTModel.execute(**_loader_kwargs())[0]
        metadata = {
            "fps": 16,
            "actual_height": 480,
            "actual_width": 832,
        }
        with (
            mock.patch(
                "xdit_comfyui.worker._run_xdit",
                return_value=(torch.zeros((5, 480, 832, 3)), "video", metadata),
            ) as run_xdit,
            mock.patch(
                "xdit_comfyui.sampling._generation_outputs_from_frames",
                return_value=(None, mock.sentinel.video),
            ) as convert,
        ):
            images, video, result_metadata = _execute_sample(
                runtime,
                dry_run=False,
                output_type="pil",
                resize_input_images=True,
                output_fps=0,
                return_metadata=True,
                **_execution_kwargs(num_frames=1),
            )

        self.assertIsNone(images)
        self.assertIs(video, mock.sentinel.video)
        self.assertEqual(result_metadata["selected_fps"], 16)
        self.assertEqual(convert.call_args.args[1:], ("video", 16))
        self.assertTrue(run_xdit.call_args.kwargs["runner_config"]["resize_input_images"])

    def test_execute_sample_output_fps_overrides_playback_only(self):
        import torch

        runtime = XDiTModel.execute(**_loader_kwargs())[0]
        with (
            mock.patch(
                "xdit_comfyui.worker._run_xdit",
                return_value=(
                    torch.zeros((5, 8, 8, 3)),
                    "video",
                    {"fps": 16, "actual_height": 8, "actual_width": 8},
                ),
            ),
            mock.patch(
                "xdit_comfyui.sampling._generation_outputs_from_frames",
                return_value=(None, mock.sentinel.video),
            ) as convert,
        ):
            _execute_sample(
                runtime,
                dry_run=False,
                output_type="pil",
                output_fps=12,
                return_metadata=True,
                **_execution_kwargs(),
            )

        self.assertEqual(convert.call_args.args[2], 12)
