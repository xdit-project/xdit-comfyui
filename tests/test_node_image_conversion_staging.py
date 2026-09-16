import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

pytestmark = pytest.mark.usefixtures("mock_loader_worker_warm", "synthetic_preset_catalog")

from tests.node_test_helpers import (
    _execution_kwargs,
    _loader_kwargs,
    _preset_spec,
)
from xdit_comfyui.images import (
    _comfy_image_batch_to_paths,
    _generation_input_staging_directory,
    _is_preset_placeholder_image,
    _load_preset_reference_image,
    _paths_to_comfy_image,
)
from xdit_comfyui.nodes import (
    XDiTModel,
    XDiTPreset,
)
from xdit_comfyui.runtime_config import _preset_synced_loader_kwargs
from xdit_comfyui.sampling import _execute_sample
from xdit_comfyui.worker import _clear_all_runtime_caches


class RunnerNodesTest(unittest.TestCase):
    def setUp(self):
        _clear_all_runtime_caches()

    def _image_file(self, size=(96, 64)):
        from PIL import Image

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "reference.png"
        Image.new("RGB", size, "navy").save(path)
        return path

    def test_comfy_image_batch_to_paths_stages_pngs(self):
        import torch

        staging = _generation_input_staging_directory()
        image = torch.zeros((2, 8, 8, 3), dtype=torch.float32)
        image[1, 2:4, 2:4, 0] = 1.0
        paths = _comfy_image_batch_to_paths(image, staging)
        self.assertEqual(len(paths), 2)
        for path in paths:
            self.assertTrue(Path(path).is_file())
            self.assertTrue(path.endswith(".png"))

    def test_paths_to_comfy_image_loads_benchmark_file(self):
        tensor = _paths_to_comfy_image([str(self._image_file())])
        self.assertEqual(tuple(tensor.shape[0:1]), (1,))
        self.assertGreater(int(tensor.shape[1]), 0)
        self.assertGreater(int(tensor.shape[2]), 0)

    def test_load_preset_reference_image_returns_placeholder_without_paths(self):

        image = _load_preset_reference_image({"paths": [], "required": False})
        self.assertEqual(tuple(image.shape), (1, 64, 64, 3))
        self.assertTrue(_is_preset_placeholder_image(image))

    def test_benchmark_preset_loads_reference_image_output(self):
        with mock.patch(
            "xdit_comfyui.benchmark_data.resolve_benchmark_data_path",
            return_value=str(self._image_file((1024, 720))),
        ):
            raw = XDiTPreset.execute("gfx1201", 4, "wan2_2_ti2v_5b.i2v.4gpu.rdna4", "")
        if isinstance(raw, dict):
            _, image, _ = raw["result"]
        else:
            _, image, _ = raw
        self.assertGreater(int(image.shape[0]), 0)
        self.assertFalse(_is_preset_placeholder_image(image))
        self.assertGreater(int(image.shape[1]), 64)

    def test_benchmark_preset_t2i_image_output_is_placeholder(self):
        raw = XDiTPreset.execute("gfx1201", 1, "flux.1gpu.rdna4", "")
        if isinstance(raw, dict):
            _, image, _ = raw["result"]
        else:
            _, image, _ = raw
        self.assertTrue(_is_preset_placeholder_image(image))

    def test_paths_to_comfy_image_pads_mixed_sizes(self):
        tensor = _paths_to_comfy_image(
            [
                str(self._image_file((1024, 1024))),
                str(self._image_file((1024, 1104))),
            ]
        )
        self.assertEqual(int(tensor.shape[0]), 2)
        self.assertEqual(int(tensor.shape[1]), 1104)
        self.assertEqual(int(tensor.shape[2]), 1024)

    def test_benchmark_preset_multi_image_tensor_does_not_crash(self):
        with mock.patch(
            "xdit_comfyui.benchmark_data.resolve_benchmark_data_path",
            return_value=str(self._image_file()),
        ):
            raw = XDiTPreset.execute("gfx942", 8, "flux2.t-multi-i2i_1k", "")
        if isinstance(raw, dict):
            _, image, _ = raw["result"]
        else:
            _, image, _ = raw
        self.assertEqual(int(image.shape[0]), 2)

    def test_execute_sample_uses_preset_paths_when_images_unwired(self):
        spec = _preset_spec("wan2_2_ti2v_5b.i2v.4gpu.rdna4")
        runtime = XDiTModel.execute(
            preset=spec,
            **_preset_synced_loader_kwargs(
                spec,
                hf_cache_mode="comfy_models_shared",
            ),
        )[0]
        with mock.patch("xdit_comfyui.worker._run_xdit") as run_xdit:
            import torch as torch_mod

            run_xdit.return_value = torch_mod.zeros((1, 64, 64, 3), dtype=torch_mod.float32)
            _execute_sample(
                runtime,
                dry_run=False,
                output_type="pil",
                preset=spec,
                **_execution_kwargs(),
            )
        config = run_xdit.call_args.kwargs["runner_config"]
        self.assertEqual(len(config["input_images"]), 1)
        self.assertTrue(str(config["input_images"][0]).endswith("wan_input.jpg"))

    def test_execute_sample_uses_staged_image_paths(self):
        import torch

        runtime = XDiTModel.execute(**_loader_kwargs(hf_cache_mode="comfy_models_shared"))[0]
        image = torch.full((1, 16, 16, 3), 0.5, dtype=torch.float32)
        with mock.patch("xdit_comfyui.worker._run_xdit") as run_xdit:
            import torch as torch_mod

            run_xdit.return_value = torch_mod.zeros((1, 64, 64, 3), dtype=torch.float32)
            _execute_sample(
                runtime,
                dry_run=False,
                output_type="pil",
                images=image,
                **_execution_kwargs(),
            )
        config = run_xdit.call_args.kwargs["runner_config"]
        self.assertEqual(len(config["input_images"]), 1)
        self.assertTrue(Path(config["input_images"][0]).is_file())

    def test_quick_diffusion_output_is_non_black(self):
        from xdit_comfyui.images import _diffusion_output_to_comfy_image
        from xdit_comfyui.quick_run import quick_diffusion_output

        output = quick_diffusion_output({"height": 128, "width": 128, "seed": 7, "num_frames": 1})
        frames = _diffusion_output_to_comfy_image(output)
        self.assertGreater(float(frames.mean()), 0.01)

    def test_diffusion_output_preserves_float_video_range(self):
        import numpy as np
        from xfuser.model_executor.models.runner_models.base_model import DiffusionOutput

        from xdit_comfyui.images import _diffusion_output_to_comfy_image

        video = np.full((4, 8, 8, 3), 0.75, dtype=np.float32)
        output = DiffusionOutput(videos=[video], pipe_args={})
        frames = _diffusion_output_to_comfy_image(output)
        self.assertAlmostEqual(float(frames.mean()), 0.75, places=3)
        self.assertAlmostEqual(float(frames.max()), 0.75, places=3)

    def test_diffusion_output_rejects_multiple_videos(self):
        import numpy as np
        from xfuser.model_executor.models.runner_models.base_model import DiffusionOutput

        from xdit_comfyui.images import _diffusion_output_to_comfy_image

        videos = [
            np.zeros((2, 8, 8, 3), dtype=np.float32),
            np.zeros((2, 8, 8, 3), dtype=np.float32),
        ]
        with self.assertRaisesRegex(ValueError, "Multiple video outputs"):
            _diffusion_output_to_comfy_image(DiffusionOutput(videos=videos, pipe_args={}))

    def test_generate_splits_video_output(self):
        import torch

        from xdit_comfyui.images import _generation_outputs_from_frames

        frames = torch.zeros((3, 8, 8, 3))
        image, video = _generation_outputs_from_frames(frames, "image")
        self.assertEqual(tuple(image.shape), (3, 8, 8, 3))
        self.assertIsNone(video)

        video_mod = mock.MagicMock()
        video_mod.VideoFromComponents = mock.Mock(return_value=mock.sentinel.video)
        util_mod = mock.MagicMock()
        util_mod.VideoComponents = mock.Mock(return_value=mock.sentinel.components)
        with mock.patch.dict(
            sys.modules,
            {
                "comfy_api.latest._input_impl.video_types": video_mod,
                "comfy_api.latest._util.video_types": util_mod,
            },
        ):
            image2, video2 = _generation_outputs_from_frames(frames, "video", 16)
        self.assertIsNone(image2)
        self.assertIs(video2, mock.sentinel.video)

    def test_frames_are_never_dropped_when_comfy_cannot_build_a_video(self):
        """Returning nothing would throw away a finished run without a word."""

        import torch

        from xdit_comfyui.images import _generation_outputs_from_frames

        with mock.patch.dict(sys.modules, {"comfy_api.latest._input_impl.video_types": None}):
            with self.assertRaisesRegex(RuntimeError, "3 generated frames"):
                _generation_outputs_from_frames(torch.zeros((3, 8, 8, 3)), "video", 16)


class ScratchDirectoryTest(unittest.TestCase):
    """Every run makes a scratch directory; nothing used to remove them."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch("tempfile.gettempdir", return_value=self._tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_run_directories_do_not_accumulate(self):
        from xdit_comfyui.images import _SCRATCH_KEEP, _generation_output_directory

        made = [_generation_output_directory() for _ in range(_SCRATCH_KEEP + 5)]
        base = Path(self._tmp.name) / "xdit_comfyui"
        remaining = sorted(path.name for path in base.glob("run_*"))
        self.assertEqual(_SCRATCH_KEEP + 1, len(remaining))
        self.assertIn(made[-1].name, remaining)

    def test_staged_inputs_do_not_accumulate(self):
        from xdit_comfyui.images import (
            _SCRATCH_KEEP,
            _generation_input_staging_directory,
        )

        for _ in range(_SCRATCH_KEEP + 5):
            _generation_input_staging_directory()
        base = Path(self._tmp.name) / "xdit_comfyui" / "inputs"
        self.assertEqual(_SCRATCH_KEEP + 1, len(list(base.glob("in_*"))))

    def test_pruning_keeps_the_newest_and_never_touches_the_inputs_tree(self):
        from xdit_comfyui.images import (
            _generation_input_staging_directory,
            _generation_output_directory,
        )

        staged = _generation_input_staging_directory()
        (staged / "input_000.png").write_bytes(b"")
        for _ in range(20):
            _generation_output_directory()
        self.assertTrue(staged.is_dir(), "input staging is not run scratch and must survive")
        self.assertTrue((staged / "input_000.png").is_file())
