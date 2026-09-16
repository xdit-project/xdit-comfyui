"""Worker parking feasibility and device transfers."""

import unittest
from unittest import mock

from xdit_comfyui.xdit_ext.residency_park import (
    _component_parameter_bytes,
    park_feasible,
    park_runner,
    restore_runner,
)

try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False


class ParkFeasibleTest(unittest.TestCase):
    def test_single_gpu_replicated_layout_is_feasible(self):
        ok, reason = park_feasible({"fully_shard_degree": 1, "ulysses_degree": 2})
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_fsdp_blocks_park(self):
        ok, reason = park_feasible({"fully_shard_degree": 8})
        self.assertFalse(ok)
        self.assertIn("FSDP", reason)

    def test_pipefusion_blocks_park(self):
        ok, reason = park_feasible({"pipefusion_parallel_degree": 4})
        self.assertFalse(ok)
        self.assertIn("pipefusion", reason)

    def test_inference_cpu_offload_blocks_park(self):
        ok, reason = park_feasible({"enable_model_cpu_offload": True})
        self.assertFalse(ok)
        self.assertIn("offload", reason)


class ParkRunnerTest(unittest.TestCase):
    def test_parameter_accounting_uses_components_not_pipeline_parameters(self):
        from torch import nn

        component = nn.Linear(3, 2, bias=False)
        pipe = mock.Mock()
        pipe.components = {"transformer": component, "alias": component}
        pipe.parameters = {"not": "callable"}
        self.assertEqual(
            _component_parameter_bytes(pipe, "cpu"),
            component.weight.numel() * component.weight.element_size(),
        )

    def _fake_pipe(self):
        import torch
        from torch import nn

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(2, 2, device="cuda"))

        pipe = mock.Mock()
        pipe.components = {"transformer": Block()}
        pipe.parameters = pipe.components["transformer"].parameters
        return pipe

    @unittest.skipUnless(_HAS_CUDA, "needs CUDA for device moves")
    @mock.patch("torch.cuda.is_available", return_value=True)
    @mock.patch("torch.cuda.synchronize")
    @mock.patch("torch.cuda.empty_cache")
    @mock.patch("torch.cuda.current_device", return_value=0)
    @mock.patch("torch.cuda.memory_reserved", return_value=1024)
    @mock.patch("torch.cuda.memory_allocated", return_value=512)
    def test_park_moves_components_to_cpu(self, *_mocks):
        pipe = self._fake_pipe()
        runner = mock.Mock()
        runner.model.pipe = pipe
        result = park_runner(runner, {"fully_shard_degree": 1})
        self.assertTrue(result["ok"])
        self.assertEqual(next(pipe.components["transformer"].parameters()).device.type, "cpu")

    @unittest.skipUnless(_HAS_CUDA, "needs CUDA for device moves")
    @mock.patch("torch.cuda.is_available", return_value=True)
    @mock.patch("torch.cuda.synchronize")
    @mock.patch("torch.cuda.empty_cache")
    @mock.patch("torch.cuda.current_device", return_value=0)
    @mock.patch("torch.cuda.memory_reserved", return_value=2048)
    @mock.patch("torch.cuda.memory_allocated", return_value=1024)
    def test_restore_moves_components_back_to_cuda(self, *_mocks):
        pipe = self._fake_pipe()
        pipe.components["transformer"] = pipe.components["transformer"].to("cpu")
        runner = mock.Mock()
        runner.model.pipe = pipe
        result = restore_runner(runner, {"fully_shard_degree": 1})
        self.assertTrue(result["ok"])
        self.assertEqual(next(pipe.components["transformer"].parameters()).device.type, "cuda")
