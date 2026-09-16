"""Residency policy selection and post-run actions."""

import unittest
from unittest import mock

from xdit_comfyui import worker
from xdit_comfyui.registry import REGISTRY
from xdit_comfyui.residency_allocator import (
    demote_loader_after_run,
    demote_policy_after_run,
    normalize_residency,
    residency_choices_for_runtime,
    residency_pins_gpu,
)
from xdit_comfyui.runner_contract import (
    RESIDENCY_CHOICES,
    RESIDENCY_KEEP_GPU,
    RESIDENCY_PARK_CPU,
    RESIDENCY_RELEASE,
)


class ResidencyChoiceTest(unittest.TestCase):
    def test_the_user_picks_from_exactly_three_policies(self):
        self.assertEqual(
            RESIDENCY_CHOICES,
            [RESIDENCY_KEEP_GPU, RESIDENCY_PARK_CPU, RESIDENCY_RELEASE],
        )

    def test_anything_unrecognised_keeps_the_gpu(self):
        for value in ("nonsense", "auto", "keep_warm", "free_after_run", None, ""):
            self.assertEqual(normalize_residency(value), RESIDENCY_KEEP_GPU)

    def test_only_keep_gpu_pins_vram(self):
        self.assertTrue(residency_pins_gpu(RESIDENCY_KEEP_GPU))
        self.assertFalse(residency_pins_gpu(RESIDENCY_PARK_CPU))
        self.assertFalse(residency_pins_gpu(RESIDENCY_RELEASE))

    def test_each_policy_maps_to_one_action_after_the_last_sample(self):
        self.assertIsNone(demote_policy_after_run(RESIDENCY_KEEP_GPU))
        self.assertEqual(demote_policy_after_run(RESIDENCY_PARK_CPU), "park")
        self.assertEqual(demote_policy_after_run(RESIDENCY_RELEASE), "release")

    def test_park_is_offered_for_replicated_layouts(self):
        choices, reason = residency_choices_for_runtime({"fully_shard_degree": 1})
        self.assertEqual(choices, [RESIDENCY_KEEP_GPU, RESIDENCY_PARK_CPU, RESIDENCY_RELEASE])
        self.assertEqual(reason, "")

    def test_park_is_not_offered_for_sharded_layouts(self):
        choices, reason = residency_choices_for_runtime({"fully_shard_degree": 4})
        self.assertEqual(choices, [RESIDENCY_KEEP_GPU, RESIDENCY_RELEASE])
        self.assertIn("FSDP", reason)


class DemoteAfterRunTest(unittest.TestCase):
    def setUp(self):
        worker.register_prompt_loader_consumers({})

    def _demote(self, policy, **runtime):
        park = mock.Mock(return_value=True)
        evict = mock.Mock(return_value=True)
        demote_loader_after_run(
            {"_loader_node_id": "2", "_residency": policy, **runtime},
            park_fn=park,
            evict_fn=evict,
        )
        return park, evict

    def test_keep_gpu_holds_the_worker(self):
        park, evict = self._demote(RESIDENCY_KEEP_GPU)
        park.assert_not_called()
        evict.assert_not_called()

    def test_release_stops_the_worker(self):
        park, evict = self._demote(RESIDENCY_RELEASE)
        park.assert_not_called()
        evict.assert_called_once_with("2")

    def test_park_cpu_moves_the_weights_to_host_ram(self):
        park, evict = self._demote(RESIDENCY_PARK_CPU, fully_shard_degree=1)
        park.assert_called_once_with("2")
        evict.assert_not_called()

    def test_park_cpu_keeps_gpu_warm_when_sharding_blocks_it(self):
        park, evict = self._demote(RESIDENCY_PARK_CPU, fully_shard_degree=8)
        park.assert_not_called()
        evict.assert_not_called()

    def test_park_cpu_keeps_gpu_warm_when_the_worker_refuses(self):
        park = mock.Mock(return_value=False)
        evict = mock.Mock(return_value=True)
        demote_loader_after_run(
            {
                "_loader_node_id": "2",
                "_residency": RESIDENCY_PARK_CPU,
                "fully_shard_degree": 1,
            },
            park_fn=park,
            evict_fn=evict,
        )
        park.assert_called_once_with("2")
        evict.assert_not_called()

    def test_the_worker_survives_until_the_last_sample_of_the_prompt(self):
        worker.register_prompt_loader_consumers({"2": 2})
        park, evict = self._demote(RESIDENCY_RELEASE)
        evict.assert_not_called()
        park, evict = self._demote(RESIDENCY_RELEASE)
        evict.assert_called_once_with("2")

    def test_a_runtime_without_a_model_node_id_is_ignored(self):
        park = mock.Mock()
        evict = mock.Mock()
        demote_loader_after_run({"_residency": RESIDENCY_RELEASE}, park_fn=park, evict_fn=evict)
        evict.assert_not_called()


class NoImplicitEvictionTest(unittest.TestCase):
    """A second Model node must never take VRAM from the first one."""

    def setUp(self):
        worker._clear_all_runtime_caches()

    def tearDown(self):
        worker._clear_all_runtime_caches()

    def test_the_pack_has_no_vram_broker_left(self):
        import xdit_comfyui.residency_allocator as allocator

        for removed in (
            "ensure_vram_for_load",
            "resolve_gpu_conflicts",
            "plan_conflict_resolution",
            "can_coexist_without_eviction",
            "memory_pressure_for_load",
        ):
            self.assertFalse(
                hasattr(allocator, removed),
                f"{removed} is back; residency must not evict on its own",
            )

    def test_warming_a_second_loader_leaves_the_first_alone(self):
        REGISTRY.cache_keys["2"] = "key-a"
        REGISTRY.snapshots["2"] = {"gpus": ["0"], "residency": RESIDENCY_KEEP_GPU}
        REGISTRY.workers["2"] = {"last_used_at": 1.0, "gpus": ["0"]}
        with mock.patch.object(worker, "_evict_loader_worker") as evict:
            with mock.patch.object(worker, "_park_loader_worker") as park:
                worker._register_loader_cache(
                    "5",
                    "key-b",
                    {"model": "m", "_cuda_visible_devices": "0", "_residency": RESIDENCY_KEEP_GPU},
                )
        evict.assert_not_called()
        park.assert_not_called()
        self.assertIn("2", REGISTRY.workers)
