"""Residency tracking, reporting, and cleanup."""

import unittest
from unittest import mock

from xdit_comfyui import nodes, residency, worker
from xdit_comfyui.prompt_hooks import apply_preset_prompt_overrides
from xdit_comfyui.registry import REGISTRY, clear_node_status, record_node_status
from xdit_comfyui.residency import (
    _oom_attribution_text,
    _pair_memory_rows,
    record_sample_run_memory,
    residency_report,
    sample_run_memory,
)
from xdit_comfyui.residency_allocator import normalize_residency
from xdit_comfyui.runner_contract import (
    RESIDENCY_KEEP_GPU,
    RESIDENCY_PARK_CPU,
    RESIDENCY_RELEASE,
)
from xdit_comfyui.worker import (
    _release_loader_after_run,
    register_prompt_loader_consumers,
)


def _runtime(node_id="2", residency=RESIDENCY_RELEASE):
    return {"_loader_node_id": node_id, "_residency": residency}


class ResidencyPolicyTest(unittest.TestCase):
    def setUp(self):
        register_prompt_loader_consumers({})

    def test_unknown_residency_keeps_the_gpu(self):
        self.assertEqual(normalize_residency("nonsense"), RESIDENCY_KEEP_GPU)
        self.assertEqual(normalize_residency(None), RESIDENCY_KEEP_GPU)
        self.assertEqual(normalize_residency(RESIDENCY_RELEASE), RESIDENCY_RELEASE)
        self.assertEqual(normalize_residency(RESIDENCY_PARK_CPU), RESIDENCY_PARK_CPU)

    def test_keep_gpu_never_evicts(self):
        with mock.patch.object(worker, "_evict_loader_worker") as evict:
            with mock.patch.object(worker, "_park_loader_worker") as park:
                _release_loader_after_run(_runtime(residency=RESIDENCY_KEEP_GPU))
        evict.assert_not_called()
        park.assert_not_called()

    def test_release_waits_for_the_last_consumer(self):
        register_prompt_loader_consumers({"2": 2})
        with mock.patch.object(worker, "_evict_loader_worker") as evict:
            _release_loader_after_run(_runtime())
            evict.assert_not_called()
            _release_loader_after_run(_runtime())
            evict.assert_called_once_with("2")

    def test_release_without_registered_consumers_evicts_immediately(self):
        with mock.patch.object(worker, "_evict_loader_worker") as evict:
            _release_loader_after_run(_runtime())
        evict.assert_called_once_with("2")

    def test_release_ignores_runtime_without_loader_id(self):
        with mock.patch.object(worker, "_evict_loader_worker") as evict:
            _release_loader_after_run({"_residency": RESIDENCY_RELEASE})
        evict.assert_not_called()

    def test_park_cpu_calls_park_fn(self):
        with mock.patch.object(worker, "_park_loader_worker", return_value=True) as park:
            with mock.patch.object(worker, "_evict_loader_worker") as evict:
                _release_loader_after_run(
                    {
                        **_runtime(residency=RESIDENCY_PARK_CPU),
                        "fully_shard_degree": 1,
                    }
                )
        park.assert_called_once_with("2")
        evict.assert_not_called()

    def test_residency_is_excluded_from_the_worker_cache_key(self):
        warm = {"model": "FLUX.1-dev", "_residency": RESIDENCY_KEEP_GPU}
        freed = {"model": "FLUX.1-dev", "_residency": RESIDENCY_RELEASE}
        self.assertEqual(worker._runtime_cache_key(warm), worker._runtime_cache_key(freed))

    def test_residency_change_re_executes_the_model_node(self):
        base = {"model": "FLUX.1-dev", "gpu_device_ids": "0"}
        warm = nodes.XDiTModel.fingerprint_inputs(**base, residency=RESIDENCY_KEEP_GPU)
        freed = nodes.XDiTModel.fingerprint_inputs(**base, residency=RESIDENCY_RELEASE)
        self.assertNotEqual(warm, freed)


class PromptConsumerCountTest(unittest.TestCase):
    def test_prompt_hook_counts_sample_nodes_per_model_node(self):
        prompt = {
            "1": {"class_type": "xDiT.Model", "inputs": {"model": "black-forest-labs/FLUX.1-dev"}},
            "2": {"class_type": "xDiT.Sample", "inputs": {"model": ["1", 0], "prompt": "a"}},
            "3": {"class_type": "xDiT.Sample", "inputs": {"model": ["1", 0], "prompt": "b"}},
            "4": {"class_type": "PreviewImage", "inputs": {"images": ["2", 0]}},
        }
        with mock.patch.object(worker, "register_prompt_loader_consumers") as register:
            apply_preset_prompt_overrides({"prompt": prompt})
        register.assert_called_once_with({"1": 2})


class MemoryReportTest(unittest.TestCase):
    def test_footprint_rows_account_for_the_whole_device(self):
        warm = residency._memory_rows([{"gpu": "0", "held_bytes": 11 * 1024**3}])
        latest = residency._memory_rows([{"gpu": "0", "held_bytes": 12 * 1024**3}])
        devices = {"0": {"used_gib": 14.0, "free_gib": 18.0, "total_gib": 32.0}}
        row = residency._footprint_rows(warm, latest, devices)[0]
        self.assertEqual(row["model_gib"], 12.0)
        self.assertEqual(row["weights_gib"], 11.0)
        self.assertEqual(row["other_gib"], 2.0)
        self.assertEqual(row["model_gib"] + row["other_gib"] + row["free_gib"], row["total_gib"])

    def test_footprint_falls_back_to_warm_rows_before_any_run(self):
        warm = residency._memory_rows(
            [{"gpu": "3", "held_bytes": 9 * 1024**3, "live_bytes": 8 * 1024**3}]
        )
        row = residency._footprint_rows(warm, [], {})[0]
        self.assertEqual(row["gpu"], "3")
        self.assertEqual(row["model_gib"], 9.0)
        self.assertIsNone(row["other_gib"])

    def test_pair_memory_rows_splits_weights_from_activations(self):
        warm = [
            {
                "gpu": "0",
                "held_bytes": 10 * 1024**3,
                "live_bytes": 9 * 1024**3,
                "alloc_retries": 1,
            }
        ]
        run = [
            {
                "gpu": "0",
                "held_bytes": 18 * 1024**3,
                "live_bytes": 11 * 1024**3,
                "peak_bytes": 16 * 1024**3,
                "peak_held_bytes": 17 * 1024**3,
                "device_total_bytes": 32 * 1024**3,
                "device_free_bytes": 14 * 1024**3,
                "alloc_retries": 4,
            }
        ]
        rows = _pair_memory_rows(residency._memory_rows(warm), residency._memory_rows(run))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["weights_gib"], 10.0)
        self.assertEqual(row["peak_gib"], 17.0)
        self.assertEqual(row["activation_gib"], 7.0)
        self.assertEqual(row["device_used_gib"], 18.0)
        self.assertEqual(row["alloc_retries"], 4)

    def test_run_peak_is_never_below_the_resident_footprint(self):
        warm = residency._memory_rows([{"gpu": "0", "held_bytes": 8 * 1024**3}])
        run = residency._memory_rows(
            [{"gpu": "0", "peak_bytes": 6 * 1024**3, "peak_held_bytes": 8 * 1024**3}]
        )
        row = _pair_memory_rows(warm, run)[0]
        resident = residency._footprint_rows(warm, [], {})[0]["model_gib"]
        self.assertGreaterEqual(row["peak_gib"], resident)
        self.assertEqual(row["peak_allocated_gib"], 6.0)

    def test_activation_delta_never_goes_negative(self):
        warm = residency._memory_rows([{"gpu": "0", "held_bytes": 12 * 1024**3}])
        run = residency._memory_rows([{"gpu": "0", "peak_held_bytes": 11 * 1024**3}])
        self.assertEqual(_pair_memory_rows(warm, run)[0]["activation_gib"], 0.0)

    def test_record_sample_run_memory_round_trips_for_the_node(self):
        metadata = {
            "warm_memory": [{"gpu": "0", "held_bytes": 10 * 1024**3}],
            "run_memory": [{"gpu": "0", "peak_held_bytes": 16 * 1024**3}],
        }
        text = record_sample_run_memory("7", metadata)
        self.assertIn("GPU0 peak 16.0GiB", text)
        rows = sample_run_memory("7")["rows"]
        self.assertEqual(rows[0]["activation_gib"], 6.0)

    def test_record_sample_run_memory_ignores_runs_without_stats(self):
        self.assertEqual(record_sample_run_memory("8", {}), "")
        self.assertEqual(sample_run_memory("8"), {})

    def test_stopping_the_worker_drops_its_recorded_run_peaks(self):
        metadata = {
            "warm_memory": [{"gpu": "0", "held_bytes": 10 * 1024**3}],
            "run_memory": [{"gpu": "0", "peak_held_bytes": 16 * 1024**3}],
        }
        record_sample_run_memory("3", metadata, loader_uid="2")
        record_sample_run_memory("5", metadata, loader_uid="4")
        self.assertTrue(sample_run_memory("3"))

        worker._evict_loader_worker("2")
        self.assertEqual(sample_run_memory("3"), {})
        self.assertTrue(sample_run_memory("5"))


class DeletedNodeReapTest(unittest.TestCase):
    def setUp(self):
        worker._clear_all_runtime_caches()

    def _register(self, *node_ids):
        for node_id in node_ids:
            worker._register_loader_cache(node_id, f"key-{node_id}")

    def test_reap_releases_workers_whose_node_is_gone(self):
        self._register("2", "7")
        with mock.patch.object(worker, "_evict_loader_worker", return_value=True) as evict:
            result = worker._reap_loaders_except(["2"])
        self.assertEqual(result["released"], ["7"])
        evict.assert_called_once_with("7")

    def test_reap_keeps_every_worker_still_present_in_the_graph(self):
        self._register("2", "7")
        with mock.patch.object(worker, "_evict_loader_worker", return_value=True) as evict:
            result = worker._reap_loaders_except(["7", "2"])
        self.assertEqual(result["released"], [])
        evict.assert_not_called()

    def test_reap_with_no_live_nodes_releases_everything(self):
        self._register("2", "7")
        with mock.patch.object(worker, "_evict_loader_worker", return_value=True):
            result = worker._reap_loaders_except([])
        self.assertEqual(sorted(result["released"]), ["2", "7"])

    def test_release_all_frees_every_registered_loader(self):
        self._register("2", "7")
        with mock.patch.object(worker, "_evict_loader_worker", return_value=True):
            result = worker._release_all_loaders()
        self.assertEqual(sorted(result["released"]), ["2", "7"])
        self.assertEqual(worker._registered_loader_ids(), [])


class OomAttributionTest(unittest.TestCase):
    def test_oom_text_names_the_resident_loader_and_device_usage(self):
        report = {
            "loaders": [
                {
                    "node_id": "2",
                    "model": "Tongyi-MAI/Z-Image-Turbo",
                    "warm": True,
                    "parked": False,
                    "state": "gpu_warm",
                    "gpus": ["0", "1"],
                    "footprint": [
                        {
                            "gpu": "0",
                            "model_gib": 10.2,
                            "weights_gib": 9.8,
                            "other_gib": 11.2,
                            "free_gib": 10.5,
                            "total_gib": 31.9,
                        }
                    ],
                },
                {"node_id": "5", "model": "Wan2.2", "warm": False, "parked": False, "gpus": ["0"]},
            ],
            "devices": {"0": {"used_gib": 21.4, "total_gib": 31.9, "free_gib": 10.5}},
        }
        with mock.patch.object(residency, "residency_report", return_value=report):
            text = _oom_attribution_text()
        self.assertIn("GPU 0: 21.4/31.9 GiB used, 10.5 GiB free", text)
        self.assertIn("Model node 2: Tongyi-MAI/Z-Image-Turbo on GPU 0,1", text)
        self.assertIn("gpu_warm", text)
        self.assertIn("10.2GiB this model", text)
        self.assertNotIn("Wan2.2", text)
        self.assertIn("residency=release", text.replace("set ", ""))

    def test_oom_text_is_empty_without_any_known_loader(self):
        with mock.patch.object(
            residency, "residency_report", return_value={"loaders": [], "devices": {}}
        ):
            self.assertEqual(_oom_attribution_text(), "")


class ResidencyReportTest(unittest.TestCase):
    def test_report_pairs_loader_snapshots_with_device_usage(self):
        stats = {
            "warm": [
                {
                    "gpu": "0",
                    "held_bytes": 12 * 1024**3,
                    "live_bytes": 11 * 1024**3,
                    "device_total_bytes": 32 * 1024**3,
                    "device_free_bytes": 20 * 1024**3,
                }
            ]
        }
        devices = {"0": {"free_gib": 20.0, "total_gib": 32.0, "used_gib": 12.0}}
        with (
            mock.patch.dict(REGISTRY.cache_keys, {"2": "abc"}, clear=True),
            mock.patch.dict(
                REGISTRY.snapshots,
                {"2": {"model": "FLUX.1-dev", "gpus": ["0"], "residency": RESIDENCY_KEEP_GPU}},
                clear=True,
            ),
            mock.patch.dict(
                REGISTRY.workers,
                {"2": {"residency_state": "gpu_warm"}},
                clear=True,
            ),
            mock.patch.object(residency, "_worker_memory_stats", return_value=stats),
            mock.patch.object(worker, "_distributed_worker_alive", return_value=True),
            mock.patch.object(residency, "_device_memory", return_value=devices),
        ):
            report = residency_report()
        self.assertEqual(report["devices"], devices)
        entry = report["loaders"][0]
        self.assertTrue(entry["warm"])
        self.assertEqual(entry["node_id"], "2")
        self.assertEqual(entry["footprint"][0]["model_gib"], 12.0)
        self.assertEqual(entry["residency"], RESIDENCY_KEEP_GPU)
        self.assertEqual(entry["state"], "gpu_warm")

    def test_report_shows_no_footprint_for_a_stopped_worker(self):
        with (
            mock.patch.dict(REGISTRY.cache_keys, {"2": "abc"}, clear=True),
            mock.patch.dict(
                REGISTRY.snapshots,
                {"2": {"model": "FLUX.1-dev", "gpus": ["0"]}},
                clear=True,
            ),
            mock.patch.object(worker, "_distributed_worker_alive", return_value=False),
            mock.patch.object(residency, "_device_memory", return_value={}),
        ):
            entry = residency_report()["loaders"][0]
        self.assertFalse(entry["warm"])
        self.assertEqual(entry["footprint"], [])

    def test_report_includes_node_status(self):
        record_node_status("2", "loading weights")
        record_node_status("3", "denoising · 12s")
        try:
            report = residency_report()
            self.assertEqual(report["node_status"]["2"]["text"], "loading weights")
            self.assertEqual(report["node_status"]["3"]["text"], "denoising · 12s")
        finally:
            clear_node_status("2")
            clear_node_status("3")

    def test_report_includes_all_sample_run_summaries(self):
        with mock.patch.dict(
            REGISTRY.run_stats,
            {"3": {"rows": [{"gpu": "0", "peak_gib": 12.5}]}},
            clear=True,
        ):
            report = residency_report()
        self.assertEqual(report["sample_runs"]["3"]["rows"][0]["peak_gib"], 12.5)


if __name__ == "__main__":
    unittest.main()
