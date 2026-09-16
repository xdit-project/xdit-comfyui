"""The ComfyUI side of the worker: spawn, wait, dispatch, reuse, and evict.

Everything here normally happens around a torchrun child holding a model on a GPU, so it
was only ever exercised by hand or by a live run. A stand-in child that speaks the same
socket protocol makes the whole lifecycle testable on any machine, which is where the
expensive bugs live: a worker that is never reused, one that is never released, or a
failed start that leaves a socket behind.
"""

import hashlib
import json
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from tests.support.fake_worker import REAL_POPEN, spawn_fake_worker, worker_init_config
from xdit_comfyui import worker

_REAL_POPEN = REAL_POPEN
_spawn_fake = spawn_fake_worker
_init_config = worker_init_config


class WorkerLifecycleTest(unittest.TestCase):
    def setUp(self):
        worker._clear_all_runtime_caches()
        self.addCleanup(worker._clear_all_runtime_caches)
        self.behaviour = {"FAKE_WORKER_BEHAVIOUR": "serve"}

        def _popen(cmd, **kwargs):
            # Only the torchrun launch is replaced; worker.py also shells out to pgrep.
            if "torch.distributed.run" not in list(cmd):
                return _REAL_POPEN(cmd, **kwargs)
            env = {**(kwargs.pop("env", None) or {}), **self.behaviour}
            return _spawn_fake(cmd, env=env, **kwargs)

        patch = mock.patch.object(worker.subprocess, "Popen", side_effect=_popen)
        patch.start()
        self.addCleanup(patch.stop)

    def _warm(self, loader_uid="7", cache_key="key-a", config=None):
        runtime = config or _init_config()
        # The Model node registers the key before it warms; the browser-facing release
        # paths walk that registration to find the workers.
        worker._register_loader_cache(loader_uid, cache_key, runtime)
        return worker._get_or_create_distributed_worker(
            cache_key,
            runtime,
            {},
            1,
            loader_uid=loader_uid,
            timeout_seconds=60,
        )

    def test_the_suite_cannot_address_a_real_comfyui_worker(self):
        """Workers are found by a token hashed from the node id, and killed by it.

        Without the isolation in conftest, this suite warming node "7" would reap the
        worker of node "7" in a ComfyUI running beside it — including mid-run.
        """
        unsalted = hashlib.sha256(b"7").hexdigest()[:16]
        self.assertNotEqual(unsalted, worker._loader_worker_token("7"))

    def test_warming_a_loader_leaves_a_worker_ready_to_run(self):
        entry, created = self._warm()
        self.assertTrue(created)
        self.assertTrue(Path(entry["socket_path"]).exists())
        self.assertTrue(Path(entry["ready_path"]).is_file())
        self.assertEqual(
            json.loads(Path(entry["config_path"]).read_text())["model"], _init_config()["model"]
        )

    def test_the_same_configuration_reuses_the_warm_worker(self):
        first, created_first = self._warm()
        second, created_second = self._warm()
        self.assertTrue(created_first)
        self.assertFalse(created_second, "a second queue of the same Model respawned the worker")
        self.assertIs(first, second)

    def test_changing_the_configuration_replaces_the_worker(self):
        first, _ = self._warm(cache_key="key-a")
        second, created = self._warm(cache_key="key-b")
        self.assertTrue(created)
        self.assertIsNot(first, second)
        self.assertIsNotNone(first["proc"].poll(), "the previous worker was left running")

    def test_two_model_nodes_hold_two_workers(self):
        first, _ = self._warm(loader_uid="7")
        second, _ = self._warm(loader_uid="8", cache_key="key-b")
        self.assertNotEqual(first["socket_path"], second["socket_path"])
        self.assertEqual({"7", "8"}, set(worker._registered_loader_ids()))

    def test_a_run_reaches_the_worker_and_comes_back(self):
        entry, _ = self._warm()
        output, timings, metadata = worker._run_distributed_worker(
            entry,
            {**_init_config(), "prompt": "a cat"},
            timeout_seconds=60,
            loader_uid="7",
        )
        self.assertEqual({"echo": "a cat"}, output)
        self.assertEqual([0.25], timings)
        self.assertEqual(24, metadata["fps"])
        self.assertEqual(1, entry["run_count"])

    def test_a_timed_out_run_stops_and_unregisters_the_worker(self):
        self.behaviour["FAKE_WORKER_BEHAVIOUR"] = "hang_run"
        entry, _ = self._warm()
        with self.assertRaisesRegex(TimeoutError, "run timeout"):
            worker._run_distributed_worker(
                entry,
                {**_init_config(), "prompt": "never returns"},
                timeout_seconds=0.1,
                loader_uid="7",
            )
        self.assertIsNotNone(entry["proc"].poll())
        self.assertFalse(worker._distributed_worker_alive("7"))

    def test_runs_on_one_worker_are_serialized(self):
        active = 0
        peak = 0
        guard = threading.Lock()

        def fake_run(*_args, **_kwargs):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with guard:
                active -= 1
            return "output", [0.1], {}

        entry = {"run_lock": threading.Lock()}
        with mock.patch.object(worker, "_run_distributed_worker_locked", side_effect=fake_run):
            threads = [
                threading.Thread(
                    target=worker._run_distributed_worker,
                    args=(entry, {}, 1),
                )
                for _ in range(2)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(peak, 1)

    def test_unloading_a_model_releases_its_worker(self):
        entry, _ = self._warm()
        result = worker._clear_loader_cache("7")
        self.assertTrue(result["evicted"])
        self.assertIsNotNone(entry["proc"].poll(), "the worker outlived the Unload button")
        self.assertFalse(Path(entry["socket_path"]).exists())
        self.assertEqual([], list(worker._registered_loader_ids()))

    def test_deleting_the_node_from_the_graph_releases_its_worker(self):
        """The Unload button lives on the node that just went away."""
        entry, _ = self._warm(loader_uid="7")
        worker._reap_loaders_except(["8", "9"])
        self.assertEqual([], list(worker._registered_loader_ids()))
        self.assertIsNotNone(entry["proc"].poll())

    def test_a_worker_that_dies_on_startup_reports_its_own_error(self):
        self.behaviour["FAKE_WORKER_BEHAVIOUR"] = "die"
        with self.assertRaises(RuntimeError) as raised:
            self._warm()
        self.assertIn("no such model", str(raised.exception))

    def test_a_failed_start_leaves_nothing_behind(self):
        self.behaviour["FAKE_WORKER_BEHAVIOUR"] = "die"
        with self.assertRaises(RuntimeError):
            self._warm()
        socket_path, ready_path, _config = worker._distributed_worker_paths(
            worker._loader_worker_token("7")
        )
        self.assertFalse(Path(socket_path).exists())
        self.assertFalse(Path(ready_path).exists())
        self.assertFalse(worker._distributed_worker_alive("7"))

    def test_a_worker_that_never_reports_ready_is_not_waited_on_forever(self):
        self.behaviour["FAKE_WORKER_BEHAVIOUR"] = "hang"
        with mock.patch.object(worker, "_WORKER_INIT_TIMEOUT_SECONDS", 2):
            with self.assertRaises(Exception) as raised:
                self._warm()
        self.assertIn("timed out", str(raised.exception).lower())

    def _forget_workers(self):
        """Simulate a ComfyUI restart: the processes survive, our bookkeeping does not."""
        entries = dict(worker.REGISTRY.workers)
        worker.REGISTRY.workers.clear()
        worker.REGISTRY.cache_keys.clear()
        return entries

    def test_a_worker_left_by_a_previous_comfyui_is_reused(self):
        first, _ = self._warm(cache_key="key-a")
        self._forget_workers()
        second, created = self._warm(cache_key="key-a")
        self.assertFalse(created, "a warm worker was respawned after the restart")
        self.assertIsNone(first["proc"].poll(), "the reusable worker was killed")
        self.assertEqual(first["socket_path"], second["socket_path"])

    def test_a_left_over_worker_is_released_when_the_settings_changed(self):
        """Its token comes from the node id, so a stale one would answer for the old config."""
        first, _ = self._warm(cache_key="key-a")
        self._forget_workers()
        _second, created = self._warm(cache_key="key-b")
        self.assertTrue(created)
        self.assertIsNotNone(first["proc"].poll(), "the stale worker kept holding its GPUs")

    def _orphaned(self, *entries):
        """Make exactly these workers look reparented, and nothing else on the machine.

        The sweep asks the process table who is orphaned, so answering "everyone" would
        volunteer any real worker running beside the test suite.
        """
        pids = {entry["proc"].pid for entry in entries}
        real_parent_pid = worker._parent_pid
        return mock.patch.object(
            worker,
            "_parent_pid",
            lambda pid: 1 if pid in pids else real_parent_pid(pid),
        )

    def test_orphaned_workers_are_released_before_a_new_model_loads(self):
        """A crashed ComfyUI cannot free its GPUs; the next load has to do it."""
        orphan, _ = self._warm(loader_uid="7", cache_key="key-a")
        self._forget_workers()
        with self._orphaned(orphan):
            self._warm(loader_uid="8", cache_key="key-b")
        self.assertIsNotNone(orphan["proc"].poll(), "the orphan kept holding its GPUs")

    def test_a_worker_this_comfyui_owns_is_never_swept(self):
        entry, _ = self._warm(loader_uid="7", cache_key="key-a")
        with self._orphaned(entry):
            released = worker._sweep_orphan_workers()
        self.assertEqual([], released)
        self.assertIsNone(entry["proc"].poll())

    def test_releasing_everything_leaves_no_worker_running(self):
        first, _ = self._warm(loader_uid="7")
        second, _ = self._warm(loader_uid="8", cache_key="key-b")
        worker._release_all_loaders()
        self.assertEqual([], list(worker._registered_loader_ids()))
        for entry in (first, second):
            self.assertIsNotNone(entry["proc"].poll())


if __name__ == "__main__":
    unittest.main()
