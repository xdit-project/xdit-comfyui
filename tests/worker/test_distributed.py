"""The worker child: wire protocol, memory probes, and job loop.

This module only ever executes inside the torchrun child, so until now nothing outside a
live GPU run touched it. The two sides of the socket also live in different files
(`worker.py` in ComfyUI, this one in the child), so a framing change on one side can
only be caught by testing them against each other.
"""

import json
import os
import pickle
import signal
import socket
import struct
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

from xdit_comfyui import worker_server as dw
from xdit_comfyui.worker import _socket_read_exact, _socket_write_json


def _read_framed(conn):
    """Read one framed message the way ComfyUI does."""
    (size,) = struct.unpack("!I", _socket_read_exact(conn, 4, poll_interval=0.05))
    return _socket_read_exact(conn, size, poll_interval=0.05)


def _read_framed_bounded(conn, timeout=30.0):
    """As above, but gives up: ComfyUI polls forever so a dead worker would hang here."""
    import time

    deadline = time.time() + timeout
    chunks = bytearray()

    def _take(size):
        while len(chunks) < size:
            if time.time() > deadline:
                raise AssertionError("worker sent no reply")
            conn.settimeout(0.1)
            try:
                received = conn.recv(size - len(chunks))
            except socket.timeout:
                continue
            if not received:
                raise ConnectionError("worker closed the connection")
            chunks.extend(received)
        taken = bytes(chunks[:size])
        del chunks[:size]
        return taken

    (size,) = struct.unpack("!I", _take(4))
    return _take(size)


class WireProtocolTest(unittest.TestCase):
    """ComfyUI writes with worker.py helpers; the child reads with its own."""

    def setUp(self):
        self.parent, self.child = socket.socketpair()
        self.addCleanup(self.parent.close)
        self.addCleanup(self.child.close)

    def test_a_job_written_by_comfyui_is_read_by_the_worker(self):
        job = {"op": "run", "config": {"prompt": "a cat", "num_inference_steps": 8}}
        _socket_write_json(self.parent, job)
        self.assertEqual(job, dw._read_message(self.child))

    def test_a_result_written_by_the_worker_is_read_by_comfyui(self):
        result = {"output": [1, 2, 3], "timings": [0.5], "metadata": {"fps": 24}}
        dw._write_pickle(self.child, result)
        self.assertEqual(result, pickle.loads(_read_framed(self.parent)))

    def test_a_json_reply_from_the_worker_is_read_by_comfyui(self):
        dw._write_message(self.child, {"ok": False, "error": "unknown op: nope"})
        self.assertEqual(
            {"ok": False, "error": "unknown op: nope"},
            json.loads(bytes(_read_framed(self.parent))),
        )

    def test_a_payload_larger_than_one_recv_is_reassembled(self):
        """A video result is hundreds of MiB and never arrives in one chunk."""
        payload = {"output": "x" * (4 << 20)}
        writer = threading.Thread(target=dw._write_pickle, args=(self.child, payload))
        writer.daemon = True
        writer.start()
        self.assertEqual(payload, pickle.loads(_read_framed(self.parent)))
        writer.join(timeout=10)

    def test_values_json_cannot_encode_are_stringified_rather_than_dropped(self):
        _socket_write_json(self.parent, {"op": "run", "config": {"path": Path("/tmp/x")}})
        self.assertEqual("/tmp/x", dw._read_message(self.child)["config"]["path"])

    def test_a_closed_connection_is_reported_not_hung(self):
        self.parent.sendall(struct.pack("!I", 64))
        self.parent.close()
        with self.assertRaises(ConnectionError):
            dw._read_message(self.child)


class MemoryProbeTest(unittest.TestCase):
    def test_physical_gpu_id_maps_through_the_visible_device_list(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5"}):
            self.assertEqual("5", dw._physical_gpu_id(1))

    def test_physical_gpu_id_falls_back_to_the_local_index(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            self.assertEqual("2", dw._physical_gpu_id(2))
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}):
            self.assertEqual("1", dw._physical_gpu_id(1))

    def test_memory_probes_are_quiet_without_a_gpu(self):
        """CI has no GPU, and neither does a CPU-only debug run."""
        import torch

        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertIsNone(dw._memory_stats())
            self.assertEqual(0, dw._alloc_retries())
            dw._reset_peak_memory()
            world = types.SimpleNamespace(rank=0, world_size=1)
            self.assertEqual([], dw._gather_memory_stats(world))

    def test_retries_are_reported_relative_to_the_run(self):
        import torch

        stats = {"num_alloc_retries": 7}
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            mock.patch.object(torch.cuda, "memory_stats", return_value=stats),
            mock.patch.object(torch.cuda, "mem_get_info", return_value=(1, 2)),
            mock.patch.object(torch.cuda, "memory_reserved", return_value=3),
            mock.patch.object(torch.cuda, "memory_allocated", return_value=4),
            mock.patch.object(torch.cuda, "max_memory_allocated", return_value=5),
            mock.patch.object(torch.cuda, "max_memory_reserved", return_value=6),
        ):
            self.assertEqual(5, dw._memory_stats(retry_baseline=2)["alloc_retries"])
            self.assertEqual(0, dw._memory_stats(retry_baseline=99)["alloc_retries"])

    def test_every_rank_is_represented_in_a_multi_gpu_report(self):
        import torch.distributed as dist

        def _fill(gathered, _snapshot):
            gathered[0] = {"gpu": "0"}
            gathered[1] = {"gpu": "1"}

        with (
            mock.patch.object(dw, "_memory_stats", return_value={"gpu": "0"}),
            mock.patch.object(dist, "all_gather_object", side_effect=_fill),
        ):
            world = types.SimpleNamespace(rank=0, world_size=2)
            self.assertEqual([{"gpu": "0"}, {"gpu": "1"}], dw._gather_memory_stats(world))

    def test_a_single_rank_reports_without_a_collective(self):
        with mock.patch.object(dw, "_memory_stats", return_value={"gpu": "0"}):
            world = types.SimpleNamespace(rank=0, world_size=1)
            self.assertEqual([{"gpu": "0"}], dw._gather_memory_stats(world))


class BroadcastTest(unittest.TestCase):
    def test_every_rank_runs_the_job_the_output_rank_received(self):
        import torch.distributed as dist

        def _fill(obj, src):
            self.assertEqual(1, src)
            obj[0] = {"op": "run", "config": {"prompt": "from rank 1"}}

        with mock.patch.object(dist, "broadcast_object_list", side_effect=_fill):
            self.assertEqual(
                {"op": "run", "config": {"prompt": "from rank 1"}},
                dw._broadcast_job(None, 1),
            )

    def test_worker_commands_use_the_cpu_control_group(self):
        import torch.distributed as dist

        group = object()
        with mock.patch.object(dist, "broadcast_object_list") as broadcast:
            dw._broadcast_job({"op": "shutdown"}, 7, group)
        broadcast.assert_called_once_with([{"op": "shutdown"}], src=7, group=group)

    def test_multi_rank_control_group_uses_gloo(self):
        import torch.distributed as dist

        group = object()
        with mock.patch.object(dist, "new_group", return_value=group) as new_group:
            self.assertIs(dw._create_control_group(8), group)
        new_group.assert_called_once_with(backend="gloo")

    def test_single_rank_worker_needs_no_control_group(self):
        self.assertIsNone(dw._create_control_group(1))


class StatsFileTest(unittest.TestCase):
    def test_stats_are_written_as_json(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "worker.sock.stats"
            dw._write_stats_file(path, {"warm": [{"gpu": "0"}]})
            self.assertEqual({"warm": [{"gpu": "0"}]}, json.loads(path.read_text()))

    def test_an_unwritable_stats_path_never_fails_the_run(self):
        dw._write_stats_file(Path("/proc/nope/worker.stats"), {"warm": []})


class _FakeRunner:
    """Stands in for xFuserModelRunner."""

    def __init__(self, _config):
        self.model = types.SimpleNamespace(
            pipe=types.SimpleNamespace(_interrupt=False),
            settings=types.SimpleNamespace(fps=16, model_output_type="video"),
        )
        self.cleaned = False
        self.runs = []

    def preprocess_args(self, config):
        return dict(config, height=480, width=832)

    def initialize(self, _args):
        return None

    def run(self, args):
        self.runs.append(args)
        return "OUTPUT", [1.5]

    def cleanup(self):
        self.cleaned = True


class JobLoopTest(unittest.TestCase):
    """Drive main() end to end with a stand-in runner, no GPU and no xfuser."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.socket_path = str(Path(self.tmp.name) / "w.sock")
        self.config_path = Path(self.tmp.name) / "init.json"
        self.config_path.write_text(json.dumps({"model": "m", "cache_method": "none"}))
        self.runners = []

        world = types.SimpleNamespace(rank=0, world_size=1)

        def _make_runner(config):
            runner = _FakeRunner(config)
            self.runners.append(runner)
            return runner

        # Patch the names on the real xfuser modules rather than shadowing them in
        # sys.modules: a stub package breaks every later xfuser import inside main().
        self.patches = [
            mock.patch("xfuser.core.distributed.get_world_group", return_value=world),
            mock.patch("xfuser.runner.xFuserModelRunner", _make_runner),
            # main() installs a SIGUSR1 handler, which only the main thread may do.
            mock.patch("signal.signal"),
            # A single rank has nothing to broadcast, and torch.distributed refuses to
            # run a collective without a process group.
            mock.patch.object(dw, "_broadcast_job", side_effect=lambda job, _src, _group=None: job),
            mock.patch.object(dw, "_gather_memory_stats", return_value=[{"gpu": "0"}]),
            mock.patch("xdit_comfyui.xdit_ext.step_cache.maybe_refresh_step_cache"),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

        self.thread = threading.Thread(target=dw.main, daemon=True)
        self.errors = []
        self.thread = threading.Thread(target=self._run_main, daemon=True)
        self.thread.start()
        self._wait_for_ready()

    def _run_main(self):
        with mock.patch.object(sys, "argv", ["dw", self.socket_path, str(self.config_path)]):
            try:
                dw.main()
            except BaseException as exc:  # surfaced by the assertions below
                self.errors.append(exc)

    def _wait_for_ready(self, timeout=10.0):
        import time

        ready = Path(f"{self.socket_path}.ready")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if ready.is_file():
                return
            if self.errors:
                raise self.errors[0]
            time.sleep(0.02)
        raise AssertionError("worker never became ready")

    def _connect(self):
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.settimeout(10)
        conn.connect(self.socket_path)
        return conn

    def _reply(self, conn):
        try:
            return _read_framed_bounded(conn)
        except (AssertionError, ConnectionError):
            if self.errors:
                raise self.errors[0]
            raise

    def _shutdown(self):
        conn = self._connect()
        _socket_write_json(conn, {"op": "shutdown"})
        conn.close()
        self.thread.join(timeout=10)
        self.assertFalse(self.thread.is_alive())
        self.assertEqual([], self.errors)

    def test_a_run_job_answers_with_the_output_and_its_metadata(self):
        conn = self._connect()
        _socket_write_json(
            conn,
            {"op": "run", "config": {"prompt": "a cat", "num_inference_steps": 4}},
        )
        result = pickle.loads(self._reply(conn))
        conn.close()
        self.assertEqual("OUTPUT", result["output"])
        self.assertEqual([1.5], result["timings"])
        self.assertEqual(16, result["metadata"]["fps"])
        self.assertEqual(480, result["metadata"]["preprocessed_height"])
        self.assertEqual([{"gpu": "0"}], result["metadata"]["run_memory"])
        self.assertEqual("a cat", self.runners[0].runs[0]["prompt"])
        self._shutdown()

    def test_the_stats_file_carries_the_warm_and_run_footprints(self):
        conn = self._connect()
        _socket_write_json(conn, {"op": "run", "config": {"prompt": "x"}})
        pickle.loads(self._reply(conn))
        conn.close()
        stats = json.loads(Path(f"{self.socket_path}.stats").read_text())
        self.assertEqual([{"gpu": "0"}], stats["warm"])
        self.assertEqual([{"gpu": "0"}], stats["run"])
        self._shutdown()

    def test_an_unknown_op_is_answered_rather_than_ignored(self):
        conn = self._connect()
        _socket_write_json(conn, {"op": "nonsense"})
        reply = json.loads(bytes(self._reply(conn)))
        conn.close()
        self.assertFalse(reply["ok"])
        self.assertIn("nonsense", reply["error"])
        self._shutdown()

    def test_the_worker_keeps_serving_after_a_bad_job(self):
        conn = self._connect()
        _socket_write_json(conn, {"op": "nonsense"})
        self._reply(conn)
        conn.close()

        conn = self._connect()
        _socket_write_json(conn, {"op": "run", "config": {"prompt": "still here"}})
        self.assertEqual("OUTPUT", pickle.loads(self._reply(conn))["output"])
        conn.close()
        self._shutdown()

    def test_cancelling_from_comfyui_interrupts_the_pipeline(self):
        """ComfyUI's Cancel reaches the worker as SIGUSR1."""
        handler = signal.signal.call_args[0][1]
        pipe = self.runners[0].model.pipe
        pipe._interrupt = False
        handler(signal.SIGUSR1, None)
        self.assertTrue(pipe._interrupt)

        conn = self._connect()
        _socket_write_json(conn, {"op": "run", "config": {"prompt": "next"}})
        self._reply(conn)
        conn.close()
        self.assertFalse(pipe._interrupt, "a new run must start uninterrupted")
        self._shutdown()

    def test_shutdown_releases_the_socket_and_the_model(self):
        self._shutdown()
        self.assertFalse(Path(self.socket_path).exists())
        self.assertFalse(Path(f"{self.socket_path}.ready").exists())
        self.assertTrue(self.runners[0].cleaned)

    def test_a_quick_run_skips_inference_but_still_replies(self):
        conn = self._connect()
        _socket_write_json(
            conn,
            {"op": "run", "config": {"prompt": "x", "quick_run": True, "num_frames": 1}},
        )
        result = pickle.loads(self._reply(conn))
        conn.close()
        self.assertEqual([], self.runners[0].runs, "quick_run must not touch the model")
        self.assertIsNotNone(result["output"])
        self._shutdown()


class EntryPointTest(unittest.TestCase):
    def test_the_worker_refuses_to_start_without_its_two_paths(self):
        with mock.patch.object(sys, "argv", ["distributed_worker.py"]):
            with self.assertRaises(SystemExit):
                dw.main()


if __name__ == "__main__":
    unittest.main()
