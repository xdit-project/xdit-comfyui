"""Worker process lifecycle, diagnostics, and progress reporting."""

import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from xdit_comfyui.progress import (
    _INIT_DISPLAY_TOTAL,
    _INIT_STEP_COUNT,
    _PHASE_WAIT_LOG_SECONDS,
    _drain_fragments,
    _handle_subprocess_fragment,
    _init_stage_display,
    _is_comfy_interrupt,
    _parse_fetch_progress,
    _parse_step_progress,
    _XDiTProgressTracker,
)
from xdit_comfyui.runtime_env import _allocate_master_port
from xdit_comfyui.worker import (
    _abort_distributed_worker,
    _clear_all_runtime_caches,
    _reap_children_nonblocking,
    _run_distributed_worker,
    _terminate_group,
    _terminate_worker_proc,
)


class ProgressParseTest(unittest.TestCase):
    def test_parse_step_progress_last_match(self):
        self.assertEqual(_parse_step_progress("Loading 2/2 | step 5/28"), (5, 28))

    def test_parse_step_progress_tqdm_eta_does_not_steal_denoise_step(self):
        self.assertEqual(
            _parse_step_progress("  2%|▏         | 1/50 [00:04<04:04,  4.98s/it]"),
            (1, 50),
        )
        self.assertEqual(
            _parse_step_progress(" 34%|███▍      | 17/50 [00:55<01:43,  3.14s/it]"),
            (17, 50),
        )

    def test_parse_step_progress_equal(self):
        self.assertEqual(_parse_step_progress("28/28"), (28, 28))

    def test_parse_step_progress_none(self):
        self.assertIsNone(_parse_step_progress("no numbers here"))

    def test_parse_step_progress_rejects_overflow(self):
        self.assertIsNone(_parse_step_progress("30/28"))

    def test_parse_step_progress_rejects_zero_total(self):
        self.assertIsNone(_parse_step_progress("0/0"))

    def test_drain_fragments_cr_and_lf(self):
        frags, remainder = _drain_fragments("a\rb\nc")
        self.assertEqual(frags, ["a", "b"])
        self.assertEqual(remainder, "c")

    def test_drain_fragments_keeps_partial(self):
        frags, remainder = _drain_fragments("partial")
        self.assertEqual(frags, [])
        self.assertEqual(remainder, "partial")

    def test_parse_fetch_progress_percent(self):
        self.assertEqual(
            _parse_fetch_progress("Fetching 32 files:  50%|█████     | 16/32"), (50, 100)
        )

    def test_handle_subprocess_fragment_warmup(self):
        tracker = _XDiTProgressTracker(node_id="node-1", inference_steps=0, include_init=True)
        _handle_subprocess_fragment("Warmup iteration 1/2", tracker)
        self.assertGreaterEqual(tracker._init.completed, _init_stage_display(5))

    def test_progress_tracker_init_tqdm_ticks_bump_one_percent(self):
        tracker = _XDiTProgressTracker(init_node_id="loader", inference_steps=0, include_init=True)
        tracker.on_fragment("Loading model pipeline")
        start = tracker._init.completed
        tracker.on_tqdm(1, 100, "Loading weights 1/100")
        self.assertEqual(tracker._init.completed, start + 1)
        tracker.on_tqdm(2, 100, "Loading weights 2/100")
        self.assertEqual(tracker._init.completed, start + 2)
        tracker.on_tqdm(2, 100, "Loading weights 2/100")
        self.assertEqual(tracker._init.completed, start + 2)

    def test_handle_subprocess_fragment_step(self):
        tracker = _XDiTProgressTracker(
            inference_node_id="node-1", inference_steps=4, include_init=False
        )
        _handle_subprocess_fragment("  3/4", tracker)
        self.assertEqual(tracker.completed, 3)

    def test_handle_subprocess_fragment_tqdm_with_eta(self):
        tracker = _XDiTProgressTracker(
            inference_node_id="node-1", inference_steps=50, include_init=False
        )
        _handle_subprocess_fragment("  20%|██        | 10/50 [00:33<02:06,  3.15s/it]", tracker)
        self.assertEqual(tracker.completed, 10)

    def test_progress_tracker_init_plus_inference_total(self):
        tracker = _XDiTProgressTracker(node_id="1", inference_steps=4, include_init=True)
        self.assertEqual(tracker.total, _INIT_STEP_COUNT + 6)

    def test_progress_tracker_phase_floors_compile_without_tqdm(self):
        tracker = _XDiTProgressTracker(node_id="1", inference_steps=4, include_init=True)
        tracker.on_fragment("Torch.compile enabled")
        self.assertEqual(tracker.completed, _init_stage_display(4))

    def test_progress_tracker_cumulative_never_regresses(self):
        tracker = _XDiTProgressTracker(node_id="1", inference_steps=4, include_init=True)
        tracker.on_fragment("Initializing model")
        tracker.on_fragment("Torch.compile enabled")
        tracker.on_fragment("  1/2")
        first = tracker.completed
        tracker.on_fragment("Initializing model")
        self.assertEqual(tracker.completed, first)

    def test_loader_node_id_fallback_from_cache_key(self):
        from xdit_comfyui.registry import _loader_node_id_for_runtime
        from xdit_comfyui.worker import _register_loader_cache

        _register_loader_cache("loader-99", "cachekey99")
        self.assertEqual(
            _loader_node_id_for_runtime({"_cache_key": "cachekey99"}),
            "loader-99",
        )
        self.assertEqual(
            _loader_node_id_for_runtime(
                {
                    "_loader_node_id": "explicit",
                    "_cache_key": "cachekey99",
                }
            ),
            "explicit",
        )

    def test_progress_tracker_splits_init_and_inference_nodes(self):
        tracker = _XDiTProgressTracker(
            init_node_id="loader-1",
            inference_node_id="gen-1",
            inference_steps=4,
            include_init=True,
        )
        self.assertIsNotNone(tracker._init)
        self.assertIsNotNone(tracker._infer)
        self.assertEqual(tracker._init.node_id, "loader-1")
        self.assertEqual(tracker._infer.node_id, "gen-1")
        self.assertEqual(tracker._init.total, _INIT_DISPLAY_TOTAL)
        self.assertEqual(tracker._infer.total, 6)
        tracker.on_scheduler_step(2)
        self.assertEqual(tracker._infer.completed, 2)
        self.assertEqual(tracker._init.completed, _INIT_DISPLAY_TOTAL)

    def test_progress_tracker_reserves_decode_and_finalization_slots(self):
        tracker = _XDiTProgressTracker(
            inference_node_id="sample-1",
            inference_steps=4,
            include_init=False,
        )
        tracker.on_scheduler_step(4)
        self.assertEqual(tracker._phase, "decode")
        self.assertEqual(tracker._infer.completed, 4)
        self.assertLess(tracker._infer.completed, tracker._infer.total)
        tracker.on_decode_complete()
        self.assertEqual(tracker._phase, "finalization")
        self.assertEqual(tracker._infer.completed, 5)
        tracker.on_finalization_complete()
        self.assertEqual(tracker._infer.completed, 6)

    def test_the_phases_after_denoising_say_what_they_are_doing(self):
        """A VAE decode prints nothing of its own, so the run looks hung without this."""
        tracker = _XDiTProgressTracker(
            inference_node_id="sample-1",
            inference_steps=4,
            include_init=False,
        )
        with self.assertLogs("xdit", level="INFO") as logged:
            tracker.on_scheduler_step(4)
            tracker.on_decode_complete()
        messages = "\n".join(logged.output)
        self.assertIn("decoding latents", messages)
        self.assertIn("VAE decode complete", messages)

    def test_the_phase_reaches_the_info_block_and_not_the_bottom_of_the_node(self):
        """One place per fact: the info block renders this, so ComfyUI must not repeat it.

        `send_progress_text` draws the same string under the node, which put the phase on
        screen twice for the whole run.
        """
        from xdit_comfyui.registry import clear_node_status, node_status_snapshot

        sent = []
        server = types.ModuleType("server")
        server.PromptServer = types.SimpleNamespace(
            instance=types.SimpleNamespace(
                send_progress_text=lambda text, node_id: sent.append((node_id, text))
            )
        )
        clear_node_status("sample-1")
        self.addCleanup(clear_node_status, "sample-1")
        with mock.patch.dict(sys.modules, {"server": server}):
            tracker = _XDiTProgressTracker(
                inference_node_id="sample-1",
                inference_steps=4,
                include_init=False,
            )
            tracker.on_scheduler_step(4)
            tracker.finish()

        self.assertEqual(sent, [])
        self.assertEqual(node_status_snapshot()["sample-1"]["text"], "done")

    def test_the_status_lands_on_the_node_comfyui_is_running(self):
        """Text addressed to any other node is dropped, and warm-up can start under Sample."""
        context = types.ModuleType("comfy_execution.utils")
        context.get_executing_context = lambda: types.SimpleNamespace(node_id="sample-1")
        parent = types.ModuleType("comfy_execution")
        with mock.patch.dict(
            sys.modules,
            {"comfy_execution": parent, "comfy_execution.utils": context},
        ):
            tracker = _XDiTProgressTracker(init_node_id="loader-1", include_init=True)
        self.assertEqual(tracker._status_node_id, "sample-1")

    def test_a_long_decode_keeps_reporting_that_it_is_alive(self):
        tracker = _XDiTProgressTracker(
            inference_node_id="sample-1",
            inference_steps=4,
            include_init=False,
        )
        tracker.on_scheduler_step(4)
        tracker._phase_logged_at -= _PHASE_WAIT_LOG_SECONDS + 1
        with self.assertLogs("xdit", level="INFO") as logged:
            tracker.heartbeat()
        self.assertIn("Still decoding", "\n".join(logged.output))
        # One line per interval, not one per poll.
        with self.assertRaises(AssertionError):
            with self.assertLogs("xdit", level="INFO"):
                tracker.heartbeat()

    def test_progress_tracker_inference_tqdm_before_run_markers(self):
        tracker = _XDiTProgressTracker(
            inference_node_id="sample-1",
            inference_steps=50,
            include_init=False,
        )
        tracker._phase = "loading"
        tracker.on_tqdm(12, 50, " 24%|██▍       | 12/50 [00:39<02:00,  3.14s/it]")
        self.assertEqual(tracker._infer.completed, 12)

    def test_progress_tracker_compile_warmup_stays_on_init_node(self):
        tracker = _XDiTProgressTracker(
            init_node_id="loader-1",
            inference_node_id="sample-1",
            inference_steps=50,
            include_init=True,
        )
        tracker.on_fragment("Torch.compile enabled")
        tracker.on_tqdm(10, 50, "Warming up torch compiler")
        self.assertEqual(tracker._infer.completed, 0)
        self.assertGreaterEqual(tracker._init.completed, _init_stage_display(5))
        tracker.on_fragment("Running model...")
        tracker.on_tqdm(10, 50, "Running iteration 1/1")
        self.assertEqual(tracker._infer.completed, 10)

    def test_background_progress_cannot_consume_the_comfy_interrupt(self):
        tracker = _XDiTProgressTracker(
            init_node_id="loader-1",
            inference_node_id="sample-1",
            inference_steps=4,
            include_init=True,
        )
        with mock.patch(
            "xdit_comfyui.progress._raise_if_interrupted",
            side_effect=AssertionError("background thread consumed interrupt"),
        ) as check:
            tracker.heartbeat(check_interrupt=False)
            tracker.on_fragment(
                "100%|██████████| 4/4 [00:01<00:00]",
                check_interrupt=False,
            )
        check.assert_not_called()

    def test_cooperative_interrupt_finds_the_renamed_worker_ranks(self):
        import signal
        import types

        from xdit_comfyui.worker import _interrupt_distributed_worker

        rank = mock.Mock()
        rank.cmdline.return_value = [
            sys.executable,
            "-m",
            "xdit_comfyui.worker_server",
        ]
        parent = mock.Mock()
        parent.children.return_value = [rank]
        psutil = types.SimpleNamespace(Process=mock.Mock(return_value=parent))
        proc = mock.Mock(pid=123)
        proc.poll.return_value = None
        with mock.patch.dict(sys.modules, {"psutil": psutil}):
            self.assertTrue(_interrupt_distributed_worker({"proc": proc}))
        rank.send_signal.assert_called_once_with(signal.SIGUSR1)

    def test_xfuser_model_choices_offer_every_key_once(self):
        """Aliases are xfuser's to define; the list only has to be unique and resolvable."""
        from xdit_comfyui.runtime_config import xdit_model_choices

        choices = [c for c in xdit_model_choices() if c != "Custom (HF repo id)"]
        self.assertEqual(len(choices), len(set(choices)))
        self.assertIn("black-forest-labs/FLUX.1-dev", choices)
        self.assertIn("Wan-AI/Wan2.2-I2V-A14B-Diffusers", choices)

    def test_distributed_worker_ready_path_matches_worker(self):
        from xdit_comfyui.worker import _distributed_worker_paths

        socket_path, ready_path, _config_path = _distributed_worker_paths("abc123")
        self.assertTrue(socket_path.endswith(".sock"))
        self.assertEqual(ready_path, f"{socket_path}.ready")

    def test_worker_paths_are_private_and_namespaced_per_comfy_instance(self):
        import stat

        from xdit_comfyui.worker import _distributed_worker_paths

        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch("xdit_comfyui.worker.tempfile.gettempdir", return_value=tmp),
                mock.patch.dict(os.environ, {"XDIT_WORKER_NAMESPACE": "server-a"}),
            ):
                first = _distributed_worker_paths("same-node")
                runtime_dir = Path(first[0]).parent
                self.assertEqual(stat.S_IMODE(runtime_dir.stat().st_mode), 0o700)
            with (
                mock.patch("xdit_comfyui.worker.tempfile.gettempdir", return_value=tmp),
                mock.patch.dict(os.environ, {"XDIT_WORKER_NAMESPACE": "server-b"}),
            ):
                second = _distributed_worker_paths("same-node")
        self.assertNotEqual(first, second)

    def test_wait_for_worker_ready_rejects_stale_ready(self):
        from xdit_comfyui.worker import _wait_for_worker_ready

        with tempfile.TemporaryDirectory() as tmp:
            ready = Path(tmp) / "worker.ready"
            ready.write_text("ok", encoding="utf-8")
            past = time.time() - 60
            os.utime(ready, (past, past))
            proc = mock.Mock()
            proc.poll.return_value = None
            with self.assertRaises(TimeoutError):
                _wait_for_worker_ready(
                    str(ready), proc, timeout_seconds=0.5, started_at=time.time()
                )

    def test_startup_oom_is_named_even_when_torchrun_buries_it(self):
        """A 4-rank failure ends with elastic's summary, far past the OOM that caused it.

        The tail-only check reported a bare "exited during startup (code 1)" for exactly
        the multi-GPU case where a co-resident model is the likely cause, so the user got
        neither the allocator error nor the list of who was holding the GPUs.
        """
        from xdit_comfyui.worker import _wait_for_worker_ready

        oom = (
            "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 1.25 GiB. "
            "GPU 2 has a total capacity of 31.86 GiB of which 334.00 MiB is free."
        )
        # One traceback per surviving rank, then elastic's summary: ~90 lines of noise
        # after the real error, against a 40-line tail window.
        burial = "\n".join(
            ["Traceback (most recent call last):", "  File ..., line 1, in <module>"] * 20
            + ["torch.distributed.elastic.multiprocessing.errors.ChildFailedError:"]
            + [f"  rank {rank}: SIGTERM" for rank in range(1, 4)]
            + ["=" * 60] * 45
        )

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "worker.log"
            log_path.write_text(f"loading weights\n{oom}\n{burial}\n", encoding="utf-8")
            proc = mock.Mock()
            proc.poll.return_value = 1
            proc.returncode = 1

            with self.assertRaises(RuntimeError) as caught:
                _wait_for_worker_ready(
                    str(Path(tmp) / "worker.ready"),
                    proc,
                    timeout_seconds=5,
                    started_at=time.time(),
                    log_path=str(log_path),
                )

        message = str(caught.exception)
        self.assertIn("OOM during startup", message)
        self.assertIn("334.00 MiB is free", message)

    def test_a_startup_failure_that_is_not_an_oom_is_not_reported_as_one(self):
        from xdit_comfyui.worker import _wait_for_worker_ready

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "worker.log"
            log_path.write_text("ValueError: no such model\n", encoding="utf-8")
            proc = mock.Mock()
            proc.poll.return_value = 1
            proc.returncode = 1

            with self.assertRaises(RuntimeError) as caught:
                _wait_for_worker_ready(
                    str(Path(tmp) / "worker.ready"),
                    proc,
                    timeout_seconds=5,
                    started_at=time.time(),
                    log_path=str(log_path),
                )

        message = str(caught.exception)
        self.assertIn("exited during startup", message)
        self.assertNotIn("OOM", message)

    def test_normalize_timeout_seconds_rejects_bool_and_low_values(self):
        from xdit_comfyui.runtime_config import _normalize_timeout_seconds

        self.assertEqual(_normalize_timeout_seconds(True), 900)
        self.assertEqual(_normalize_timeout_seconds(1), 900)
        self.assertEqual(_normalize_timeout_seconds(""), 900)
        self.assertEqual(_normalize_timeout_seconds(120), 120)

    def test_allocate_master_port_unique_per_call_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MASTER_PORT", None)
            with mock.patch("xdit_comfyui.runtime_env._port_is_free", return_value=True):
                p1 = _allocate_master_port()
                p2 = _allocate_master_port()
        self.assertEqual(p1, p2)


class ChildReaperTest(unittest.TestCase):
    def test_reap_children_nonblocking_empty(self):
        self.assertEqual(_reap_children_nonblocking(), 0)

    def test_no_sigchld_handler_is_installed(self):
        """A waitpid(-1) SIGCHLD handler steals exit statuses from subprocess."""
        import signal

        from xdit_comfyui import worker as worker_mod  # noqa: F401

        self.assertIn(signal.getsignal(signal.SIGCHLD), (signal.SIG_DFL, signal.SIG_IGN, None))

    def test_subprocess_failures_are_not_swallowed(self):
        import subprocess

        from xdit_comfyui import worker as worker_mod  # noqa: F401

        for _ in range(20):
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.check_output(["/bin/sh", "-c", "exit 7"])

    def test_terminate_group_reaps_exited(self):
        proc = mock.Mock()
        proc.poll.return_value = 0
        _terminate_group(proc)
        proc.wait.assert_not_called()

    def test_is_comfy_interrupt_detects_exception(self):
        exc = type("InterruptProcessingException", (BaseException,), {})()
        self.assertTrue(_is_comfy_interrupt(exc))

    def test_progress_tracker_checks_interrupt_on_heartbeat(self):
        tracker = _XDiTProgressTracker(init_node_id=1, include_init=True)
        with mock.patch("xdit_comfyui.progress._raise_if_interrupted") as interrupted:
            tracker.heartbeat()
        interrupted.assert_called_once()

    def test_abort_distributed_worker_terminates_live_proc(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        proc.pid = 4242
        entry = {"proc": proc, "socket_path": "/tmp/x.sock", "ready_path": "/tmp/x.sock.ready"}
        with mock.patch("xdit_comfyui.worker._terminate_worker_proc") as terminate:
            with mock.patch("pathlib.Path.unlink"):
                _abort_distributed_worker(entry, loader_uid="loader-42")
        terminate.assert_called_once_with(proc, fast=False)

    def test_run_distributed_worker_cooperatively_interrupts_worker(self):
        entry = {
            "proc": mock.Mock(poll=mock.Mock(return_value=None)),
            "socket_path": "/tmp/x.sock",
            "ready_path": "/tmp/x.sock.ready",
            "log_path": None,
            "run_count": 0,
            "last_used_at": 0,
        }
        interrupt = type("InterruptProcessingException", (BaseException,), {})()
        with mock.patch("xdit_comfyui.worker._is_comfy_interrupt", return_value=True):
            with mock.patch(
                "xdit_comfyui.worker._interrupt_distributed_worker",
                return_value=True,
            ) as cancel:
                with mock.patch(
                    "xdit_comfyui.worker._wait_for_cooperative_cancel",
                    return_value=True,
                ):
                    with mock.patch("xdit_comfyui.worker._abort_distributed_worker") as abort:
                        with mock.patch("xdit_comfyui.worker.socket.socket") as socket_cls:
                            conn = socket_cls.return_value
                            conn.connect.side_effect = interrupt
                            with self.assertRaises(BaseException):
                                _run_distributed_worker(
                                    entry,
                                    {"num_inference_steps": 4},
                                    30,
                                    loader_uid="loader-42",
                                )
        cancel.assert_called_once_with(entry)
        abort.assert_not_called()

    def test_run_distributed_worker_aborts_when_cooperative_interrupt_fails(self):
        entry = {
            "proc": mock.Mock(),
            "socket_path": "/tmp/x.sock",
            "run_count": 0,
            "last_used_at": 0,
        }
        interrupt = type("InterruptProcessingException", (BaseException,), {})()
        with mock.patch("xdit_comfyui.worker._is_comfy_interrupt", return_value=True):
            with mock.patch(
                "xdit_comfyui.worker._interrupt_distributed_worker",
                return_value=False,
            ):
                with mock.patch("xdit_comfyui.worker._abort_distributed_worker") as abort:
                    with mock.patch("xdit_comfyui.worker.socket.socket") as socket_cls:
                        conn = socket_cls.return_value
                        conn.connect.side_effect = interrupt
                        with self.assertRaises(BaseException):
                            _run_distributed_worker(
                                entry,
                                {"num_inference_steps": 4},
                                30,
                                loader_uid="loader-42",
                            )
        abort.assert_called_once_with(entry, loader_uid="loader-42", fast=True)

    def test_run_distributed_worker_aborts_interrupt_during_decode(self):
        entry = {
            "proc": mock.Mock(poll=mock.Mock(return_value=None)),
            "socket_path": "/tmp/x.sock",
            "run_count": 0,
            "last_used_at": 0,
        }
        tracker = mock.Mock()
        tracker._phase = "decode"
        interrupt = type("InterruptProcessingException", (BaseException,), {})()
        with (
            mock.patch("xdit_comfyui.worker._is_comfy_interrupt", return_value=True),
            mock.patch("xdit_comfyui.worker._interrupt_distributed_worker") as cancel,
            mock.patch("xdit_comfyui.worker._abort_distributed_worker") as abort,
            mock.patch("xdit_comfyui.worker.socket.socket") as socket_cls,
        ):
            socket_cls.return_value.connect.side_effect = interrupt
            with self.assertRaises(BaseException):
                _run_distributed_worker(
                    entry,
                    {"num_inference_steps": 4},
                    30,
                    tracker=tracker,
                    loader_uid="loader-42",
                )
        cancel.assert_not_called()
        abort.assert_called_once_with(entry, loader_uid="loader-42", fast=True)

    def test_abort_worker_startup_kills_process_tree(self):
        from xdit_comfyui.worker import _abort_worker_startup

        proc = mock.Mock()
        proc.poll.return_value = None
        proc.pid = 4242
        with mock.patch("xdit_comfyui.worker._terminate_workers_for_token") as kill_tree:
            with mock.patch("xdit_comfyui.worker._terminate_worker_proc") as terminate:
                with mock.patch("pathlib.Path.unlink"):
                    _abort_worker_startup(proc, "abc123deadbeef")
        kill_tree.assert_called_once_with("abc123deadbeef", fast=True)
        terminate.assert_called_once_with(proc, fast=True)

    def test_get_or_create_worker_aborts_startup_on_interrupt(self):
        from xdit_comfyui.worker import _get_or_create_distributed_worker

        interrupt = type("InterruptProcessingException", (BaseException,), {})()
        proc = mock.Mock()
        proc.poll.return_value = None
        with mock.patch.dict("xdit_comfyui.registry.REGISTRY.workers", {}, clear=True):
            with mock.patch("xdit_comfyui.worker._try_adopt_orphan_worker", return_value=None):
                with mock.patch("xdit_comfyui.worker._cleanup_worker_artifacts"):
                    with mock.patch("xdit_comfyui.worker._prune_stale_aiter_jit_build"):
                        with mock.patch(
                            "xdit_comfyui.worker._subprocess_child_env",
                            return_value={},
                        ):
                            with mock.patch(
                                "xdit_comfyui.worker._allocate_master_port", return_value="12355"
                            ):
                                with mock.patch("xdit_comfyui.worker._start_worker_log_forwarder"):
                                    with mock.patch(
                                        "xdit_comfyui.worker.subprocess.Popen",
                                        return_value=proc,
                                    ):
                                        with mock.patch(
                                            "xdit_comfyui.worker._wait_for_worker_with_progress",
                                            side_effect=interrupt,
                                        ):
                                            with mock.patch(
                                                "xdit_comfyui.worker._abort_worker_startup"
                                            ) as abort_startup:
                                                with self.assertRaises(BaseException):
                                                    _get_or_create_distributed_worker(
                                                        "abc123deadbeef",
                                                        {
                                                            "model": "m",
                                                            "_loader_node_id": "loader-99",
                                                        },
                                                        {},
                                                        1,
                                                        loader_uid="loader-99",
                                                    )
        from xdit_comfyui.worker import _loader_worker_token

        abort_startup.assert_called_once_with(proc, _loader_worker_token("loader-99"))

    def test_terminate_worker_proc_uses_killpg_for_isolated_session(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        proc.pid = 999
        with mock.patch("xdit_comfyui.worker.os.getpgid", side_effect=[111, 222]):
            with mock.patch("xdit_comfyui.worker._terminate_group") as terminate_group:
                _terminate_worker_proc(proc)
        terminate_group.assert_called_once_with(proc, fast=False)


class RunnerNodesTest(unittest.TestCase):
    def setUp(self):
        _clear_all_runtime_caches()

    def test_handle_subprocess_fragment_parses_compile_tqdm(self):
        from xdit_comfyui.progress import _parse_step_progress

        self.assertEqual(_parse_step_progress("  1/2 [00:05"), (1, 2))
        self.assertEqual(_parse_step_progress("100%|██| 4/4"), (4, 4))

    def test_socket_read_exact_reassembles_a_split_payload(self):
        from xdit_comfyui.worker import _socket_read_exact

        payload = bytes(range(256)) * 32
        remaining = [payload[:100], payload[100:5000], payload[5000:]]

        def _recv_into(view, size):
            chunk = remaining.pop(0)[:size]
            view[: len(chunk)] = chunk
            return len(chunk)

        conn = mock.MagicMock()
        conn.recv_into.side_effect = _recv_into
        self.assertEqual(bytes(_socket_read_exact(conn, len(payload))), payload)

    def test_socket_read_exact_honors_an_absolute_deadline(self):
        import socket
        import time

        from xdit_comfyui.worker import _socket_read_exact

        reader, writer = socket.socketpair()
        self.addCleanup(reader.close)
        self.addCleanup(writer.close)
        started = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "xDiT worker response"):
            _socket_read_exact(
                reader,
                1,
                poll_interval=0.01,
                deadline=started + 0.05,
            )
        self.assertLess(time.monotonic() - started, 0.5)

    def test_worker_failure_detail_extracts_the_last_traceback(self):
        from xdit_comfyui.worker import _worker_failure_detail

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "worker.log"
            log.write_text(
                "\n".join(
                    [
                        "Loading model pipeline",
                        "Traceback (most recent call last):",
                        '  File "old.py", line 1, in <module>',
                        "ValueError: an earlier failure",
                        "retrying",
                        "Traceback (most recent call last):",
                        '  File "pipeline_z_image.py", line 240, in _encode_prompt',
                        "    prompt_embeds = self.text_encoder(",
                        "TypeError: 'NoneType' object is not subscriptable",
                    ]
                ),
                encoding="utf-8",
            )
            detail = _worker_failure_detail(log)

        self.assertTrue(detail.startswith("Traceback (most recent call last):"))
        self.assertIn("TypeError: 'NoneType' object is not subscriptable", detail)
        self.assertNotIn("an earlier failure", detail)
        self.assertNotIn("Loading model pipeline", detail)

    def test_worker_failure_detail_is_empty_without_a_log(self):
        from xdit_comfyui.worker import _worker_failure_detail

        self.assertEqual(_worker_failure_detail(None), "")
        self.assertEqual(_worker_failure_detail("/tmp/xdit-does-not-exist.log"), "")

    def test_closed_worker_socket_reports_the_worker_traceback(self):
        import socket as socket_mod

        from xdit_comfyui import worker as worker_mod

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "worker.log"
            log.write_text(
                "Traceback (most recent call last):\nRuntimeError: denoiser exploded\n",
                encoding="utf-8",
            )
            entry = {"socket_path": str(Path(tmp) / "worker.sock"), "log_path": str(log)}
            conn = mock.MagicMock()
            conn.recv_into.return_value = 0
            with (
                mock.patch.object(socket_mod, "socket", return_value=conn),
                mock.patch.object(worker_mod, "_socket_write_json"),
            ):
                with self.assertRaises(ConnectionError) as caught:
                    _run_distributed_worker(entry, {"num_inference_steps": 4}, 30)

        message = str(caught.exception)
        self.assertIn("worker connection closed", message)
        self.assertIn("RuntimeError: denoiser exploded", message)

    def test_log_xdit_line_dedupes_and_filters_noise(self):
        from xdit_comfyui import progress as progress_mod

        progress_mod._reset_progress_log_dedupe()
        with mock.patch.object(progress_mod._RUN_LOG, "info") as log_info:
            progress_mod._log_xdit_line(
                "WARNING 07-27 12:27:41 [runtime_state.py:127] Using AITER_FLYDSL as attention backend."
            )
            progress_mod._log_xdit_line(
                "WARNING 07-27 12:27:41 [runtime_state.py:127] Using AITER_FLYDSL as attention backend."
            )
            progress_mod._log_xdit_line("[Gloo] Rank 0 is connected to 3 peer ranks.")
            progress_mod._log_xdit_line("warnings.warn(")
            progress_mod._log_xdit_line("[rank2]: Loading model pipeline")
            progress_mod._log_xdit_line("Loading model pipeline")
            self.assertEqual(log_info.call_count, 2)
            log_info.assert_any_call(
                "WARNING 07-27 12:27:41 [runtime_state.py:127] Using AITER_FLYDSL as attention backend."
            )
            log_info.assert_any_call("Loading model pipeline")
