"""Worker log parsing, Comfy progress bars, and cancel propagation."""

import re
import time
from contextlib import contextmanager
from functools import lru_cache

from .log_context import (
    configure_xdit_logging,
    copy_run_context,
    reset_run_context,
    run_context_from_dict,
    run_logger,
    set_run_context,
    worker_out_logger,
    xdit_logger,
)

configure_xdit_logging()
_XDIT_LOG = xdit_logger("xdit")
_RUN_LOG = run_logger()
_WORKER_OUT_LOG = worker_out_logger()


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text):
    return _ANSI_RE.sub("", text or "")


def _comfy_update_progress(node_id, value, total):
    if not node_id or total <= 0:
        return
    try:
        from comfy_execution.progress import get_progress_state

        get_progress_state().update_progress(node_id, float(value), float(total))
    except Exception:
        pass


def _comfy_finish_progress(node_id):
    if not node_id:
        return
    try:
        from comfy_execution.progress import get_progress_state

        get_progress_state().finish_progress(node_id)
    except Exception:
        pass


_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
_FRAG_SPLIT = re.compile(r"[\r\n]")
_LOADING_MARKERS = (
    "Loading model pipeline",
    "Initializing model",
    "Initializing distributed",
    "Initializing runtime state",
    # xFuser logs "self-fill <subfolder> block <i>", so match on the prefix only.
    "self-fill ",
    "Broadcast-filled",
    "Loading model on ",
)
_COMPILE_MARKERS = ("Torch.compile enabled", "Warming up torch compiler")
_RUN_MARKERS = ("Running model", "Running iteration")
_WARMUP_ITER_RE = re.compile(r"Warmup iteration\s+(\d+)\s*/\s*(\d+)", re.I)
_FETCH_PCT_RE = re.compile(r"Fetching\s+\d+\s+files:.*?(\d+)%")

_INIT_STEP_COUNT = 6
_INIT_DISPLAY_TOTAL = 100
_INIT_START_DISPLAY = 5
_COMPILE_PROGRESS_CAP = 85
_last_progress_log_step = None
_RANK_PREFIX_RE = re.compile(r"^\[rank\d+\]:\s*", re.I)
_LOG_SUPPRESS_RES = (
    re.compile(r"^\[Gloo\]", re.I),
    re.compile(r"^warnings\.warn\(", re.I),
)
_LOG_DEDUPE_SECONDS = 3.0
_last_logged_canonical = None
_last_logged_at = 0.0


def _canonical_log_line(text):
    line = (text or "").strip()
    if not line:
        return ""
    return _RANK_PREFIX_RE.sub("", line)


def _should_suppress_log_line(line):
    if not line:
        return True
    return any(pat.search(line) for pat in _LOG_SUPPRESS_RES)


def _elapsed_text(seconds):
    if seconds is None or seconds < 0.05:
        return ""
    if seconds < 90:
        return f" after {seconds:.1f}s"
    return f" after {seconds / 60:.1f}min"


def _init_stage_display(stage):
    stage = max(0, min(int(stage), _INIT_STEP_COUNT))
    return int(round(stage * _INIT_DISPLAY_TOTAL / _INIT_STEP_COUNT))


def _node_id_str(node_id):
    if node_id is None:
        return None
    return str(node_id)


def _log_xdit_line(text):
    global _last_progress_log_step, _last_logged_canonical, _last_logged_at
    line = _canonical_log_line(text)
    if not line or _should_suppress_log_line(line):
        return
    now = time.monotonic()
    if line == _last_logged_canonical and now - _last_logged_at < _LOG_DEDUPE_SECONDS:
        return
    prog = _parse_step_progress(line)
    if prog is not None:
        if prog == _last_progress_log_step:
            return
        _last_progress_log_step = prog
    elif "%|" not in line and not _STEP_RE.search(line):
        _last_progress_log_step = None
    _last_logged_canonical = line
    _last_logged_at = now
    _RUN_LOG.info(line)


def _log_worker_out_line(text, *, log_context=None):
    global _last_progress_log_step, _last_logged_canonical, _last_logged_at
    line = _canonical_log_line(text)
    if not line or _should_suppress_log_line(line):
        return
    now = time.monotonic()
    if line == _last_logged_canonical and now - _last_logged_at < _LOG_DEDUPE_SECONDS:
        return
    _last_logged_canonical = line
    _last_logged_at = now
    token = None
    ctx = run_context_from_dict(log_context) or copy_run_context()
    if ctx is not None:
        token = set_run_context(
            prompt_id=ctx.prompt_id,
            loader_node_id=ctx.loader_node_id,
            sample_node_id=ctx.sample_node_id,
            cache_key_short=ctx.cache_key_short,
        )
    try:
        _WORKER_OUT_LOG.debug(line)
    finally:
        reset_run_context(token)


def _reset_progress_log_dedupe():
    global _last_progress_log_step, _last_logged_canonical, _last_logged_at
    _last_progress_log_step = None
    _last_logged_canonical = None
    _last_logged_at = 0.0


class _NodeProgress:
    __slots__ = ("node_id", "total", "completed", "_started", "_bar")

    def __init__(self, node_id, total):
        self.node_id = _node_id_str(node_id)
        self.total = max(int(total or 0), 0)
        self.completed = 0
        self._started = False
        self._bar = (
            _make_progress_bar(self.total, self.node_id)
            if self.total > 0 and self.node_id
            else None
        )
        if self.total > 0 and self.node_id:
            self._start()

    def _publish(self, value):
        if not self.node_id or self.total <= 0:
            return
        if self._bar is not None:
            self._bar.update_absolute(value, self.total)
        else:
            _comfy_update_progress(self.node_id, value, self.total)

    def _start(self):
        if self._started or not self.node_id or self.total <= 0:
            return
        self._started = True
        start = _INIT_START_DISPLAY if self.total >= _INIT_DISPLAY_TOTAL else 0
        self._publish(start)
        self.completed = start

    def set_completed(self, value):
        value = max(0, min(int(value), self.total))
        if value <= self.completed:
            return
        self.completed = value
        if not self._started:
            self._start()
        self._publish(self.completed)

    def map_slot(self, base, size, cur, tot):
        if tot <= 0 or size <= 0:
            return
        frac = min(max(cur / tot, 0.0), 1.0)
        offset = int(round(frac * size))
        self.set_completed(base + offset)

    def finish(self):
        if self.total > 0:
            self.set_completed(self.total)
        if self._bar is None:
            _comfy_finish_progress(self.node_id)


_PHASE_STATUS_TEXT = {
    "loading": "loading weights",
    "parallel": "sharding across GPUs",
    "compile": "compiling (first run)",
    "inference": "denoising",
    "decode": "decoding with the VAE",
    "finalization": "collecting the result",
}
_STATUS_TICK_SECONDS = 5.0


def _send_status_text(node_id, text):
    """Say what the worker is doing on the node ComfyUI is running.

    ComfyUI's progress bar carries a number and nothing else, so a five-minute weight
    load and a five-minute VAE decode look identical while they happen. This goes to the
    info block at the top of the node, which polls /xdit/residency while the node runs —
    not to ComfyUI's own `send_progress_text`, which would repeat it at the bottom.
    """
    if not node_id or not text:
        return
    from .registry import record_node_status

    record_node_status(node_id, text)


def _send_node_status(node_id, phase, elapsed):
    text = _PHASE_STATUS_TEXT.get(phase)
    if not text:
        return
    if elapsed and elapsed >= 5:
        text = f"{text} ·{_elapsed_text(elapsed).replace(' after', '')}"
    _send_status_text(node_id, text)


def _executing_node_id():
    """The node ComfyUI is running, read on the node's own thread.

    Progress arrives on the log-tailing thread, which carries none of the execution
    context, so this is captured when the tracker is built and reused after.
    """
    try:
        from comfy_execution.utils import get_executing_context

        context = get_executing_context()
    except Exception:
        return None
    return _node_id_str(context.node_id) if context is not None else None


_PHASE_ENTRY_MESSAGES = {
    "compile": "Warming up the torch compiler (first run for this configuration)",
    "decode": "Denoising complete{elapsed}; decoding latents with the VAE",
    "finalization": "VAE decode complete{elapsed}; collecting the result",
}
_PHASE_WAIT_MESSAGES = {
    "compile": "Still compiling",
    "decode": "Still decoding with the VAE",
    "finalization": "Still collecting the result",
}
_PHASE_WAIT_LOG_SECONDS = 30.0


class _XDiTProgressTracker:
    """Track model initialization or denoising, decode/collect, and finalization.

    The phases after denoising produce no output of their own: a VAE decode can run for
    minutes on video, and without this the run looks hung between the last step and the
    image appearing.
    """

    __slots__ = (
        "inference_steps",
        "init_steps",
        "_phase",
        "_phase_started_at",
        "_phase_logged_at",
        "_phase_status_at",
        "_init",
        "_infer",
        "_init_tqdm_key",
        "_init_tqdm_cur",
        "_status_node_id",
        "node_id",
        "total",
        "completed",
    )

    def __init__(
        self,
        init_node_id=None,
        inference_node_id=None,
        inference_steps=0,
        include_init=True,
        node_id=None,
    ):
        if node_id is not None:
            init_node_id = inference_node_id = node_id
        self.inference_steps = max(int(inference_steps or 0), 0)
        self.init_steps = _INIT_STEP_COUNT if include_init and init_node_id else 0
        self._phase = "init" if self.init_steps > 0 else "inference"
        self._init = (
            _NodeProgress(init_node_id, _INIT_DISPLAY_TOTAL) if self.init_steps > 0 else None
        )
        self._infer = (
            _NodeProgress(inference_node_id, self.inference_steps + 2)
            if self.inference_steps > 0 and inference_node_id
            else None
        )
        self._phase_started_at = time.monotonic()
        self._phase_logged_at = 0.0
        self._phase_status_at = 0.0
        self._init_tqdm_key = None
        self._init_tqdm_cur = -1
        self.node_id = _node_id_str(inference_node_id or init_node_id)
        self._status_node_id = _executing_node_id() or self.node_id
        self.total = self.init_steps + (self.inference_steps + 2 if self._infer is not None else 0)
        self.completed = 0

    def _set_phase(self, phase):
        if phase == self._phase:
            return
        now = time.monotonic()
        elapsed = now - self._phase_started_at
        self._phase = phase
        self._phase_started_at = now
        self._phase_logged_at = now
        self._phase_status_at = now
        message = _PHASE_ENTRY_MESSAGES.get(phase)
        if message:
            _XDIT_LOG.info(message.format(elapsed=_elapsed_text(elapsed)))
        _send_node_status(self._status_node_id, phase, 0)

    def _tick_phase(self):
        """Keep a phase that produces no output of its own visibly alive."""
        now = time.monotonic()
        elapsed = now - self._phase_started_at
        if now - self._phase_status_at >= _STATUS_TICK_SECONDS:
            self._phase_status_at = now
            _send_node_status(self._status_node_id, self._phase, elapsed)
        message = _PHASE_WAIT_MESSAGES.get(self._phase)
        if not message:
            return
        if now - self._phase_logged_at < _PHASE_WAIT_LOG_SECONDS:
            return
        self._phase_logged_at = now
        _XDIT_LOG.info("%s%s", message, _elapsed_text(elapsed))

    def _init_tick_ceiling(self):
        if self._phase == "loading":
            return _init_stage_display(3) - 1
        if self._phase == "parallel":
            return _init_stage_display(4) - 1
        if self._phase == "compile":
            return _COMPILE_PROGRESS_CAP
        return _COMPILE_PROGRESS_CAP

    def _bump_init_on_tick(self, cur, tot):
        if self._init is None:
            return
        cur = max(0, int(cur))
        tot = max(0, int(tot))
        key = (tot, self._phase)
        if key != self._init_tqdm_key:
            self._init_tqdm_key = key
            self._init_tqdm_cur = -1
        if cur <= self._init_tqdm_cur:
            return
        self._init_tqdm_cur = cur
        ceiling = self._init_tick_ceiling()
        new_val = min(self._init.completed + 1, ceiling)
        if new_val > self._init.completed:
            self._init.set_completed(new_val)
        self._sync_aggregate_completed()

    def _finish_init(self):
        if self._init is None:
            return
        self._init.set_completed(_INIT_DISPLAY_TOTAL)
        self._init.finish()
        self._sync_aggregate_completed()

    def _sync_aggregate_completed(self):
        parts = []
        if self._init is not None:
            parts.append(self._init.completed)
        if self._infer is not None:
            parts.append(self._infer.completed)
        self.completed = max(parts) if parts else 0

    def heartbeat(self, *, check_interrupt=True):
        if check_interrupt:
            _raise_if_interrupted()
        self._tick_phase()
        if self._init is None:
            return
        if self._phase == "compile" and self._init.completed < _COMPILE_PROGRESS_CAP:
            self._init.set_completed(min(self._init.completed + 1, _COMPILE_PROGRESS_CAP))
            self._sync_aggregate_completed()

    def on_scheduler_step(self, step_index):
        _raise_if_interrupted()
        if self._infer is None:
            return
        self._set_phase("inference")
        if self._init is not None and self._init.completed < _INIT_DISPLAY_TOTAL:
            self._finish_init()
        step = min(max(int(step_index), 0), self.inference_steps)
        self._infer.set_completed(step)
        self._sync_aggregate_completed()
        if step >= self.inference_steps:
            self._set_phase("decode")

    def _is_inference_tqdm(self, cur, tot, text):
        if self.inference_steps <= 0 or self._infer is None:
            return False
        text = text or ""
        if "Warming up" in text or "Loading" in text or "Fetching" in text:
            return False
        if tot == self.inference_steps:
            if self._phase == "inference" or any(marker in text for marker in _RUN_MARKERS):
                return True
            if self._init is None:
                return True
            if "%|" in text and "it/s" in text:
                return True
            return False
        if self._phase != "inference":
            return False
        if any(marker in text for marker in _RUN_MARKERS):
            return True
        return False

    def on_tqdm(self, cur, tot, fragment="", *, check_interrupt=True):
        if check_interrupt:
            _raise_if_interrupted()
        if tot <= 0:
            return
        text = fragment or ""
        if self._is_inference_tqdm(cur, tot, text):
            self._set_phase("inference")
            if self._init is not None and self._init.completed < _INIT_DISPLAY_TOTAL:
                self._finish_init()
            self._infer.set_completed(min(max(int(cur), 0), self.inference_steps))
            self._sync_aggregate_completed()
            if int(cur) >= self.inference_steps:
                self._set_phase("decode")
            return

        if self._init is None:
            return
        if "Warming up" in text:
            self._init.set_completed(max(self._init.completed, _init_stage_display(5)))
            self._sync_aggregate_completed()
        if (
            "Loading pipeline" in text
            or "Loading weights" in text
            or "Loading checkpoint" in text
            or "Fetching" in text
            or self._phase == "loading"
        ):
            self._init.set_completed(max(self._init.completed, _init_stage_display(1)))
            self._sync_aggregate_completed()
        if self._phase == "parallel" or "self-fill" in text or "Broadcast" in text:
            self._init.set_completed(max(self._init.completed, _init_stage_display(3)))
            self._sync_aggregate_completed()
        self._bump_init_on_tick(cur, tot)

    def on_fragment(self, fragment, *, check_interrupt=True):
        text = fragment or ""

        if any(marker in text for marker in _LOADING_MARKERS):
            self._set_phase("loading")
            if self._init is not None:
                if "Initializing runtime state" in text:
                    self._init.set_completed(max(self._init.completed, _init_stage_display(3)))
                elif "self-fill " in text or "Broadcast-filled" in text:
                    self._set_phase("parallel")
                    self._init.set_completed(max(self._init.completed, _init_stage_display(3)))
                else:
                    self._init.set_completed(max(self._init.completed, _init_stage_display(1)))
                self._sync_aggregate_completed()

        if any(marker in text for marker in _COMPILE_MARKERS):
            self._set_phase("compile")
            if self._init is not None:
                self._init.set_completed(max(self._init.completed, _init_stage_display(4)))
                self._sync_aggregate_completed()

        if any(marker in text for marker in _RUN_MARKERS):
            self._set_phase("inference")
            if self._init is not None and self._init.completed < _INIT_DISPLAY_TOTAL:
                self._finish_init()
            if self._infer is not None:
                self._infer._start()
            self._sync_aggregate_completed()

        if "Model initialization complete" in text:
            self._finish_init()

        warmup = _WARMUP_ITER_RE.search(text)
        if warmup and self._init is not None:
            cur, tot = int(warmup.group(1)), int(warmup.group(2))
            self._init.set_completed(max(self._init.completed, _init_stage_display(5)))
            self._bump_init_on_tick(cur, tot)
            return

        fetch = _parse_fetch_progress(text)
        if fetch and self._init is not None:
            cur, tot = fetch
            self._phase = "loading"
            self._init.set_completed(max(self._init.completed, _init_stage_display(1)))
            self._bump_init_on_tick(cur, tot)
            return

        prog = _parse_step_progress(text)
        if prog:
            cur, tot = prog
            self.on_tqdm(cur, tot, text, check_interrupt=check_interrupt)

    def on_decode_complete(self):
        if self._infer is None:
            return
        self._set_phase("finalization")
        self._infer.set_completed(self.inference_steps + 1)
        self._sync_aggregate_completed()

    def on_finalization_complete(self):
        if self._infer is None:
            return
        self._set_phase("complete")
        self._infer.set_completed(self.inference_steps + 2)
        self._sync_aggregate_completed()

    def finish(self):
        if self._init is not None:
            self._init.finish()
        if self._infer is not None:
            self._infer.finish()
        self._sync_aggregate_completed()
        # The frontend leaves the last line on the node, so end on a finished one
        # rather than whatever phase happened to be running.
        _send_status_text(
            self._status_node_id, "done" if self._infer is not None else "model ready"
        )


@contextmanager
def _xdit_progress(
    init_node_id=None,
    inference_node_id=None,
    inference_steps=0,
    include_init=True,
    node_id=None,
):
    tracker = _XDiTProgressTracker(
        init_node_id=init_node_id,
        inference_node_id=inference_node_id,
        inference_steps=inference_steps,
        include_init=include_init,
        node_id=node_id,
    )
    try:
        yield tracker
    except BaseException:
        raise
    else:
        tracker.finish()


def _parse_step_progress(text):
    """Extract (current, total) from a tqdm-style 'cur/total' fragment, or None.

    When several cur/total pairs appear on one line (e.g. denoising ``1/50`` plus
    an ETA fragment ``04/04`` inside ``[00:04<04:04]``), prefer the pair with the
    largest total so denoising steps win over time estimates."""
    best = None
    for match in _STEP_RE.finditer(text):
        cur, tot = int(match.group(1)), int(match.group(2))
        if tot <= 0 or cur > tot:
            continue
        if best is None or tot > best[1] or (tot == best[1] and cur >= best[0]):
            best = (cur, tot)
    return best


def _parse_fetch_progress(text):
    match = _FETCH_PCT_RE.search(text)
    if match:
        pct = int(match.group(1))
        return max(0, min(100, pct)), 100
    prog = _parse_step_progress(text)
    if prog and "Fetching" in text:
        return prog
    return None


def _handle_subprocess_fragment(fragment, tracker, *, check_interrupt=True):
    if tracker is not None:
        tracker.on_fragment(_strip_ansi(fragment), check_interrupt=check_interrupt)


def _drain_fragments(buf):
    """Split a text buffer on CR/LF into completed fragments, returning
    (fragments, remainder). tqdm rewrites its line with '\\r', so CR is a boundary too."""
    parts = _FRAG_SPLIT.split(buf)
    remainder = parts.pop()
    return parts, remainder


def _make_progress_bar(total, node_id=None):
    """Comfy progress bar, or None when Comfy isn't importable (tests/CLI)."""
    if not total or total <= 0:
        return None
    try:
        from comfy.utils import ProgressBar  # type: ignore[reportMissingImports]

        return ProgressBar(total, node_id=_node_id_str(node_id))
    except Exception:
        return None


@lru_cache(maxsize=1)
def _comfy_interrupt_api():
    """(raise_if_interrupted, InterruptProcessingException), both None outside ComfyUI.

    Resolved once: the interrupt check runs on every socket poll of every run.
    """
    try:
        from comfy.model_management import (  # type: ignore[reportMissingImports]
            InterruptProcessingException,
            throw_exception_if_processing_interrupted,
        )
    except Exception:
        return None, None
    return throw_exception_if_processing_interrupted, InterruptProcessingException


def _raise_if_interrupted():
    """Honor a Comfy cancel request; no-op outside ComfyUI."""
    check, _ = _comfy_interrupt_api()
    if check is not None:
        check()


def _is_comfy_interrupt(exc):
    if type(exc).__name__ == "InterruptProcessingException":
        return True
    _, interrupt_cls = _comfy_interrupt_api()
    return interrupt_cls is not None and isinstance(exc, interrupt_cls)
