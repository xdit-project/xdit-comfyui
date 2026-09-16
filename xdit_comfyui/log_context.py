"""Structured xDiT logging: child loggers, run context, and env level overrides."""

from __future__ import annotations

import contextvars
import logging
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunLogContext:
    prompt_id: str | None = None
    loader_node_id: str | None = None
    sample_node_id: str | None = None
    cache_key_short: str | None = None

    def prefix(self) -> str:
        parts = []
        if self.prompt_id:
            parts.append(f"p={self.prompt_id[:8]}")
        if self.loader_node_id:
            parts.append(f"M={self.loader_node_id}")
        if self.sample_node_id:
            parts.append(f"S={self.sample_node_id}")
        if self.cache_key_short:
            parts.append(f"k={self.cache_key_short}")
        if not parts:
            return ""
        return f"[{' '.join(parts)}] "


_RUN_CONTEXT: contextvars.ContextVar[RunLogContext | None] = contextvars.ContextVar(
    "xdit_run_context", default=None
)

_LOG = logging.getLogger("xdit")
_WORKER_LOG = logging.getLogger("xdit.worker")
_RUN_LOG = logging.getLogger("xdit.run")
_RESIDENCY_LOG = logging.getLogger("xdit.residency")
_WORKER_OUT_LOG = logging.getLogger("xdit.worker.out")

_CONFIGURED = False


def _level_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip().upper()
    if not raw:
        return default
    return getattr(logging, raw, default)


def configure_xdit_logging():
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger("xdit")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
    base_level = _level_from_env("XDIT_LOG_LEVEL", logging.INFO)
    root.setLevel(base_level)
    _WORKER_LOG.setLevel(base_level)
    _RUN_LOG.setLevel(base_level)
    _RESIDENCY_LOG.setLevel(base_level)
    out_level = _level_from_env("XDIT_WORKER_OUT_LEVEL", logging.DEBUG)
    _WORKER_OUT_LOG.setLevel(out_level)
    _CONFIGURED = True


class _ContextAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        ctx = _RUN_CONTEXT.get()
        prefix = ctx.prefix() if ctx is not None else ""
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("xdit_prefix", prefix)
        return f"{prefix}{msg}", kwargs


def xdit_logger(name: str = "xdit") -> logging.LoggerAdapter:
    configure_xdit_logging()
    return _ContextAdapter(logging.getLogger(name), {})


def worker_logger() -> logging.LoggerAdapter:
    configure_xdit_logging()
    return _ContextAdapter(_WORKER_LOG, {})


def run_logger() -> logging.LoggerAdapter:
    configure_xdit_logging()
    return _ContextAdapter(_RUN_LOG, {})


def residency_logger() -> logging.LoggerAdapter:
    configure_xdit_logging()
    return _ContextAdapter(_RESIDENCY_LOG, {})


def worker_out_logger() -> logging.LoggerAdapter:
    configure_xdit_logging()
    return _ContextAdapter(_WORKER_OUT_LOG, {})


def set_run_context(
    *,
    prompt_id: str | None = None,
    loader_node_id: str | None = None,
    sample_node_id: str | None = None,
    cache_key_short: str | None = None,
) -> contextvars.Token:
    current = _RUN_CONTEXT.get()
    merged = RunLogContext(
        prompt_id=prompt_id or (current.prompt_id if current else None),
        loader_node_id=loader_node_id or (current.loader_node_id if current else None),
        sample_node_id=sample_node_id or (current.sample_node_id if current else None),
        cache_key_short=cache_key_short or (current.cache_key_short if current else None),
    )
    return _RUN_CONTEXT.set(merged)


def reset_run_context(token: contextvars.Token | None):
    if token is not None:
        _RUN_CONTEXT.reset(token)


def copy_run_context() -> RunLogContext | None:
    return _RUN_CONTEXT.get()


def run_context_from_dict(data: dict[str, Any] | None) -> RunLogContext | None:
    if not data:
        return None
    return RunLogContext(
        prompt_id=data.get("prompt_id"),
        loader_node_id=data.get("loader_node_id"),
        sample_node_id=data.get("sample_node_id"),
        cache_key_short=data.get("cache_key_short"),
    )


def run_context_to_dict(ctx: RunLogContext | None) -> dict[str, str] | None:
    if ctx is None:
        return None
    payload = {}
    if ctx.prompt_id:
        payload["prompt_id"] = ctx.prompt_id
    if ctx.loader_node_id:
        payload["loader_node_id"] = ctx.loader_node_id
    if ctx.sample_node_id:
        payload["sample_node_id"] = ctx.sample_node_id
    if ctx.cache_key_short:
        payload["cache_key_short"] = ctx.cache_key_short
    return payload or None
