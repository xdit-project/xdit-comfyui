"""Every piece of per-node state the pack keeps, behind one lock."""

import threading
import time

from .progress import _node_id_str


class _Registry:
    __slots__ = (
        "_lock",
        "workers",
        "pending",
        "cache_keys",
        "snapshots",
        "consumers",
        "run_stats",
        "node_status",
    )

    def __init__(self):
        self._lock = threading.RLock()
        self.workers = {}
        self.pending = {}
        self.cache_keys = {}
        self.snapshots = {}
        self.consumers = {}
        self.run_stats = {}
        self.node_status = {}

    def lock(self):
        """Reentrant so a holder can call another registry helper without deadlocking."""
        return self._lock


REGISTRY = _Registry()


def _loader_node_id_for_cache_key(cache_key):
    if not cache_key:
        return None
    with REGISTRY.lock():
        for uid, key in REGISTRY.cache_keys.items():
            if key == cache_key:
                return _node_id_str(uid)
    return None


def _loader_node_id_for_runtime(runner_config):
    if not isinstance(runner_config, dict):
        return None
    loader_id = runner_config.get("_loader_node_id")
    if loader_id:
        return _node_id_str(loader_id)
    return _loader_node_id_for_cache_key(runner_config.get("_cache_key"))


def _progress_node_ids(runner_config, generate_node_id):
    return _loader_node_id_for_runtime(runner_config), _node_id_str(generate_node_id)


def record_node_status(node_id, text):
    node_id = _node_id_str(node_id)
    if not node_id or not text:
        return
    with REGISTRY.lock():
        REGISTRY.node_status[node_id] = {"text": str(text), "at": time.time()}


def clear_node_status(node_id):
    node_id = _node_id_str(node_id)
    if not node_id:
        return
    with REGISTRY.lock():
        REGISTRY.node_status.pop(node_id, None)


def node_status_snapshot():
    with REGISTRY.lock():
        return {node_id: dict(payload) for node_id, payload in REGISTRY.node_status.items()}
