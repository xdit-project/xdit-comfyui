"""Minimal ComfyUI HTTP client for queueing prompts like the web UI."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from PIL import Image


def _get_json(url: str, *, timeout: float = 30) -> tuple[int, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _post_json(url: str, body: dict, *, timeout: float = 30) -> tuple[int, Any]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            if not raw.strip():
                return response.status, {}
            return response.status, json.loads(raw)
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(payload) if payload.strip() else {}
        except json.JSONDecodeError:
            return exc.code, {"raw": payload}


def comfy_reachable(base_url: str) -> bool:
    try:
        status, _payload = _get_json(f"{base_url.rstrip('/')}/system_stats", timeout=3)
        return status == 200
    except Exception:
        return False


@dataclass
class ComfyUiRunResult:
    prompt_id: str
    history: dict[str, Any]
    outputs: dict[str, Any] = field(default_factory=dict)
    status: dict[str, Any] = field(default_factory=dict)
    queue_error: dict[str, Any] | None = None
    node_errors: dict[str, Any] = field(default_factory=dict)


class ComfyUiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def interrupt(self) -> None:
        _post_json(f"{self.base_url}/interrupt", {}, timeout=10)

    def queue_prompt(
        self, prompt: dict[str, Any], *, prompt_id: str | None = None
    ) -> ComfyUiRunResult:
        body: dict[str, Any] = {"prompt": prompt, "client_id": self.client_id}
        if prompt_id is not None:
            body["prompt_id"] = prompt_id
        status, payload = _post_json(f"{self.base_url}/prompt", body, timeout=60)
        if status != 200:
            return ComfyUiRunResult(
                prompt_id=prompt_id or "",
                history={},
                queue_error=payload.get("error") or payload,
                node_errors=payload.get("node_errors") or {},
            )
        pid = payload["prompt_id"]
        return ComfyUiRunResult(
            prompt_id=pid,
            history={},
            node_errors=payload.get("node_errors") or {},
        )

    def wait_for_prompt(
        self,
        prompt_id: str,
        *,
        timeout_seconds: float = 900,
        poll_seconds: float = 2.0,
    ) -> ComfyUiRunResult:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                _status, history = _get_json(f"{self.base_url}/history/{prompt_id}", timeout=30)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    time.sleep(poll_seconds)
                    continue
                raise
            if prompt_id not in history:
                time.sleep(poll_seconds)
                continue
            entry = history[prompt_id]
            status = entry.get("status") or {}
            if status.get("completed") or entry.get("outputs"):
                return ComfyUiRunResult(
                    prompt_id=prompt_id,
                    history=entry,
                    outputs=entry.get("outputs") or {},
                    status=status,
                )
            if status.get("status_str") == "error":
                return ComfyUiRunResult(
                    prompt_id=prompt_id,
                    history=entry,
                    outputs=entry.get("outputs") or {},
                    status=status,
                )
            time.sleep(poll_seconds)
        raise TimeoutError(
            f"Timed out waiting for ComfyUI prompt {prompt_id} after {timeout_seconds}s"
        )

    def fetch_view_image(self, entry: dict[str, Any]) -> Image.Image:
        params = urllib.parse.urlencode(
            {
                "filename": entry["filename"],
                "subfolder": entry.get("subfolder") or "",
                "type": entry.get("type") or "output",
            }
        )
        with urllib.request.urlopen(f"{self.base_url}/view?{params}", timeout=60) as response:
            return Image.open(BytesIO(response.read())).convert("RGB")

    def queue_and_wait(
        self,
        prompt: dict[str, Any],
        *,
        timeout_seconds: float = 900,
    ) -> ComfyUiRunResult:
        queued = self.queue_prompt(prompt)
        if queued.queue_error:
            return queued
        finished = self.wait_for_prompt(queued.prompt_id, timeout_seconds=timeout_seconds)
        finished.node_errors = queued.node_errors
        return finished
