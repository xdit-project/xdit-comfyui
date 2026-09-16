from __future__ import annotations

import hashlib
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

_CONTAINER_DATA_DIR = Path("/app/data")
REMOTE_CACHE_DIR = Path(tempfile.gettempdir()) / "xdit_comfyui" / "remote_images"
MAX_REMOTE_IMAGE_BYTES = 32 * 1024 * 1024
from .identity import USER_AGENT as _USER_AGENT


def is_remote_image_ref(path: str) -> bool:
    lowered = str(path or "").strip().lower()
    return lowered.startswith("https://")


def _remote_cache_path(url: str) -> Path:
    parsed = urlparse(url)
    stem = Path(parsed.path).name or "remote.jpg"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    safe = re.sub(r"[^0-9A-Za-z._-]", "_", stem)
    return REMOTE_CACHE_DIR / f"{digest}-{safe}"


def fetch_remote_image(url: str, *, timeout: float = 30.0) -> Path:
    raw = str(url or "").strip()
    if not is_remote_image_ref(raw):
        raise ValueError(f"Not a remote image URL: {url!r}")

    cache_path = _remote_cache_path(raw)
    if cache_path.is_file() and cache_path.stat().st_size > 0:
        return cache_path

    REMOTE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(raw, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read(MAX_REMOTE_IMAGE_BYTES + 1)
    except urllib.error.URLError as exc:
        raise FileNotFoundError(f"Failed to download preset reference image: {raw}") from exc

    if not data:
        raise FileNotFoundError(f"Empty preset reference image download: {raw}")
    if len(data) > MAX_REMOTE_IMAGE_BYTES:
        raise ValueError(f"Preset reference image exceeds {MAX_REMOTE_IMAGE_BYTES} bytes: {raw}")

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=REMOTE_CACHE_DIR, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        temporary.replace(cache_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return cache_path


def resolve_benchmark_data_path(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return raw

    if is_remote_image_ref(raw):
        return str(fetch_remote_image(raw))

    name = Path(raw).name
    original = Path(raw)
    if original.is_file():
        return str(original)

    container_path = _CONTAINER_DATA_DIR / name
    if container_path.is_file():
        return str(container_path)

    return raw


def resolve_benchmark_data_paths(paths: list[str]) -> list[str]:
    return [resolve_benchmark_data_path(path) for path in paths if str(path).strip()]


def _preview_display_name(raw: str, resolved: Path) -> str:
    if is_remote_image_ref(raw):
        parsed = urlparse(raw)
        remote_name = Path(parsed.path).name
        if remote_name:
            return remote_name
    resolved = resolved.resolve()
    if resolved.parent == REMOTE_CACHE_DIR.resolve() and "-" in resolved.name:
        return resolved.name.split("-", 1)[1]
    return resolved.name


def _preview_url(raw: str, resolved: Path) -> str | None:
    resolved = resolved.resolve()
    if not resolved.is_file():
        return None
    if resolved.parent == REMOTE_CACHE_DIR.resolve():
        return f"/xdit/benchmark_cache/{resolved.name}"
    if is_remote_image_ref(raw):
        return f"/xdit/benchmark_cache/{resolved.name}"
    return None


def benchmark_image_preview_entries(paths: list[str]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for raw in paths:
        if not str(raw).strip():
            continue
        resolved = Path(resolve_benchmark_data_path(raw))
        if not resolved.is_file():
            continue
        url = _preview_url(raw, resolved)
        if url is None:
            continue
        entries.append({"name": _preview_display_name(raw, resolved), "url": url})
    return entries
