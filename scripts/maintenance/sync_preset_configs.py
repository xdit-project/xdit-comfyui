#!/usr/bin/env python3
"""Sync the bundled offline preset snapshot from the pinned AMD-AGI revision."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
DESTINATION = ROOT / "xdit_comfyui" / "preset_configs"
MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024


def _source() -> tuple[str, str, str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["xdit-comfyui"][
        "preset-configs"
    ]
    repository = str(data["repository"]).removesuffix(".git")
    owner_repo = repository.removeprefix("https://github.com/")
    return owner_repo, str(data["commit"]), str(data["path"]).strip("/")


def _download(url: str) -> bytes:
    if not url.startswith("https://"):
        raise ValueError(f"Refusing non-HTTPS source: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "xdit-comfyui-config-sync"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"Download exceeds {MAX_DOWNLOAD_BYTES} bytes: {url}")
    return data


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail instead of updating files.")
    args = parser.parse_args()
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    owner_repo, commit, source_path = _source()
    api = f"https://api.github.com/repos/{owner_repo}/contents/{source_path}?ref={commit}"
    import json

    entries = json.loads(_download(api))
    upstream = {
        str(entry["name"]): _download(str(entry["download_url"]))
        for entry in entries
        if entry.get("type") == "file" and str(entry.get("name", "")).endswith(".yaml")
    }
    asset_config = project["tool"]["xdit-comfyui"]["preset-assets"]
    asset_path = str(asset_config["path"]).strip("/")
    wan_image_url = (
        f"https://raw.githubusercontent.com/{owner_repo}/{commit}/{asset_path}/wan_input.jpg"
    )
    upstream = {
        name: content.replace(b"/app/data/wan_input.jpg", wan_image_url.encode())
        for name, content in upstream.items()
    }
    # Validate both immutable asset URLs during sync/check. They stay upstream so the
    # repository has one source of truth; the runtime downloads the image on demand.
    for name in (asset_config["image"], asset_config["license"]):
        _download(f"https://raw.githubusercontent.com/{owner_repo}/{commit}/{asset_path}/{name}")
    local_names = {path.name for path in DESTINATION.glob("*.yaml")}
    changed = sorted(
        name
        for name, content in upstream.items()
        if not (DESTINATION / name).is_file() or (DESTINATION / name).read_bytes() != content
    )
    removed = sorted(local_names - set(upstream))
    if args.check:
        if changed or removed:
            raise SystemExit(f"preset snapshot is stale: changed={changed}, removed={removed}")
        return
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, content in upstream.items():
        _write_atomic(DESTINATION / name, content)
    for name in removed:
        (DESTINATION / name).unlink()
    print(f"synced {len(upstream)} configs and verified WAN assets from {owner_repo}@{commit}")


if __name__ == "__main__":
    main()
