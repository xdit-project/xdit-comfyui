#!/usr/bin/env python3
"""Print the base image for a maintained Docker profile."""

import argparse
import json
import shlex
from pathlib import Path

PROFILES = Path(__file__).resolve().parents[2] / "docker" / "profiles.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("--xdit-spec", action="store_true")
    args = parser.parse_args()
    data = json.loads(PROFILES.read_text(encoding="utf-8"))
    profiles = data["profiles"]
    if args.profile not in profiles:
        choices = ", ".join(sorted(profiles))
        raise SystemExit(f"Unknown xDiT profile {args.profile!r}; choose one of: {choices}")
    profile = profiles[args.profile]
    if args.xdit_spec:
        repository = str(profile["xdit_repository"]).removesuffix(".git")
        print("xfuser[cache-dit,opencv-python] " f"@ git+{repository}.git@{profile['xdit_ref']}")
    else:
        print(f"export BASE_IMAGE={shlex.quote(str(profile['base_image']))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
