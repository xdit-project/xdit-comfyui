import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def profiles():
    return json.loads((ROOT / "docker/profiles.json").read_text(encoding="utf-8"))


def test_derived_install_pins_match_the_manifest():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    manifest = profiles()["profiles"]
    instinct = manifest["instinct"]
    rdna4 = manifest["rdna4"]
    instinct_pin = f"git+{instinct['xdit_repository']}@{instinct['xdit_ref']}"
    rdna4_pin = f"git+{rdna4['xdit_repository']}@{rdna4['xdit_ref']}"
    rdna4_spec = subprocess.check_output(
        [sys.executable, ROOT / "scripts/docker/profile.py", "rdna4", "--xdit-spec"],
        text=True,
    )
    assert instinct_pin in requirements
    assert instinct_pin in pyproject
    assert rdna4_pin in rdna4_spec
