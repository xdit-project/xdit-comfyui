"""Make this machine look like a GitHub runner: no GPU, no ROCm packages.

CI installs xfuser on a CPU-only box where `aiter` is simply absent and xfuser's guarded
imports fall back. Hiding the GPUs on a pod is not enough to reproduce that: aiter is
installed here and raises at import when it finds no device, a failure mode CI never
sees. So drop it off the import path entirely, the way an uninstalled package behaves —
including for the `find_spec` probes that ask whether it is available.

    /opt/venv/bin/python -m pytest tests -q --ignore=tests/integration \
        -p tests.cpu_only_plugin

Not loaded by default; it only makes sense on a machine that does have a GPU.
"""

import sys
from importlib.machinery import PathFinder
from pathlib import Path

_UNINSTALLED_ON_CI = ("aiter", "flash_attn")


def _drop_path_entry(spec):
    origin = spec.origin or next(iter(spec.submodule_search_locations or []), None)
    if not origin:
        return False
    # A package's origin is .../<entry>/<name>/__init__.py.
    entry = str(Path(origin).resolve().parent.parent)
    remaining = [item for item in sys.path if str(Path(item).resolve()) != entry]
    if len(remaining) == len(sys.path):
        return False
    sys.path[:] = remaining
    return True


def _drop_finder(name):
    """Editable installs answer through sys.meta_path, not through sys.path."""
    for finder in list(sys.meta_path):
        if finder is PathFinder or not hasattr(finder, "find_spec"):
            continue
        try:
            found = finder.find_spec(name, None)
        except Exception:
            found = None
        if found is not None:
            sys.meta_path.remove(finder)
            return True
    return False


def _hide(name):
    import importlib.util

    for loaded in [key for key in sys.modules if key.split(".", 1)[0] == name]:
        del sys.modules[loaded]
    while True:
        spec = importlib.util.find_spec(name)
        if spec is None:
            return
        if not (_drop_finder(name) or _drop_path_entry(spec)):
            raise RuntimeError(f"cannot hide {name} from the import system")


def pytest_configure(config):
    for name in _UNINSTALLED_ON_CI:
        _hide(name)

    import torch

    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
