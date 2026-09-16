import os
import sys

# ComfyUI imports a custom node by file location and never adds its directory to
# `sys.path`. Adding it here is what lets the pack run from a plain clone, with no
# pip install. The absolute import below is deliberate: the worker child and the
# tests import `xdit_comfyui` by that name too, and the package holds
# process-wide state (worker registry, loader cache) that must be one instance.
_PACK_ROOT = os.path.dirname(os.path.realpath(__file__))
if _PACK_ROOT not in sys.path:
    sys.path.insert(0, _PACK_ROOT)

from xdit_comfyui import api as _xdit_api  # noqa: F401,E402
from xdit_comfyui import comfy_entrypoint  # noqa: E402

WEB_DIRECTORY = "./web"

__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
