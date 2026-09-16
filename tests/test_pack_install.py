"""The pack has to work from a plain clone into `custom_nodes/`, with no pip install.

ComfyUI imports a custom node by file location and never adds its directory to
`sys.path`, so both the node import and the worker child's `-m` module path depend on
the pack putting itself on the import path. A regression here only shows up as a node
pack that silently fails to load on every install that is not the dev pod.
"""

import importlib.util
import json
import os
import sys
import unittest
from unittest import mock


class RegistryMetadataTest(unittest.TestCase):
    def test_registry_identity_and_icon_are_publishable(self):
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - Python 3.10
            import tomli as tomllib

        root = Path(__file__).resolve().parents[1]
        metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        comfy = metadata["tool"]["comfy"]
        self.assertEqual(comfy["PublisherId"], "xdit-project")
        self.assertEqual(comfy["DisplayName"], "xDiT")
        self.assertEqual(comfy["requires-comfyui"], ">=0.28.0")
        self.assertTrue(comfy["Icon"].startswith("https://"))
        self.assertNotIn("TODO", json.dumps(comfy))


from pathlib import Path

_PACK_ROOT = Path(__file__).resolve().parents[1]


def _load_pack_like_comfyui():
    """Import the pack exactly as `nodes.load_custom_node` does."""
    module_name = str(_PACK_ROOT).replace(".", "_x_")
    spec = importlib.util.spec_from_file_location(module_name, _PACK_ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


class ComfyUILoadTest(unittest.TestCase):
    def test_comfyui_style_load_registers_the_nodes(self):
        import asyncio

        module = _load_pack_like_comfyui()
        extension = asyncio.run(module.comfy_entrypoint())
        nodes = asyncio.run(extension.get_node_list())
        self.assertEqual(
            {"xDiT.Preset", "xDiT.Model", "xDiT.Sample"},
            {node.define_schema().node_id for node in nodes},
        )
        self.assertEqual("./web", module.WEB_DIRECTORY)

    def test_loading_puts_the_pack_root_on_the_import_path(self):
        original = list(sys.path)
        try:
            sys.path[:] = [entry for entry in sys.path if entry != str(_PACK_ROOT)]
            _load_pack_like_comfyui()
            self.assertIn(str(_PACK_ROOT), sys.path)
        finally:
            sys.path[:] = original


class WorkerChildImportPathTest(unittest.TestCase):
    """`python -m xdit_comfyui.worker_server` needs the pack root."""

    def test_child_env_carries_the_pack_root(self):
        from xdit_comfyui.runtime_env import _ensure_runtime_env

        env = _ensure_runtime_env()
        self.assertIn(str(_PACK_ROOT), env["PYTHONPATH"].split(os.pathsep))

    def test_existing_pythonpath_is_kept(self):
        from xdit_comfyui.runtime_env import _ensure_runtime_env

        env = _ensure_runtime_env({"PYTHONPATH": "/opt/somewhere"})
        entries = env["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(str(_PACK_ROOT), entries[0])
        self.assertIn("/opt/somewhere", entries)


class DeclaredDependenciesTest(unittest.TestCase):
    """Manager installs requirements.txt; a plain `pip install .` reads pyproject."""

    def test_the_two_dependency_lists_agree(self):
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - Python 3.10
            self.skipTest("tomllib needs Python 3.11+")

        pyproject = tomllib.loads((_PACK_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        from_pyproject = set(pyproject["project"]["dependencies"])
        from_requirements = {
            line.strip()
            for line in (_PACK_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
        self.assertEqual(from_pyproject, from_requirements)


class MissingRunnerTest(unittest.TestCase):
    """ComfyUI loads the pack before the user has installed xfuser at least once."""

    def test_a_node_whose_schema_needs_xfuser_says_how_to_install_it(self):
        from xdit_comfyui.model_info import default_input_value_names
        from xdit_comfyui.runtime_config import xdit_model_choices

        for reader in (xdit_model_choices, default_input_value_names):
            reader.cache_clear()
            self.addCleanup(reader.cache_clear)
            with mock.patch.dict(
                sys.modules,
                {"xfuser.model_executor.models.runner_models.base_model": None},
            ):
                with self.assertRaisesRegex(RuntimeError, "pip install -r requirements.txt"):
                    reader()


if __name__ == "__main__":
    unittest.main()
