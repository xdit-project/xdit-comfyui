# xDiT ComfyUI

xDiT for ComfyUI runs supported diffusion models through the
[xDiT](https://github.com/xdit-project/xDiT) unified runner. It provides nodes for selecting
a tested hardware preset, loading a distributed model, and generating images or video.

## Requirements

- Linux
- A ComfyUI version satisfying the constraint in `pyproject.toml`
- A CUDA or ROCm environment supported by xDiT
- Enough GPU memory for the selected model and parallel configuration

## Installation

### ComfyUI Manager

**This has not yet been published to the Comfy Registry, manual installation is required.**

### Manual installation

Clone the repository into `ComfyUI/custom_nodes` and install its dependencies with the
same Python interpreter that runs ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xdit-project/xdit-comfyui.git
python -m pip install -r xdit-comfyui/requirements.txt
```

Restart ComfyUI after installation.

### Docker

The development launcher can use either maintained hardware profile:

```bash
XDIT_PROFILE=instinct bash scripts/docker/run_dev.sh
XDIT_PROFILE=rdna4 bash scripts/docker/run_dev.sh
```

Or supply another compatible xDiT image:

```bash
BASE_IMAGE=your-published-xdit-image bash scripts/docker/run_dev.sh
```

See [docker/README.md](docker/README.md) for the base-image contract, runtime behavior, and
available overrides.

## Quick start

Add the **xDiT Starter** template or connect the nodes manually.

1. In **xDiT Preset**, select the detected GPU family, GPU count, and a matching preset.
2. In **xDiT Model**, review the model and runtime settings.
3. In **xDiT Sample**, enter a prompt and queue the workflow.
4. Connect the result to its matching Save node. Unload models from the sidebar when finished.

The xDiT sidebar shows resident workers, assigned GPUs, memory use, and unload controls.

## Nodes

| Node                | Purpose                                                              |
| ------------------- | -------------------------------------------------------------------- |
| **xDiT Preset**     | Applies a benchmark-tested model, GPU, and generation configuration. |
| **xDiT Model**      | Configures and manages a model worker.                               |
| **xDiT Sample**     | Supplies generation inputs and produces an image or video.           |
| **Save xDiT Image** | Saves image results.                                                 |
| **Save xDiT Video** | Saves video results.                                                 |

The Model node owns model initialization and runtime controls, including VAE settings. The
Sample node owns generation inputs. Advanced controls are grouped and filtered according to
the installed xDiT revision, selected model, and related settings.

## Model storage

With the default `auto` setting, downloads use the first configured Hugging Face location:

1. `HF_HUB_CACHE` or `HUGGINGFACE_HUB_CACHE`
2. `HF_HOME`
3. A populated container cache at `/cache/huggingface`
4. `ComfyUI/models/huggingface`

The Model node shows the effective source and path. `comfy_models_shared` always uses the
ComfyUI models path, `system_default` leaves Hugging Face paths unchanged, and `custom_path`
uses the specified cache root. Authentication uses `HF_TOKEN` or `hf auth login`; credentials
are not stored in workflows.

Models are selected by Hugging Face repository ID. Arbitrary directories under
`ComfyUI/models/diffusers` are not discovered; this plugin does not reorganize model files or
create compatibility symlinks.

## Development

Install the locked development environment and run the checks:

```bash
uv sync --dev
uv run ruff check .
uv run black --check -W 1 .
corepack pnpm install --frozen-lockfile
corepack pnpm format:check
corepack pnpm lint
uv run pytest tests -q --ignore=tests/integration
uv run pytest tests/integration -q -m contract
```

### Preset snapshot

Preset YAML files are bundled in `xdit_comfyui/preset_configs`; the plugin reads this local
snapshot at runtime and does not fetch presets from upstream. The source repository and commit
are pinned in `pyproject.toml`. After changing that pin, refresh and commit the snapshot with:

```bash
uv run python scripts/maintenance/sync_preset_configs.py
```

CI runs the same command with `--check` to ensure the committed files still match the pin.
Regenerate the committed starter workflow with
`uv run python scripts/maintenance/build_starter_workflow.py` after changing its builder.

GPU integration tests are opt-in:

```bash
XDIT_RUN_GPU_TESTS=1 uv run pytest tests/integration -m gpu_live
bash scripts/browser/run_tests.sh --install  # install Chromium once
bash scripts/browser/run_tests.sh
```

For GPU container development and launcher overrides, see
[docker/README.md](docker/README.md). Python node changes require a ComfyUI restart with
`bash scripts/docker/restart.sh`.

## Releases

The package version and Registry metadata are maintained in `pyproject.toml`. Releases are
published to the Comfy Registry through the protected `publish` workflow.

## License

Licensed under Apache-2.0. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
