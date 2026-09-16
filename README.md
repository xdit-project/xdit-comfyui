# xDiT

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

The included Dockerfile expects an xDiT base image containing the matching GPU stack:

```bash
docker build -t xdit-comfyui -f docker/Dockerfile .
```

## Quick start

Add the **xDiT Starter** template or connect the nodes manually.

1. In **xDiT Preset**, select the detected GPU family, GPU count, and a matching preset.
2. In **xDiT Model**, review the model, GPU assignment, and residency policy.
3. In **xDiT Sample**, enter a prompt and queue the workflow.
4. Use **Unload Model (Free VRAM)** when you no longer need the warm model.

The xDiT sidebar shows resident workers, assigned GPUs, memory use, and unload controls.

## Nodes

| Node            | Purpose                                                              |
| --------------- | -------------------------------------------------------------------- |
| **xDiT Preset** | Applies a benchmark-tested model, GPU, and generation configuration. |
| **xDiT Model**  | Loads and keeps an xDiT worker ready for inference.                  |
| **xDiT Sample** | Generates images or video from the loaded model.                     |

Model, residency, and compilation are always visible. Less frequently changed controls
are grouped into model cache, parallelism, memory, VAE, attention, GEMM precision, step
cache, and optional model-specific sections. Unsupported controls are disabled or hidden
for the selected model.

## Model storage

With the default `auto` setting, downloads use the first configured Hugging Face location:

1. `HF_HUB_CACHE` or `HUGGINGFACE_HUB_CACHE`
2. `HF_HOME`
3. A populated container cache at `/cache/huggingface`
4. `ComfyUI/models/huggingface`

The Model node shows the effective source and path. Choose `custom_path` to specify another
Hugging Face cache root. Authentication uses `HF_TOKEN` or `hf auth login`; credentials are
not stored in workflows.

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

For a GPU development container, run `bash scripts/docker/run_dev.sh`. The most useful
overrides are:

| Variable                               | Purpose                                                      |
| -------------------------------------- | ------------------------------------------------------------ |
| `XDIT_DOCKER_ROCM=0`                   | Use CUDA (`--gpus all`) instead of the ROCm device defaults. |
| `XDIT_REPOSITORY`, `XDIT_REF`          | Build against another xDiT repository or revision.           |
| `HF_CACHE_HOST`                        | Host Hugging Face cache to mount in the container.           |
| `XDIT_DEV_IMAGE`, `XDIT_DEV_CONTAINER` | Override the development image or container name.            |
| `COMFYUI_URL`                          | Target an existing ComfyUI server in live tests.             |
| `XDIT_BROWSER_COMFYUI_PORT`            | Override the provisioned browser-test port.                  |

Python node changes require a ComfyUI restart: `bash scripts/docker/restart.sh`.
See [docker/README.md](docker/README.md) for container examples and the complete Docker
configuration.

## Releases

The package version and Registry metadata are maintained in `pyproject.toml`. Releases are
published to the Comfy Registry through the protected `publish` workflow.

## License

Licensed under Apache-2.0. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
