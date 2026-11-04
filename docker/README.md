# Docker

The Dockerfile adds ComfyUI and this plugin to an image that already provides the GPU stack
and xDiT. Maintained image pins live only in `docker/profiles.json`.

## Quick start

Run one of the maintained profiles:

```bash
XDIT_PROFILE=instinct bash scripts/docker/run_dev.sh
XDIT_PROFILE=rdna4 bash scripts/docker/run_dev.sh
```

Or use any compatible xDiT image:

```bash
BASE_IMAGE=your-xdit-image bash scripts/docker/run_dev.sh
```

## Base-image contract

A compatible base image must provide:

- A working `python` environment and `xdit` executable.
- xDiT source metadata at `/app/xDiT/pyproject.toml`, unless both `XDIT_REPOSITORY` and
  `XDIT_REF` are supplied to install a replacement revision.

The build honors `/opt/build-constraints.txt` when the base image provides it, preventing
dependency installation from replacing its GPU stack.

## Runtime behavior

The launcher bind-mounts this repository at `/workspace/plugin` and the host Hugging Face
cache at `/.cache/huggingface`. ComfyUI runs from `/workspace/comfyui` on port `8188`.

Maintained profiles use ROCm devices. A custom image uses ROCm when `/dev/kfd` is present and
otherwise uses the NVIDIA `--gpus all` runtime. Set `XDIT_DOCKER_ROCM=1` or `0` to override
that choice.

The launcher accepts these overrides:

- `XDIT_PROFILE`: select a maintained profile.
- `BASE_IMAGE`: use a compatible image instead of resolving a profile.
- `XDIT_REPOSITORY` and `XDIT_REF`: replace the base image's xDiT revision; both are required.
- `HF_CACHE_HOST` and `HF_CACHE_ROOT`: change the host and container cache paths.
- `XDIT_DOCKER_ROCM`: force ROCm (`1`) or NVIDIA (`0`) container options.
- `XDIT_REBUILD=1`: rebuild even when the resolved development image already exists.
- `XDIT_DEV_IMAGE` and `XDIT_DEV_CONTAINER`: override generated image and container names.
- `COMFY_PORT`: change the host-facing ComfyUI port.

`CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, and `GPU_ARCHS` are forwarded when set.

After editing Python, restart ComfyUI with:

```bash
bash scripts/docker/restart.sh
```

To test another xDiT revision on the same base image:

```bash
XDIT_PROFILE=instinct \
XDIT_REPOSITORY=https://github.com/OWNER/xDiT.git \
XDIT_REF=BRANCH_OR_COMMIT \
bash scripts/docker/run_dev.sh
```

## Building directly

Resolve a maintained base image and build:

```bash
eval "$(python3 scripts/docker/profile.py instinct)"
docker build --build-arg "BASE_IMAGE=${BASE_IMAGE}" -t xdit-comfyui -f docker/Dockerfile .
```

Custom base images must provide a working Python environment and `xdit` executable.
