# Docker

The Dockerfile extends an image that already provides the GPU stack and xDiT. The
digest-pinned maintained images live in `docker/profiles.json`.

## Development

Run one of the maintained profiles:

```bash
XDIT_PROFILE=instinct bash scripts/docker/run_dev.sh
XDIT_PROFILE=rdna4 bash scripts/docker/run_dev.sh
```

Or use any compatible xDiT image:

```bash
BASE_IMAGE=your-xdit-image bash scripts/docker/run_dev.sh
```

The launcher uses ROCm devices for maintained profiles and custom images on hosts with
`/dev/kfd`; otherwise it uses the NVIDIA `--gpus all` runtime. Override that detection
with `XDIT_DOCKER_ROCM=1` or `0`.

Common overrides are `HF_CACHE_HOST`, `CUDA_VISIBLE_DEVICES`, `XDIT_DEV_IMAGE`,
`XDIT_DEV_CONTAINER`, and `XDIT_REBUILD=1`. After editing Python, restart ComfyUI with:

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
