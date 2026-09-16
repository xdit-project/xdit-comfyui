# Docker templates

These Dockerfiles target an **xDiT PyTorch base image** that already ships the GPU stack,
Python, `xfuser`, and the `xdit` CLI. They only add ComfyUI and this plugin layer.

Default base: **`rocm/pytorch-xdit:v26.7`**

Override when your site uses a different tag with the same layout:

```bash
docker build --build-arg BASE_IMAGE=amdsiloai/pytorch-xdit:v26.5.1 ...
```

## Assumed base image layout

The templates expect the base image to provide:

| Path                         | Purpose                                                                   |
| ---------------------------- | ------------------------------------------------------------------------- |
| `/opt/venv/bin/python`       | Common layout in some xDiT images (auto-detected if present)              |
| `xdit` on `PATH`             | Used to locate the Python env when `/opt/venv` is absent                  |
| `/opt/build-constraints.txt` | Optional pip constraints (some xDiT images ship this to block PyPI torch) |

If your image uses different paths, set `VIRTUAL_ENV` or `PIP_CONSTRAINT` at build or run time.
When the constraints file is absent, pip runs without it.

## Paths this layer adds

| Path                  | Purpose                                                                  |
| --------------------- | ------------------------------------------------------------------------ |
| `/workspace/plugin`   | This repo (`PLUGIN_ROOT`)                                                |
| `/workspace/comfyui`  | ComfyUI checkout (`COMFYUI_ROOT`)                                        |
| `/.cache/huggingface` | HF hub cache mount point (`HF_CACHE_ROOT`, matches other xDiT workloads) |

ComfyUI listens on **8188**. Lifecycle scripts live under `/workspace/plugin/scripts/`.

## Dev (bind-mount the repo)

**Cluster node** (ROCm devices, host network, shared `/raid` cache):

```bash
docker build -t xdit-comfyui-dev -f docker/Dockerfile .
HF_CACHE_HOST=/raid/huggingface \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/docker/run_dev.sh
```

Or equivalently:

```bash
docker run \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --user root \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --network host \
  --privileged \
  --shm-size 128G \
  -v "$PWD:/workspace/plugin" \
  -v /raid/huggingface:/.cache/huggingface \
  -e HF_HOME=/.cache/huggingface \
  -e HF_CACHE_ROOT=/.cache/huggingface \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e OMP_NUM_THREADS=16 \
  -p 8188:8188 \
  xdit-comfyui-dev
```

(`scripts/docker/run_dev.sh` sets these ROCm flags by default; use `--network host` so `-p`
is unnecessary.)

**Laptop / CUDA** — simpler runtime:

```bash
XDIT_DOCKER_ROCM=0 bash scripts/docker/run_dev.sh
```

After editing node Python: `bash /workspace/plugin/scripts/docker/restart.sh`

Images install the xDiT revision pinned by the Dockerfile and development launcher. To build
against a branch, tag, or commit from a fork, pass the repository and Git ref separately:

```bash
XDIT_REPOSITORY=https://github.com/OWNER/xDiT.git \
XDIT_REF=BRANCH_OR_COMMIT \
  bash scripts/docker/run_dev.sh
```

The helper rebuilds whenever either override is explicitly set.

### Run-time overrides

| Variable               | Default                                    | Purpose                                                        |
| ---------------------- | ------------------------------------------ | -------------------------------------------------------------- |
| `XDIT_DEV_CONTAINER`   | `xdit-comfyui-$USER`                       | Dev container name                                             |
| `XDIT_REPOSITORY`      | `https://github.com/xdit-project/xDiT.git` | Git repository from which xDiT is installed                    |
| `XDIT_REF`             | pinned revision                            | xDiT branch, tag, or commit installed by pip                   |
| `HF_CACHE_HOST`        | `$HOME/.cache/huggingface`                 | Host path bind-mounted into the container                      |
| `HF_CACHE_ROOT`        | `/.cache/huggingface`                      | In-container HF cache mount point                              |
| `XDIT_DOCKER_ROCM`     | `1`                                        | `0` → `--gpus all` + port publish instead of ROCm device flags |
| `XDIT_DOCKER_SHM_SIZE` | `128g`                                     | Shared memory (`--shm-size`)                                   |
| `CUDA_VISIBLE_DEVICES` | unset                                      | GPU selection (ROCm images often reuse this name)              |
| `OMP_NUM_THREADS`      | `16`                                       | CPU thread cap for BLAS/OpenMP                                 |
| `BASE_IMAGE`           | `rocm/pytorch-xdit:v26.7`                  | xDiT base image tag at build time                              |

## Prod (bake the plugin into the image)

```bash
docker build -t xdit-comfyui -f docker/Dockerfile .
docker run --rm -it \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --network host --privileged --shm-size 128G \
  -v /raid/huggingface:/.cache/huggingface \
  -e HF_HOME=/.cache/huggingface \
  xdit-comfyui
```
