#!/usr/bin/env bash
# Build (if needed) and run the dev container with the repo bind-mounted.
set -euo pipefail

XDIT_OVERRIDE_SET=0
if [ -n "${XDIT_REPOSITORY+x}" ] || [ -n "${XDIT_REF+x}" ]; then
  XDIT_OVERRIDE_SET=1
fi
BASE_IMAGE="${BASE_IMAGE:-rocm/pytorch-xdit:v26.7}"
XDIT_REPOSITORY="${XDIT_REPOSITORY:-https://github.com/xdit-project/xDiT.git}"
XDIT_REF="${XDIT_REF:-68bca878f454c718d23b3764ded1de6add474eb4}"
CONTAINER_NAME="${XDIT_DEV_CONTAINER:-xdit-comfyui-${USER:-$(id -un)}}"
COMFY_PORT="${COMFY_PORT:-8188}"
PLUGIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${XDIT_DEV_IMAGE:-xdit-comfyui-dev}"

# Host path → in-container HF cache (/.cache/huggingface matches other xDiT workloads).
HF_CACHE_HOST="${HF_CACHE_HOST:-${HOME}/.cache/huggingface}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/.cache/huggingface}"

# ROCm when the host exposes /dev/kfd, otherwise NVIDIA. Set XDIT_DOCKER_ROCM to force.
if [ -z "${XDIT_DOCKER_ROCM:-}" ]; then
  if [ -e /dev/kfd ]; then XDIT_DOCKER_ROCM=1; else XDIT_DOCKER_ROCM=0; fi
fi
XDIT_DOCKER_SHM_SIZE="${XDIT_DOCKER_SHM_SIZE:-128g}"
XDIT_DOCKER_NETWORK="${XDIT_DOCKER_NETWORK:-host}"
XDIT_DOCKER_IPC="${XDIT_DOCKER_IPC:-host}"

mkdir -p "${HF_CACHE_HOST}"

if [ "${XDIT_OVERRIDE_SET}" = "1" ] || ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Building ${IMAGE} from docker/Dockerfile (BASE_IMAGE=${BASE_IMAGE})..."
  docker build -t "${IMAGE}" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "XDIT_REPOSITORY=${XDIT_REPOSITORY}" \
    --build-arg "XDIT_REF=${XDIT_REF}" \
    -f "${PLUGIN_ROOT}/docker/Dockerfile" \
    "${PLUGIN_ROOT}"
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

run_opts=( -d --name "${CONTAINER_NAME}" )

if [ "${XDIT_DOCKER_ROCM}" = "1" ]; then
  run_opts+=(
    --cap-add=SYS_PTRACE
    --security-opt seccomp=unconfined
    --user root
    --device=/dev/kfd
    --device=/dev/dri
    --group-add video
    --ipc="${XDIT_DOCKER_IPC}"
    --network "${XDIT_DOCKER_NETWORK}"
    --privileged
    --shm-size "${XDIT_DOCKER_SHM_SIZE}"
  )
else
  # --gpus all is what injects the NVIDIA userspace driver libs. Without it NCCL
  # fails on a missing libnvidia-ml.so.1 even though CUDA itself works.
  run_opts+=(
    --cap-add=SYS_PTRACE
    --security-opt seccomp=unconfined
    --user root
    --gpus all
    --ipc="${XDIT_DOCKER_IPC}"
    --network "${XDIT_DOCKER_NETWORK}"
    --shm-size "${XDIT_DOCKER_SHM_SIZE}"
  )
fi

if [ "${XDIT_DOCKER_NETWORK}" != "host" ]; then
  run_opts+=( -p "${COMFY_PORT}:8188" )
fi

run_opts+=(
  -v "${PLUGIN_ROOT}:/workspace/plugin"
  -v "${HF_CACHE_HOST}:${HF_CACHE_ROOT}"
  -e "HF_CACHE_ROOT=${HF_CACHE_ROOT}"
  -e "HF_HOME=${HF_CACHE_ROOT}"
  -e "OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}"
)

if [ "${XDIT_DOCKER_ROCM}" = "1" ]; then
  run_opts+=( -e "HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}" )
fi

for var in CUDA_VISIBLE_DEVICES GPU_ARCHS HIP_VISIBLE_DEVICES; do
  if [ -n "${!var:-}" ]; then
    run_opts+=( -e "${var}=${!var}" )
  fi
done

docker run "${run_opts[@]}" "${IMAGE}"

sleep 2
if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} exited during startup. Logs:" >&2
  docker logs "${CONTAINER_NAME}" >&2 || true
  exit 1
fi

if [ "${XDIT_DOCKER_NETWORK}" = "host" ]; then
  echo "ComfyUI: http://localhost:${COMFY_PORT} (host network)"
else
  echo "ComfyUI: http://localhost:${COMFY_PORT}"
fi
echo "Logs: docker logs -f ${CONTAINER_NAME}"
echo "Restart after node edits:"
echo "  docker exec ${CONTAINER_NAME} bash /workspace/plugin/scripts/docker/restart.sh"
