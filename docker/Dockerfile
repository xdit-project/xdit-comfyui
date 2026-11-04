# syntax=docker/dockerfile:1
#
# Production-style image: ComfyUI + this pack baked in.
#
#   docker build --build-arg BASE_IMAGE=<xdit-image> \
#     -t xdit-comfyui -f docker/Dockerfile .

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG COMFYUI_REF=v0.44.0
ARG XDIT_REPOSITORY
ARG XDIT_REF

ENV PLUGIN_ROOT=/workspace/plugin \
    COMFYUI_ROOT=/workspace/comfyui \
    HF_CACHE_ROOT=/cache/huggingface \
    COMFY_PORT=8188

# Do not let ComfyUI's unpinned torch/vision/audio replace the base image stack.
RUN git clone --depth 1 --branch ${COMFYUI_REF} \
        https://github.com/comfyanonymous/ComfyUI ${COMFYUI_ROOT} \
    && grep -vE '^(torch|torchvision|torchaudio)([=<>!~ ].*)?$' \
         ${COMFYUI_ROOT}/requirements.txt > /tmp/comfy-reqs.txt \
    && if [ -f /opt/build-constraints.txt ]; then \
         PIP_CONSTRAINT=/opt/build-constraints.txt \
           python -m pip install --no-cache-dir -r /tmp/comfy-reqs.txt; \
       else \
         python -m pip install --no-cache-dir -r /tmp/comfy-reqs.txt; \
       fi

# ComfyUI imports torchaudio at startup. Keep the base image's wheel when it
# imports (ROCm). If missing or CUDA-mismatched (NGC 13.3 vs PyPI cu130),
# install the CPU wheel with --no-deps so pip cannot replace GPU torch.
RUN if ! python -c "import torchaudio" >/dev/null 2>&1; then \
      python -m pip install --no-cache-dir --no-deps --force-reinstall \
        torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Validated profile images own xDiT and may carry architecture-specific patches.
# Only replace that copy when the caller explicitly overrides repository/ref.
RUN if [ -n "${XDIT_REPOSITORY}" ] && [ -n "${XDIT_REF}" ]; then \
      xdit_repo="${XDIT_REPOSITORY%.git}.git"; \
      xdit_spec="xfuser[cache-dit,opencv-python] @ git+${xdit_repo}@${XDIT_REF}"; \
      if [ -f /opt/build-constraints.txt ]; then \
        PIP_CONSTRAINT=/opt/build-constraints.txt \
          python -m pip install --no-cache-dir --upgrade "${xdit_spec}"; \
      else \
        python -m pip install --no-cache-dir --upgrade "${xdit_spec}"; \
      fi; \
    fi

# Base images may install xfuser without its optional runtime dependencies.
# Resolve those through the exact xDiT source bundled in the image so its
# compatibility bounds remain authoritative.
RUN if [ -z "${XDIT_REPOSITORY}" ] && [ -z "${XDIT_REF}" ]; then \
      test -f /app/xDiT/pyproject.toml; \
      xdit_spec="/app/xDiT[cache-dit,opencv-python]"; \
      if [ -f /opt/build-constraints.txt ]; then \
        PIP_CONSTRAINT=/opt/build-constraints.txt \
          python -m pip install --no-cache-dir "${xdit_spec}"; \
      else \
        python -m pip install --no-cache-dir "${xdit_spec}"; \
      fi; \
    fi

COPY . ${PLUGIN_ROOT}

WORKDIR ${COMFYUI_ROOT}
EXPOSE 8188
ENTRYPOINT ["bash", "/workspace/plugin/scripts/docker/entrypoint.sh"]
