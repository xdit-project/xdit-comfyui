# syntax=docker/dockerfile:1
#
# Production-style image: ComfyUI + this pack baked in.
#
#   docker build -t xdit-comfyui -f docker/Dockerfile .

ARG BASE_IMAGE=rocm/pytorch-xdit:v26.7
FROM ${BASE_IMAGE}

ARG COMFYUI_REF=v0.28.0
ARG XDIT_REPOSITORY=https://github.com/xdit-project/xDiT.git
ARG XDIT_REF=68bca878f454c718d23b3764ded1de6add474eb4

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

# Install the requested xDiT branch, tag, or commit over the base image's copy.
RUN xdit_repo="${XDIT_REPOSITORY%.git}.git" \
    && xdit_spec="xfuser[cache-dit,opencv-python] @ git+${xdit_repo}@${XDIT_REF}" \
    && if [ -f /opt/build-constraints.txt ]; then \
         PIP_CONSTRAINT=/opt/build-constraints.txt \
           python -m pip install --no-cache-dir --upgrade "${xdit_spec}"; \
       else \
         python -m pip install --no-cache-dir --upgrade "${xdit_spec}"; \
       fi

COPY . ${PLUGIN_ROOT}

WORKDIR ${COMFYUI_ROOT}
EXPOSE 8188
ENTRYPOINT ["bash", "/workspace/plugin/scripts/docker/entrypoint.sh"]
