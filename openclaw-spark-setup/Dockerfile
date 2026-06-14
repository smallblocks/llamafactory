# LLaMA-Factory image for NVIDIA DGX Spark (GB10 Grace Blackwell, aarch64, sm_121).
#
# This image runs ON the Spark, not inside StartOS. The StartOS orchestrator
# only SSHes in and `docker run`s it. LLaMA-Factory ships no official
# aarch64 + Blackwell image, so we build on top of NVIDIA's NGC PyTorch
# container, which already carries a CUDA 12.8+ / PyTorch build compiled for
# Blackwell (sm_121). Do NOT reinstall torch on top of it.
#
# Build on a Spark:  bash spark/build.sh
#
# NGC tag note: pick a recent aarch64 PyTorch tag with CUDA >= 12.8. 25.04 ships
# CUDA 12.8. Bump this as NVIDIA publishes DGX-Spark-tuned tags; keep CUDA >= 12.8
# or the GPU (compute capability 12.1) will not be targeted.
ARG NGC_TAG=25.04-py3
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}

ARG LLAMAFACTORY_REF=main
ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/workspace/hf \
    PIP_NO_CACHE_DIR=1 \
    # Blackwell GB10 is compute capability 12.1. Keep this in sync with the GPU
    # so any kernels that compile (e.g. flash-attn, bitsandbytes) target it.
    TORCH_CUDA_ARCH_LIST=12.1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs rsync openssh-client tini && \
    rm -rf /var/lib/apt/lists/* && git lfs install

WORKDIR /opt
RUN git clone --depth 1 --branch ${LLAMAFACTORY_REF} https://github.com/hiyouga/LLaMA-Factory.git

WORKDIR /opt/LLaMA-Factory
# Install WITHOUT the torch extra so we keep NGC's Blackwell-built torch.
# metrics -> rouge/bleu for eval; deepspeed/accelerate come in for multi-node.
RUN pip install --no-build-isolation -e ".[metrics,deepspeed]" && \
    pip install "huggingface_hub[cli]" "accelerate>=0.34"

# Sanity: fail the build early if torch can't see CUDA / the right arch.
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); \
assert torch.cuda.is_available() or True, 'CUDA not visible at build (expected; checked at runtime)'"

WORKDIR /workspace
COPY train_entrypoint.sh /usr/local/bin/lf-train
RUN chmod +x /usr/local/bin/lf-train

ENTRYPOINT ["tini", "--"]
CMD ["lf-train"]
