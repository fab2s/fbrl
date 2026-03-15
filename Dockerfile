# fbrl development image
# Rust + libtorch + CUDA, ready for flodl/fbrl development and training.

# --- Layer 1: Base image ---
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# --- Layer 2: System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    unzip \
    ca-certificates \
    git \
    gcc \
    g++ \
    pkg-config \
    graphviz \
    cuda-nsight-systems-12-6 \
    && rm -rf /var/lib/apt/lists/*

# --- Layer 3: Rust (installed globally so non-root user can access) ---
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH="/usr/local/cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable \
    && chmod -R a+rX /usr/local/rustup /usr/local/cargo

# --- Layer 4: libtorch + CUDA ---
ARG LIBTORCH_VERSION=2.10.0
ARG CUDA_TAG=cu126
RUN wget -q https://download.pytorch.org/libtorch/${CUDA_TAG}/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2B${CUDA_TAG}.zip \
    && unzip -q libtorch-shared-with-deps-${LIBTORCH_VERSION}+${CUDA_TAG}.zip -d /usr/local \
    && rm libtorch-shared-with-deps-${LIBTORCH_VERSION}+${CUDA_TAG}.zip

# --- Layer 5: Environment ---
ENV LIBTORCH_PATH="/usr/local/libtorch"
ENV LD_LIBRARY_PATH="${LIBTORCH_PATH}/lib:${LD_LIBRARY_PATH}"
ENV LIBRARY_PATH="${LIBTORCH_PATH}/lib:${LIBRARY_PATH}"

ENV NVIDIA_PRODUCT_NAME=""
ENTRYPOINT []

WORKDIR /workspace
