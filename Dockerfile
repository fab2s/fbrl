# fbrl development image
# Rust + libtorch + CUDA, ready for flodl/fbrl development and training.

# --- Layer 1: Base image ---
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

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
    && rm -rf /var/lib/apt/lists/*

# --- Layer 3: Rust (installed globally so non-root user can access) ---
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH="/usr/local/cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable \
    && chmod -R a+rwx /usr/local/rustup /usr/local/cargo

# --- Layer 4: libtorch (CUDA 12.8) ---
ARG LIBTORCH_VERSION=2.10.0
RUN --mount=type=cache,target=/tmp/libtorch-cache \
    ZIPFILE="libtorch-shared-with-deps-${LIBTORCH_VERSION}+cu128.zip" && \
    if [ ! -f "/tmp/libtorch-cache/${ZIPFILE}" ]; then \
        wget -q "https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcu128.zip" \
            -O "/tmp/libtorch-cache/${ZIPFILE}"; \
    fi && \
    unzip -q "/tmp/libtorch-cache/${ZIPFILE}" -d /usr/local

# --- Layer 5: Environment ---
ENV LIBTORCH_PATH="/usr/local/libtorch"
ENV LD_LIBRARY_PATH="${LIBTORCH_PATH}/lib:/usr/local/cuda/lib64"
ENV LIBRARY_PATH="${LIBTORCH_PATH}/lib:/usr/local/cuda/lib64"
ENV CUDA_HOME="/usr/local/cuda"

ENTRYPOINT []

WORKDIR /workspace
