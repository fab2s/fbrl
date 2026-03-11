# fbrl development image
# Go + libtorch + CUDA, ready for CGo development and training.
#
# Same environment as goDl — copied rather than referenced so fbrl
# is fully self-contained and buildable from its own repo.

# --- Layer 1: Base image ---
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# --- Layer 2: System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    ca-certificates \
    git \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# --- Layer 3: Go ---
ARG GO_VERSION=1.24.1
RUN wget -q https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz \
    && rm go${GO_VERSION}.linux-amd64.tar.gz

ENV PATH="/usr/local/go/bin:/root/go/bin:${PATH}"

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
ENV C_INCLUDE_PATH="${LIBTORCH_PATH}/include:${LIBTORCH_PATH}/include/torch/csrc/api/include:${C_INCLUDE_PATH}"
ENV CPLUS_INCLUDE_PATH="${LIBTORCH_PATH}/include:${LIBTORCH_PATH}/include/torch/csrc/api/include:${CPLUS_INCLUDE_PATH}"

ENV CGO_CFLAGS="-I${LIBTORCH_PATH}/include -I${LIBTORCH_PATH}/include/torch/csrc/api/include"
ENV CGO_LDFLAGS="-L${LIBTORCH_PATH}/lib"
ENV CGO_ENABLED=1

ENV GOFLAGS="-tags=cuda"

ENV NVIDIA_PRODUCT_NAME=""
ENTRYPOINT []

WORKDIR /workspace
