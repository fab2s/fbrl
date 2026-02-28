FROM python:3.10-slim AS deps

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip pip install torch==2.5.1
RUN --mount=type=cache,target=/root/.cache/pip pip install numpy 'matplotlib<3.8' pillow fonttools

FROM deps

RUN apt-get update && \
    apt-get install -y --no-install-recommends fonts-dejavu-core fonts-liberation fonts-liberation-sans-narrow && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["python", "vision_training.py"]
