FROM python:3.10-slim

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip pip install torch==2.5.1
RUN --mount=type=cache,target=/root/.cache/pip pip install numpy 'matplotlib<3.8' pillow

COPY vision_training.py .

ENTRYPOINT ["python", "vision_training.py"]
