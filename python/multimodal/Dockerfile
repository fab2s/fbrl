FROM python:3.10-slim

# Install system deps for MoviePy/FFmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Install Python deps with pip cache mount for fast rebuilds.
# Heaviest / most stable first, lightest / most likely to change last.
RUN --mount=type=cache,target=/root/.cache/pip pip install torch==2.5.1 torchaudio==2.5.1
RUN --mount=type=cache,target=/root/.cache/pip pip install torchcodec
RUN --mount=type=cache,target=/root/.cache/pip pip install numpy
RUN --mount=type=cache,target=/root/.cache/pip pip install 'matplotlib<3.8'
RUN --mount=type=cache,target=/root/.cache/pip pip install pillow
RUN --mount=type=cache,target=/root/.cache/pip pip install gtts
RUN --mount=type=cache,target=/root/.cache/pip pip install 'moviepy<2'
RUN --mount=type=cache,target=/root/.cache/pip pip install librosa

# Copy script
COPY multimodal_training.py .

# Run script (override with docker run args)
ENTRYPOINT ["python", "multimodal_training.py"]
