# RTX 50-series (Blackwell, sm_120) needs CUDA 12.8+ and PyTorch 2.7+.
# Needs a host NVIDIA driver >= 570.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg sox libsndfile1 build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Pin the image's torch/torchaudio so no dependency silently swaps in a CUDA build without sm_120.
RUN pip freeze | grep -E '^(torch|torchaudio)==' > /tmp/constraints.txt \
    && pip install --no-cache-dir -c /tmp/constraints.txt -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -c /tmp/constraints.txt -e .

# Data, runs and caches live on a mounted volume: docker run --gpus all -v /data:/data ...
VOLUME ["/data"]
CMD ["bash"]
