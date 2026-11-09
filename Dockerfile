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

# ---- teacher pipeline (offline arrangement), see docs/MODEL_LICENSES.md
COPY requirements-teacher.txt .
RUN pip install --no-cache-dir -c /tmp/constraints.txt -r requirements-teacher.txt \
    && pip install --no-cache-dir --no-deps basic-pitch==0.4.0 \
    && pip install --no-cache-dir --no-deps \
       "beat_this @ https://github.com/CPJKU/beat_this/archive/b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c.zip"
# BTC chord recognition (MIT), pinned commit; its weights ship in the repo. Patch a NumPy-2 removal.
RUN git clone -q https://github.com/jayg996/BTC-ISMIR19 /opt/btc && cd /opt/btc \
    && git checkout -q 2682317be668032e6e4b269ded36adaa2ad57df0 \
    && sed -i -E 's/np\.float([^0-9_a-z])/float\1/g' utils/transformer_modules.py \
    && rm -rf .git test/example.mp3
# Bake weights into the image so a server never downloads at runtime.
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs'); \
from beat_this.inference import load_checkpoint; load_checkpoint('final0')"

COPY . .
RUN pip install --no-cache-dir -c /tmp/constraints.txt -e .

# Data, runs and caches live on a mounted volume: docker run --gpus all -v /data:/data ...
VOLUME ["/data"]
CMD ["bash"]
