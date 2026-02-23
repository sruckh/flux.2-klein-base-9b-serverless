# RunPod Serverless for FLUX.2-klein-base-9B with LoRA support
FROM runpod/base:1.0.3-cuda1281-ubuntu2404

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/runpod-volume/huggingface \
    HF_HUB_CACHE=/runpod-volume/huggingface/hub \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface/hub \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_XET_NUM_CONCURRENT_RANGE_GETS=32 \
    HF_HUB_ETAG_TIMEOUT=30 \
    HF_HUB_DOWNLOAD_TIMEOUT=300 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

# Install Python 3.12 and system dependencies (no python3-pip — use get-pip.py instead)
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Bootstrap pip via get-pip.py — installs outside apt so upgrades work freely
RUN wget -q https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py \
    && python3 /tmp/get-pip.py \
    && rm /tmp/get-pip.py \
    && pip install --upgrade setuptools wheel

# Install PyTorch with CUDA 12.8 support
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install Flash Attention
RUN pip install --no-cache-dir \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (includes boto3 for S3 support)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models, LoRA weights, and HuggingFace cache
RUN mkdir -p /workspace/models /workspace/lora /runpod-volume/huggingface

# Set proper permissions for cache directory
RUN chmod -R 777 /runpod-volume

# Set the handler as the entrypoint
CMD ["python3", "handler.py"]
