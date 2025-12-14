# ==============================
# automl-runtime:latest Dockerfile
# ==============================

# 1. Base image (runtime 버전 사용 권장)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 2. 환경 설정
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

SHELL ["/bin/bash", "-c"]

# 3. 기본 패키지 설치 + Python 3.10 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    vim \
    build-essential \
    ca-certificates \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python 심볼릭 링크 설정
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# 4. Virtual environment 생성
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 5. pip, setuptools, wheel 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 6. ML / AutoML 필수 라이브러리 설치
RUN pip install --no-cache-dir \
    "numpy<2" \
    pandas \
    scikit-learn \
    scipy \
    matplotlib \
    seaborn \
    lightgbm \
    xgboost \
    catboost \
    optuna \
    gradio \
    transformers \
    datasets \
    accelerate \
    Pillow

# 7. PyTorch + torchvision + torchaudio 설치 (CUDA 12.1 기준)
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 8. 추가 유용한 패키지
RUN pip install --no-cache-dir \
    jupyter \
    ipython \
    tensorboard \
    wandb \
    mlflow

# 9. HuggingFace 캐시 디렉토리 설정
ENV HF_HOME=/workspace/hf_cache \
    TRANSFORMERS_CACHE=/workspace/hf_cache

# 10. 작업 디렉토리
WORKDIR /workspace

# 11. 비root 사용자 생성
RUN useradd -m -u 1000 -s /bin/bash automl && \
    chown -R automl:automl /workspace
USER automl

# 12. 기본 실행 명령
CMD ["sleep", "infinity"]