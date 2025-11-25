# ==============================
# automl-runtime:latest Dockerfile
# ==============================

# 1. Base image
FROM nvidia/cuda:12.8.0-base-ubuntu24.04

# 2. 환경 설정
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# 3. 기본 패키지 설치 + Python 3.12 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
       wget curl git vim build-essential ca-certificates \
       python3 python3-venv python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 4. Virtual environment 생성
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 5. pip, setuptools, wheel 업그레이드
RUN pip install --upgrade pip setuptools wheel

# 6. ML / AutoML 필수 라이브러리 설치
RUN pip install numpy pandas scikit-learn lightgbm catboost optuna gradio

# 7. PyTorch + torchvision + torchaudio 설치 (CUDA 12.8 기준)
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch torchvision torchaudio --upgrade

# 8. 작업 디렉토리
WORKDIR /workspace

# 9. 기본 실행 명령
CMD ["/bin/bash"]
