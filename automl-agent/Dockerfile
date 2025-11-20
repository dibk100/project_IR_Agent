# ==============================
# automl-runtime:latest Dockerfile
# ==============================

# 1. Base image
FROM nvidia/cuda:12.8.0-base-ubuntu24.04

# 2. 환경 설정
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# 3. 기본 패키지 설치 + Python 3.11 설치
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       wget curl git vim build-essential \
       ca-certificates python3.11 python3.11-venv python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 4. Virtual environment 생성
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 5. pip, setuptools, wheel 업그레이드
RUN pip install --upgrade pip setuptools wheel

# 6. ML / AutoML 필수 라이브러리 설치
# (원하는 패키지들을 requirements.txt에 작성 후 COPY + 설치)
# 예시:
# COPY requirements.txt /workspace/
# RUN pip install -r /workspace/requirements.txt
RUN pip install numpy pandas scikit-learn lightgbm catboost optuna

# 7. 작업 디렉토리
WORKDIR /workspace

# 8. 기본 실행 명령
CMD ["/bin/bash"]
