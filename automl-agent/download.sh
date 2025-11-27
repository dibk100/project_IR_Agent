#!/bin/bash

# .env 파일에서 HuggingFace 토큰 불러오기
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d '=' -f2)
# echo $HF_TOKEN

sudo mkdir -p /mnt/hdd/hf_cache
sudo chown $USER:$USER /mnt/hdd/hf_cache

python -c "
import os
from transformers import AutoModel, AutoTokenizer

model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
cache_dir = '/mnt/hdd/hf_cache'  # 절대 경로 사용

# 캐시 디렉토리 없으면 생성
os.makedirs(cache_dir, exist_ok=True)

AutoModel.from_pretrained(model_name, device_map='auto', use_safetensors=True, cache_dir=cache_dir)
AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
"

rm -rf /home/dibaeck/.cache/huggingface