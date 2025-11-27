#!/bin/bash

# .env 파일에서 HuggingFace 토큰 불러오기
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d '=' -f2)
# echo $HF_TOKEN

sudo mkdir -p /mnt/hdd/hf_cache
sudo chown $USER:$USER /mnt/hdd/hf_cache

python -c "
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'Qwen/Qwen2.5-Coder-7B-Instruct'
cache_dir = '/mnt/hdd/hf_cache'  # 절대 경로 사용

# 캐시 디렉토리 없으면 생성
os.makedirs(cache_dir, exist_ok=True)

print(f'Downloading {model_name} to {cache_dir}...')

# Tokenizer 다운로드
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=os.getenv('HF_TOKEN'), cache_dir=cache_dir)

# 모델 다운로드 (safetensors 사용)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    torch_dtype='auto',
    use_safetensors=True,
    cache_dir=cache_dir,
    token=os.getenv('HF_TOKEN')
)

print('✅ Download complete.')
"

rm -rf /home/dibaeck/.cache/huggingface