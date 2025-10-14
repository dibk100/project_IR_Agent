#!/bin/bash

# .env 파일에서 HuggingFace 토큰 불러오기
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d '=' -f2)
# echo $HF_TOKEN

python -c "
import os
from transformers import AutoModel, AutoTokenizer

model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
cache_dir = '/home/dibaeck/hf_cache'  # 절대 경로 사용

AutoModel.from_pretrained(model_name, device_map='auto', use_safetensors=True, cache_dir=cache_dir)
AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
"