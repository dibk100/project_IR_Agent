#!/bin/bash
set -e  # 오류 발생 시 즉시 종료

# .env 파일에서 HuggingFace 토큰 불러오기
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d '=' -f2)
# echo $HF_TOKEN

CACHE_DIR="/mnt/hdd/hf_cache"
sudo mkdir -p "$CACHE_DIR"
sudo chown -R $USER:$USER "$CACHE_DIR"

export HF_HOME="$CACHE_DIR"
export TRANSFORMERS_CACHE="$CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"

echo "📁 Cache directory set to $CACHE_DIR"

python -c '
import os
from transformers import AutoModel, AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
cache_dir = os.getenv("HF_HOME")

print(f"Downloading {model_name} to {cache_dir}")

# Tokenizer
try:
    tok = AutoTokenizer.from_pretrained(
        model_name,
        token=os.getenv("HF_TOKEN"),
        cache_dir=cache_dir
    )
    print("Tokenizer downloaded.")
except Exception as e:
    print("Tokenizer download failed:", e)
    raise

# Model
try:
    model = AutoModel.from_pretrained(
        model_name,
        device_map=None,   # 다운로드만 하므로 auto 불필요
        use_safetensors=True,
        token=os.getenv("HF_TOKEN"),
        cache_dir=cache_dir
    )
    print("Model downloaded.")
except Exception as e:
    print("Model download failed:", e)
    raise

print("All downloads complete.")
'

if [ -d "/home/$USER/.cache/huggingface" ]; then
  echo "Cleaning ~/.cache/huggingface ..."
  rm -rf "/home/$USER/.cache/huggingface"
fi

echo "Done!"