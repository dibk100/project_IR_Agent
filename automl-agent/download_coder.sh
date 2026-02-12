#!/bin/bash
set -e  # 에러 발생 시 즉시 종료

# .env 파일에서 HuggingFace 토큰 불러오기
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d '=' -f2)
# echo $HF_TOKEN

CACHE_DIR="/mnt/hdd/hf_cache"

sudo mkdir -p "$CACHE_DIR"
sudo chown -R $USER:$USER "$CACHE_DIR"

export HF_HOME="$CACHE_DIR"
export TRANSFORMERS_CACHE="$CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"

echo "📁 Cache directory set to: $CACHE_DIR"

python -c '
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
cache_dir = os.getenv("HF_HOME")

print(f"🚀 Downloading {model_name} to {cache_dir} ...")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
        token=os.getenv("HF_TOKEN"),
        cache_dir=cache_dir
    )
    print("✔️ Tokenizer downloaded.")
except Exception as e:
    print("❌ Tokenizer download failed:", e)
    raise

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",  # ✅ 추가 권장
        device_map=None,
        token=os.getenv("HF_TOKEN"),
        cache_dir=cache_dir,
        low_cpu_mem_usage=True
    )
    print("✔️ Model downloaded.")
except Exception as e:
    print("❌ Model download failed:", e)
    raise

print("✅ All downloads complete.")
'

# 로컬 캐시 정리
if [ -d "/home/$USER/.cache/huggingface" ]; then
    echo "🧹 Removing local HF cache..."
    rm -rf "/home/$USER/.cache/huggingface"
fi

echo "🎉 Done! Model saved to: $CACHE_DIR"