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

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
cache_dir = os.getenv("HF_HOME")

print(f"🚀 Downloading {model_name} to {cache_dir} ...")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
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
        torch_dtype="auto",
        device_map=None,
        use_safetensors=True,
        token=os.getenv("HF_TOKEN"),
        cache_dir=cache_dir
    )
    print("✔️ Model downloaded.")
except Exception as e:
    print("❌ Model download failed:", e)
    raise

print("✅ All downloads complete.")
'

if [ -d "/home/$USER/.cache/huggingface" ]; then
    echo "🧹 Removing local HF cache..."
    rm -rf "/home/$USER/.cache/huggingface"
fi

echo "🎉 Done!"