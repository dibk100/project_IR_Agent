#!/bin/bash
set -e

export HF_HOME=/mnt/hdd/hf_cache
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d '=' -f2)

MODEL_PATH="/mnt/hdd/hf_cache/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71"

echo "📁 Using model: $MODEL_PATH"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name mistral \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --port 8000
