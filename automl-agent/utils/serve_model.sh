#!/bin/bash
# serve_model.sh
set -e

MODEL_NAME=$1
PORT=8000

export HF_HOME=/mnt/hdd/hf_cache
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d '=' -f2)

mkdir -p logs

echo "🧹 Stopping old vLLM servers..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true

if [ "$MODEL_NAME" == "mistral" ]; then
    MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.3"
elif [ "$MODEL_NAME" == "deepseek_coder" ]; then
    MODEL_PATH="deepseek-ai/deepseek-coder-7b-instruct-v1.5"
else
    echo "❌ Unknown model: $MODEL_NAME"
    exit 1
fi

echo "🚀 Launching vLLM server with model: $MODEL_PATH"

nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --port $PORT \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    > logs/vllm_$MODEL_NAME.log 2>&1 &

echo "⏳ Waiting for vLLM server to be ready..."
sleep 5

echo "✅ vLLM serving $MODEL_NAME at http://localhost:$PORT/v1"
