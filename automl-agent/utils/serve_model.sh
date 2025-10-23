#!/bin/bash
# serve_model.sh
# chmod +x serve_model.sh
MODEL_NAME=$1
PORT=8000

# logs 디렉토리 확인 및 생성
mkdir -p logs

echo "🧹 Stopping old vLLM servers..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null

if [ "$MODEL_NAME" == "mistral" ]; then
    MODEL_PATH="/home/dibaeck/hf_cache/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a/"
elif [ "$MODEL_NAME" == "qwen_coder" ]; then
    MODEL_PATH="/home/dibaeck/hf_cache/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242/"

else
    echo "❌ Unknown model: $MODEL_NAME"
    exit 1
fi

echo "🚀 Launching vLLM server with model: $MODEL_NAME"
nohup python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --port $PORT \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8 \
    > logs/vllm_$MODEL_NAME.log 2>&1 &

echo "✅ vLLM now serving $MODEL_NAME at http://localhost:$PORT/v1"
