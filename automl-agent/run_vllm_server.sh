#!/bin/bash
# chmod +x run_vllm_server.sh
# ./run_vllm_server.sh

export HF_HOME=/home/dibaeck/hf_cache

# .env 파일에서 HuggingFace 토큰 불러오기
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d '=' -f2)
# echo $HF_TOKEN

# vLLM 서버 실행 
python -m vllm.entrypoints.openai.api_server \
    --model /home/dibaeck/hf_cache/mistralai/Mistral-7B-Instruct-v0.3 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.6 \
    --gpu-memory-utilization 0.6 \
    --port 8000