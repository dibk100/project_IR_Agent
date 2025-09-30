#!/bin/bash
# chmod +x run_vllm_server.sh
# ./run_vllm_server.sh

export HF_HOME=/home/dibaeck/hf_cache

# .env 파일에서 HuggingFace 토큰 불러오기
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d '=' -f2)
# echo $HF_TOKEN

# vLLM 서버 실행 
python -m vllm.entrypoints.openai.api_server \
    --model /home/dibaeck/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a/ \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.7 \
    --port 8000