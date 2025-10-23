import os
import requests
import subprocess
import time

AVAILABLE_LLMs = {
    "prompt-llm": {"model": "mistral", "base_url": "http://localhost:8000/v1"},
    "coder-llm": {"model": "qwen_coder", "base_url": "http://localhost:8000/v1"},
}

def wait_vllm_ready(base_url="http://localhost:8000/v1", timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{base_url}/models", timeout=3)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(3)
    raise RuntimeError("vLLM server did not become ready in time")

def switch_model(llm_name):
    model_info = AVAILABLE_LLMs[llm_name]
    model_name = model_info["model"]

    # 현재 vLLM 서버 모델 상태 확인
    try:
        res = requests.get(f"{model_info['base_url']}/models", timeout=3)
        res.raise_for_status()
        current_models = [m["id"] for m in res.json().get("data", [])]
    except Exception:
        current_models = []

    # 서빙 중인 모델이 다르면 전환
    if model_name not in current_models:
        print(f"🔄 Switching vLLM server to {model_name} ...")
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(["bash", os.path.join(SCRIPT_DIR, "serve_model.sh"), model_name])
        wait_vllm_ready(model_info['base_url'])
        print(f"✅ {model_name} is now active on vLLM server.")
    else:
        print(f"✅ {model_name} already active.")
