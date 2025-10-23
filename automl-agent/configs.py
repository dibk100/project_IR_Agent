import os
from dotenv import load_dotenv

load_dotenv()

class Configs:
    OPENAI_KEY = ""  # your openai's account api key
    HF_KEY = os.getenv("HF_TOKEN")
    PWC_KEY = ""
    SEARCHAPI_API_KEY = ""
    TAVILY_API_KEY = ""
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")

AVAILABLE_LLMs = {  
    "prompt-llm": {
        "api_key": "empty",                 # # vLLM 서버는 api_key 필요 없음
        "model": "mistral",
        "base_url": "http://localhost:8000/v1",
    },
    "coder-llm": {
        "api_key": "empty",                     
        "model": "qwen_coder",                  # vLLM 서버에서 등록한 served_model_name
        "base_url": "http://localhost:8000/v1"  # 한 서버에서 두 모델 서빙
    },
    "gpt-4": {"api_key": Configs.OPENAI_KEY, "model": "gpt-4o"},
    "gemini-flash": {"api_key": Configs.GEMINI_KEY, "model": "gemini-2.5-flash","provider": "google"},      #   "base_url": "https://api.generativeai.google/v1/models",  # 실제 엔드포인트 작성 필요
    
}

TASK_METRICS = {
    "image_classification": "accuracy",
    "text_classification": "accuracy",
    "tabular_classification": "F1",
    "tabular_regression": "RMSLE",
    "tabular_clustering": "RI",
    "node_classification": "accuracy",
    "ts_forecasting": "RMSLE",
}
