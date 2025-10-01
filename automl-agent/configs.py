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
    "gpt-4": {"api_key": Configs.OPENAI_KEY, "model": "gpt-4o"},
    "gemini-flash": {"api_key": Configs.GEMINI_KEY, "model": "gemini-2.5-flash","provider": "google"},
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
