# Experiments_AutoML Agent 💽
- models : 
    - mistralai/Mistral-7B-Instruct-v0.3
    - bigcode/starcoder2-3b
- dataset : 
    - Tabular Classification Banana Quality
- fast-TEST:
    - AutoMLAgent.ipynb

## 📁 Folder Structure
```
project/
├── automl-agent/
│   ├── agent_manager/
│   │   ├── __init__.py
│   │   └── retriever.py
│   ├── data_agent/
│   │   ├── __init__.py
│   │   └── retriever.py
│   ├── model_agent/
│   │   ├── __init__.py
│   │   └── retriever.py
│   ├── operation_agent/
│   │   ├── __init__.py
│   │   └── execution.py
│   ├── prompt_agent/
│   │   ├── __init__.py
│   │   ├── WizardLAMP/
│   │   └── schema.json
│   ├── prompt_pool/
│   │   ├── __init__.py
│   │   └── tabular_classification.py
│   ├── configs.py
│   └── AutoMLAgent.ipynb

└── requirements.txt

```

# AutoML-Agent
>오리지널 깃허브 레포지토리 :   
This is the official implementation of **AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML** (ICML 2025) 
> [[Paper](https://arxiv.org/abs/2410.02958)][[Poster](/static/pdfs/poster.pdf)][[Website](https://deepauto-ai.github.io/automl-agent/)]
