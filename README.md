# Log
sketch

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


### 🚀 구현 / 진행 상황
- [YYYY-MM-DD] 환경 세팅 완료
- [YYYY-MM-DD] 논문 알고리즘 재현 시작
- [YYYY-MM-DD] 데이터셋 준비 완료 / 실험 1 완료

### 💽 Error 기록
<details>
<summary>Error : pip install -r requirements.txt</summary>

### 오류 상황
```
Downloading nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl (410.6 MB) ━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━ 197.4/410.6 MB 41.0 MB/s eta 0:00:06
ERROR: Could not install packages due to an OSError: 
[Errno 28] No space left on device ━━━━━━━━━━━━━━━━━━━╸━━━━━━━━━━━━━━━━━━━━ 200.5/410.6 MB 40.9 MB/s eta 0:00:06
```
### 분석
- 오류: 디스크 공간 부족 (OSError: [Errno 28] No space left on device)
- 원인: 설치하려는 패키지(nvidia_cublas_cu12)가 약 410MB로, 설치 디렉토리에 남은 공간이 부족
- 상세: pip는 wheel 패키지를 임시 디렉토리에 풀어서 설치하는데, 이 임시 디렉토리 공간이 부족하면 설치 실패

### 해결 방법
```
# pip 캐시 비우기
pip cache purge

# 임시 디렉토리 생성
mkdir -p ~/tmp
export TMPDIR=~/tmp

# 다시 설치
pip install -r requirements.txt
```
⚠️ 참고: TMPDIR을 임시 디렉토리로 지정하면 pip가 패키지를 풀 때 이 디렉토리를 사용하므로 디스크 부족 문제를 회피할 수 있음

</details>
