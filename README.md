# Project_IR_sketch 🔍
> 본 레포는 sLM_MAS 연구(1세대)이며, 고도화 연구는 [project_IR_sLM_MAS](https://github.com/dibk100/project_IR_sLM_MAS)(2026)에서 진행 중
- **Type**: 개인 연구 프로젝트 (Independent Research)
- **Subject**: OpenSourec sLM 기반 자율 에이전트 시스템 연구
- **Focus**: AgentVerse, AutoML-Agent, LightAgent 논문 구현 및 확장형 sLM(OpenSourec) 기반 지능형 Agent 구조 설계
- **Period**: 2025-09-19 ~ 2025.12.01

### 🚀 Goal
- 오픈소스 sLM 기반으로, 상용 API 모델(gpt,claude,etc) 수준의 성능을 끌어내기 위한 전략 및 설계 수립.
- SLM과 시스템 설계(프롬프트/평가/메모리/협업 구조 등)를 결합하여 비용 효율적이고 실전 적용 가능한 에이전트 성능을 목표로 함.

### 📌 Notes & Issues
- sLM-based Code Generation ISSUE :
    - Operation Agent의 **코드 생성** 병목 현상 발견
    - 코드는 생성하는데, SyntaxError/NameError/ImportError 같은 Low-Level 오류가 발생하여, 코드를 다시 새로 생성(retry)
    - (Notes) prompt에 Error-Type이나 제약조건을 추가하는 방식으로 제어하려 했으나, 코드 생성이 무너짐
    - (Notes) doker를 활용하여 ImportError를 해결하려고 했으나, Operation Agent의 **Executor**할 때 마다 도커 생성, 삭제가 반복되어 비효율적이었고, 결론적으로 prompt에 추가할 문맥을 찾는 것에 불과했음.

### 📝 Wrap-up
- Failure의 발생 원인, 결과, 실험흐름을 기록하지 않고, 실행 or Not 에 집중했음. 연구 기록을 잘 남기면 논문화하는 과정에 인사이트를 얻을 수 있을 것 같음.
    - ✍🏻 연구할 때, 실험 기록(실패-분석-방향)을 잘 남겨두기(Notion, Github READBE.md)
- Goal 설정을 고정해서 LLM과의 성능 비교에 초점이 된 연구였음. 하지만 sLM, 심지어 open-source를 활용할 생각이었다면, 성능 비교가 아닌 다른 방향으로 초점을 맞춰야할 것 같음.
    - ✍🏻 선행 연구와 Key-Paper 선정이 문제였던 것 같음. paper-review할 때, Research Landscape를 구상하며 Positioning Prior Work를 정리하고 인사이트 얻기

### 📚 논문 구현 작업(related works)
- [AgentVerse (ICLR 2023)](https://github.com/dibk100/paper_agentverse)
- [LightAgent (preprint 2025-09-10)](https://github.com/dibk100/paper_LightAgent)
- [AutoML-Agent (ICML 2025)](https://github.com/dibk100/project_IR_Agent/tree/main/automl-agent)
