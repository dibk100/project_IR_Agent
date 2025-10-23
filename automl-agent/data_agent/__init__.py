from configs import AVAILABLE_LLMs
from data_agent import retriever
from utils import print_message, get_client

# 기본 버전
# agent_profile = """You are a helpful assistant."""

# 역할 명시 버전 : 데이터 관련 책임 명시 ver
# agent_profile = """You are a helpful assistant. You have the following main responsibilities to complete.
# 1. Retrieve a dataset from the user or search for the dataset based on the user instruction.
# 2. Perform data preprocessing based on the user instruction or best practice based on the given tasks.
# 3. Perform data augmentation as neccesary.
# 4. Extract useful information and underlying characteristics of the dataset."""

# AutoML과 시각화 강조 버전 : 역할을 데이터 과학자로 승격/ AutoML프로젝트 맥락 강조/ 데이터 시각화 언급 / 데이터 이해 포괄적으로 수행 :: 기존 4가지 책임 + 시각화 가능
# agent_profile = """You are a data scientist of an automated machine learning project (AutoML) that can find the most relevant datasets,run useful preprocessing, perform suitable data augmentation, and make meaningful visulaization to comprehensively understand the data based on the user requirements. You have the following main responsibilities to complete.
# 1. Retrieve a dataset from the user or search for the dataset based on the user instruction.
# 2. Perform data preprocessing based on the user instruction or best practice based on the given tasks.
# 3. Perform data augmentation as neccesary.
# 4. Extract useful information and underlying characteristics of the dataset."""

# 경험 강조 버전 : “experienced”를 추가해 전문성을 강조
# agent_profile = """You are an experienced data scientist of an automated machine learning project (AutoML) that can find the most relevant datasets,run useful preprocessing, perform suitable data augmentation, and make meaningful visulaization to comprehensively understand the data based on the user requirements. You have the following main responsibilities to complete.
# 1. Retrieve a dataset from the user or search for the dataset based on the user instruction.
# 2. Perform data preprocessing based on the user instruction or best practice based on the given tasks.
# 3. Perform data augmentation as neccesary.
# 4. Extract useful information and underlying characteristics of the dataset."""

# 최고 전문가 강조 버전 : You are the world's best data scientist 표현 추가 / LLM한테 자신감 있게, 최적화된 방식이 나오도록 유도(부담감주기)
agent_profile = """You are the world's best data scientist of an automated machine learning project (AutoML) that can find the most relevant datasets,run useful preprocessing, perform suitable data augmentation, and make meaningful visulaization to comprehensively understand the data based on the user requirements. You have the following main responsibilities to complete.
1. Retrieve a dataset from the user or search for the dataset based on the user instruction.
2. Perform data preprocessing based on the user instruction or best practice based on the given tasks.
3. Perform data augmentation as neccesary.
4. Extract useful information and underlying characteristics of the dataset."""


class DataAgent:
    """
    DataAgent는 LLM을 활용해 데이터 중심 계획을 분석하고, 단계별 실행 전략을 상세히 설명하는 자동화 데이터 분석 에이전트
    """
    def __init__(self, user_requirements, llm="qwen", rap=True, decomp=True):       # RAP(Reasoning/Planning) 사용 여부/ Task decomposition 사용 여부
        self.agent_type = "data"
        self.llm = llm
        self.model = AVAILABLE_LLMs[llm]["model"]
        self.user_requirements = user_requirements
        self.rap = rap
        self.decomp = decomp
        self.money = {}

    def understand_plan(self, plan):
        """
        역할 : 사용자 요구사항과 프로젝트 계획(plan)을 LLM에게 전달해서 데이터 중심으로 실행 가능한 요약/세부 계획을 생성
        
        1. summary_prompt 생성
        2. LLM 호출 :: 시스템 프롬프트(agent_profile) + 사용자 프롬프트(summary_prompt)
        3. 결과 저장
        4. return :: data_plan : 데이터 중심 계획 요약 문자열
        """ 
        
        summary_prompt = f"""As a proficient data scientist, summarize the following plan given by the senior AutoML project manager according to the user's requirements and your expertise in data science.
        
        # User's Requirements
        ```json
        {self.user_requirements}
        ```
        
        # Project Plan
        {plan}
        
        The summary of the plan should enable you to fulfill your responsibilities as the answers to the following questions by focusing on the data manipulation and analysis.
        1. How to retrieve or collect the dataset(s)?
        2. How to preprocess the retrieved dataset(s)?
        3. How to efficiently augment the dataset(s)?
        4. How to extract and understand the underlying characteristics of the dataset(s)?
        
        Note that you should not perform data visualization because you cannot see it. Make sure that another data scientist can exectly reproduce the results based on your summary."""

        messages = [
            {"role": "system", "content": agent_profile},
            {"role": "user", "content": summary_prompt},
        ]

        retry = 0
        while retry < 10:
            try:
                res = get_client(self.llm).chat.completions.create(
                    model=self.model, messages=messages, temperature=0.3
                )
                break
            except Exception as e:
                print_message("system", e)
                retry += 1
                continue

        data_plan = res.choices[0].message.content.strip()
        self.money[f"Data_Plan_Decomposition"] = res.usage.to_dict(mode="json")
        return data_plan

    def execute_plan(self, plan, data_path, pid):
        """
        역할 : 이해한 데이터 계획(data_plan)을 기반으로 실제 데이터 처리 단계를 수행하기 위한 구체적인 설명/실행 계획을 생성
        
        1. 계획 이해 - llm으로 요약, data_plan
        2. 데이터셋  - 나는 로컬에 입력함., or 링크
        3. 실행 프롬프트 생성 exec_prompt
        
        """
        print_message(self.agent_type, "I am working with the given plan!", pid)
        
        # 1. 계획 이해 
        if self.decomp:
            # True : 계획을 LLM에게 이해시켜 요약
            data_plan = self.understand_plan(plan)
        else:
            # False : 사용자가 제공한 계획 그대로 사용
            data_plan = plan

        # 2. 데이터셋 수집
        # 사용자 업로드, url, 허깅페이스, 케글 등에서 데이터셋 검색/load
        available_sources = retriever.retrieve_datasets(
            self.user_requirements, data_path, get_client(self.llm), self.model
        )

        # Check whether the given source is accessible before running the execution --> reduce FileNotFound error
        # 파일 접근 권한/ 존재 여부 확인하는게 좋을거임

        # modality-based extraction ?
        # 데이터 종류에 따라 처리가 달라질거니까 그것도 확인ㄱ

        # 3. 실행 프롬프트 생성 : 
        exec_prompt = f"""As a proficient data scientist, your task is to explain **detailed** steps for data manipulation and analysis parts by executing the following machine learning development plan.
        
        # Plan
        {data_plan}
        
        # Potential Source of Dataset
        {available_sources}
        
        Make sure that your explanation follows these instructions:
        - All of your explanation must be self-contained without using any placeholder to ensure that other data scientists can exactly reproduce all the steps, but do not include any code.
        - Include how and where to retrieve or collect the data.
        - Include how to preprocess the data and which tools or libraries are used for the preprocessing.
        - Include how to do the data augmentation with details and names.
        - Include how to extract and understand the characteristics of the data.
        - Include reasons why each step in your explanations is essential to effectively complete the plan.        
        Note that you should not perform data visualization because you cannot see it. Make sure to focus only on the data part as it is your expertise. Do not conduct or perform anything regarding modeling or training.
        After complete the explanations, explicitly specify the (expected) outcomes and results both quantitative and qualitative of your explanations."""

        messages = [
            {"role": "system", "content": agent_profile},
            {"role": "user", "content": exec_prompt},
        ]

        # 5. LLM 호출해서 결과 얻기(최대 10회 재시도)
        retry = 0
        while retry < 10:
            try:
                res = get_client(self.llm).chat.completions.create(
                    model=self.model, messages=messages, temperature=0.3
                )
                break
            except Exception as e:
                print_message("system", e)
                retry += 1
                continue
        
        # 6. 결과 저장 :: 사용량 기록 (비용 추적용)도 기록함
        # Data LLaMA summarizes the given plan for optimizing data relevant processes
        action_result = res.choices[0].message.content.strip()                              # LLM이 생성한 실제 데이터 처리 단계 설명(str)
        self.money[f"Data_Plan_Execution_{pid}"] = res.usage.to_dict(mode="json")

        print_message(self.agent_type, "I have done with my execution!", pid)
        return action_result                                                                 # LLM이 생성한 실제 데이터 처리 단계 설명((문자열, 전처리·증강·특성 추출 단계 포함))
