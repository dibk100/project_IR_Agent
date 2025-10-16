import json, os, time
import pandas as pd

from multiprocessing import Pool, current_process
from configs import AVAILABLE_LLMs
from prompt_agent import PromptAgent
from data_agent import DataAgent
from model_agent import ModelAgent
from operation_agent import OperationAgent
from utils import print_message, get_client
from num2words import num2words
from agent_manager.retriever import retrieve_knowledge
from glob import glob

from experiments import FREE_PROMPTS

# 기본 버전
# agent_profile = """You are a helpful assistant."""

# 역할 명시 버전: assistant
# agent_profile = """You are a helpful assistant. You have two main responsibilities as follows.
# 1. Receive requirements and/or inquiries from users through a well-structured JSON object.
# 2. Using recent knowledge and state-of-the-art studies to devise promising high-quality plans for data scientists, machine learning research engineers, and MLOps engineers in your team to execute subsequent processes based on the user requirements you have received.
# """

#  AutoML 프로젝트의 시니어 PM 강조 버전 : 단순 도움을 넘어서 팀 내 데이터·모델·MLOps 담당자 관리
# agent_profile = """You are a senior project manager of a automated machine learning project (AutoML). You have two main responsibilities as follows.
# 1. Receive requirements and/or inquiries from users through a well-structured JSON object.
# 2. Using recent knowledge and state-of-the-art studies to devise promising high-quality plans for data scientists, machine learning research engineers, and MLOps engineers in your team to execute subsequent processes based on the user requirements you have received.
# """

# 경험 강조 버전
agent_profile = """You are an experienced senior project manager of a automated machine learning project (AutoML). You have two main responsibilities as follows.
1. Receive requirements and/or inquiries from users through a well-structured JSON object.
2. Using recent knowledge and state-of-the-art studies to devise promising high-quality plans for data scientists, machine learning research engineers, and MLOps engineers in your team to execute subsequent processes based on the user requirements you have received.
"""
# 세계 최고 PM버전
# agent_profile = """You are the world's best senior project manager of a automated machine learning project (AutoML). You have two main responsibilities as follows.
# 1. Receive requirements and/or inquiries from users through a well-structured JSON object.
# 2. Using recent knowledge and state-of-the-art studies to devise promising high-quality plans for data scientists, machine learning research engineers, and MLOps engineers in your team to execute subsequent processes based on the user requirements you have received.
# """

## json 형태로 ML 개발 계획을 작성하도록 지침
json_plan = """Each of the following plans should cover the entire process of machine learning model development when applicable based on the given requirements, i.e., from problem formulation to deployment.
Please ansewer your plans in list of the JSON object with `title` and `steps` keys."""

# _is_relevant,_is_enough()에서 system prompt
basic_profile = """You are a helpful, respectful and honest "human" assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

plan_conditions = """
- Ensure that your plan is up-to-date with current state-of-the-art knowledge.
- Ensure that your plan is based on the requirements and objectives described in the above JSON object.
- Ensure that your plan is designed for AI agents instead of human experts. These agents are capable of conducting machine learning and artificial intelligence research.
- Ensure that your plan is self-contained with sufficient instructions to be executed by the AI agents. 
- Ensure that your plan includes all the key points and instructions (from handling data to modeling) so that the AI agents can successfully implement them. Do NOT directly write the code.
- Ensure that your plan completely include the end-to-end process of machine learning or artificial intelligence model development pipeline in detail (i.e., from data retrieval to model training and evaluation) when applicable based on the given requirements."""

possible_states = {
    "INIT": "",
    "PLAN": "",
    "ACT": "",
    "PRE_EXEC": "",
    "EXEC": "",
    "POST_EXEC": "",
    "REV": "",
    "RES": "",
}

"""
에이전트/시스템 상태 관리용 딕셔너리

INIT: 초기 상태
PLAN: 계획 수립
ACT: 활동 단계
PRE_EXEC: 실행 전 준비
EXEC: 실행 중
POST_EXEC: 실행 후
REV: 검토
RES: 결과
"""

# prompt Agent객체 생성 ::: manager객체 아님!
# 사용자 자연어를 JSON으로 변환하는 역할
# ./prompt_agent >> WizardLAMP 확인하기
parser = PromptAgent()

class AgentManager:
    """
    전체 워크플로우를 총괄하는 매니저 역할
    - 사용자 요구사항을 바탕으로 계획(plan)을 관리
    - 여러 에이전트(DataAgent, ModelAgent, OperationAgent)와 연계
    - 계획 실행과 결과 수집
    - 상태 관리, 재시도, 검증 등 전체 AutoML 파이프라인 조정
    
    함수 :
    make_plans
    execute_plan --> Data,Model Agent 실행
    verify_solution
    implement_solution
    generate_reply
    _is_relevant : ML/AI 관련 내용인지 판단하는 함수(llm-based)
    _is_enough() : Auto<ML 수행에 충분한 정보를 포함하는지 판단하는 함수(llm-based)
    
    
    """
    
    def __init__(
        self,
        task,                                           # task == downstream = 'tabular_classification'
        n_plans=3,                                      # 생성할 계획
        n_candidates=3,                                 # 계획 후보
        n_revise=3,                                     # 수정 횟수
        device=0,
        interactive=False,                              # True면, 상태 전환 시 사용자 확인 허용
        llm="qwen",
        user_requirements=None,                         # user_requirements : json 형태 / 없다면 []로 반환됨
        plans=None,                                     # plans: json 형태 / 없다면 []로 반환됨 
        plan_knowledge='./example_plans/plan_knowledge.md',                                 # './example_plans/plan_knowledge.md',   
        data_path=None,                                 # 테스트할 때 작성 필요
        full_pipeline=True,
        rap=True,
        decomp=True,
        verification=True,
        result_path=None,
        instruction_path=None,                          # 코드 및 실행 지침 파일 경로(추측)
        exp_configs=None,
        uid=None,                                           # 사용자/실험 ID?
        inj=None
    ):
        # Setup Agent Manager
        self.agent_type = "manager"
        self.exp_config = exp_configs  # {"task": "", "prompt_type": "", "uid": 0}
        self.rap = rap
        self.decomp = decomp
        self.verification = verification
        self.full_pipeline = full_pipeline
        self.llm = llm
        self.model = AVAILABLE_LLMs[llm]["model"]
        self.chats = []                                         # generate_reply() :: Agent가 유지 중인 전체 대화 기록 (대화 히스토리)
        self.state = "INIT"                     # 초기 상태로 시작
        
        # Plan and Result
        if plans != None:
            f = open(plans)
            self.plans = json.load(f)
        else:
            self.plans = []
        self.action_results = []
        
        # AgentManager가 사용자와 상호작용(interactive) 모드인지 여부
        # 한 번의 입력/출력으로 끝나는 것이 아니라, 여러 턴의 대화를 강제할 수 있음 : 자동화 파이프라인 중간중간에 사용자 확인/승인을 받을 수 있는 모드
        self.interactive = interactive  # enable the manager to ask user before proceeding to the next state (force multi-turn dialogue)
        # ex : 계획 단계에서 LLM이 생성한 계획을 확인하고 사용자가 수정할 수 있음 
        # 자동 모드(interactive=False)일 경우 → 자동으로 다음 상태 진행, 사용자 개입 없음
        if user_requirements != None:
            f = open(user_requirements)
            self.user_requirements = json.load(f)
        else:
            self.user_requirements = None
        self.req_summary = ""
        self.has_valid_requirement = False
        self.n_plans = n_plans
        self.n_candidates = n_candidates
        self.n_revise = n_revise
        self.is_solution_found = False
        if plan_knowledge != None:
            with open(plan_knowledge, "r") as f:                    # plan_knowledge, "r"
                self.plan_knowledge = f.read()
        else:
            self.plan_knowledge = None
            print("############## 확인용 : plan_knowledge==None")
        self.data_path = data_path
        self.device = device
        
        # result_path 이전 결과가 있다면 로드, EXEC상태로 전환
        if result_path:
            plans = glob(result_path + "/*")
            for plan in plans:
                f = open(plan)
                self.action_results.append(json.load(f))
                self.state = "EXEC"
        if instruction_path:
            with open(instruction_path + "/code_instruction.txt", "r") as f:
                self.code_instruction = f.read()
        else:
            self.code_instruction = None
        if exp_configs:
            self.code_path = f"/{self.llm}_{exp_configs.task}_{exp_configs.prompt_type}_{exp_configs.uid}"
        else:
            self.code_path = f"/{uid}_{self.llm}_p{self.n_plans}_{'rap' if self.rap else ''}_{'decomp' if self.decomp else ''}_{'ver' if self.verification else ''}_{'full' if self.full_pipeline else ''}"
        self.n_attempts = 0
        self.task = task
        self.inj = inj
        self.timer = {}
        self.money = {}

    def make_plans(self, is_revision=False):
        """
        new plan 생성
        사용자 요구사항(self.user_requirements) + 관련 지식(self.plan_knowledge)으로 여러 개의 end-to-actionable plan 생성
        
        is_revision
        이전 계획이 실패했을 때, 실패 원인 분석 후 개선된 계획 생성
        
        instruction: Prompt Agent가 사용자 입력을 파싱하도록 안내
        """
        # planning should include action_id, completion_status, action_dependencies (with required prior action ids), and
        # instruction (i.e., prompt to tell how Prompt Agent should parse user's input prompt (e.g., what keys should be included etc.)) for the repsective agent(s) responding to the given tasks
        
        ### 계획 수정
        if is_revision:
            start_time = time.time()
            fail_prompt = "I found that all the plans you provided are failed or unsatisfied with the given requirements. Now, your task is to find the reasons 'why' and 'how' the above plans were unsatisfied by carefully comparing them with the requirements again. Please answer me your findings and insights as we will use them to create the new set of plans."
            fail_rationale = self.generate_reply(
                system_prompt=agent_profile,
                user_prompt=fail_prompt,
                return_content=True,
                caller_id='manager_fail_plan_reflection'
            )
            print_message(
                self.agent_type,
                "Sorry, I am revising the plans for you 💭.",
            )
            plan_prompt = f"""Now, you will be asked to revise and rethink {num2words(self.n_plans)} different end-to-end actionable plans according to the user's requirements described in the JSON object below.
            
            ```json
            {self.user_requirements}
            ```
            
            Please use to the following findings and insights summarized from the previously failed plans. Try as much as you can to avoid the same failure again.
            {fail_rationale}
            
            Finally, when devising a plan, follow these instructions and do not forget them:
            {plan_conditions}
            """
            self.plans = []
            self.timer[f'fail_plan_reflection_{self.n_attempts}'] = time.time() - start_time
        else:
            start_time = time.time()
            # retrieve relevant knowledge/expereince (from internal and external sources) for effective planning
            
            ## 수정 False,
            ### 새 계획 생성
            if self.plan_knowledge == None and self.rap and self.inj in [None, 'pre']:      # 아직 계획을 위한 참조 지식이 없는 경우
                self.plan_knowledge = retrieve_knowledge(self.user_requirements, self.req_summary, llm=self.llm, inj=self.inj)
            else:
                # 추가 후처리(post_noise) :: retrieve_knowledge 함수 ver2 사용함
                # self.plan_knowledge, self.post_noise = retrieve_knowledge(self.user_requirements, self.req_summary, llm=self.llm, inj=self.inj)
                #self.plan_knowledge = f""""{self.plan_knowledge}\r\nHere is a list of knowledge written by an AI agent for a relevant task:\r\n{self.post_noise}"""
                pass
            print_message(
                self.agent_type,
                f"Now, I am making a set of plans for you based on your requirements and the following knowledge 💭.\n{self.plan_knowledge}",
            )
            self.timer['retrieve_knowledge'] = time.time() - start_time
            
            # Independent Planning (i.e., The agent does not know how it previously made the plans. Pros: Significantly less contexnt length consumption --> have room for knowledge sources, Cons: Diversity is not guaranteed.)
            plan_prompt = f"""Now, I want you to devise an end-to-end actionable plan according to the user's requirements described in the following JSON object.
            
            ```json
            {self.user_requirements}
            ```
            
            Here is a list of past experience cases and knowledge written by an human expert for a relevant task: 
            {self.plan_knowledge}

            When devising a plan, follow these instructions and do not forget them:
            {plan_conditions}
            """

        ### LLM호출 및 계획 생성
        start_time = time.time()
        for i in range(1, self.n_plans + 1):
            print(f"########################## make_plans에서 실행되고 있음. : {i}번째 LLM \n")
            messages = [
                {"role": "system", "content": agent_profile},
                {"role": "user", "content": plan_prompt},
            ]
            while True:
                try:
                    response = get_client(self.llm).chat.completions.create(
                        model=self.model, messages=messages, temperature=0.7
                    )
                    break
                except Exception as e:
                    print_message("system", e)
                    continue
            plan = response.choices[0].message.content.strip()
            self.plans.append(plan)
            self.money[f'manager_plan_{i}'] = response.usage.to_dict(mode='json')
            print(f"##########################: {i}번째 LLM 계획 : \n{plan}\n###################################### {i}번째 LLM 계획 END plan part \n")
        self.timer['planning'] = time.time() - start_time

    def execute_plan(self, plan):
        """
        단일 계획을 실제로 수행하는 역할(################# 3. ACT 단계)
        
        1. Data Agent
        2. Model Agent
        
        return ::
        Dict(결과 str)
        """
        
        print(f"################# 3. ACT 단계(확인용) execute_plan 함수 : \n")
        ############### 여기 수정 고민중
        # plan = FREE_PROMPTS["tabular"]["tabular_classification"][0]

        # langauge (text) based execution
        pid = current_process()._identity[0]  # for checking the current plan : 현재 실행 중인 프로세스 ID 가져옴(각 계획 실행 결과를 구분용)

        ######### 1. Data Agent 실행 #########################
        start_time = time.time()
        # Data Agent generates the results after execute the given plan
        data_llama = DataAgent(
            user_requirements=self.user_requirements,
            llm=self.llm,
            rap=self.rap,
            decomp=self.decomp,
        )
        ############## 실험을 위한
        add_prompt = "\n\n **Important Instruction: All data loading must strictly use the absolute path specified. Do not attempt relative paths or alternative locations.**\n**Be careful not to import from incorrect modules. Do not import from non-existent paths.**\n\n"
        
        data_result = data_llama.execute_plan(plan, self.data_path, pid)      
        data_result +=add_prompt

        print(f"################# 3. ACT 단계(Data Agent모델 수행) execute_plan - 결과 : \n{data_result}\n################# Data Agent모델 data_result END ########\n")  
        # # data_result(str) :
        # # LLM이 생성한 실제 데이터 처리 단계 설명((문자열, 전처리·증강·특성 추출 단계 포함))
        
        self.timer[f'data_execution_{pid}'] = time.time() - start_time
        self.money['Data'] = data_llama.money

        ######### 2. Model Agent 실행 #########################
        # Model Agent summarizes the given plan for optimizing data relevant processes : 모델 에이전트는 주어진 계획을 요약하여 데이터 관련 프로세스를 최적화
        # Model Agent generates the results after execute the given plan : 모델 에이전트는 주어진 계획을 실행한 후 결과를 생성
        start_time = time.time()
        model_llama = ModelAgent(
            user_requirements=self.user_requirements,
            llm=self.llm,
            rap=self.rap,
            decomp=self.decomp,
        )
        model_result = model_llama.execute_plan(
            k=self.n_candidates, project_plan=plan, data_result=data_result, pid=pid            # data_result :: Data Agent가 출력한 결과(str, 인사이트)기반 데이터 기반 모델링 진행
        )
        print(f"################# 3. ACT 단계(Model Agent모델 수행) execute_plan 함수 : {model_result}\n")  
        ## model_result 결과 타입 확인하기
        self.timer[f'model_execution_{pid}'] = time.time() - start_time
        self.money['Model'] = model_llama.money
        
        return {"data": data_result, "model": model_result}

    def verify_solution(self, solution):
        """
        
        return ::
        (str) pass or fail
        """
        
        
        pid = current_process()._identity[0]  # for checking the current plan
        
        start_time = time.time()
        
        is_pass = False

        # pre-execution verification
        # LLM에게 **솔루션이 사용자 요구사항을 만족하는지** 판단하도록 요청  
        verification_prompt = """Given the proposed solution and user's requirements, please carefully check and verify whether the proposed solution 'pass' or 'fail' the user's requirements.
        
        **Proposed Solution and Its Implementation**
        Data Manipulation and Analysis: {}
        Modeling and Optimization: {}
        
        **User Requirements**
        ```json
        {}
        ```
                
        Answer only 'Pass' or 'Fail'
        """ 

        prompt = verification_prompt.format(
            solution["data"], solution["model"], self.user_requirements
        )
        messages = [
            {"role": "system", "content": basic_profile},
            {"role": "user", "content": prompt},
        ]

        
        # res : 결과 메세지는 prompt에 의해 pass or fail
        while True:
            try:
                res = get_client(self.llm).chat.completions.create(
                    model=self.model, messages=messages, temperature=0
                )
                break
            except Exception as e:
                print_message("system # Error Function :: Agent Manager verify_solution #", e)
                continue
        ans = res.choices[0].message.content.strip()                                       
        is_pass = "pass" in ans.lower()
        self.money['manager_execution_verification'] = res.usage.to_dict(mode='json')
        
        self.timer[f'execution_verification_{pid}'] = time.time() - start_time

        return is_pass

    def implement_solution(self, selected_solution):
        """
        ############ 실제로 python 코드를 수행하는 구간 :: ops_llama.implement_solution의 결과값 ops_result
        Manager Agent가 선정된 계획/솔루션(selected_solution)을 가져오고, 코드 템플릿도 가져옴. OperationAgent에게 실제 실행을 맡기는 구조
        selected_solution는 generate_replay로 생성됨
        
        prompt_pool/{self.task}.py : “OperationAgent가 실제로 코드를 삽입하고 실행할 틀(frame)”
        code_path : 생성된 코드가 저장될 경로
        code_instructions == selected_solution = Model/Data Agent가 만든 최종 솔루션의 지시문 (pseudo code나 step-by-step)
        
        """
        with open(f"prompt_pool/{self.task}.py") as file:
            template_code = file.read()        
        # code-based execution
        ops_llama = OperationAgent(
            user_requirements=self.user_requirements,
            llm=self.llm,
            code_path=self.code_path,
            device=self.device,
        )
        ops_result = ops_llama.implement_solution(
            code_instructions=selected_solution, 
            full_pipeline=self.full_pipeline, 
            code=template_code
        )
        self.money['Operation'] = ops_llama.money
        return ops_result

    def generate_reply(
        self,
        user_prompt,
        system_prompt=basic_profile,
        return_content=False,                       # True면 return 값이 llm순수 텍스트, False면 응답구조(메타데이터)
        system_use=False,
        caller_id=None
    ):
        """
        AgentManager, DataAgent, ModelAgent 등 모든 에이전트들이 공통적으로 LLM에게 “질문 → 답변”을 요청할 때 사용하는 핵심 유틸 함수
        LLM에게 질문하고, 답변을 받아, 대화 히스토리를 업데이트하는 역할
        
        주어진 프롬프트(user_prompt) + 시스템 프롬프트(system_prompt) -> LLM -> 응답(response)
        
        self.chats :  현재 Agent가 유지 중인 전체 대화 기록 (대화 히스토리) :: 리스트로 저장됨
        
        return ::
        reply(str)   LLM이 생성한 응답구조(메타데이터)
        
        
        log 
        ### 과거 대화 기록을 LLM input에 복원하는 것
        # n_calls max_lenght == self.chats의 수
        
        """
        n_calls = 0
        self.chats.append({"role": "user", "content": user_prompt})
        messages = [{"role": "system", "content": system_prompt}]
        
        test_trriger = False
        for msg in self.chats:
            
            if msg["role"] in ["function", "tool"]:                 
                print(f"@@@@@@@@@@@@@@@ 챗 확인하기 :\n {msg}\n@@@@@@@@@@@@@@@@@@@@\n")
                test_trriger = True
                n_calls = n_calls + 1
            if n_calls > 0:
                messages.append(msg)
            else:
                messages.append({"role": msg["role"], "content": msg["content"]})   
        
        if test_trriger :
            raise SystemExit("⛔ generate_reply 중단: LLM 호출 직전에서 종료됨")
        ## LLM 호출 
        retry = 0
        response = None
        while retry < 5:
            try:
                response = get_client(self.llm).chat.completions.create(
                    model=self.model, messages=messages, temperature=0.3
                )
                break
            except Exception as e:
                print_message("system :: Error Type :: Manager - generate_reply", e)
                retry += 1
                continue
        
        if response:
            reply = response.choices[0].message.content.strip() if return_content else response
        else:
            reply = ''
            print("generate_reply에서 5회 이상 시도 - 응답 없음(빈 값)")
        # add a new response message
        ### 대화 기록 갱신
        # system_use=False일 때만 대화 기록에 추가함
        # "내부적인 LLM 호출" (ex. 데이터 검증용 등)은 기록하지 않음
        if not system_use and response:
            self.chats.append(dict(response.choices[0].message))
        
        ## 비용 기록
        if caller_id and response:
            self.money[caller_id] = response.usage.to_dict(mode='json')
        return reply

    def _is_relevant(self, msg):
        """
        msg (문자열)이 "머신러닝(Machine Learning)" 또는 "인공지능(AI)" 과 관련된 내용인지 LLM에게 물어보고 판단하는 함수
        """
        init_prompt = f"""Is the following statement relevant to machine learning or artificial intelligence?
        
        `{msg}`
        
        Answer only 'Yes' or 'No'
        """
        messages = [
            {"role": "system", "content": basic_profile},
            {"role": "user", "content": init_prompt},
        ]
        retry = 0
        while retry < 5:
            try:
                response = get_client(self.llm).chat.completions.create(
                    model=self.model, messages=messages, temperature=0
                )
                break
            except Exception as e:
                print_message("system", e)
                retry += 1
                continue
        return "yes" in response.choices[0].message.content.strip().lower()

    def _is_enough(self, msg):
        """
        _is_relevant()보다 한 단계 “더 깊은 필터링 단계”에 해당 : "AutoML 파이프라인을 실제로 실행할 만큼 정보가 충분한가?"를 판단하는 역할
        
        """
        
        
        init_prompt = f"""
        Given the following JSON object representing the user's requirement for a potential ML or AI project, please tell me whether we have essential information (e.g., problem and dataset) to be used for a AutoML project?
        Please note that our users are not AI experts, you must focus only on the essential requirements, e.g., problem and brief dataset descriptions.
        You do not need to check every details of the requirements. You must also answer 'yes' even though it lacks detailed and specific information.
        
        ```json
        {msg}
        ```
        
        Please answer with this format: `a 'yes' or 'no' answer; your reasons for the answer` by using ';' to separate between the answer and its reasons.
        If the answer is 'no', you must tell me the alternative solutions or examples for completing such missing information.
        """
        messages = [
            {"role": "system", "content": basic_profile},
            {"role": "user", "content": init_prompt},
        ]
        while True:
            try:
                response = get_client(self.llm).chat.completions.create(
                    model=self.model, messages=messages, temperature=0
                )
                break
            except Exception as e:
                print_message("system", e)
                continue
        ans, reason = response.choices[0].message.content.strip().split(";")
        self.money['manager_request_verification'] = response.usage.to_dict(mode='json')
        
        if "yes" in ans.strip().lower():
            return True, reason.strip()
        else:
            return False, reason.strip()

    def _on_stop(self, msg):
        """
        대화 종료 트리거
        사용자가 입력한 msg가 **대화 종료를 의도한 명령**인지 판별하는 역할
        """
        
        return msg.lower() in [
            "stop",
            "close",
            "exit",
            "terminate",
            "end",
            "done",
            "finish",
            "complete",
            "bye",
            "goodbye",
        ]
### Main Loop 함수
    def initiate_chat(self, prompt, plan_path=None, instruction_path=None):
        """
        [ 사용자의 요청 → 분석 → 계획 수립 → 실행 → 검증 → 수정 → 완료 ] 전 단계를 관리함.
        self.state 값에 따라 동작이 바뀌며 단계를 실행함 : possible_states 참고
        
        INPUT parameters
        - prompt: 사용자가 처음 입력한 요청 (ex: “MNIST 분류 모델 만들어줘”)
        - plan_path: 계획 결과를 저장할 경로 (옵션)
        - instruction_path: 코드 생성 지침을 저장할 경로 (옵션)
        
        Q. pool
        Q. input() 는 어디서 나온거지?
        
        """
        
        last_msg = prompt
        pool = Pool(self.n_plans)           # multiprocessing 파일에 정의되는 거 같은데 못찾음
        
        start_time = time.time() 
        init_time = time.time() # init time
        
        # last_msg, self.state로 루프 멈춤
        while not self._on_stop(last_msg) and self.state != "END":
            
            # 여기가 루프 도는 트리거 위치(state 상태, last_msg)
            # reply process: current state + current state description + response : 응답 처리 과정: 현재 상태 + 현재 상태 설명 + 응답
            if last_msg == "":
                sys_query = "Please give feedback or answer to proceed. You may type 'exit' to end the session."
                last_msg = input(sys_query)
                if last_msg == "" or self._on_stop(last_msg):
                    continue
                else:
                    prompt = last_msg

            ########### 1. INIT 단계 ###########
            if self.state == "INIT":
                print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 1. INIT 단계(확인용) 시작 위치\n")
                # display user's input prompt
                self.chats.append({"role": "user", "content": prompt})
                print_message("user", prompt)
                
                """
                1. _is_relevant : 요청이 AI/ML 관련인지 분류
                1-1. “Yes” → 분석 계속
                1-2. “No” → 그냥 대화(generate_reply)로 응답 후 종료
                """
                if self._is_relevant(prompt) or self.verification == False:
                    ############# Request Vericication : 검증 단계(_is_relevant,_is_enough)
                    """
                    1-1-1. 요청을 JSON으로 파싱
                    1-1-2. _is_enough() : 요청이 충분히 구체적인지 검사
                    1-1-3. request_summary 만드는 단계
                    state 바꾸고 루프 도는 위치로.
                    """
                    # 1-1-1. 요청을 JSON으로 파싱
                    if self.user_requirements == None:
                        ######### parser.parse는 ./prompt_agent의 WizardLAMP에 의해?
                        self.user_requirements = parser.parse(prompt, return_json=True) # or parser.parse_openai(prompt, return_json=True)
                        # check user's requirement quality (JSON schema validation)
                        self.timer['prompt_parsing'] = time.time() - start_time 
                        ######################## 확인용
                        print(f"################# 1. INIT 단계(확인용) user_requirements : {self.user_requirements}\n")
                
                        start_time = time.time()
                        ########### 1-1-2. _is_enough() : 요청이 충분히 구체적인지 검사       
                        is_enough, reasons = self._is_enough(self.user_requirements)
                        self.timer['request_verification'] = time.time() - start_time
                    else:
                        # user_requirements에 값을 넣으면 is_enough는 그냥 pass
                        is_enough = True
                        
                    if 'confidence' in self.user_requirements.keys():
                        del self.user_requirements["confidence"]
                    
                    ########### 1-1-3. LLM이 JSON 내용을 한 문단으로 요약하는 단계(PLAN 단계 바로 전) :: request_summary
                    #######      user_requirements가 이미 있음.
                    if is_enough or self.verification == False:
                        start_time = time.time()
                        messages = [
                            {"role": "system", "content": agent_profile},
                            {
                                "role": "user",
                                "content": f"Please briefly summarize the user's request represented in the following JSON object into a single paragraph based on how you understand it.\n\r{self.user_requirements}",
                            },
                        ]
                        retry = 0
                        while retry < 5:
                            try:
                                res = get_client(self.llm).chat.completions.create(
                                    model=self.model, messages=messages, temperature=0.3
                                )
                                break
                            except Exception as e:
                                print_message("system", e)
                                retry += 1
                                continue
                        self.req_summary = res.choices[0].message.content.strip()
                        self.timer['request_summary'] = time.time() - start_time
                        self.money['manager_request_summary'] = res.usage.to_dict(mode='json')

                        ########################### 확인용
                        print(f"################# 1. INIT 단계(확인용) request_summary : {self.req_summary}\n")
                        print_message(
                            "prompt",
                            f"""I understand your request as follows.\n\r{self.req_summary}""",
                        )
                        self.chats.append(
                            {
                                "role": "assistant",
                                "content": f"""I understand your request as follows.\n\r{self.req_summary}""",
                            }
                        )
                        if self.interactive:
                            ans = input(
                                "Please check whether I understand your requirements correctly? (Yes/No)"
                            )
                            if ans.lower() in ["yes", "correct", "right", "sure"]:
                                self.state = "PLAN"
                        else:
                            self.state = "PLAN"
                            # key_point : while문으로 돌아감(다음 단계 PLAN로)
                            
                    else:
                        print_message(
                            self.agent_type,
                            f"""Based on the analysis result from 🦙 Prompt Agent, it seems that, for the following reasons, we cannot process your request or solve your task.\n{reasons}""",
                        )
                        last_msg = "" if self.interactive else "stop"
                else:
                    # chit-chat case : 일반 대화로 전환 후 스톱
                    res = self.generate_reply(user_prompt=prompt, return_content=True, caller_id='manager_chitchat')
                    print_message(self.agent_type, res)
                    last_msg = "" if self.interactive else "stop"
                    
            ########### 2. PLAN 단계 ###########
            elif self.state == "PLAN":
                """
                make_plans() : 여러 계획(self.n_plans)을 생성    - ex. XGBoost 기반 파이프라인, CNN기반 딥러닝 파이프라인
                * interactive 모드일 경우, 생성된 plan들을 user의 승인이 필요
                """
                print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 2. PLAN 단계(확인용) 시작 위치\n")
                start_time = time.time()
                # Planning Stage
                self.make_plans()
                self.timer['planning_total'] = time.time() - start_time

                display_plans = "I have the following plan(s) for your task 📜!\n\n"
                for plan in self.plans:
                    display_plans = display_plans + plan + "\n\n"

                # display all plans to the users
                print_message(self.agent_type, display_plans)
                if self.interactive:
                    # ask user before executing the plans
                    ans = input(
                        "Do you want me to proceed with the above plan(s)? (Yes/No)"
                    )
                    if ans.lower() in ["yes", "correct", "right", "sure"]:
                        self.state = "ACT"
                else:
                    self.state = "ACT"
                    
            ########### 3. ACT 단계 ########### 계획 실행 단계(계획구현하는거지 실제 실행은 아님)
            elif self.state == "ACT":
                # Action (executing the plans) Stage
                print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 3. ACT 단계(확인용) 시작 위치\n")
                print_message(
                    self.agent_type,
                    "With the above plan(s), our 🦙 Data Agent and 🦙 Model Agent are going to find the best solution for you!",
                )
                start_time = time.time()
                
                """
                각 plan은 내부적으로 Data Agent, Model Agent 호출.
                # Data Agent : 데이터 준비/ 전처리 파이프라인 생성
                # Model Agent : 델 탐색 / 학습 계획
                self.action_results에 저장됨
                """

                # Parallelization
                with Pool(self.n_plans) as pool:
                    # print("##################### test print 확인용 ############# : ",self.plans) :: 여기서 오류
                    self.action_results = pool.map(self.execute_plan, self.plans)
                self.timer['plan_execution_total'] = time.time() - start_time
                
                self.state = "PRE_EXEC"

            ############ 4. PRE_EXEC 단계 ########### 실행 결과 검증 단계
            elif self.state == "PRE_EXEC":
                # Pre-(Code)Execution Verification stage
                """
                각 action_result(self.action_results)를 verify_solution()으로 검증.
                통과한 plan만 pass=True 설정.
                
                5.EXEC 단계 or 7.REV 단계 결정
                """
                print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 4. PRE_EXEC 단계(확인용) 시작 위치\n")
                if self.verification:
                    print_message(
                        self.agent_type,
                        "I am now verifying the solutions found by our Agent team 🦙.",
                    )
                    
                    start_time = time.time()
                    # Parallelization
                    with Pool(self.n_plans) as pool:
                        verification_result = pool.map(self.verify_solution, self.action_results)
                    self.timer['execution_verification_total'] = time.time() - start_time
                    
                    # 하나라도 True인 계획이 있다면 ## 5. EXEC 단계로 넘어감
                    for i, result in enumerate(verification_result):
                        self.action_results[i]["pass"] = result
                        if result:
                            self.is_solution_found = True

                    if self.is_solution_found:
                        ###################### 5. EXEC 단계
                        result_text = f"""Thanks to all the hard-working 🦙 Agents 🦙, we have found \033[4m{num2words(sum([result['pass'] for result in self.action_results]))}\033[0m suitable solution(s) for you 🥳.\nThen, let our Operation Agent 🦙 implement and evaluate these solutions 👨🏻‍💻!"""
                        self.state = "EXEC"
                    else:
                        ###################### 7. REV 단계
                        result_text = f"""Despite all the hard work by 🦙 Agents 🦙, we have not found a suitable solution that matches your requirements yet 😭."""
                        self.state = "REV"
                else:
                    result_text = f"""Thanks to all the hard-working 🦙 Agents 🦙. Then, let our Operation Agent 🦙 implement and evaluate these solutions 👨🏻‍💻!"""
                    for action in self.action_results:
                        action["pass"] = True
                    self.state = "EXEC"

                if plan_path:
                    for i, action in enumerate(self.action_results):
                        if action["pass"]:
                            # save pass plan
                            filename = f"{plan_path}/plan_{i}.json"
                            os.makedirs(os.path.dirname(filename), exist_ok=True)
                            with open(filename, "w") as f:
                                json.dump(action, f)
                                print_message(
                                    self.agent_type,
                                    f"Saved a pass plan: {plan_path}/plan_{i}.json",
                                )
                print_message(self.agent_type, result_text)

            ############ 5. EXEC 단계 ########### Operation Agent에게 실제로 코드 수행
            elif self.state == "EXEC":
                # Code Execution stage
                """
                1. summary_prompt : 여러 개획(plan) 중 통과(Ture)된 것만 모아서 요약
                2. generate_reply : Operation Agent에게 지침 주는 프롬프트 문장 생성(직접 코드 작성 금지!) -> code_instruction 생성됨(코드 생성 지침문).
                3. implementation_result : implement_solution함수 통해서 Operation Agent가 코드 작성(여기서 skeleton code활용해서 틀 채움)
                """
                print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 5. EXEC 단계(확인용) 시작 위치\n")
                if not self.code_instruction:
                    start_time = time.time()
                    
                    data_plan_for_execution = ""
                    model_plan_for_execution = ""
                    for action in self.action_results:
                        if action["pass"]:
                            data_plan_for_execution = (
                                data_plan_for_execution + action["data"] + "\n"
                            )
                            model_plan_for_execution = (
                                model_plan_for_execution + action["model"] + "\n"
                            )

                    # Summarize the passed plan for operation llama to write and execute the code
                    upload_path = (
                        f"This is the retrievable data path: {self.data_path}."
                        if self.data_path
                        else ""
                    )
                    summary_prompt = f"""As the project manager, please carefully read and understand the following instructions suggested by data scientists and machine learning engineers. Then, select the best solution for the given user's requirements.
                    
                    - Instructions from Data Scientists
                    {data_plan_for_execution}
                    If there is no predefined data split or the data scientists suggest the data split other than train 70%, validation 20%, and test 10%, please use 70%, 20%, and 10% instead for consistency across different tasks. {upload_path}
                    You should exclude every suggestion related to data visualization as you will be unable to see it.
                    - Instructions from Machine Learning Engineers
                    {model_plan_for_execution}                    
                    - User's Requirements
                    {self.req_summary}
                    
                    Note that you must select only ONE promising solution (i.e., one data processing pipeline and one model from the top-{num2words(self.n_candidates)} models) based on the above suggestions.
                    After choosing the best solution, give detailed instructions and guidelines for MLOps engineers who will write the code based on your instructions. Do not write the code by yourself. Since PyTorch is preferred for implementing deep learning and neural networks models, please guide the MLOPs engineers accordingly.
                    Make sure your instructions are sufficient with all essential information (e.g., complete path for dataset source and model location) for any MLOps or ML engineers to enable them to write the codes using existing libraries and frameworks correctly."""
                    self.code_instruction = self.generate_reply(
                        system_prompt=agent_profile,
                        user_prompt=summary_prompt,
                        return_content=True,
                        system_use=True,
                        caller_id='manager_code_instruction'
                    )
                    self.timer['code_instruction'] = time.time() - start_time
                    
                    if instruction_path:
                        with open(f"{instruction_path}/code_instruction.txt", "w") as f:
                            f.write(self.code_instruction)

                start_time = time.time()
                ### 위에서 code_instruction(코드 생성 지침문, 프롬프트)를 활용해서 실제 수행(코드 생성)
                ### 실제 코드가 작성 및 생성됨 : implementation_result
                self.implementation_result = self.implement_solution(self.code_instruction)
                print_message('system', f'{self.code_path}, <<< END CODING, TIME USED: {time.time() - init_time} SECS >>>')
                self.timer['implementation'] = time.time() - start_time
                
                self.n_attempts += 1
                self.state = "POST_EXEC"

            ############ 6. POST_EXEC 단계 ########### 마지막
            elif self.state == "POST_EXEC":                
                # Post-(Code)Execution Verification stage
                print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 6. POST_EXEC 단계(확인용) 시작 위치\n")
                if self.implementation_result["rcode"] == 0:
                    start_time = time.time()
                    verification_prompt = f"""As the project manager, please carefully verify whether the given Python code and results satisfy the user's requirements.
                    
                    - Python Code
                    ```python
                    {self.implementation_result['code']}
                    ```
                    
                    - Code Execution Result
                    {self.implementation_result['action_result']}
                    
                    - User's Requirements
                    {self.user_requirements}
                    
                    Answer only 'Pass' or 'Fail'"""
                    messages = [
                        {"role": "system", "content": agent_profile},
                        {"role": "user", "content": verification_prompt},
                    ]
                    while True:
                        try:
                            res = get_client(self.llm).chat.completions.create(
                                model=self.model, messages=messages, temperature=0
                            )
                            break
                        except Exception as e:
                            print_message("system", e)
                            continue
                    ans = res.choices[0].message.content.strip()
                    is_pass = "pass" in ans.lower()
                    if is_pass:
                        self.state = "END"
                        self.solution = self.implementation_result["code"]
                        print_message(
                            self.agent_type,
                            f"We have successfully built your pipeline as follows!\n{self.solution}",
                        )
                    else:
                        self.state = "REV"
                    self.timer['implementation_verification'] = time.time() - start_time
                    self.money['manager_implementation_verification'] = res.usage.to_dict(mode='json')
                else:
                    if self.n_revise >= 0:
                        start_time = time.time()
                        print_message(
                            self.agent_type,
                            f"It seems that the previous attempt (# {self.n_attempts}) has failed. I am start revising it for you!",
                        )
                        # Summarize the passed plan for operation llama to write and execute the code
                        upload_path = (
                            f"This is the retrievable data path: {self.data_path}."
                            if self.data_path
                            else ""
                        )
                        summary_prompt = f"""As the project manager, you have provided an instruction that was not good enough for the MLOps engineer to write a correct code for the user's requirements.
                        Please carefully check your previous instruction, the written Python, the execution results, and the user's requirements.
                        
                        - Your Previous Instruction
                        {self.code_instruction}
                        
                        - Python Code
                        ```python
                        {self.implementation_result['code']}
                        ```
                        
                        - Code Execution Result (Error)
                        {self.implementation_result['error_logs']}
                        
                        - User's Requirements
                        {self.user_requirements}
                        
                        {upload_path}
                        After you figure out the causes, give detailed instructions and guidelines for MLOps engineers who will write the code based on your instructions. Do not write the code by yourself.
                        Make sure your instructions are sufficient with all essential information (e.g., complete path for dataset source and model location) for any MLOps or ML engineers to enable them to write the codes using existing libraries and frameworks correctly."""
                        messages = [
                            {"role": "system", "content": agent_profile},
                            {"role": "user", "content": summary_prompt},
                        ]
                        self.code_instruction = self.generate_reply(
                            system_prompt=agent_profile,
                            user_prompt=summary_prompt,
                            return_content=True,
                            system_use=True,
                            caller_id='manager_code_revision'
                        )
                        self.state = "EXEC"
                        self.n_revise = self.n_revise - 1
                        self.timer['code_revision'] = time.time() - start_time
                    else:
                        print_message(
                            self.agent_type,
                            "Sorry, even after a round of revision, we could not find a suitable solution for your problem 🙏🏻.",
                        )
                        break

            elif self.state == "REV":
                # Plan Revision stage
                print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 7. REV 단계(확인용) 시작 위치\n")
                if self.n_revise > 0:
                    start_time = time.time()
                    self.make_plans(is_revision=True)
                    self.n_revise = self.n_revise - 1
                    print_message(
                        "system", f"Remaining revision: {self.n_revise} round."
                    )
                    self.timer[f'plan_revision_{self.n_attempts}'] = time.time() - start_time
                else:
                    print_message(self.agent_type, "Sorry, even after a round of revision, we could not find a suitable solution for your problem 🙏🏻.",)
                    break

            elif self.state == "END":
                break
