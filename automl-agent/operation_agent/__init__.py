import os
import shutil

from configs import AVAILABLE_LLMs
from utils import print_message, get_client
from operation_agent.execution import execute_script

import time

# test용
agent_profile = """You are the world's best MLOps engineer of an automated machine learning project (AutoML) that can implement the optimal solution for model training and saving, given any datasets and models. You have the following main responsibilities to complete.
1. Write accurate Python codes to retrieve/load the given dataset from the corresponding source.
2. Write effective Python codes to preprocess the retrieved dataset.
3. Write precise Python codes to retrieve/load the given model and optimize it with the suggested hyperparameters.
4. Write efficient Python codes to train/finetune the retrieved model.
5. Write suitable Python codes to save the trained model to the appropriate directory.
6. Run the model evaluation using the given Python functions and summarize the results for validation against the user's requirements.
"""

class OperationAgent:
    def __init__(self, user_requirements, llm, code_path, device=0):
        # setup Farm Manager
        self.agent_type = "operation"
        self.llm = llm
        self.model = AVAILABLE_LLMs[llm]["model"]
        self.experiment_logs = []
        self.user_requirements = user_requirements
        self.root_path = "agent_workspace" # + f"{code_path}"
        self.code_path = code_path
        self.device = device
        self.money = {}
        

    def self_validation(self, filename):
        """
        생성한 코드를 확인
        """
        rcode, log = execute_script(filename, device=self.device)
        return rcode, log

    def implement_solution(self, code_instructions, full_pipeline=False, code="", n_attempts=5):
        """
        실제 수행 : LLM과 실제 Python 실행(subprocess)을 연결하는 중심부
        ## full_pipeline : False :: 모델링 파이프라인 (데이터 가져오기부터 모델 저장까지)
        Operation 단계는 qwen_coder 모델 사용 후 다시 mistral로 복귀
        """
    
        print_message(
            self.agent_type,
            f"I am implementing the following instruction:\n\r{code_instructions}",
        )
        from utils.switch_model import switch_model
        print(f"[OperationAgent] 🔄 Switching model: mistral → qwen_coder")
        switch_model(self.llm)
        
        log = "Nothing. This is your first attempt."
        error_logs = []
        code = code  # if a template/skeleton code is provided
        iteration = 0
        completion = None
        action_result = ""
        rcode = -1
        
        while iteration < n_attempts:
            try:
                ##################### 1. 수행 프롬프트 : LLM에게 “이런 instruction대로 코드를 써봐” 요청
                exec_prompt = """Carefully read the following instructions to write Python code for {} task.
                {}
                
                # Previously Written Code
                ```python
                {}
                ```
                
                # Error from the Previously Written Code
                {}
                
                Note that you need to write the python code for the {}. If saving model is required, you must save the trained model to "./agent_workspace/trained_models" directory.
                Start the python code with "```python". Please ensure the completeness of the code so that it can be run without additional modifications.
                If there is any error from the previous attempt, please carefully fix it first."""
                pipeline = (
                    "entire machine learning pipeline (from data retrieval to model deployment via Gradio)"
                    if full_pipeline
                    else "modeling pipeline (from data retrieval to model saving)"
                )
                exec_prompt = exec_prompt.format(
                    self.user_requirements["problem"]["downstream_task"],
                    code_instructions,
                    code,                                   # 초기 코드 or 잘못 작성되었던 코드
                    log,
                    pipeline,
                )

                messages = [
                    {"role": "system", "content": agent_profile},
                    {"role": "user", "content": exec_prompt},
                ]
                print("재확인 ::: self.model : ",self.model)
                res = get_client(self.llm).chat.completions.create(
                    model=self.model, messages=messages, temperature=0.3
                )
                ############################ 2. LLM이 Python 코드를 생성
                raw_completion = res.choices[0].message.content.strip()
                completion = raw_completion.split("```python")[1].split("```")[0]
                self.money[f'Operation_Coding_{iteration}'] = res.usage.to_dict(mode='json')

                if not completion.strip(" \n"):
                    print("### ?????? 오류가 여기서 잡히나?")
                    continue
                
                ############################# 3. 코드 저장
                filename = f"{self.root_path}{self.code_path}_{iteration}.py"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "wt") as file:
                    file.write(completion)
                code = completion                       ## 생성된 코드 입력될텐데?
                #print(">>>>>>>>>>>>>> 확인용 :\n",code)
                
                ############################## 4. 진짜 실행 ::: self_validation -> execute_script(얘가 수행됨)
                rcode, log = self.self_validation(filename)
                if rcode == 0:
                    action_result = log
                    break
                
                else:
                    #### 실패
                    log = log
                    error_logs.append(log)
                    action_result = log
                    print_message(self.agent_type, f"I got this error (itr #{iteration}): {log}")
                    iteration += 1                    
                    # while문
            except Exception as e:
                iteration += 1
                print_message(self.agent_type, f"===== Retry: {iteration} =====")
                print_message(self.agent_type, f"Executioin error occurs: {e}")
            continue
        
        # 모델 복귀
        print(f"[OperationAgent] 🔁 Restoring model: qwen_coder → mistral")
        switch_model("prompt-llm")
        
        if not completion:
            completion = ""

        print_message(
            self.agent_type,
            f"I executed the given plan and got the follow results:\n\n{action_result}",
        )
        return {"rcode": rcode, "action_result": action_result, "code": completion, "error_logs": error_logs}
