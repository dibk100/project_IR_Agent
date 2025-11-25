import os
import re
import subprocess
import sys
import importlib.util
import selectors

"""
subprocess.Popen()을 사용해서 python -u {script_path} 명령을 진짜 로컬에서 실행함.
"""
def execute_script(script_name, work_dir = ".", device="0"):    
    if not os.path.exists(os.path.join(work_dir, script_name)):
        raise Exception(f"The file {script_name} does not exist.")
    try:
        script_path = script_name
        device = device        
        cmd = f"CUDA_VISIBLE_DEVICES={device} python -u {script_path}"
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, cwd=work_dir)

        stdout_lines = []
        stderr_lines = []

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        selector.register(process.stderr, selectors.EVENT_READ)

        while process.poll() is None and selector.get_map():
            events = selector.select(timeout=1)

            for key, _ in events:
                line = key.fileobj.readline()
                if key.fileobj == process.stdout:
                    # print("STDOUT:", line, end =" ")
                    stdout_lines.append(line)
                else:
                    # print("STDERR:", line, end =" ")
                    stderr_lines.append(line)

        for line in process.stdout:
            line = line
            # print("STDOUT:", line, end =" ")
            stdout_lines.append(line)
        for line in process.stderr:
            line = line
            # print("STDERR:", line, end =" ")
            stderr_lines.append(line)

        return_code = process.returncode

        if return_code != 0:
            observation = "".join(stderr_lines)
        else:
            observation = "".join(stdout_lines)
        if observation == "" and return_code == 0:
            # printed to stderr only
            observation = "".join(stderr_lines)
        return return_code, "The script has been executed. Here is the output:\n" + observation
    
    except Exception as e:
        print("++++", "Wrong!")
        # raise Exception(f"Something went wrong in executing {script_name}: {e}. Please check if it is ready to be executed.")
        return -1, f"Something went wrong in executing {script_name}: {e}. Please check if it is ready to be executed."

PACKAGE_MAP = {
    "sklearn": "scikit-learn",
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "gradio": "gradio",
    "pandas": "pandas",
    "numpy": "numpy",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "optuna": "optuna"
}

def is_std_lib(pkg_name: str) -> bool:
    if pkg_name in sys.builtin_module_names:
        return True
    try:
        spec = importlib.util.find_spec(pkg_name)
        if spec is None:
            return False
        # origin이 None이거나 Python 기본 경로에 있으면 표준 라이브러리
        return spec.origin is None or "site-packages" not in spec.origin
    except ModuleNotFoundError:
        return False

def execute_script(script_name, work_dir=".", device="0"):
    """실제 로컬에서 스크립트를 실행하고 stdout/stderr 실시간 출력"""
    script_path = os.path.join(work_dir, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"{script_path} does not exist")

    cmd = f"CUDA_VISIBLE_DEVICES={device} python -u {script_name}"
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, shell=True, cwd=work_dir)

    stdout_lines, stderr_lines = [], []

    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    selector.register(process.stderr, selectors.EVENT_READ)

    while process.poll() is None and selector.get_map():
        for key, _ in selector.select(timeout=1):
            line = key.fileobj.readline()
            if not line:
                continue
            if key.fileobj == process.stdout:
                stdout_lines.append(line)
                print(line, end='')
            else:
                stderr_lines.append(line)
                print(line, end='')

    # 남은 stdout/stderr 처리
    for line in process.stdout:
        stdout_lines.append(line)
        print(line, end='')
    for line in process.stderr:
        stderr_lines.append(line)
        print(line, end='')

    return_code = process.returncode
    observation = "".join(stdout_lines if return_code == 0 else stderr_lines)
    if observation == "" and return_code == 0:
        observation = "".join(stderr_lines)

    return return_code, "The script has been executed. Here is the output:\n" + observation

def auto_install_packages_docker(script_path, work_dir=".", device="0", image="automl-runtime:latest"):
    """Docker 환경에서 필요한 패키지 자동 설치 후 스크립트 실행"""
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"{script_path} does not exist")

    with open(script_path, "r") as f:
        code = f.read()

    # import 구문 추출
    imports = re.findall(r'^\s*import (\S+)|^\s*from (\S+) import', code, re.MULTILINE)
    packages = {pkg for grp in imports for pkg in grp if pkg}

    # 표준 라이브러리 제외
    packages = {pkg for pkg in packages if not is_std_lib(pkg)}

    for pkg in packages:
        base_pkg = pkg.split('.')[0]
        install_pkg = PACKAGE_MAP.get(base_pkg, base_pkg)
        print(f"[AutoInstaller] Installing {install_pkg} in Docker container ...")

        # 설치 전 확인
        check_cmd = [
            "docker", "run", "--rm", "--gpus", f"device={device}",
            "-v", f"{os.path.abspath(work_dir)}:/workspace", image,
            "python", "-c", f"import {base_pkg}"
        ]
        try:
            subprocess.run(check_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[AutoInstaller] {install_pkg} already installed, skipping.")
            continue
        except subprocess.CalledProcessError:
            pass

        # 설치
        install_cmd = [
            "docker", "run", "--rm", "--gpus", f"device={device}",
            "-v", f"{os.path.abspath(work_dir)}:/workspace", image,
            "pip", "install", "--no-cache-dir", install_pkg
        ]
        try:
            subprocess.run(install_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[AutoInstaller] ⚠️ Failed to install {install_pkg}: {e}")

    # 스크립트 실행
    run_cmd = [
        "docker", "run", "--rm", "--gpus", f"device={device}",
        "-v", f"{os.path.abspath(work_dir)}:/workspace", image,
        "python", f"/workspace/{os.path.basename(script_path)}"
    ]
    with subprocess.Popen(run_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            print(line, end='')
        proc.wait()
        
def docker_execute_script(script_name, work_dir=".", device="0", image="automl-runtime:latest"):
    """Docker에서 스크립트를 실행하고 stdout/stderr 실시간 출력"""
    script_path = os.path.join(work_dir, script_name)
    if not os.path.exists(script_path):
        return -1, f"The file {script_path} does not exist."

    cmd = [
        "docker", "run", "--rm", "--gpus", f"device={device}",
        "-v", f"{os.path.abspath(work_dir)}:/workspace", image,
        "python", "-u", f"/workspace/{script_name}"
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    stdout_lines, stderr_lines = [], []
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    selector.register(process.stderr, selectors.EVENT_READ)

    while process.poll() is None and selector.get_map():
        for key, _ in selector.select(timeout=1):
            line = key.fileobj.readline()
            if not line:
                continue
            if key.fileobj == process.stdout:
                stdout_lines.append(line)
                print(line, end='')
            else:
                stderr_lines.append(line)
                print(line, end='')

    for line in process.stdout:
        stdout_lines.append(line)
    for line in process.stderr:
        stderr_lines.append(line)

    return_code = process.returncode
    observation = "".join(stdout_lines if return_code == 0 else stderr_lines)
    if observation == "" and return_code == 0:
        observation = "".join(stderr_lines)

    return return_code, "The script has been executed. Here is the output:\n" + observation