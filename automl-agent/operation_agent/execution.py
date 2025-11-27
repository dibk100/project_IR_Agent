import os
import re
import subprocess
import sys
import importlib.util
import selectors
import time

"""
subprocess.Popen()을 사용해서 python -u {script_path} 명령을 진짜 로컬에서 실행함.
"""

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
    
def ensure_persistent_container(container_name="automl_worker", image="automl-runtime:latest", device="0", work_dir="."):
    # Check container exists
    check_container = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
        capture_output=True, text=True
    )
    if container_name not in check_container.stdout:
        print(f"[Docker] Creating persistent container: {container_name}")
        create_cmd = [
            "docker", "create", "--gpus", f"device={device}",
            "-v", f"{os.path.abspath(work_dir)}:/workspace",
            "--name", container_name,
            image, "sleep", "infinity"
        ]
        try:
            subprocess.run(create_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[Docker] Failed to create container: {e}")

    # Start if not running
    check_running = subprocess.run(
        ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
        capture_output=True, text=True
    )
    if container_name not in check_running.stdout:
        print(f"[Docker] Starting container: {container_name}")
        subprocess.run(["docker", "start", container_name], check=True)
    return container_name

def auto_install_packages_persistent(script_path, container_name="automl_worker",retry=True):
    """
    Docker 내에서 필요한 패키지 설치.
    실패 시 1회 재시도 가능(retry=True).
    
    Returns:
        failed_packages (list): 설치 실패한 패키지 리스트
    """
    
    failed_packages = []  # 실패 패키지 기록
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"{script_path} does not exist")
    
    with open(script_path, "r") as f:
        code = f.read()
    imports = re.findall(r'^\s*import (\S+)|^\s*from (\S+) import', code, re.MULTILINE)
    packages = {pkg.split('.')[0] for grp in imports for pkg in grp if pkg and not is_std_lib(pkg)}
    
    for pkg in packages:
        install_pkg = PACKAGE_MAP.get(pkg, pkg)
        # check if installed
        check_cmd = ["docker", "exec", container_name, "python", "-c", f"import {pkg}"]
        result = subprocess.run(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print(f"[AutoInstaller] {install_pkg} already installed.")
            continue
        # install
        # 설치 시도
        for attempt in range(1, 3 if retry else 2):
            try:
                print(f"[AutoInstaller] Installing {install_pkg} (attempt {attempt})...")
                install_cmd = ["docker", "exec", container_name, "pip", "install", "--no-cache-dir", install_pkg]
                subprocess.run(install_cmd, check=True)
                # 설치 성공 시 break
                break
            except subprocess.CalledProcessError as e:
                print(f"[AutoInstaller] Failed to install {install_pkg} (attempt {attempt}): {e}")
                if attempt == 2 or not retry:
                    failed_packages.append(install_pkg)
    
    return failed_packages

def docker_exec_script_persistent(script_name, container_name="automl_worker", timeout=120):
    """
    Docker 내 Python 스크립트 실행 + timeout 처리
    :param timeout: 최대 실행 시간 (초)
    """
    
    cmd = ["docker", "exec", "-w", "/workspace", container_name, "python", "-u", script_name]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    stdout_lines, stderr_lines = [], []
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    selector.register(process.stderr, selectors.EVENT_READ)
    
    start_time = time.time()
    while process.poll() is None and selector.get_map():
        
        if time.time() - start_time > timeout:
            process.kill()
            process.wait()
            return -1, f"Execution timed out after {timeout} seconds."
        
        
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
                
    return_code = process.returncode
    observation = "".join(stdout_lines if return_code == 0 else stderr_lines)
    if observation == "" and return_code == 0:
        observation = "".join(stdout_lines + stderr_lines)
    return return_code, "The script has been executed. Here is the output:\n" + observation
