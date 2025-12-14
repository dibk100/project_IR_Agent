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

def extract_imports(code: str):
    imports = set()
    
    # handle "import a", "import a, b", "import a.b"
    for line in re.findall(r'^\s*import\s+(.+)$', code, re.MULTILINE):
        mods = [m.strip().split('.')[0] for m in line.split(',')]
        imports.update(mods)
    
    # handle "from a.b import c, d"
    for line in re.findall(r'^\s*from\s+(\S+)\s+import', code, re.MULTILINE):
        imports.add(line.split('.')[0])

    return imports

def is_std_lib(pkg_name: str) -> bool:
    if pkg_name in sys.builtin_module_names:
        return True
    try:
        spec = importlib.util.find_spec(pkg_name)
        if spec is None or spec.origin is None:
            return True
        
        origin = spec.origin
        return (
            "site-packages" not in origin and
            "dist-packages" not in origin
        )
    except Exception:
        return False

def ensure_persistent_container(container_name="automl_worker", image="automl-runtime:latest", device="0", work_dir="agent_workspace"):
    """
    Persistent Docker container 생성 또는 재사용
    """
    # 작업 디렉토리 절대 경로
    work_dir_host = os.path.abspath(work_dir)
    os.makedirs(work_dir_host, exist_ok=True)
    
    # 1. 컨테이너 존재 여부 확인
    print(f"[Docker] Checking for container '{container_name}'...")
    inspect_cmd = ["docker", "inspect", "-f", "{{.State.Status}}", container_name]
    result = subprocess.run(inspect_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # 컨테이너가 존재하는 경우
        status = result.stdout.strip()
        print(f"[Docker] Container '{container_name}' exists. Status: {status}")
        
        if status == "running":
            print(f"[Docker] Container is already running.")
            return container_name
        elif status in ["exited", "created"]:
            print(f"[Docker] Starting existing container...")
            try:
                subprocess.run(["docker", "start", container_name], check=True, capture_output=True)
                print(f"[Docker] Container started successfully.")
                return container_name
            except subprocess.CalledProcessError as e:
                # 시작 실패 시 컨테이너 삭제 후 재생성
                print(f"[Docker] Failed to start container. Removing and recreating...")
                subprocess.run(["docker", "rm", "-f", container_name], check=False)
                # 아래 생성 로직으로 진행
        else:
            # 알 수 없는 상태면 삭제
            print(f"[Docker] Container in unknown state '{status}'. Removing...")
            subprocess.run(["docker", "rm", "-f", container_name], check=False)
    else:
        # 컨테이너가 없는 경우
        print(f"[Docker] Container '{container_name}' does not exist.")

    # 2. 컨테이너 생성 (위에서 리턴되지 않은 경우 실행)
    print(f"[Docker] Creating new container: {container_name}")
    
    # 이미지 존재 확인
    check_image = subprocess.run(
        ["docker", "images", "-q", image],
        capture_output=True,
        text=True
    )
    
    if not check_image.stdout.strip():
        raise RuntimeError(
            f"Docker image '{image}' not found. Please build it first:\n"
            f"  docker build -t {image} ."
        )
    
    create_cmd = [
        "docker", "create",
        "--gpus", f"device={device}",
        "-v", f"{work_dir_host}:/workspace/agent_workspace",
        "-v", "/mnt/hdd/hf_cache:/workspace/hf_cache",
        "-e", "HF_HOME=/workspace/hf_cache",
        "--workdir", "/workspace",            # 중요
        "--name", container_name,
        image,
        "sleep", "infinity"
    ]
    
    try:
        subprocess.run(create_cmd, check=True, capture_output=True)
        subprocess.run(["docker", "start", container_name], check=True, capture_output=True)
        print(f"[Docker] Container '{container_name}' created and started successfully.")
        return container_name
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError(
            f"[Docker] Failed to create container:\n{error_msg}\n"
            f"Troubleshooting:\n"
            f"  1. Check if image exists: docker images | grep {image}\n"
            f"  2. Check GPU availability: nvidia-smi\n"
            f"  3. Try without GPU: Remove '--gpus' option\n"
            f"  4. Check existing containers: docker ps -a"
        )

def auto_install_packages_persistent(script_path, container_name="automl_worker", retry=True, skip_packages=None):
    """
    Docker 내에서 필요한 패키지 설치.
    실패 시 1회 재시도 가능(retry=True).
    
    Args:
        script_path (str): 스크립트 경로
        container_name (str): Docker 컨테이너 이름
        retry (bool): 설치 실패 시 재시도 여부
        skip_packages (set): 이미 설치된 패키지 set (설치 건너뛰기용) ✅ 추가
    
    Returns:
        failed_packages (list): 설치 실패한 패키지 리스트
    """
    
    failed_packages = []  # 실패 패키지 기록
    
    # ✅ 추가: skip_packages가 None이면 빈 set으로 초기화
    if skip_packages is None:
        skip_packages = set()
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"{script_path} does not exist")
    
    with open(script_path, "r") as f:
        code = f.read()
    imports = re.findall(r'^\s*import (\S+)|^\s*from (\S+) import', code, re.MULTILINE)
    packages = {pkg.split('.')[0] for grp in imports for pkg in grp if pkg and not is_std_lib(pkg)}
    
    for pkg in packages:
        install_pkg = PACKAGE_MAP.get(pkg, pkg)
        
        # ✅ 추가: 이미 설치 확인된 패키지는 건너뛰기
        if install_pkg in skip_packages:
            print(f"[AutoInstaller] {install_pkg} already tracked as installed. Skipping check.")
            continue
        
        # check if installed
        check_cmd = ["docker", "exec", container_name, "python", "-c", f"import {pkg}"]
        result = subprocess.run(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print(f"[AutoInstaller] {install_pkg} already installed.")
            skip_packages.add(install_pkg)  # ✅ 추가: 설치 확인된 패키지 추적
            continue
        
        # install
        # 설치 시도
        for attempt in range(1, 3 if retry else 2):
            try:
                print(f"[AutoInstaller] Installing {install_pkg} (attempt {attempt})...")
                install_cmd = ["docker", "exec", "-u", "0", container_name, "pip", "install", "--no-cache-dir", install_pkg]
                subprocess.run(install_cmd, check=True)
                # ✅ 추가: 설치 성공 시 추적
                skip_packages.add(install_pkg)
                print(f"[AutoInstaller] ✅ {install_pkg} installed successfully.")
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
    모든 stdout / stderr를 EOF까지 캡처

    """
    
    cmd = ["docker", "exec", "-w", "/workspace/agent_workspace", container_name, "python", "-u", script_name]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,bufsize=1)
    
    stdout_lines, stderr_lines = [], []
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    selector.register(process.stderr, selectors.EVENT_READ)
    
    start_time = time.time()
    while selector.get_map():
        # timeout 체크
        if time.time() - start_time > timeout:
            process.kill()
            process.wait()
            return -1, f"Execution timed out after {timeout} seconds."

        for key, _ in selector.select(timeout=1):
            line = key.fileobj.readline()

            # ✅ EOF 처리
            if not line:
                selector.unregister(key.fileobj)
                key.fileobj.close()
                continue

            if key.fileobj is process.stdout:
                stdout_lines.append(line)
                print(line, end="")
            else:
                stderr_lines.append(line)
                print(line, end="")

    # 스트림 다 읽은 뒤 프로세스 종료 보장
    return_code = process.wait()

    # observation 구성
    if return_code == 0:
        observation = "".join(stdout_lines + stderr_lines)
    else:
        observation = "".join(stderr_lines if stderr_lines else stdout_lines)

    return (
        return_code,
        "The script has been executed. Here is the output:\n" + observation
    )