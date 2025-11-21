import os
import subprocess
import sys
from pathlib import Path

def create_env_and_install(req_file="requirements.txt", env_name=".venv"):
    env_path = Path(env_name)
    python_exe = env_path / "bin" / "python" if os.name != "nt" else env_path / "Scripts" / "python.exe"

    # Create venv
    print(f"Creating virtual environment: {env_name}")
    subprocess.run([sys.executable, "-m", "venv", env_name], check=True)

    # Upgrade pip
    print("Upgrading pip...")
    subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)

    # Install requirements
    if Path(req_file).exists():
        print(f"Installing from {req_file}...")
        subprocess.run([python_exe, "-m", "pip", "install", "-r", req_file], check=True)
    else:
        print(f"{req_file} not found.")

    print(f"Environment created and dependencies installed in '{env_name}'")

if __name__ == "__main__":
    create_env_and_install()
