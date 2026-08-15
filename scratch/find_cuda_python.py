#!/usr/bin/env python3
"""
find_cuda_python.py — Search for Python environments with CUDA PyTorch installed.
"""
import sys
import os
import subprocess
import glob

def check_env(py_path):
    try:
        cmd = [py_path, "-c", "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            print(f"[FOUND] {py_path} -> {res.stdout.strip()}")
            return True
        else:
            print(f"[NO TORCH] {py_path}")
    except Exception as e:
        print(f"[ERROR] {py_path}: {e}")
    return False

def search():
    print("=== Searching System Python Environments ===")
    candidates = [
        "python",
        "python3",
        r"C:\Python314\python.exe",
        r"C:\Python312\python.exe",
        r"C:\Python311\python.exe",
        r"C:\Python310\python.exe",
    ]
    
    # Add user AppData python executables
    appdata_py = glob.glob(r"C:\Users\Rishi\AppData\Local\Programs\Python\Python*\python.exe")
    candidates.extend(appdata_py)

    # Add Anaconda/Miniconda executables
    conda_py = glob.glob(r"C:\Users\Rishi\anaconda3\python.exe") + glob.glob(r"C:\Users\Rishi\miniconda3\python.exe") + glob.glob(r"C:\ProgramData\anaconda3\python.exe")
    candidates.extend(conda_py)

    for py in set(candidates):
        if os.path.exists(py) or py in ["python", "python3"]:
            check_env(py)

if __name__ == "__main__":
    search()
