#!/usr/bin/env python3
"""
verify_python312_cuda.py — Comprehensive Python 3.12 CUDA Environment Verification.
Asserts that PyTorch is running under Python 3.12 with CUDA acceleration enabled.
"""
import sys
import os
import torch

def verify():
    print("==========================================================================")
    print("   ENVIRONMENT VERIFICATION: PYTHON 3.12 CUDA ACCELERATION              ")
    print("==========================================================================\n")

    py_exe = sys.executable
    py_ver = sys.version.split()[0]
    torch_ver = torch.__version__
    cuda_avail = torch.cuda.is_available()

    print(f"Python Executable:        {py_exe}")
    print(f"Python Version:           {py_ver}")
    print(f"PyTorch Version:          {torch_ver}")
    print(f"PyTorch CUDA Version:     {torch.version.cuda}")
    print(f"torch.cuda.is_available():{cuda_avail}")

    assert "3.12" in py_ver, f"Expected Python 3.12, got {py_ver}"
    assert cuda_avail is True, "CRITICAL: torch.cuda.is_available() is False! CUDA acceleration required."
    
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Device Name:          {device_name}")
    print(f"Total VRAM Available:     {vram_gb:.2f} GB")

    # Perform a test tensor matrix multiplication on GPU
    a = torch.randn(1000, 1000, device="cuda")
    b = torch.randn(1000, 1000, device="cuda")
    c = torch.matmul(a, b)
    torch.cuda.synchronize()

    print("\n[SUCCESS] Python 3.12 CUDA Environment Verified & Active!")
    print("==========================================================================\n")

if __name__ == "__main__":
    verify()
