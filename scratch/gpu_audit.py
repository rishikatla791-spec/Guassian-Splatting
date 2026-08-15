#!/usr/bin/env python3
"""
gpu_audit.py — Audit system GPU / CUDA availability and PyTorch device configuration.
"""
import sys
import os
import subprocess
import torch

def audit():
    print("==========================================================================")
    print("   GPU & CUDA ACCELERATION AUDIT REPORT                                  ")
    print("==========================================================================\n")

    print(f"Python Executable:    {sys.executable}")
    print(f"Python Version:       {sys.version}")
    print(f"PyTorch Version:      {torch.__version__}")
    print(f"PyTorch CUDA Built:   {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"CUDA Device Count:    {device_count}")

    if torch.cuda.is_available() and device_count > 0:
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU [{i}]: {name} | Compute Cap: {cap[0]}.{cap[1]} | VRAM: {total_mem:.2f} GB")
    else:
        print("  [ALERT] PyTorch reports torch.cuda.is_available() == False!")

    print("\n--- System nvidia-smi Check ---")
    try:
        res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if res.returncode == 0:
            print(res.stdout)
        else:
            print(f"nvidia-smi exited with code {res.returncode}:\n{res.stderr}")
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")

if __name__ == "__main__":
    audit()
