@echo off
call "%~dp0msvc\setup_x64.bat"
set "CUDA_HOME=C:\Users\Rishi\anaconda3\envs\gaussian_cuda"
set "CUDA_PATH=C:\Users\Rishi\anaconda3\envs\gaussian_cuda"
set "PATH=C:\Users\Rishi\anaconda3\envs\gaussian_cuda\bin;C:\Users\Rishi\anaconda3\envs\gaussian_cuda\Library\bin;%PATH%"
set "DISTUTILS_USE_SDK=1"
%*
