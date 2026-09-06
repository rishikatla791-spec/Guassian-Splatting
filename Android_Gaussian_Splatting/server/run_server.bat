@echo off
echo ========================================================
echo   Starting Mobile 3DGS GPU Training Server
echo ========================================================

set PATH=C:\Users\Rishi\anaconda3\envs\gaussian_cuda\bin;C:\Users\Rishi\anaconda3\envs\gaussian_cuda\Scripts;%PATH%
set PYTHON_EXE=C:\Users\Rishi\anaconda3\envs\gaussian_cuda\python.exe

echo Server listening on http://0.0.0.0:8000
echo Connect your Android device on the same Wi-Fi network.
echo.

cd /d "%~dp0"
"%PYTHON_EXE%" -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
