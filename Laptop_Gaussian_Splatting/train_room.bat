@echo off
echo ========================================================
echo   Training 3D Gaussian Splatting Scene: Room (3000 Steps)
echo ========================================================
set SCENE_DIR=%~dp0Room_dataset
set OUTPUT_DIR=%~dp0output\room

call "%~dp0build_env.bat" C:\Users\Rishi\anaconda3\envs\gaussian_cuda\python.exe "%~dp0gaussian-splatting\train.py" -s "%SCENE_DIR%" -m "%OUTPUT_DIR%" --eval --iterations 3000 --save_iterations 1000 2000 3000 --resolution 2
