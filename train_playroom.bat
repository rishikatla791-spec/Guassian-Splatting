@echo off
echo ========================================================
echo   Training 3D Gaussian Splatting Scene: Playroom
echo ========================================================
set SCENE_DIR=C:\Users\Rishi\Downloads\tandt_db\db\playroom
set OUTPUT_DIR=C:\Users\Rishi\Downloads\test\output\playroom

if not "%1"=="" set SCENE_DIR=%1
if not "%2"=="" set OUTPUT_DIR=%2

call C:\Users\Rishi\Downloads\test\build_env.bat C:\Users\Rishi\anaconda3\envs\gaussian_cuda\python.exe C:\Users\Rishi\Downloads\test\gaussian-splatting\train.py -s "%SCENE_DIR%" -m "%OUTPUT_DIR%" --eval --iterations 3000 --save_iterations 1000 2000 3000
