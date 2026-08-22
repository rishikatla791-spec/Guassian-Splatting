@echo off
echo ========================================================
echo   Training 3D Gaussian Splatting Scene: Room (3000 Steps)
echo ========================================================
set SCENE_DIR=C:\Users\Rishi\Downloads\test\Room_dataset
set OUTPUT_DIR=C:\Users\Rishi\Downloads\test\output\room

call C:\Users\Rishi\Downloads\test\build_env.bat C:\Users\Rishi\anaconda3\envs\gaussian_cuda\python.exe C:\Users\Rishi\Downloads\test\gaussian-splatting\train.py -s "%SCENE_DIR%" -m "%OUTPUT_DIR%" --eval --iterations 3000 --save_iterations 1000 2000 3000 --resolution 2
