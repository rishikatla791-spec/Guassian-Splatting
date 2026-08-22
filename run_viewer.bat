@echo off
echo ========================================================
echo   Launching 3D Gaussian Splatting Interactive Viewer
echo ========================================================

set PATH=C:\Users\Rishi\Downloads\test\viewers\bin;C:\Users\Rishi\anaconda3\envs\gaussian_cuda\bin;%PATH%
set SIBR_BIN=C:\Users\Rishi\Downloads\test\viewers\bin\SIBR_gaussianViewer_app.exe
set MODEL_DIR=C:\Users\Rishi\Downloads\test\output\pretrained_train\train

if not "%1"=="" set MODEL_DIR=%1

echo Loading model: %MODEL_DIR%
echo Controls:
echo   - T key: Switch to Trackball 360 Orbit mode
echo   - SPACEBAR: Toggle single full-viewport (expand to 100%% screen)
echo   - Left Click + Drag: Rotate camera
echo   - Right Click + Drag: Pan camera
echo   - Scroll: Move forward/backward
echo.

cd /d C:\Users\Rishi\Downloads\test\viewers\bin
SIBR_gaussianViewer_app.exe -m "%MODEL_DIR%" --fullscreen --width 1920 --height 1080 --hd

