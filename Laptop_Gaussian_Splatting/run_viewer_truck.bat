@echo off
echo ========================================================
echo   Launching SIBR 3D Viewer - Truck Scene
echo ========================================================

set PATH=%~dp0viewers\bin;C:\Users\Rishi\anaconda3\envs\gaussian_cuda\bin;%PATH%
set SIBR_BIN=%~dp0viewers\bin\SIBR_gaussianViewer_app.exe
set MODEL_DIR=%~dp0output\truck

echo Loading Truck model: %MODEL_DIR%
echo Controls:
echo   - T key: Switch to Trackball 360 Orbit mode (Orbit around truck)
echo   - SPACEBAR: Toggle single full-viewport (expand to 100%% screen)
echo   - Left Click + Drag: Rotate / Orbit in 360
echo   - Right Click + Drag: Pan camera
echo   - Scroll: Zoom in / out
echo   - WASD: Fly navigation
echo.

cd /d "%~dp0viewers\bin"
SIBR_gaussianViewer_app.exe -m "%MODEL_DIR%" --fullscreen --width 1920 --height 1080 --hd

