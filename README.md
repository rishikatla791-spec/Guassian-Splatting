# 3D Gaussian Splatting (3DGS) Multi-Platform Workspace

This repository is organized into two independent, specialized environments:

```
c:\Users\Rishi\Downloads\test\
├── Android_Gaussian_Splatting/    # 📱 Android Mobile 3DGS Stack (Studio Project & Mobile Sync)
└── Laptop_Gaussian_Splatting/     # 💻 Laptop / PC 3DGS Pipeline (CUDA Training, Viewers, Studio)
```

---

## 📱 1. Android Gaussian Splatting (`Android_Gaussian_Splatting/`)

A standalone Android Studio application and mobile training ecosystem:
- **Android App Project (`app/`)**:
  - Full Android Studio project with `build.gradle` and `settings.gradle` at folder root.
  - Real-time 60 FPS WebGL / WebGPU 3DGS touch renderer (`assets/viewer/`).
  - Native C++ Brush runtime integration (`cpp/brush_bridge.cpp` + `jniLibs/`).
  - Multi-view camera capture with gyroscope-assisted pose tracking (`CaptureActivity.kt`).
  - Prebuilt standalone APK (`Mobile3DGS.apk`).
- **Mobile Sync & Training Server (`server/`)**:
  - FastAPI server for receiving mobile scans over Wi-Fi, training Gaussian splats, and streaming `.splat` files to the phone.
  - High-speed PLY to 32-byte `.splat` binary converter (`server/ply_to_splat.py`).
- **Cloud Trainer (`cloud_server/`)**:
  - Google Colab notebook (`mobile_3dgs_cloud_trainer.ipynb`) and cloud training backend (`server_3dgs.py`).
- **Mobile Capture Guides (`mobile_guides/`)**:
  - ARCore (Android Kotlin) and ARKit (iOS Swift) camera guide implementations.

### Quick Start (Android):
1. Open **Android Studio** -> Select **Open** -> Choose `Android_Gaussian_Splatting`.
2. Connect your Android phone via USB (or start an emulator) and click **Run**.
3. (Optional) Run `server\run_server.bat` on your PC to host the training server for your phone.

---

## 💻 2. Laptop / PC Gaussian Splatting (`Laptop_Gaussian_Splatting/`)

The full high-performance CUDA training, rendering, and web studio pipeline:
- **Core 3DGS Engine (`gaussian-splatting/`)**:
  - CUDA differentiable rasterizer, spherical harmonics, anisotropic covariance, and dense geometry.
- **Interactive Viewers**:
  - **SIBR Interactive 3D Viewer**: High-framerate desktop viewer (`run_viewer.bat`, `run_viewer_truck.bat`, `run_viewer_playroom.bat`).
  - **Viser Web Viewer**: Browser-based 3D scene visualizer (`viser_viewer.py`).
  - **Web 3D Studio**: Browser UI (`index.html` + `server.py`).
- **One-Click Training Scripts**:
  - `train_fast.bat`: Fast 3k-iteration half-resolution training (ideal for 6GB VRAM GPUs like RTX 3050).
  - `train_room.bat`: Smartphone room dataset training pipeline.
- **Unified 3DGS Studio (`gaussian_studio.py`)**:
  - Floater pruning, 360° orbit / turntable rendering, and WebGL `.splat` export.

### Quick Start (Laptop):
1. Open terminal in `Laptop_Gaussian_Splatting`.
2. Train a scene:
   ```cmd
   .\train_fast.bat data\lego_scene\lego output\lego_fast
   ```
3. Launch the desktop viewer:
   ```cmd
   .\run_viewer.bat
   ```
4. Launch the Web Studio:
   ```cmd
   python server.py
   ```
