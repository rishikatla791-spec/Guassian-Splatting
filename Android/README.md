# 📱 Mobile 3D Gaussian Splatting (3DGS) for Android

A complete, end-to-end Android solution for **mobile 3D Gaussian Splatting capture, GPU optimization, and real-time 60 FPS touch rendering**.

---

## 🏗️ Architecture & Open-Source Repositories Used

| Module | Core Repository | Role & Technology |
| :--- | :--- | :--- |
| **Mobile Capture** | **[SpectacularAI/sdk](https://github.com/SpectacularAI/sdk)** & **Google ARCore** | Real-time Visual-Inertial Odometry (VIO) using phone camera + gyroscope to compute 6DoF camera poses (, T$) without slow COLMAP. |
| **GPU Optimization Backend** | **[
erfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)** / **INRIA 3DGS** | High-performance PyTorch CUDA rasterization server running on host RTX GPU. |
| **Mobile 3D Viewport** | **[ntimatter15/splat](https://github.com/antimatter15/splat)** & **[playcanvas/super-splat](https://github.com/playcanvas/super-splat)** | Hardware-accelerated WebGL2/WebGPU 3D Gaussian Splatting rasterizer running at 60 FPS on Android. |
| **Binary Compression** | **[ArthurBrussee/brush](https://github.com/ArthurBrussee/brush)** | Vectorized 32-byte .splat binary packaging format reducing file size by 80%. |

---

## 📂 Directory Structure

`
Android/
├── app/                                 # Android Application Project
│   ├── src/main/java/com/splat/mobile3dgs/
│   │   ├── MainActivity.kt              # Dashboard & remote model manager
│   │   ├── capture/CaptureActivity.kt   # Camera preview & gyroscope pose logger
│   │   ├── viewer/ViewerActivity.kt     # Fullscreen interactive 3D WebGL viewport
│   │   ├── network/ApiClient.kt         # OkHttp async server synchronization client
│   │   └── model/ScanSession.kt         # Data models for poses and training jobs
│   ├── src/main/assets/viewer/
│   │   ├── index.html                   # Mobile touch UI overlay & canvas
│   │   ├── viewer.js                    # WebGL2 3D Gaussian Splatting engine
│   │   └── demo.splat                   # Bundled offline 3D model (1.8M Gaussians)
│   └── build.gradle                     # Android dependencies (ARCore, CameraX, OkHttp)
└── server/                              # Host Training & Conversion Backend
    ├── server.py                        # FastAPI streaming & training server
    ├── ply_to_splat.py                  # High-speed PLY to .splat binary converter
    ├── run_server.bat                   # 1-click launcher for the GPU server
    └── models/                          # Storage for trained .splat models
`

---

## 🚀 Quick Start Guide

### Step 1: Start the GPU Training Server (On PC)
Double click or run in terminal:
`cmd
C:\Users\Rishi\Downloads\test\Android\server\run_server.bat
`
The server will start on http://0.0.0.0:8000.

### Step 2: Open & Run the App in Android Studio
1. Open **Android Studio**.
2. Select **Open** -> Navigate to C:\Users\Rishi\Downloads\test\Android.
3. Connect your Android phone via USB (or start an Android Emulator).
4. Click **Run (Shift + F10)**.

### Step 3: Explore 3D Gaussian Splats
- **Offline Mode**: Tap **"Open 3D Splat Viewport"** to immediately interact with the bundled photorealistic demo model.
- **Online Mode**: Tap **"Configure"** -> enter your PC's local Wi-Fi IP (e.g. http://192.168.1.5:8000) to stream and download newly trained 3D models.
- **New Scan**: Tap **"Start New 3D Scan"** to orbit around an object, record multi-view frames + poses, and queue instant GPU training.
