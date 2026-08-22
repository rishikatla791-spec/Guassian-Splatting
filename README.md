# Next-Generation 3D Gaussian Splatting & Web/Mobile Studio Pipeline

A high-performance, modular Python, Web, and Mobile toolkit for real-time **3D Gaussian Splatting (3DGS)**, fast CUDA training, floater pruning, WebGL `.splat` compression, dense surface mesh extraction, interactive web visualization, and mobile dataset capture (iOS ARKit & Android ARCore).

---

## 🌟 Key Features

- **Unified 3D Gaussian Studio (`gaussian_studio.py`)**:
  - Load, inspect, filter, prune, and transform 3D Gaussian models.
  - Floater removal via opacity filtering (`--min_opacity`) and scale bounds (`--max_scale`).
  - Automated 360° Orbit / Turntable, Dataset Path, and Spiral trajectory generation.
  - Multi-format export: MP4 high-FPS video, animated GIF, PNG sequence, cleaned PLY, and WebGL `.splat` format.
- **Fast GPU-Vectorized Renderer (`fast_render.py`)**:
  - Vectorized NumPy and PyTorch CUDA tensor ingestion (10x faster startup than iterative loaders).
  - High-throughput rendering (160+ FPS at 1080p on consumer GPUs like RTX 3050).
- **Core Differentiable Rasterizer**:
  - Full CUDA rasterization engine (`diff-gaussian-rasterization`, `simple-knn`, `fused-ssim`).
  - Tile-based sorting, spherical harmonics (SH) up to degree 3, anisotropic 3D covariance, and alpha blending.
- **Optimization & Memory Management**:
  - Optimized for mid-tier GPUs (RTX 3050 / 4050 6GB VRAM) with zero OOM errors.
  - Half-resolution training (`--resolution 2`), fast densification (3k–7k iterations).
  - 88%–93% file size reduction via `.splat` format (from ~350MB PLY down to ~30MB).
- **Interactive Viewers**:
  - Built-in **SIBR Interactive Viewer** launcher (`run_viewer.bat`).
  - Modern browser-based **Viser Web Viewer** (`viser_viewer.py`).
  - HTML5 / WebGL Three.js studio (`index.html`).
- **Mobile Camera Capture Guides**:
  - **iOS (ARKit / Swift)**: Guided camera capture app with real-time blur detection, dome trajectory planning, and COLMAP dataset export.
  - **Android (ARCore / Kotlin)**: ARCore-assisted spatial guidance and dataset validation.

---

## 📁 Repository Structure

```
.
├── gaussian_studio.py          # Unified 3DGS Studio (Pruning, Orbit, Splat export, Video)
├── fast_render.py              # Ultra-fast GPU-vectorized frame renderer & benchmark
├── viser_viewer.py             # Interactive Viser Web 3D Viewer
├── train_fast.bat              # One-click Windows fast training script (3k steps, res/2)
├── train_room.bat              # One-click Room dataset training script
├── run_viewer.bat              # SIBR Interactive 3D Viewer launcher
├── build_env.bat               # Windows MSVC + CUDA environment setup
├── gaussian-splatting/         # Core 3DGS engine with CUDA submodules & Windows fixes
│   ├── arguments/
│   ├── gaussian_renderer/
│   ├── scene/
│   ├── submodules/             # diff-gaussian-rasterization, fused-ssim, simple-knn
│   └── train.py / render.py
├── core/                       # Mathematical primitives, camera model, GaussianModel
├── pipeline/                   # COLMAP loading, pose estimation, dense geometry
├── renderer/                   # Differentiable tile-based & Gaussian rasterizer
├── training/                   # Loss functions (SSIM, L1, D-SSIM), trainer, config
├── mobile_app/                 # Native iOS (ARKit/Swift) & Android (ARCore/Kotlin) guides
│   ├── Android_ARCore_Camera_Guide/
│   └── iOS_ARKit_Camera_Guide/
├── tests/                      # Comprehensive unit tests & performance benchmarks
├── pyproject.toml              # Build setup & metadata
├── requirements.txt            # System dependencies
└── index.html                  # Ultra-Fidelity 3D Gaussian Web Studio
```

---

## ⚡ Quick Start & Usage

### 1. Fast Model Training (6GB VRAM friendly)
Train a 3D Gaussian Splatting scene in under 5 minutes on RTX 3050 / 4050:
```powershell
.\train_fast.bat data\lego_scene\lego output\lego_fast
```

### 2. 3D Gaussian Studio (Filtering, Orbit Video & WebGL Splat Export)
Clean floaters, render a 360-degree orbit video, and export web-ready `.splat` file:
```bash
python gaussian_studio.py \
  --model output/room \
  --mode orbit \
  --frames 60 \
  --fps 30 \
  --min_opacity 0.04 \
  --max_scale 0.15 \
  --export_splat output/room/room_web.splat \
  --export_ply output/room/room_clean.ply
```

### 3. Fast Vectorized Render Benchmark
```bash
python fast_render.py
```

### 4. Interactive Web Viewer (Viser)
```bash
python viser_viewer.py --ply output/room/point_cloud/iteration_3000/point_cloud.ply
```

---

## 📊 Optimization & Performance Benchmarks

*(Benchmarked on NVIDIA GeForce RTX 3050 6GB Laptop GPU)*

| Metric / Dimension | Baseline (Default 30k PLY) | Optimized (7k Steps + Pruned + Splat) | Ultra-Fast (3k Steps + Downscaled) |
| :--- | :--- | :--- | :--- |
| **Model Disk Size** | `340 MB – 520 MB` | **`32 MB – 48 MB` (-90%)** | **`14 MB – 22 MB` (-95%)** |
| **Active Gaussians** | `1,800,000 – 2,500,000` | **`450,000 – 700,000`** | **`220,000 – 350,000`** |
| **Training Time** | `42 mins – 1 hr 10 mins` | **`6 mins – 9 mins` (6x faster)** | **`2 mins – 3.5 mins` (15x faster)**|
| **VRAM Peak (Train)** | `5.8 GB – 6.1 GB` (Near OOM) | **`3.4 GB` (Safe on 6GB)** | **`2.6 GB`** |
| **Render FPS (1080p)** | `41 FPS` | **`161 FPS` (4x boost)** | **`294 FPS`** |

---

## 📱 Mobile Dataset Capture & Deployment

Find complete native mobile project code and setup instructions:
- [Android ARCore Camera Guide](mobile_app/Android_ARCore_Camera_Guide/README.md)
- [iOS ARKit Camera Guide](mobile_app/iOS_ARKit_Camera_Guide/README.md)

---

## 📄 License
This project is licensed under the MIT License and standard open-source research licenses.
