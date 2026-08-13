# Next-Generation 3D Gaussian Splatting & Mesh Reconstruction System

A high-performance, modular Python & Mobile toolkit for real-time **3D Gaussian Splatting**, dense surface mesh extraction, interactive web visualization, and mobile dataset capture (iOS ARKit & Android ARCore).

---

## 🌟 Key Features

- **Core Differentiable Rasterizer**: Custom PyTorch & CUDA rasterization engine supporting tile-based sorting, spherical harmonics (SH) up to degree 3, anisotropic 3D covariance, and alpha blending.
- **End-to-End Reconstruction Pipeline**: Includes COLMAP loader, pose estimation, dense geometry estimation, background masking, and dataset validation.
- **Advanced & Experimental Modules**:
  - **Adaptive FPS & VRAM Management**: Dynamic level-of-detail (LOD) pruning and memory-aware batch optimization.
  - **Predictive Streaming & Compression**: Vector quantization and pruning for web streaming.
  - **Temporal Dynamics & Neural Balancer**: Support for dynamic scenes and neural-guided gradient densification.
- **Mobile Camera Capture Guides**:
  - **iOS (ARKit / Swift)**: Guided camera capture app with real-time blur detection, dome trajectory planning, and COLMAP dataset export.
  - **Android (ARCore / Kotlin)**: ARCore-assisted spatial guidance and dataset validation.
- **Interactive Web & 3D Viewer**: Built-in Three.js studio (`index.html`) for 3D Gaussian point clouds and extracted surface mesh rendering with camera orbit controls.

---

## 📁 Repository Structure

```
.
├── core/                       # Mathematical primitives, camera model, GaussianModel
├── pipeline/                   # COLMAP loading, pose estimation, dense geometry, pipeline
├── renderer/                   # Differentiable tile-based & Gaussian rasterizer
├── training/                   # Loss functions (SSIM, L1, D-SSIM), trainer, config
├── experimental/               # Compression, LOD, VRAM manager, predictive streaming
├── scene/                      # Spatial hashing & point cloud primitives
├── ui/                         # Interactive viewer & offline rendering utilities
├── mobile_app/                 # Native iOS (ARKit/Swift) & Android (ARCore/Kotlin) guides
│   ├── Android_ARCore_Camera_Guide/
│   └── iOS_ARKit_Camera_Guide/
├── tests/                      # Comprehensive unit tests & performance benchmarks
├── pyproject.toml              # Build setup & metadata
├── requirements.txt            # System dependencies
├── index.html                  # Ultra-Fidelity 3D Gaussian & Surface Mesh Web Studio
└── train.py / reconstruct.py   # Training & end-to-end entry points
```

---

## ⚙️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rishikatla791-spec/Guassian-Splatting.git
   cd Guassian-Splatting
   ```

2. **Set Up Python Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Quick Start

### 1. Training a Gaussian Model
```bash
python train.py --source_path /path/to/colmap/dataset --model_path ./output/my_model
```

### 2. Extract 3D Surface Mesh
```bash
python export_3d_mesh.py --model_path ./output/my_model --output_mesh ./output/mesh.obj
```

### 3. Interactive Web Viewer
Open `index.html` in any web browser to view 3D Gaussian models and extracted surface meshes interactively.

### 4. Running Unit Tests
```bash
pytest tests/
```

---

## 📱 Mobile Dataset Capture

Find complete native mobile project code and setup instructions:
- [Android ARCore Camera Guide](mobile_app/Android_ARCore_Camera_Guide/README.md)
- [iOS ARKit Camera Guide](mobile_app/iOS_ARKit_Camera_Guide/README.md)

---

## 📄 License

This repository is distributed under standard open-source licensing.
