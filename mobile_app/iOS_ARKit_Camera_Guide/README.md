# iOS ARKit 3DGS Guided Camera Capture App

This native iOS Swift application guides users in real time to capture optimal camera angles, high coverage, and non-blurred photos for training **3D Gaussian Splatting (`gaussian`)**.

---

## 📱 App Architecture & Features

1. **6-DoF ARKit SLAM Tracking (`ARCameraGuideViewController.swift`):** Real-time 60 FPS camera pose tracking relative to physical space.
2. **3D Geodesic Orbit Target Nodes (`GeodesicDomePlanner.swift`):** Projects 3D target nodes (red spheres) around the object at $15^\circ$, $45^\circ$, and $75^\circ$ elevation rings.
3. **Automatic Target Satisfaction & Haptics:** As the user moves the phone within $20^\circ$ of a target node, the node snaps green, triggers haptic feedback, and auto-captures the photo.
4. **Real-time Blur Quality Filter (`BlurDetector.swift`):** Calculates Laplacian variance $\text{Var}(\nabla^2 I)$ using Apple's Accelerate `vImage` framework on `CVPixelBuffer` frames to reject motion blur.
5. **Direct 3DGS Dataset Exporter (`DatasetExporter.swift`):** Saves captured images alongside `transforms.json` containing 4x4 camera pose matrices and pinhole camera intrinsics.

---

## 🚀 Xcode Setup & Build Instructions

### Requirements
- Mac with Xcode 15+ installed.
- Physical iOS Device (iPhone 11+, iPad Pro with LiDAR recommended).
- iOS 16.0+ SDK.

### Xcode Project Setup
1. Create a new **iOS App** project in Xcode:
   - **Product Name:** `3DGSCameraGuide`
   - **Interface:** Storyboard or SwiftUI
   - **Language:** Swift
2. Copy all `.swift` files from `mobile_app/iOS_ARKit_Camera_Guide/` into your Xcode project directory:
   - [`ARCameraGuideViewController.swift`](file:///C:/Users/Rishi/Downloads/gaussian/mobile_app/iOS_ARKit_Camera_Guide/ARCameraGuideViewController.swift)
   - [`GeodesicDomePlanner.swift`](file:///C:/Users/Rishi/Downloads/gaussian/mobile_app/iOS_ARKit_Camera_Guide/GeodesicDomePlanner.swift)
   - [`BlurDetector.swift`](file:///C:/Users/Rishi/Downloads/gaussian/mobile_app/iOS_ARKit_Camera_Guide/BlurDetector.swift)
   - [`DatasetExporter.swift`](file:///C:/Users/Rishi/Downloads/gaussian/mobile_app/iOS_ARKit_Camera_Guide/DatasetExporter.swift)
3. Add `NSCameraUsageDescription` to your `Info.plist`:
   ```xml
   <key>NSCameraUsageDescription</key>
   <string>Camera access is required for 6D SLAM tracking and capturing 3DGS dataset photos.</string>
   ```
4. Build and Run on your physical iPhone/iPad device.

---

## 🔄 Connecting Mobile Captured Datasets to our `gaussian` Pipeline

Once you finish capturing photos in the mobile app, the dataset folder will be stored in your iPhone Documents directory:

```
3dgs_session_1723420000/
├── images/
│   ├── frame_0000.jpg
│   ├── frame_0001.jpg
│   └── ...
└── transforms.json
```

Transfer the folder to your PC and run our training script directly:
```bash
python train.py --source_path /path/to/3dgs_session_1723420000
```
