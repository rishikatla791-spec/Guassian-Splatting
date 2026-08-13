# Android ARCore 3DGS Guided Camera Capture App

This native Android Kotlin application provides real-time 60 FPS 6-DoF positional SLAM tracking using **Google ARCore** to guide users in capturing optimal camera angles, high coverage, and non-blurred photos for training **3D Gaussian Splatting (`gaussian`)**.

---

## 📱 App Architecture & Features

1. **Google ARCore 6-DoF Tracking (`ARCoreCameraGuideActivity.kt`):** High-precision SLAM camera tracking using ARCore on Android devices.
2. **3D Geodesic Orbit Target Nodes (`GeodesicDomePlanner.kt`):** Computes 3D target nodes around the object at $15^\circ$, $45^\circ$, and $75^\circ$ elevation rings.
3. **Motion Blur Quality Filter (`BlurDetector.kt`):** Computes Laplacian variance on camera image buffers to reject blurry frames in real time.
4. **Direct 3DGS Dataset Exporter (`DatasetExporter.kt`):** Saves JPEG photos alongside `transforms.json` containing 4x4 camera pose matrices and pinhole camera intrinsics.

---

## 🚀 Android Studio Build & Setup

### Requirements
- Android Studio Hedgehog / Iguana or newer.
- Physical Android Device supporting **Google Play Services for AR (ARCore)** (e.g. Pixel 4+, Samsung Galaxy S10+ / A52+, OnePlus 8+).
- Android 9.0 (API Level 28) or higher.

### Android Studio Setup
1. Create a new **Empty Activity** project in Android Studio:
   - **Name:** `3DGSCameraGuide`
   - **Package Name:** `com.gaussian.cameraguide`
   - **Language:** Kotlin
   - **Minimum SDK:** API 28 (Android 9.0)
2. Add Google ARCore dependency in your `build.gradle.kts` (Module: app):
   ```kotlin
   dependencies {
       implementation("com.google.ar:core:1.41.0")
       implementation("androidx.appcompat:appcompat:1.6.1")
       implementation("com.google.android.material:material:1.11.0")
   }
   ```
3. Add ARCore permissions to `AndroidManifest.xml`:
   ```xml
   <uses-permission android.permission.CAMERA />
   <uses-feature android.hardware.camera.ar android:required="true" />

   <application ...>
       <meta-data android:name="com.google.ar.core" android:value="required" />
   </application>
   ```
4. Copy all `.kt` files from `mobile_app/Android_ARCore_Camera_Guide/` into `app/src/main/java/com/gaussian/cameraguide/`:
   - [`ARCoreCameraGuideActivity.kt`](file:///C:/Users/Rishi/Downloads/gaussian/mobile_app/Android_ARCore_Camera_Guide/ARCoreCameraGuideActivity.kt)
   - [`GeodesicDomePlanner.kt`](file:///C:/Users/Rishi/Downloads/gaussian/mobile_app/Android_ARCore_Camera_Guide/GeodesicDomePlanner.kt)
   - [`BlurDetector.kt`](file:///C:/Users/Rishi/Downloads/gaussian/mobile_app/Android_ARCore_Camera_Guide/BlurDetector.kt)
   - [`DatasetExporter.kt`](file:///C:/Users/Rishi/Downloads/gaussian/mobile_app/Android_ARCore_Camera_Guide/DatasetExporter.kt)
5. Build and Run on your Android phone!

---

## 🔄 Connecting Android Captured Datasets to our `gaussian` Pipeline

Captured datasets are saved directly under your device's app storage folder:
`Android/data/com.gaussian.cameraguide/files/3dgs_session_1723420000/`

```
3dgs_session_1723420000/
├── images/
│   ├── frame_0000.jpg
│   ├── frame_0001.jpg
│   └── ...
└── transforms.json
```

Copy the folder via USB / ADB to your PC and launch training directly:
```bash
python train.py --source_path /path/to/3dgs_session_1723420000
```
