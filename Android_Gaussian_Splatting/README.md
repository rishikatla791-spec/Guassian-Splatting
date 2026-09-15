# Mobile 3D Gaussian Splatting for Android

A **standalone, on-device** 3D Gaussian Splatting app. Capture, training and
viewing all happen on a single phone. There is no server, no PC, no cloud and no
CUDA machine anywhere in the pipeline — the phone is the whole system.

```
   ARCore capture              On-device training              Viewer
  ┌────────────────┐         ┌────────────────────┐       ┌──────────────┐
  │ camera frames  │         │ libbrush_c.so      │       │ WebGL2 splat │
  │ 6-DoF poses    │  ────▶  │ Rust / wgpu        │ ────▶ │ renderer in  │
  │ depth + points │ dataset │ Vulkan compute     │ .splat│ a WebView    │
  └────────────────┘         └────────────────────┘       └──────────────┘
```

1. **Capture** — `CaptureActivity` drives ARCore for 6-DoF visual-inertial
   tracking. Each keyframe is stored with its camera-to-world matrix and
   per-frame pinhole intrinsics. ARCore depth and feature points accumulate into
   a seed point cloud with real metric scale (1 unit = 1 metre).
2. **Export** — `DatasetExporter` writes a nerfstudio-style dataset to the app's
   external files directory: `transforms.json`, `images/`, `points3d.ply`
   (the seed cloud the optimiser initialises from) and `points3D_initial.json`.
3. **Train** — `TrainingService` runs a foreground service that calls into
   `libbrush_c.so` (a prebuilt Rust/wgpu Gaussian splatting trainer) through the
   JNI bridge in `cpp/brush_bridge.cpp`. Output is a 32-byte-per-Gaussian
   `.splat` file.
4. **View** — `ViewerActivity` loads the `.splat` into a WebGL2 renderer hosted
   in a WebView (`assets/viewer/`). `ARPlacementActivity` can re-anchor the
   trained model onto a detected real-world plane.

---

## Device requirements

These are enforced by the manifest and by the shipped native code — the app will
not install or will not train if they are not met.

| Requirement | Value | Why |
| :--- | :--- | :--- |
| ABI | **`arm64-v8a`** | The prebuilt training engine is arm64-only. |
| ARCore | **ARCore-certified device**, Google Play Services for AR installed | 6-DoF pose tracking and depth. See the [supported device list](https://developers.google.com/ar/devices). |
| Vulkan | **1.1+** (`android.hardware.vulkan.version` 0x401000, required) | The trainer dispatches Vulkan compute. |
| OpenGL ES | **3.0+** (required) | AR camera background and the splat viewer. |
| Android | **8.0 / API 26** minimum | `minSdk 26`. |
| RAM | 8 GB minimum, 12 GB comfortable | Training a few hundred thousand Gaussians is memory-hungry; the app requests `largeHeap`. |

Reference device for all measured numbers below: **Snapdragon 8 Gen 2 / Adreno 740.**

> The **emulator** can run capture, pose tracking and dataset export (the debug
> build ships an `x86_64` slice for exactly this), but it **cannot train** —
> `libbrush_c.so` exists only for `arm64-v8a`, so the engine reports itself
> unavailable there.

---

## Getting the source

The training engine `android/app/src/main/jniLibs/arm64-v8a/libbrush_c.so` is a
**184 MB binary stored in Git LFS**. A plain `git clone` without LFS leaves a
~130-byte pointer file in its place and **the build will produce an APK that
crashes on launch**.

```bash
git lfs install
git clone <repo-url>
cd <repo>
git lfs pull
```

Verify you got the real thing — it must be ~184 MB and an ELF, not text:

```bash
ls -l Android_Gaussian_Splatting/android/app/src/main/jniLibs/arm64-v8a/libbrush_c.so
file Android_Gaussian_Splatting/android/app/src/main/jniLibs/arm64-v8a/libbrush_c.so
# expected: ELF 64-bit LSB shared object, ARM aarch64
```

If it came down as a pointer file, run `git lfs pull` again.

---

## Build

The Gradle project root is `android/`.

```bash
cd android
./gradlew.bat :app:assembleDebug      # Windows
./gradlew     :app:assembleDebug      # macOS / Linux
```

Output: `android/app/build/outputs/apk/debug/app-debug.apk` (~127 MB — almost
all of it is the training engine).

Android Studio: **Open** → select the `android/` directory (not the repository
root). `local.properties` is not tracked; Studio writes your own `sdk.dir` on
first sync.

### Release build

```bash
cd android
./gradlew.bat :app:assembleRelease
```

The release build is minified and resource-shrunk (R8, see `app/proguard-rules.pro`)
and ships **`arm64-v8a` only**. It is **never** signed with the debug key.
Provide signing material through Gradle properties or environment variables:

```properties
# ~/.gradle/gradle.properties  — never commit this
RELEASE_STORE_FILE=/absolute/path/to/release.jks
RELEASE_STORE_PASSWORD=...
RELEASE_KEY_ALIAS=...
RELEASE_KEY_PASSWORD=...
```

Without them the build still succeeds but emits `app-release-unsigned.apk` and
warns loudly.

---

## Install and run

```bash
python deploy_to_physical_phone.py
```

It locates `adb` inside the Android SDK (checking `ANDROID_HOME`,
`ANDROID_SDK_ROOT`, `android/local.properties`, then the platform default), picks
a connected physical device over an emulator, installs the debug APK and launches
the app. No sample models are pushed — the app is standalone, so you capture your
own scan.

On the phone: **Start New 3D Scan** → orbit the object → stop → training starts
as a foreground service (it keeps running with the screen off) → open the result
in the viewer.

---

## Inspecting a scan

When a reconstruction comes out wrong, the cause is almost always the capture,
not the trainer. Pull the scan and run the report before blaming the optimiser:

```bash
adb pull /sdcard/Android/data/com.splat.mobile3dgs/files ./scans
python tools/scan_report.py ./scans/<session>
python tools/scan_report.py ./scans/<session> --splat model.splat
```

It reports frame count, camera baseline, median scene depth, triangulation angle
(parallax), seed-point count and bounding box, and structurally validates a
`.splat` (size is exactly N×32, all floats finite, quaternions ~unit norm,
opacity distribution). Exit status is non-zero if any hard check fails, so it
works in CI. `--json` emits machine-readable output.

The two numbers that matter most: **triangulation angle ≥ 5°** and a sane
**baseline / depth ratio**. Low parallax produces flat, smeared splats no matter
how long you train.

---

## Performance (measured, Snapdragon 8 Gen 2)

| Preset | Steps | Wall clock |
| :--- | ---: | :--- |
| Fast | 2,000 | ~5–7 min |
| Balanced | 5,000 | ~11–15 min |
| High fidelity | 7,000 | **~15–20+ min** |

Thermal throttling is the dominant factor: sustained training heats the SoC
enough to roughly **halve** step throughput partway through a run, so the back
half of a 7k run is substantially slower than the front half. Keep the phone
cool and plugged in. These are real numbers from a physical device, not
estimates.

---

## Layout

```
Android_Gaussian_Splatting/
├── android/                                  # Gradle project root
│   ├── app/
│   │   ├── build.gradle                      # SDK/ABI/signing/minify config
│   │   ├── proguard-rules.pro                # R8 keep rules (JNI, JS bridge, Gson)
│   │   └── src/main/
│   │       ├── AndroidManifest.xml
│   │       ├── assets/viewer/                # WebGL2 splat renderer (no models bundled)
│   │       ├── cpp/brush_bridge.cpp          # JNI bridge, dlopen's libbrush_c.so
│   │       ├── jniLibs/arm64-v8a/
│   │       │   └── libbrush_c.so             # 184 MB prebuilt trainer (Git LFS)
│   │       └── java/com/splat/mobile3dgs/
│   │           ├── MainActivity.kt
│   │           ├── capture/                  # ARCore capture, quality gate, export
│   │           ├── engine/                   # Native engine binding + training service
│   │           ├── viewer/                   # WebView splat viewport
│   │           ├── ar/                       # AR relocalisation and placement
│   │           ├── hardware/                 # Device capability tiering
│   │           └── model/                    # Data models
│   └── gradle/                               # Gradle 8.7 wrapper
├── tools/scan_report.py                      # Capture + .splat verification tool
├── deploy_to_physical_phone.py               # Build-free installer/launcher
├── IMPLEMENTATION.md                         # Engineering plan and status
└── README.md
```

No `.splat` models are bundled in the APK. They used to be — ~74 MB of sample
models in `assets/viewer/` — and were removed; the app generates its own.
