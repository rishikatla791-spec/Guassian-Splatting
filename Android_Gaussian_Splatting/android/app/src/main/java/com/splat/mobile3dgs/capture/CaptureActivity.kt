package com.splat.mobile3dgs.capture

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.Image
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.ar.core.ArCoreApk
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Frame
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.ar.core.exceptions.UnavailableApkTooOldException
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException
import com.google.ar.core.exceptions.UnavailableSdkTooOldException
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException
import com.splat.mobile3dgs.R
import com.splat.mobile3dgs.engine.NativeBrushEngine
import com.splat.mobile3dgs.engine.TrainingService
import com.splat.mobile3dgs.viewer.ViewerActivity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Milestone 1 — ARCore 6-DoF capture.
 *
 * Records real camera poses (metric, OpenGL camera-to-world convention),
 * per-frame pinhole intrinsics, the ARCore feature point cloud, and the CPU
 * camera image for each kept keyframe. Frames are kept based on real camera
 * motion (translation / rotation) rather than a blind timer. On stop, a
 * NeRF/3DGS `transforms.json` dataset is written via [DatasetExporter] and
 * handed to the on-device Vulkan/wgpu training engine.
 */
class CaptureActivity : AppCompatActivity(), GLSurfaceView.Renderer {

    private lateinit var glSurfaceView: GLSurfaceView
    private lateinit var btnRecord: Button
    private lateinit var tvStatus: TextView
    private lateinit var tvFrameCount: TextView
    private lateinit var progressBar: ProgressBar

    private var session: Session? = null
    private var userRequestedInstall = true
    private val backgroundRenderer = ARBackgroundRenderer()

    @Volatile private var isRecording = false
    private lateinit var datasetExporter: DatasetExporter
    private val saveExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val pendingSaves = AtomicInteger(0)

    // Real intrinsics captured once tracking is stable: [fx, fy, cx, cy, w, h]
    @Volatile private var haveIntrinsics = false
    private var fx = 0f; private var fy = 0f; private var cx = 0f; private var cy = 0f
    private var imgW = 0; private var imgH = 0

    // Motion gate — pose of the last kept keyframe.
    private var lastKeptPos: FloatArray? = null
    private var lastKeptQuat: FloatArray? = null
    @Volatile private var keptFrameCount = 0

    /**
     * Upper bound on keyframes, chosen from the hardware tier. Training memory is
     * dominated by (frames x resolution), and Android's low-memory killer reacts to
     * free RAM rather than total RAM -- so a budget device must stop collecting
     * before the dataset outgrows what is actually available.
     */
    private var maxKeyframes = 120
    @Volatile private var frameLimitNotified = false

    // Display geometry sync (activity is locked portrait).
    private var viewportWidth = 0
    private var viewportHeight = 0
    @Volatile private var viewportChanged = false

    companion object {
        private const val TAG = "CaptureActivity"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private const val REQUEST_CODE_NOTIFICATIONS = 11
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)

        // Keyframe motion thresholds (real ARCore metric poses).
        private const val MIN_TRANSLATION_M = 0.02f      // 2 cm
        private const val MIN_ROTATION_RAD = 0.052f       // ~3 degrees
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_capture)

        glSurfaceView = findViewById(R.id.gl_surface_capture)
        btnRecord = findViewById(R.id.btn_record_scan)
        tvStatus = findViewById(R.id.tv_capture_status)
        tvFrameCount = findViewById(R.id.tv_frame_count)
        progressBar = findViewById(R.id.progress_upload)

        glSurfaceView.preserveEGLContextOnPause = true
        glSurfaceView.setEGLContextClientVersion(2)
        glSurfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0)
        glSurfaceView.setRenderer(this)
        glSurfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY

        btnRecord.setOnClickListener {
            if (!isRecording) startRecordingSession() else stopRecordingSession()
        }

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        // Needed so the training foreground service can show progress on Android 13+.
        if (android.os.Build.VERSION.SDK_INT >= 33) {
            val notifPerm = "android.permission.POST_NOTIFICATIONS"
            if (ContextCompat.checkSelfPermission(this, notifPerm) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, arrayOf(notifPerm), REQUEST_CODE_NOTIFICATIONS)
            }
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS && !allPermissionsGranted()) {
            Toast.makeText(this, "Camera permission required for 3D scan.", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    // -------------------------------------------------------------------------
    // ARCore session lifecycle
    // -------------------------------------------------------------------------

    override fun onResume() {
        super.onResume()
        if (!allPermissionsGranted()) return

        if (session == null) {
            try {
                when (ArCoreApk.getInstance().requestInstall(this, userRequestedInstall)) {
                    ArCoreApk.InstallStatus.INSTALL_REQUESTED -> {
                        userRequestedInstall = false
                        return
                    }
                    ArCoreApk.InstallStatus.INSTALLED -> { /* proceed */ }
                    else -> {}
                }

                val newSession = Session(this)
                selectHighestResolutionCameraConfig(newSession)

                val config = Config(newSession).apply {
                    updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                    focusMode = Config.FocusMode.AUTO
                    planeFindingMode = Config.PlaneFindingMode.DISABLED
                    if (newSession.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
                        depthMode = Config.DepthMode.AUTOMATIC
                    }
                }
                newSession.configure(config)
                session = newSession
            } catch (e: UnavailableUserDeclinedInstallationException) {
                toastAndFinish("Please install Google Play Services for AR")
                return
            } catch (e: UnavailableArcoreNotInstalledException) {
                toastAndFinish("Please install ARCore (Google Play Services for AR)")
                return
            } catch (e: UnavailableDeviceNotCompatibleException) {
                toastAndFinish("This device does not support ARCore capture")
                return
            } catch (e: UnavailableApkTooOldException) {
                toastAndFinish("Please update Google Play Services for AR")
                return
            } catch (e: UnavailableSdkTooOldException) {
                toastAndFinish("App is out of date for ARCore")
                return
            } catch (e: Exception) {
                Log.e(TAG, "Failed to create ARCore session", e)
                toastAndFinish("ARCore init failed: ${e.localizedMessage}")
                return
            }
        }

        try {
            session?.resume()
        } catch (e: CameraNotAvailableException) {
            Log.e(TAG, "Camera not available", e)
            session = null
            toastAndFinish("Camera not available for ARCore")
            return
        }
        glSurfaceView.onResume()
    }

    override fun onPause() {
        super.onPause()
        if (session != null) {
            glSurfaceView.onPause()
            session?.pause()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Detach UI hooks; the service keeps training and reports via notification.
        TrainingService.progressListener = null
        TrainingService.doneListener = null
        saveExecutor.shutdown()
        session?.close()
        session = null
    }

    /** Pick the supported camera config with the largest CPU image resolution. */
    private fun selectHighestResolutionCameraConfig(session: Session) {
        try {
            val filter = CameraConfigFilter(session)
            val configs = session.getSupportedCameraConfigs(filter)
            var best: CameraConfig? = null
            var bestArea = 0
            for (cfg in configs) {
                val size = cfg.imageSize
                val area = size.width * size.height
                if (area > bestArea) {
                    bestArea = area
                    best = cfg
                }
            }
            if (best != null) {
                session.cameraConfig = best
                Log.i(TAG, "Selected camera CPU image ${best.imageSize.width}x${best.imageSize.height}")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Could not select high-res camera config: ${e.message}")
        }
    }

    // -------------------------------------------------------------------------
    // GLSurfaceView.Renderer
    // -------------------------------------------------------------------------

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0.04f, 0.05f, 0.08f, 1.0f)
        try {
            backgroundRenderer.createOnGlThread()
        } catch (e: Exception) {
            Log.e(TAG, "Background renderer init failed", e)
        }
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        viewportWidth = width
        viewportHeight = height
        viewportChanged = true
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        val session = this.session ?: return
        if (backgroundRenderer.textureId == -1) return

        try {
            session.setCameraTextureName(backgroundRenderer.textureId)
            if (viewportChanged) {
                val rotation = if (android.os.Build.VERSION.SDK_INT >= 30) {
                    display?.rotation ?: 0
                } else {
                    @Suppress("DEPRECATION") windowManager.defaultDisplay.rotation
                }
                session.setDisplayGeometry(rotation, viewportWidth, viewportHeight)
                viewportChanged = false
            }

            val frame = session.update()
            backgroundRenderer.draw(frame)

            val camera = frame.camera
            val tracking = camera.trackingState == TrackingState.TRACKING

            if (tracking && !haveIntrinsics) captureIntrinsics(frame)
            if (tracking && isRecording) maybeCaptureKeyframe(frame)

            updateStatusText(tracking)
        } catch (e: CameraNotAvailableException) {
            Log.e(TAG, "Camera not available during draw", e)
        } catch (e: Exception) {
            Log.e(TAG, "onDrawFrame error: ${e.message}")
        }
    }

    private fun captureIntrinsics(frame: Frame) {
        val intr = frame.camera.imageIntrinsics
        val f = intr.focalLength
        val pp = intr.principalPoint
        val dims = intr.imageDimensions
        fx = f[0]; fy = f[1]; cx = pp[0]; cy = pp[1]
        imgW = dims[0]; imgH = dims[1]
        haveIntrinsics = true
        Log.i(TAG, "Intrinsics: fx=$fx fy=$fy cx=$cx cy=$cy (${imgW}x${imgH})")
    }

    private fun maybeCaptureKeyframe(frame: Frame) {
        if (keptFrameCount >= maxKeyframes) {
            if (!frameLimitNotified) {
                frameLimitNotified = true
                Log.i(TAG, "Keyframe limit reached ($maxKeyframes); stopping capture")
                runOnUiThread {
                    tvStatus.text = "Enough coverage ($maxKeyframes frames) - tap Stop & Train"
                }
            }
            return
        }

        val pose = frame.camera.pose
        val pos = floatArrayOf(pose.tx(), pose.ty(), pose.tz())
        val quat = floatArrayOf(pose.qx(), pose.qy(), pose.qz(), pose.qw()) // [x, y, z, w]

        val prevPos = lastKeptPos
        val prevQuat = lastKeptQuat
        if (prevPos != null && prevQuat != null) {
            val dist = distance(pos, prevPos)
            val angle = quaternionAngle(quat, prevQuat)
            if (dist < MIN_TRANSLATION_M && angle < MIN_ROTATION_RAD) return
        }

        val image: Image = try {
            frame.acquireCameraImage()
        } catch (e: NotYetAvailableException) {
            return
        } catch (e: Exception) {
            Log.w(TAG, "acquireCameraImage failed: ${e.message}")
            return
        }

        val jpeg: ByteArray
        val meanLum: Float
        try {
            jpeg = YuvToJpeg.toJpeg(image)
            meanLum = YuvToJpeg.meanLuminance(image)
        } catch (e: Exception) {
            Log.w(TAG, "YUV->JPEG failed: ${e.message}")
            image.close()
            return
        } finally {
            image.close()
        }

        val poseMatrix = FloatArray(16)
        pose.toMatrix(poseMatrix, 0)

        // Copy the point cloud out before releasing the frame's resources.
        var pointsCopy: FloatArray? = null
        try {
            val pc = frame.acquirePointCloud()
            try {
                val buf = pc.points
                val remaining = buf.remaining()
                if (remaining > 0) {
                    pointsCopy = FloatArray(remaining)
                    buf.get(pointsCopy)
                }
            } finally {
                pc.release()
            }
        } catch (e: Exception) {
            Log.w(TAG, "acquirePointCloud failed: ${e.message}")
        }

        val timestamp = frame.timestamp
        lastKeptPos = pos
        lastKeptQuat = quat
        keptFrameCount++
        pendingSaves.incrementAndGet()

        val ptsForSave = pointsCopy
        saveExecutor.execute {
            try {
                datasetExporter.saveCapturedFrameJpeg(
                    jpegBytes = jpeg,
                    poseMatrix = poseMatrix,
                    timestampNs = timestamp,
                    sharpnessScore = 0f,
                    meanLuminance = meanLum,
                    pointsXyzc = ptsForSave
                )
            } catch (e: Exception) {
                Log.e(TAG, "Failed to save frame: ${e.message}")
            } finally {
                pendingSaves.decrementAndGet()
            }
        }
    }

    // -------------------------------------------------------------------------
    // Recording control
    // -------------------------------------------------------------------------

    private fun startRecordingSession() {
        if (!haveIntrinsics) {
            Toast.makeText(this, "Move the phone slightly to start tracking, then try again.", Toast.LENGTH_SHORT).show()
            return
        }
        datasetExporter = DatasetExporter(this, "scan_${System.currentTimeMillis() / 1000}")
        lastKeptPos = null
        lastKeptQuat = null
        keptFrameCount = 0
        frameLimitNotified = false
        pendingSaves.set(0)

        val profile = com.splat.mobile3dgs.hardware.DeviceCapabilityManager.getDeviceProfile(this)
        maxKeyframes = when (profile.tier) {
            com.splat.mobile3dgs.hardware.HardwareTier.TIER_1_FLAGSHIP -> 200
            com.splat.mobile3dgs.hardware.HardwareTier.TIER_2_BALANCED -> 120
            com.splat.mobile3dgs.hardware.HardwareTier.TIER_3_STANDALONE -> 70
        }
        val availGb = com.splat.mobile3dgs.hardware.DeviceCapabilityManager.getAvailableRamGb(this)
        Log.i(TAG, "Capture start: tier=${profile.tier.tierName} availRam=${"%.1f".format(availGb)}GB maxKeyframes=$maxKeyframes")

        isRecording = true

        btnRecord.text = "Stop & Train 3DGS"
        runOnUiThread { tvStatus.text = "Scanning — orbit slowly around the object..." }
    }

    private fun stopRecordingSession() {
        isRecording = false
        btnRecord.isEnabled = false
        btnRecord.text = "Processing..."
        progressBar.visibility = View.VISIBLE
        progressBar.isIndeterminate = true
        tvStatus.text = "Finalizing capture..."

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                var waitCycles = 0
                while (pendingSaves.get() > 0 && waitCycles < 50) {
                    delay(100); waitCycles++
                }

                val frameCount = datasetExporter.capturedFrames.size
                if (frameCount < 8) {
                    withContext(Dispatchers.Main) {
                        resetRecordUi()
                        tvStatus.text = "Not enough frames ($frameCount). Capture more coverage."
                        Toast.makeText(this@CaptureActivity, "Need at least ~8 keyframes. Try a slower, wider orbit.", Toast.LENGTH_LONG).show()
                    }
                    return@launch
                }

                withContext(Dispatchers.Main) {
                    tvStatus.text = "Writing dataset ($frameCount keyframes)..."
                }
                datasetExporter.exportDataset(fx, fy, cx, cy, imgW, imgH)

                val profile = com.splat.mobile3dgs.hardware.DeviceCapabilityManager.getDeviceProfile(this@CaptureActivity)
                val isTier3 = (profile.tier == com.splat.mobile3dgs.hardware.HardwareTier.TIER_3_STANDALONE)
                val isNativeVulkanReady = NativeBrushEngine.isNativeEngineAvailable()

                withContext(Dispatchers.Main) {
                    tvStatus.text = "Generating on-device 3D splats..."
                }

                val datasetDir = datasetExporter.outputDir
                val outputSplat = File(filesDir, "${datasetDir.name}.splat")

                // Generate standalone Direct Photometric Splat model (zero GPU/Vulkan dependency, runs in ~1.5s on CPU)
                val directSplatCount = com.splat.mobile3dgs.engine.GaussianInitializer.generatePhotometricSplatModel(
                    points = datasetExporter.accumulatedFeaturePoints,
                    datasetDir = datasetDir,
                    frames = datasetExporter.capturedFrames,
                    fx = fx, fy = fy, cx = cx, cy = cy,
                    origImgWidth = imgW, origImgHeight = imgH,
                    outputFile = outputSplat,
                    maxGaussians = profile.maxGaussians
                )
                Log.i(TAG, "Direct Photometric Model generated: $directSplatCount splats")

                val prefs = getSharedPreferences("Mobile3DGS_Prefs", Context.MODE_PRIVATE)
                val userExplicitSteps = prefs.contains("PREF_TRAINING_STEPS")
                val targetIterations = prefs.getInt("PREF_TRAINING_STEPS", profile.totalSteps)
                val targetResolution = prefs.getInt("PREF_TRAINING_RES", profile.maxResolution)

                // If native engine is NOT available, or user requested 0 steps (instant), or if this is Tier 3 without explicit steps:
                val shouldUseDirectModelImmediately = (!isNativeVulkanReady) || (targetIterations == 0) || (isTier3 && !userExplicitSteps)

                if (shouldUseDirectModelImmediately) {
                    withContext(Dispatchers.Main) {
                        progressBar.visibility = View.GONE
                        if (outputSplat.exists() && outputSplat.length() > 0) {
                            Toast.makeText(this@CaptureActivity, "🎉 3D Model Generated On-Device ($directSplatCount splats)!", Toast.LENGTH_LONG).show()
                            startActivity(Intent(this@CaptureActivity, ViewerActivity::class.java).apply {
                                putExtra("MODEL_NAME", "Standalone On-Device Scan")
                                putExtra("MODEL_PATH", outputSplat.absolutePath)
                            })
                            finish()
                        } else {
                            resetRecordUi()
                            tvStatus.text = "Capture finished, but insufficient surface points detected."
                            Toast.makeText(this@CaptureActivity, "Could not seed 3D points. Ensure well-lit surface.", Toast.LENGTH_LONG).show()
                        }
                    }
                    return@launch
                }

                // If Vulkan engine IS ready and user wants iterative refinement:
                Log.i(TAG, "Starting on-device training: tier=${profile.tier.tierName} steps=$targetIterations res=$targetResolution")
                val modelName = "On-Device Scan ($targetIterations steps)"

                // Hand the long-running optimization to a foreground service so it
                // survives the screen locking or the user leaving the app.
                withContext(Dispatchers.Main) {
                    tvStatus.text = "Training started - safe to lock the screen"
                    progressBar.isIndeterminate = false
                    progressBar.progress = 0

                    TrainingService.progressListener = { step, pct ->
                        runOnUiThread {
                            tvStatus.text = "Training: step $step / $targetIterations ($pct%)"
                            progressBar.progress = pct
                        }
                    }
                    TrainingService.doneListener = { ok, path ->
                        runOnUiThread {
                            progressBar.visibility = View.GONE
                            if (ok && File(path).exists() && File(path).length() > 0) {
                                Toast.makeText(this@CaptureActivity, "3D model refined on-device!", Toast.LENGTH_LONG).show()
                                startActivity(Intent(this@CaptureActivity, ViewerActivity::class.java).apply {
                                    putExtra("MODEL_NAME", modelName)
                                    putExtra("MODEL_PATH", path)
                                })
                                finish()
                            } else {
                                // Graceful fallback: Open the direct photometric splat model if training didn't produce a new one
                                if (outputSplat.exists() && outputSplat.length() > 0) {
                                    Toast.makeText(this@CaptureActivity, "Opening direct on-device 3D model!", Toast.LENGTH_SHORT).show()
                                    startActivity(Intent(this@CaptureActivity, ViewerActivity::class.java).apply {
                                        putExtra("MODEL_NAME", "Standalone 3D Model ($directSplatCount splats)")
                                        putExtra("MODEL_PATH", outputSplat.absolutePath)
                                    })
                                    finish()
                                } else {
                                    resetRecordUi()
                                    tvStatus.text = "Training finished (no output produced)"
                                }
                            }
                        }
                    }

                    TrainingService.start(
                        context = this@CaptureActivity,
                        datasetPath = datasetDir.absolutePath,
                        outputPath = outputSplat.absolutePath,
                        steps = targetIterations,
                        resolution = targetResolution,
                        modelName = modelName
                    )
                }
            } catch (e: Throwable) {
                Log.e(TAG, "Scan pipeline error", e)
                withContext(Dispatchers.Main) {
                    resetRecordUi()
                    tvStatus.text = "Error: ${e.message}"
                }
            }
        }
    }

    private fun resetRecordUi() {
        progressBar.visibility = View.GONE
        btnRecord.isEnabled = true
        btnRecord.text = "Start 3D Scan"
    }

    private fun updateStatusText(tracking: Boolean) {
        if (isRecording) {
            runOnUiThread { tvFrameCount.text = "$keptFrameCount Frames" }
        } else if (!tracking) {
            runOnUiThread { tvStatus.text = "Move phone slowly to start tracking..." }
        }
    }

    // -------------------------------------------------------------------------
    // Math helpers
    // -------------------------------------------------------------------------

    private fun distance(a: FloatArray, b: FloatArray): Float {
        val dx = a[0] - b[0]; val dy = a[1] - b[1]; val dz = a[2] - b[2]
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    /** Angle (radians) between two orientation quaternions [x, y, z, w]. */
    private fun quaternionAngle(q1: FloatArray, q2: FloatArray): Float {
        var dot = q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]
        dot = abs(dot)
        if (dot > 1.0f) dot = 1.0f
        return 2.0f * acos(min(1.0f, dot))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun toastAndFinish(msg: String) {
        Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
        finish()
    }
}
