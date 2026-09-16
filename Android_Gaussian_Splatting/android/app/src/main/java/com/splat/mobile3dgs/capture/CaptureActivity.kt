package com.splat.mobile3dgs.capture

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.media.Image
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Build
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
import com.google.ar.core.Anchor
import com.google.ar.core.ArCoreApk
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Frame
import com.google.ar.core.Pose
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
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
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
    private val keptPositions = ArrayList<FloatArray>()
    private val keptQuats = ArrayList<FloatArray>()

    /**
     * Sharpness / exposure gate. A motion-blurred keyframe is worse than no
     * keyframe at all: the optimizer treats its smeared pixels as ground truth
     * and bakes the smear permanently into the Gaussians.
     */
    private val frameQualityFilter = FrameQualityFilter()
    @Volatile private var lastQualityHintMs = 0L
    @Volatile private var qualityHintShowing = false

    /** OpenCV [k1, k2, p1, p2] read once from Camera2, or null on a device that reports none. */
    @Volatile private var lensDistortion: FloatArray? = null

    /**
     * ARCore keeps re-refining [Anchor]s as it learns the space, while a raw VIO
     * pose is frozen the moment it is read. Each keyframe is therefore bound to a
     * nearby anchor and recomposed against that anchor's corrected pose at export
     * time, which absorbs the drift that accumulates over a long walk.
     */
    private val sceneAnchors = mutableListOf<Anchor>()
    private var currentAnchor: Anchor? = null
    private var currentAnchorPos: FloatArray? = null
    private var keyframesSinceAnchor = 0
    @Volatile private var anchorsAvailable = false

    /**
     * Rolling record of whether each recent keyframe added real translation.
     *
     * Turning in place produces NO parallax, so those frames carry no depth
     * information at all -- a scan that is half rotation reconstructs as clouds.
     * This was previously only measured after the fact, which is too late for the
     * user to do anything about it.
     */
    private val recentMoved = ArrayDeque<Boolean>()
    @Volatile private var orbitHint: String? = null

    /** Last detailed engine result, so a failure dialog can name the actual cause. */
    @Volatile private var lastTrainingResult: com.splat.mobile3dgs.engine.TrainingResult? = null
    @Volatile private var keptFrameCount = 0

    /**
     * Upper bound on keyframes, chosen from the hardware tier. Training memory is
     * dominated by (frames x resolution), and Android's low-memory killer reacts to
     * free RAM rather than total RAM -- so a budget device must stop collecting
     * before the dataset outgrows what is actually available.
     */
    private var maxKeyframes = 120
    @Volatile private var frameLimitNotified = false
    @Volatile private var depthLogged = false
    @Volatile private var depthEnabled = false
    // Live extent of the camera path, so the user can be told to actually move.
    private var pMinX = Float.MAX_VALUE; private var pMaxX = -Float.MAX_VALUE
    private var pMinY = Float.MAX_VALUE; private var pMaxY = -Float.MAX_VALUE
    private var pMinZ = Float.MAX_VALUE; private var pMaxZ = -Float.MAX_VALUE
    /** Running estimate of subject distance, from the depth map's central median. */
    @Volatile private var subjectDistEstimate = 0f
    @Volatile private var sessionResumed = false
    @Volatile private var trainingActive = false

    // Display geometry sync (activity is locked portrait).
    private var viewportWidth = 0
    private var viewportHeight = 0
    @Volatile private var viewportChanged = false

    companion object {
        /** ARCore permits one Session per process; keep it here so retries reuse it. */
        @Volatile private var sharedSession: Session? = null

        private const val PREF_ARCORE_TIER = "PREF_ARCORE_TIER"
        private const val PREF_ARCORE_TIER_OK = "PREF_ARCORE_TIER_OK"

        /** Compatibility ladder: (name, max CPU image width, request depth). */
        private val ARCORE_TIERS = listOf(
            Triple("CPU<=1920 + depth", 1920, true),
            Triple("CPU<=1280 + depth", 1280, true),
            Triple("default camera config + depth", 0, true),
            Triple("default camera config, no depth", 0, false)
        )

        private const val TAG = "CaptureActivity"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private const val REQUEST_CODE_NOTIFICATIONS = 11
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)

        /** Minimum gap between on-screen capture-quality hints, so they stay readable. */
        private const val QUALITY_HINT_INTERVAL_MS = 1500L

        /** Drop a fresh anchor at least this often along the walk. */
        /** Consecutive keyframes closer than this added rotation but no baseline. */
        /**
         * Share of TOTAL RAM the capture may size itself against. 0.3 of a 12 GB
         * phone is ~3.6 GB, within the 5-6 GB the device owner authorised, and well
         * clear of what the engine actually needs (~750 MB RSS measured).
         */
        private const val MEMORY_BUDGET_FRACTION = 0.30f

        /**
         * Crop the seed cloud to the subject? Off: it makes the object sharper but
         * leaves the surroundings with no seed geometry at all.
         */
        private const val ISOLATE_SUBJECT = false

        /**
         * Below this the ARCore depth API stops returning anything: it reported
         * zero for all 14400 samples on a scan taken at 0.25 m, leaving the seed
         * cloud with a tenth of its usual geometry. Depth, not framing, sets the
         * lower bound on how close a scan can be.
         */
        // Measured: 0.25 m returned zero for all 14400 depth samples, while 0.37 m
        // kept 14270 of 14400. The cliff is below ~0.3 m, so warning at 0.45 m
        // nagged about scans that were working fine.
        private const val TOO_CLOSE_M = 0.30f

        /** Past this the subject is too small in frame to resolve detail. */
        private const val TOO_FAR_M = 3.0f

        private const val ROTATION_ONLY_STEP_M = 0.02f

        /** Frames kept in the rolling orbit-quality window. */
        private const val ORBIT_WINDOW = 20

        /** Warn once this share of the recent window is rotation-only. */
        private const val ORBIT_WARN_FRACTION = 0.40f

        private const val ANCHOR_SPACING_M = 0.5f
        private const val ANCHOR_KEYFRAME_INTERVAL = 15

        /** ARCore tracks anchors at a real cost; a scan never needs more. */
        private const val MAX_ANCHORS = 32

        /** Correction runs on the GL thread; never block export on it for long. */
        private const val ANCHOR_RESOLVE_TIMEOUT_MS = 2500L

        private const val SCANNING_STATUS = "Scanning — orbit slowly around the object..."

        // A candidate is redundant if some kept view is within BOTH of these.
        // Views must be separated by an ANGLE about the subject, not a fixed
        // distance. A flat 6 cm is a 13 degree arc when orbiting an object at 26 cm
        // (so a full circle allowed only ~27 frames and most candidates were
        // discarded) while being almost nothing for a subject 3 m away.
        private const val VIEW_SEPARATION_RAD_ABOUT_SUBJECT = 0.07f  // ~4 degrees of arc
        private const val MIN_VIEW_SEPARATION_FLOOR_M = 0.015f
        private const val MIN_VIEW_SEPARATION_CEIL_M = 0.30f
        private const val MIN_VIEW_SEPARATION_RAD = 0.105f  // ~6 degrees of camera turn
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

    override fun onPause() {
        super.onPause()
        // ARCore holds the camera until the session is paused. Leaving it running
        // in the background drains the battery and, worse, makes the NEXT resume
        // fail with the camera already in use -- which previously looked like an
        // ARCore incompatibility. Training deliberately keeps running: it is a
        // foreground service and no longer needs the camera.
        if (!trainingActive) releaseCaptureResources()
    }

    override fun onDestroy() {
        super.onDestroy()
        // These listeners live in a companion object, so a stale lambda would keep
        // this Activity (and its GL surface) alive for the life of the process.
        // Training that outlives this screen reports through its notification.
        TrainingService.progressListener = null
        TrainingService.doneListener = null
        TrainingService.resultListener = null
        // The ARCore Session is deliberately NOT closed here. It is a process-wide
        // singleton (ARCore permits exactly one) that later launches reuse, and
        // closing one whose camera never resumed aborts the process with an
        // uncatchable SIGABRT. Pausing is enough to free the camera.
        if (isFinishing && !trainingActive) releaseCaptureResources()
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
        // Do not restart the camera while training is running.
        if (trainingActive) return

        if (session == null) {
            // Play Services for AR can be installed on ANY device, but ARCore only
            // functions on Google-certified hardware. requestInstall() reports
            // INSTALLED regardless, so an uncertified device otherwise surfaces as a
            // confusing camera failure. Check capability explicitly first.
            val availability = try {
                ArCoreApk.getInstance().checkAvailability(this)
            } catch (t: Throwable) {
                Log.w(TAG, "checkAvailability failed: ${t.message}")
                null
            }
            Log.i(TAG, "ARCore availability = $availability")
            if (availability == ArCoreApk.Availability.UNSUPPORTED_DEVICE_NOT_CAPABLE) {
                AlertDialog.Builder(this)
                    .setTitle("3D scanning not supported")
                    .setMessage(
                        "This device is not ARCore-certified, so live 3D capture is " +
                        "unavailable. You can still open and view existing 3D models."
                    )
                    .setPositiveButton("OK") { _, _ -> finish() }
                    .setOnDismissListener { finish() }
                    .show()
                return
            }

            try {
                when (ArCoreApk.getInstance().requestInstall(this, userRequestedInstall)) {
                    ArCoreApk.InstallStatus.INSTALL_REQUESTED -> {
                        userRequestedInstall = false
                        return
                    }
                    ArCoreApk.InstallStatus.INSTALLED -> { /* proceed */ }
                    else -> {}
                }
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
            } catch (e: Throwable) {
                Log.e(TAG, "ARCore availability check failed", e)
                toastAndFinish("ARCore init failed: ${e.localizedMessage}")
                return
            }

            if (!startSessionWithFallback()) {
                toastAndFinish("ARCore could not start the camera on this device")
                return
            }
        } else {
            try {
                session?.resume()
                sessionResumed = true
            } catch (t: Throwable) {
                // Do not close the session here; retry configurations on it instead.
                Log.w(TAG, "resume() failed on existing session: ${t.javaClass.simpleName}", t)
                try { session?.pause() } catch (ignored: Throwable) { }
                if (!startSessionWithFallback()) {
                    toastAndFinish("ARCore could not restart the camera")
                    return
                }
            }
        }
        glSurfaceView.onResume()
        // releaseCaptureResources() drops the surface to RENDERMODE_WHEN_DIRTY, and
        // nothing ever marks it dirty -- so without restoring this, onDrawFrame
        // never runs again and the preview stays black with no keyframes. Reached
        // on every permission dialog, since that pauses the Activity.
        glSurfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
    }

    /**
     * Start ARCore, degrading the configuration until one actually works.
     *
     * Two hard-won constraints shape this:
     *  1. ARCore allows only ONE Session per process, so every candidate is tried
     *     by reconfiguring the SAME session rather than creating a new one.
     *  2. Calling close() on a session whose camera failed to start aborts the
     *     whole process inside ArSession_destroy ("Client must stop camera before
     *     attempting to block until stopped") - an uncatchable SIGABRT. So a
     *     failed candidate is never closed, only paused.
     *
     * Budget devices commonly cannot run a 1080p CPU image, a GPU texture stream
     * and depth at once, so CPU image size is reduced before depth is given up.
     */
    /** Stop the camera, VIO and GL rendering so training gets the GPU to itself. */
    private fun releaseCaptureResources() {
        try {
            glSurfaceView.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY
            glSurfaceView.onPause()
        } catch (t: Throwable) {
            Log.w(TAG, "Could not pause GL surface: ${t.message}")
        }
        if (sessionResumed) {
            try { session?.pause() } catch (t: Throwable) { Log.w(TAG, "session pause: ${t.message}") }
            sessionResumed = false
        }
        Log.i(TAG, "Capture pipeline released for training (camera + VIO + GL stopped)")
    }

    private fun startSessionWithFallback(): Boolean {
        val s = try {
            sharedSession ?: Session(this).also { sharedSession = it }
        } catch (t: Throwable) {
            Log.e(TAG, "Could not create ARCore session", t)
            return false
        }
        session = s

        // A failed resume() leaves ARCore's camera running, and every later
        // configure()/resume() on that session then fails with "Client must stop
        // camera before attempting to block until stopped". Retrying in-process is
        // therefore useless, so we make exactly ONE attempt per launch and walk a
        // compatibility ladder across launches instead, remembering where we got to.
        val prefs = getSharedPreferences("Mobile3DGS_Prefs", Context.MODE_PRIVATE)
        val tier = prefs.getInt(PREF_ARCORE_TIER, 0).coerceIn(0, ARCORE_TIERS.lastIndex)
        val (name, maxWidth, wantDepth) = ARCORE_TIERS[tier]

        try {
            if (maxWidth > 0) selectCameraConfig(s, maxWidth)

            val cfg = Config(s)
            cfg.updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
            cfg.focusMode = Config.FocusMode.AUTO
            cfg.planeFindingMode = Config.PlaneFindingMode.DISABLED
            cfg.depthMode =
                if (wantDepth && s.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
                    Config.DepthMode.AUTOMATIC
                } else {
                    Config.DepthMode.DISABLED
                }
            s.configure(cfg)
            s.resume()

            sessionResumed = true
            depthEnabled = cfg.depthMode == Config.DepthMode.AUTOMATIC
            prefs.edit().putInt(PREF_ARCORE_TIER, tier).putBoolean(PREF_ARCORE_TIER_OK, true).apply()
            Log.i(TAG, "ARCore session started: tier $tier '$name' (depthEnabled=$depthEnabled)")
            return true
        } catch (t: Throwable) {
            Log.w(TAG, "ARCore tier $tier '$name' rejected: ${t.javaClass.simpleName}: ${t.message}")
            // Only walk the compatibility ladder while we have never succeeded; once a
            // tier is known good, a failure is transient and must not permanently
            // downgrade capture quality.
            val everWorked = prefs.getBoolean(PREF_ARCORE_TIER_OK, false)
            if (!everWorked && tier < ARCORE_TIERS.lastIndex) {
                prefs.edit().putInt(PREF_ARCORE_TIER, tier + 1).apply()
                Log.i(TAG, "Will try tier ${tier + 1} on next launch")
            }
            return false
        }
    }

    /** Pick the largest supported CPU image at or below [maxWidth]. */
    private fun selectCameraConfig(session: Session, maxWidth: Int) {
        try {
            val filter = CameraConfigFilter(session)
            val configs = session.getSupportedCameraConfigs(filter)
            var best: CameraConfig? = null
            var bestArea = 0
            for (cfg in configs) {
                val size = cfg.imageSize
                val area = size.width * size.height
                if (size.width <= maxWidth && area > bestArea) {
                    bestArea = area
                    best = cfg
                }
            }
            if (best != null) {
                session.cameraConfig = best
                Log.i(TAG, "Camera config -> CPU ${best.imageSize.width}x${best.imageSize.height}")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "Camera config selection failed: ${t.message}")
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
        // Never touch the camera or ARCore while the optimizer owns the GPU.
        if (trainingActive) return
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
        lensDistortion = readLensDistortion()
    }

    /**
     * Read the lens distortion the camera reports, mapped into OpenCV's convention.
     *
     * ARCore's `imageIntrinsics` is a pure pinhole model and the CPU image it hands
     * out is NOT undistorted, so a dataset that declares `camera_model: "OPENCV"`
     * without coefficients tells the trainer the lens is perfect and bakes a
     * systematic, radially growing error into everything near the frame border.
     *
     * Android's `LENS_DISTORTION` is defined in NORMALISED camera coordinates:
     *   x_c = x(1 + k1 r^2 + k2 r^4 + k3 r^6) + kappa_4(2xy) + kappa_5(r^2 + 2x^2)
     * and OpenCV's is
     *   x_d = x(1 + k1 r^2 + k2 r^4 + k3 r^6) + 2 p1 x y + p2(r^2 + 2x^2)
     * so k1..k3 map straight across and p1 = kappa_4, p2 = kappa_5. Being
     * normalised, the coefficients survive whatever crop/scale ARCore applies
     * between the sensor array and the CPU image, provided the reported focal
     * length and principal point are used with them -- they are, both come from
     * `imageIntrinsics` above.
     */
    private fun readLensDistortion(): FloatArray? {
        // LENS_DISTORTION was added in API 28; older devices report nothing.
        if (Build.VERSION.SDK_INT < 28) return null
        return try {
            val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val cameraId = sharedSession?.cameraConfig?.cameraId
                ?: manager.cameraIdList.firstOrNull { id ->
                    manager.getCameraCharacteristics(id).get(CameraCharacteristics.LENS_FACING) ==
                            CameraMetadata.LENS_FACING_BACK
                }
                ?: return null
            val d = manager.getCameraCharacteristics(cameraId)
                .get(CameraCharacteristics.LENS_DISTORTION)
            if (d == null || d.size < 5 || !d.all { it.isFinite() }) {
                Log.i(TAG, "Camera $cameraId reports no LENS_DISTORTION; using pinhole model")
                return null
            }
            floatArrayOf(d[0], d[1], d[3], d[4]).also {
                Log.i(TAG, "Lens distortion (camera $cameraId): k1=${it[0]} k2=${it[1]} " +
                        "p1=${it[2]} p2=${it[3]} (k3=${d[2]} dropped, OPENCV uses 4)")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "Could not read lens distortion: ${t.message}")
            null
        }
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

        // Keep a frame only if it is a genuinely NEW viewpoint compared with every
        // frame already kept -- not merely different from the previous one. Comparing
        // against only the last frame let a slow sweep back over the same arc add
        // hundreds of near-duplicate views (179 frames in ~10s, 52% of them pure
        // turns), which inflated training time without adding any information.
        // Arc length that corresponds to a few degrees about the subject.
        val sepM = if (subjectDistEstimate > 0.05f) {
            (subjectDistEstimate * VIEW_SEPARATION_RAD_ABOUT_SUBJECT)
                .coerceIn(MIN_VIEW_SEPARATION_FLOOR_M, MIN_VIEW_SEPARATION_CEIL_M)
        } else {
            MIN_VIEW_SEPARATION_FLOOR_M
        }
        for (i in keptPositions.indices) {
            val d = distance(pos, keptPositions[i])
            val a = quaternionAngle(quat, keptQuats[i])
            if (d < sepM && a < MIN_VIEW_SEPARATION_RAD) return
        }

        val image: Image = try {
            frame.acquireCameraImage()
        } catch (e: NotYetAvailableException) {
            return
        } catch (e: Exception) {
            Log.w(TAG, "acquireCameraImage failed: ${e.message}")
            return
        }

        // Reject blurred / badly exposed frames BEFORE paying for JPEG encoding
        // and depth unprojection, and before they are recorded as covered
        // viewpoints -- otherwise a blurred frame both costs training time and
        // blocks the sharp frame that follows it from the same angle.
        val quality = frameQualityFilter.evaluateFrameQuality(image)
        if (!quality.isPassed) {
            image.close()
            val hint = when (quality.verdict) {
                FrameQualityFilter.Verdict.MOTION_BLUR -> "Hold steadier — move more slowly"
                FrameQualityFilter.Verdict.UNDEREXPOSED -> "Too dark — add more light"
                FrameQualityFilter.Verdict.OVEREXPOSED -> "Too bright — avoid direct glare"
                else -> null
            }
            val now = System.currentTimeMillis()
            if (hint != null && now - lastQualityHintMs > QUALITY_HINT_INTERVAL_MS) {
                lastQualityHintMs = now
                qualityHintShowing = true
                runOnUiThread { if (isRecording) tvStatus.text = hint }
            }
            return
        }
        // Frames are usable again -- clear a stale hint rather than leaving
        // "move more slowly" on screen for the rest of the scan.
        if (qualityHintShowing) {
            qualityHintShowing = false
            runOnUiThread { if (isRecording) tvStatus.text = SCANNING_STATUS }
        }

        val jpeg: ByteArray
        // Luminance already measured by the gate above; no second full-plane pass.
        val meanLum: Float = quality.meanLuminance
        try {
            jpeg = YuvToJpeg.toJpeg(image)
        } catch (e: Exception) {
            Log.w(TAG, "YUV->JPEG failed: ${e.message}")
            image.close()
            return
        }
        // NOTE: `image` deliberately stays open past this point so seed colours can
        // be sampled from it during depth extraction; it is closed in the finally below.

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

        // Dense depth seeding: unproject the ARCore depth map into world-space
        // points. Feature points alone give only a few hundred seeds, which is
        // far too sparse for the optimizer to start from.
        var depthPts: FloatArray? = null
        var depthImage: Image? = null
        var confImage: Image? = null
        var usedRawDepth = false
        try {
            // Prefer the SMOOTHED depth map: it is inpainted and dense, whereas raw
            // depth only returns high-confidence pixels and leaves most of the map
            // empty (~98% of samples were being discarded), which starves seeding.
            depthImage = try {
                frame.acquireDepthImage16Bits()
            } catch (e: NotYetAvailableException) {
                null
            } catch (e: Throwable) {
                try { frame.acquireRawDepthImage16Bits().also { usedRawDepth = true } }
                catch (e2: Throwable) { null }
            }
            if (depthImage != null) {
                // The confidence map describes RAW depth. The smoothed map is already
                // inpainted and valid everywhere, so masking it with raw confidence
                // discards almost every sample.
                if (usedRawDepth) {
                    confImage = try { frame.acquireRawDepthConfidenceImage() } catch (e: Throwable) { null }
                    if (confImage != null &&
                        (confImage.width != depthImage.width || confImage.height != depthImage.height)) {
                        confImage.close()
                        confImage = null
                    }
                }
                val intr = frame.camera.imageIntrinsics
                depthPts = DepthPointExtractor.extractWorldPoints(
                    depthImage = depthImage,
                    confidenceImage = confImage,
                    focal = intr.focalLength,
                    principal = intr.principalPoint,
                    imageDims = intr.imageDimensions,
                    cameraPose = pose,
                    colorImage = image
                )
                if (DepthPointExtractor.lastSubjectDepth > 0.05f) {
                    subjectDistEstimate = if (subjectDistEstimate <= 0f) {
                        DepthPointExtractor.lastSubjectDepth
                    } else {
                        subjectDistEstimate * 0.8f + DepthPointExtractor.lastSubjectDepth * 0.2f
                    }
                }
                if (!depthLogged) {
                    depthLogged = true
                    Log.i(TAG, "Depth seeding active: ${depthImage.width}x${depthImage.height} " +
                            "raw=$usedRawDepth conf=${confImage != null} -> " +
                            "${(depthPts?.size ?: 0) / 8} points/frame [${DepthPointExtractor.lastStats()}]")
                }
            } else if (!depthLogged) {
                depthLogged = true
                Log.w(TAG, "Depth image unavailable; falling back to sparse feature points only")
            }
        } catch (e: Throwable) {
            Log.w(TAG, "Depth extraction failed: ${e.message}")
        } finally {
            depthImage?.close()
            confImage?.close()
            image.close()
        }

        // Did this keyframe add real baseline, or only turn the phone? Measured the
        // same way computeQuality() scores rotationOnly, but live.
        lastKeptPos?.let { prev ->
            recentMoved.addLast(distance(pos, prev) >= ROTATION_ONLY_STEP_M)
            while (recentMoved.size > ORBIT_WINDOW) recentMoved.removeFirst()
            if (recentMoved.size >= 8) {
                val still = recentMoved.count { !it }
                val d = subjectDistEstimate
                // Distance first: too close silently kills depth seeding, which
                // costs far more than a rotation-heavy path.
                orbitHint = when {
                    d > 0.05f && d < TOO_CLOSE_M ->
                        "Too close (${(d * 100).toInt()}cm) — step back to about 60cm"
                    d > TOO_FAR_M ->
                        "Too far (${"%.1f".format(d)}m) — move closer for detail"
                    still.toFloat() / recentMoved.size > ORBIT_WARN_FRACTION ->
                        "Walk AROUND the object — turning in place adds no depth"
                    else -> null
                }
            }
        }

        val timestamp = frame.timestamp
        lastKeptPos = pos
        lastKeptQuat = quat
        keptPositions.add(pos)
        keptQuats.add(quat)
        keptFrameCount++
        if (pos[0] < pMinX) pMinX = pos[0]; if (pos[0] > pMaxX) pMaxX = pos[0]
        if (pos[1] < pMinY) pMinY = pos[1]; if (pos[1] > pMaxY) pMaxY = pos[1]
        if (pos[2] < pMinZ) pMinZ = pos[2]; if (pos[2] > pMaxZ) pMaxZ = pos[2]
        pendingSaves.incrementAndGet()

        val ptsForSave = pointsCopy
        val depthForSave = depthPts
        // Bind to a nearby anchor on the GL thread (anchors belong to the session)
        // so the pose can inherit ARCore's later drift corrections at export time.
        val anchorRef = anchorRefFor(pose)
        val frameIndex = datasetExporter.capturedFrames.size
        val blurScore = quality.blurScore
        saveExecutor.execute {
            try {
                depthForSave?.let { datasetExporter.addDepthPoints(it, frameIndex) }
                datasetExporter.saveCapturedFrameJpeg(
                    jpegBytes = jpeg,
                    poseMatrix = poseMatrix,
                    timestampNs = timestamp,
                    sharpnessScore = blurScore,
                    meanLuminance = meanLum,
                    pointsXyzc = ptsForSave,
                    anchorRef = anchorRef,
                    capturePose = pose
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
        keptPositions.clear()
        keptQuats.clear()
        frameLimitNotified = false
        pMinX = Float.MAX_VALUE; pMaxX = -Float.MAX_VALUE
        pMinY = Float.MAX_VALUE; pMaxY = -Float.MAX_VALUE
        pMinZ = Float.MAX_VALUE; pMaxZ = -Float.MAX_VALUE
        pendingSaves.set(0)
        frameQualityFilter.reset()
        recentMoved.clear()
        orbitHint = null
        lastQualityHintMs = 0L
        qualityHintShowing = false

        val profile = com.splat.mobile3dgs.hardware.DeviceCapabilityManager.getDeviceProfile(this)
        // Cap on FREE RAM rather than the SoC label: training memory is driven by
        // (frames x resolution), and a mid-range chip with memory to spare should not
        // be limited to a handful of frames just because of its model number. An
        // explicit quality choice raises the ceiling further.
        val availGb = com.splat.mobile3dgs.hardware.DeviceCapabilityManager.getAvailableRamGb(this)
        val totalGb = com.splat.mobile3dgs.hardware.DeviceCapabilityManager.getTotalRamGb(this)
        val prefsK = getSharedPreferences("Mobile3DGS_Prefs", Context.MODE_PRIVATE)
        val userChoseQuality = prefsK.contains("PREF_TRAINING_STEPS")

        // availMem reports what is free RIGHT NOW, which on a healthy phone is small
        // because Android deliberately spends RAM on page cache -- it reclaims that
        // on demand. Sizing purely on it punished a 12 GB device for being well used
        // (measured: 1.9 GB free of 11 GB -> only 113 keyframes). Frame coverage is
        // the main thing constraining reconstruction quality, so budget against a
        // share of TOTAL memory and treat availMem as a floor, not a ceiling.
        val budgetGb = maxOf(availGb, totalGb * MEMORY_BUDGET_FRACTION)
        maxKeyframes = (budgetGb * 60f).toInt().coerceIn(60, if (userChoseQuality) 250 else 150)
        Log.i(TAG, "Capture start: tier=${profile.tier.tierName} " +
                "ram=${"%.1f".format(availGb)}GB free of ${"%.1f".format(totalGb)}GB " +
                "budget=${"%.1f".format(budgetGb)}GB maxKeyframes=$maxKeyframes " +
                "userChoseQuality=$userChoseQuality")

        isRecording = true

        btnRecord.text = "Stop & Train 3DGS"
        runOnUiThread { tvStatus.text = SCANNING_STATUS }
    }

    private fun stopRecordingSession() {
        isRecording = false
        Log.i(TAG, "Frame quality: ${frameQualityFilter.summary()}")
        // Logged once per scan: the first-frame-only diagnostic hid that depth was
        // failing for the WHOLE run, not just while ARCore warmed up.
        Log.i(TAG, "Depth yield (last frame): ${DepthPointExtractor.lastStats()} " +
                "subjectDist=${"%.2f".format(subjectDistEstimate)}m " +
                "seedPoints=${datasetExporter.depthPointCount()}")
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

                // Inherit ARCore's post-hoc drift corrections BEFORE anything reads
                // the poses -- subject isolation, the quality metrics and the export
                // all consume transformMatrix.
                val anchorCorrected = resolveAnchorPoses()
                Log.i(TAG, "Anchor pose correction applied to $anchorCorrected frames")

                // Spend the seed budget on the subject, not the room behind it.
                // Subject isolation deliberately DELETES every seed outside a
                // sphere around the object, which sharpens the object at the cost
                // of starving the surroundings. The goal here is a scene that is
                // clear everywhere, and the seed budget is now large enough to
                // cover both, so it is left off rather than cropping the scene.
                if (ISOLATE_SUBJECT) datasetExporter.isolateSubject(subjectDistEstimate)

                val quality = datasetExporter.computeQuality()
                Log.i(TAG, "Capture quality: frames=${quality.frames} baseline=" +
                        "${"%.2f".format(quality.baselineM)}m depth=${"%.2f".format(quality.medianDepthM)}m " +
                        "ratio=${"%.3f".format(quality.parallaxRatio)} tri=${"%.1f".format(quality.triangulationDeg)}deg " +
                        "rotationOnly=${quality.rotationOnlyPct}%")

                // 3DGS recovers depth from parallax. If the camera barely moved relative
                // to how far away the scene is, depth is unrecoverable and extra
                // iterations only overfit -- so say so BEFORE spending 20 minutes.
                if (quality.isDegenerate && quality.frames >= 2) {
                    val proceed = kotlinx.coroutines.CompletableDeferred<Boolean>()
                    withContext(Dispatchers.Main) {
                        AlertDialog.Builder(this@CaptureActivity)
                            .setTitle("Not enough camera movement")
                            .setMessage(
                                "The camera only moved " + "%.0f".format(quality.baselineM * 100) +
                                " cm while the scene is about " + "%.1f".format(quality.medianDepthM) +
                                " m away (about " + "%.1f".format(quality.triangulationDeg) +
                                "° of parallax; 3D needs roughly 10-15°)." + "\n\n" +
                                quality.rotationOnlyPct + "% of frames were turns rather than steps." +
                                "\n\n" +
                                "Training will still run, but depth cannot be recovered from this and " +
                                "the result will look smeared. Walk around the subject instead of " +
                                "turning on the spot."
                            )
                            .setPositiveButton("Scan again") { _, _ -> proceed.complete(false) }
                            .setNegativeButton("Train anyway") { _, _ -> proceed.complete(true) }
                            .setCancelable(false)
                            .show()
                    }
                    if (!proceed.await()) {
                        withContext(Dispatchers.Main) {
                            resetRecordUi()
                            tvStatus.text = "Walk around the subject, keeping it centred"
                        }
                        return@launch
                    }
                }

                val depthSeeds = datasetExporter.depthPointCount()
                withContext(Dispatchers.Main) {
                    tvStatus.text = "Writing dataset ($frameCount keyframes, $depthSeeds depth points)..."
                }
                datasetExporter.exportDataset(fx, fy, cx, cy, imgW, imgH, lensDistortion)

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
                    // Sparse features PLUS the dense depth cloud. Features alone gave
                    // ~150 splats from a 60k-point scan; the preview is what the user
                    // sees if training fails, so it must actually resemble the scene.
                    points = datasetExporter.accumulatedFeaturePoints + datasetExporter.depthPointsAsFeatures(),
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
                // Only skip real training when the engine is genuinely unavailable or
                // the user explicitly asked for the instant photometric model. Skipping
                // it because of the SoC tier silently returned a few hundred splats that
                // look like coloured dots, with no indication training never ran.
                val shouldUseDirectModelImmediately = (!isNativeVulkanReady) || (targetIterations == 0)

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
                    // The engine only ever returns exit code 0/1; TrainingService
                    // attributes that to a phase and attaches whatever the engine
                    // printed to stderr. Keep it so the dialog below can say WHY
                    // rather than "could not run on this device".
                    TrainingService.resultListener = { r -> lastTrainingResult = r }
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
                                // Training failed. The photometric placeholder was written
                                // BEFORE training, so it always exists -- silently opening it
                                // presents a few hundred coloured dots as if it were a result.
                                // Say plainly that training failed and let the user decide.
                                if (outputSplat.exists() && outputSplat.length() > 0) {
                                    resetRecordUi()
                                    tvStatus.text = "Training failed on this device"
                                    AlertDialog.Builder(this@CaptureActivity)
                                        .setTitle("Training failed")
                                        .setMessage(
                                            buildString {
                                                append("The GPU optimizer could not run on this device, so no ")
                                                append("reconstructed model was produced. Your capture is saved ")
                                                append("and can be re-trained later from the model list. ")
                                                append("A rough $directSplatCount-point preview exists, but it is ")
                                                append("NOT a trained 3D model and will look like scattered dots.")
                                                lastTrainingResult?.let { r ->
                                                    appendLine()
                                                    appendLine()
                                                    appendLine("Reason (${r.errorCode}):")
                                                    append(r.message)
                                                }
                                            }
                                        )
                                        .setPositiveButton("View rough preview") { _, _ ->
                                            startActivity(Intent(this@CaptureActivity, ViewerActivity::class.java).apply {
                                                putExtra("MODEL_NAME", "UNTRAINED preview ($directSplatCount points)")
                                                putExtra("MODEL_PATH", outputSplat.absolutePath)
                                            })
                                            finish()
                                        }
                                        .setNegativeButton("Back", null)
                                        .show()
                                } else {
                                    resetRecordUi()
                                    tvStatus.text = "Training finished (no output produced)"
                                }
                            }
                        }
                    }

                    // Training saturates the GPU for minutes. Leaving the ARCore
                    // camera, VIO tracking and the continuous GL render loop running
                    // alongside it starves the optimizer and makes the whole device
                    // lag, so tear the capture pipeline down first.
                    trainingActive = true
                    releaseCaptureResources()

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
            val spanCm = if (pMaxX > pMinX) {
                val dx = pMaxX - pMinX; val dy = pMaxY - pMinY; val dz = pMaxZ - pMinZ
                (sqrt(dx * dx + dy * dy + dz * dz) * 100f).toInt()
            } else 0
            val hint = orbitHint
            runOnUiThread {
                tvFrameCount.text = "$keptFrameCount Frames | moved ${spanCm}cm"
                // Only overwrite the status line with the orbit warning; the blur
                // hint owns it briefly and clears itself.
                if (hint != null) tvStatus.text = hint
            }
        } else if (!tracking) {
            runOnUiThread { tvStatus.text = "Move phone slowly to start tracking..." }
        }
    }

    // -------------------------------------------------------------------------
    // Math helpers
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // Anchor-relative poses
    // -------------------------------------------------------------------------

    /**
     * This keyframe's pose relative to a nearby anchor, creating a new anchor every
     * [ANCHOR_SPACING_M] metres or [ANCHOR_KEYFRAME_INTERVAL] keyframes, whichever
     * comes first.
     *
     * Returns null whenever anchors cannot be used, in which case the caller keeps
     * the raw absolute pose and nothing downstream changes.
     */
    private fun anchorRefFor(pose: Pose): AnchorPoseRef? {
        val s = session ?: return null
        return try {
            var anchor = currentAnchor
            val anchorPos = currentAnchorPos
            val stale = anchor == null ||
                anchor.trackingState != TrackingState.TRACKING ||
                keyframesSinceAnchor >= ANCHOR_KEYFRAME_INTERVAL ||
                (anchorPos != null &&
                    distance(floatArrayOf(pose.tx(), pose.ty(), pose.tz()), anchorPos) > ANCHOR_SPACING_M)

            if (stale && sceneAnchors.size < MAX_ANCHORS) {
                val fresh = s.createAnchor(pose)
                sceneAnchors.add(fresh)
                currentAnchor = fresh
                currentAnchorPos = floatArrayOf(pose.tx(), pose.ty(), pose.tz())
                keyframesSinceAnchor = 0
                anchorsAvailable = true
                anchor = fresh
                Log.i(TAG, "Anchor ${sceneAnchors.size} created at keyframe $keptFrameCount")
            }

            val anc = anchor ?: return null
            if (anc.trackingState != TrackingState.TRACKING) return null
            keyframesSinceAnchor++
            AnchorPoseRef(anc, anc.pose.inverse().compose(pose))
        } catch (t: Throwable) {
            // Anchors are an optimisation, never a requirement.
            if (anchorsAvailable) Log.w(TAG, "Anchor unavailable, using absolute pose: ${t.message}")
            null
        }
    }

    /**
     * Recompose every anchor-backed pose against its anchor's CURRENT pose.
     *
     * Anchors belong to the ARCore session, so this is dispatched onto the GL
     * thread -- the only thread that touches the session -- and waited on with a
     * timeout; on timeout the dataset simply keeps its absolute poses.
     */
    private fun resolveAnchorPoses(): Int {
        if (!anchorsAvailable) {
            Log.i(TAG, "No anchors were created; exporting absolute VIO poses")
            return 0
        }
        val corrected = AtomicInteger(0)
        val latch = CountDownLatch(1)
        try {
            glSurfaceView.queueEvent {
                try {
                    corrected.set(datasetExporter.applyAnchorCorrections())
                } catch (t: Throwable) {
                    Log.w(TAG, "Anchor correction failed: ${t.message}")
                } finally {
                    detachAnchors()
                    latch.countDown()
                }
            }
            if (!latch.await(ANCHOR_RESOLVE_TIMEOUT_MS, TimeUnit.MILLISECONDS)) {
                Log.w(TAG, "Anchor correction timed out; exporting absolute VIO poses")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "Could not schedule anchor correction: ${t.message}")
        }
        return corrected.get()
    }

    /** Release the anchors back to ARCore once their corrections have been read. */
    private fun detachAnchors() {
        for (a in sceneAnchors) {
            try { a.detach() } catch (t: Throwable) { /* already gone */ }
        }
        sceneAnchors.clear()
        currentAnchor = null
        currentAnchorPos = null
        anchorsAvailable = false
    }

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
