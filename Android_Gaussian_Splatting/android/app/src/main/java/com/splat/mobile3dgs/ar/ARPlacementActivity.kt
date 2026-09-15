package com.splat.mobile3dgs.ar

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.util.TypedValue
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Button
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.ar.core.Anchor
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Frame
import com.google.ar.core.HitResult
import com.google.ar.core.Plane
import com.google.ar.core.Point
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.UnavailableApkTooOldException
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException
import com.google.ar.core.exceptions.UnavailableSdkTooOldException
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException
import com.splat.mobile3dgs.R
import com.splat.mobile3dgs.hardware.DeviceCapabilityManager
import com.splat.mobile3dgs.hardware.HardwareTier
import java.io.File
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * Real-world AR placement of a trained 3D Gaussian Splat model.
 *
 * The scene is composited in two passes on the GL thread:
 *  1. the ARCore camera image, as a full-screen quad from the OES external
 *     texture (see [ARCameraFeedRenderer]);
 *  2. the splat model, rasterised with a real 3DGS projection and alpha
 *     composited over it back to front (see [SplatRenderer]).
 *
 * The model hangs off an ARCore anchor, so it stays locked to the physical
 * surface it was dropped on while the phone moves.
 *
 * ### ARCore lifecycle, the hard way
 * ARCore aborts the *process* (an uncatchable SIGABRT inside `ArSession_destroy`)
 * if `close()` is called on a session whose camera never came up, and a failed
 * `resume()` leaves the camera running so retrying any other configuration on
 * the same session fails identically. Consequently: exactly one session attempt
 * per launch, `pause()`/`close()` strictly guarded by [sessionEverResumed] and
 * [resumeFailed], and any failure path just reports and finishes.
 */
class ARPlacementActivity : AppCompatActivity(), GLSurfaceView.Renderer {

    // --- Views (nullable: the layout is owned elsewhere and may change) -------
    private var surfaceView: GLSurfaceView? = null
    private var tvStatus: TextView? = null
    private var btnClose: ImageButton? = null
    private var btnScaleToggle: Button? = null
    private var btnResetAnchor: Button? = null

    // --- ARCore --------------------------------------------------------------
    private var session: Session? = null
    @Volatile private var sessionResumed = false
    @Volatile private var sessionEverResumed = false
    @Volatile private var resumeFailed = false
    private var sessionAttempted = false
    private var userRequestedInstall = true

    private var activeAnchor: Anchor? = null

    // --- Renderers -----------------------------------------------------------
    private val cameraRenderer = ARCameraFeedRenderer()
    private val splatRenderer = SplatRenderer()
    private val reticleRenderer = PlacementReticleRenderer()
    @Volatile private var glReady = false

    // --- Model ---------------------------------------------------------------
    private var modelPath: String? = null
    private var modelName: String = "Model"
    @Volatile private var pendingCloud: SplatCloud? = null
    @Volatile private var cloud: SplatCloud? = null
    private var sorter: SplatDepthSorter? = null
    @Volatile private var modelError: String? = null
    private var splatBudget = 250_000

    // --- Placement transform -------------------------------------------------
    /** Converts model units to metres; 1.0 when the model is already metric. */
    private var baseScale = 1.0f
    @Volatile private var scaleMultiplier = 1.0f
    @Volatile private var yawDegrees = 0.0f
    private var currentPresetIndex = 0
    private val scalePresets = floatArrayOf(1.0f, 0.5f, 0.25f)
    private var presetLabels = arrayOf("1:1 Metric", "50%", "25%")

    // --- Gestures ------------------------------------------------------------
    private var scaleDetector: ScaleGestureDetector? = null
    private var twistLastAngle = Float.NaN
    private var downX = 0f
    private var downY = 0f
    private var downTime = 0L
    private var draggedSinceDown = false
    private var multiTouchSinceDown = false
    private var touchSlopPx = 24f
    @Volatile private var pendingTapX = -1f
    @Volatile private var pendingTapY = -1f
    @Volatile private var hasPendingTap = false

    // --- Viewport / matrices -------------------------------------------------
    private var viewportWidth = 0
    private var viewportHeight = 0
    @Volatile private var viewportChanged = false

    private val viewMatrix = FloatArray(16)
    private val projMatrix = FloatArray(16)
    private val viewProjMatrix = FloatArray(16)
    private val anchorMatrix = FloatArray(16)
    private val localMatrix = FloatArray(16)
    private val modelMatrix = FloatArray(16)
    private val modelViewMatrix = FloatArray(16)
    private val reticleMatrix = FloatArray(16)
    private val scratchMatrix = FloatArray(16)

    // --- Adaptive level of detail -------------------------------------------
    private var lodStride = 1
    private var frameMsEma = 16f
    private var lastFrameNs = 0L
    private var lastLodChangeMs = 0L

    // --- Status --------------------------------------------------------------
    private var lastStatusMs = 0L
    private var lastPlaneCountMs = 0L
    private var lastReticleProbeMs = 0L
    private var reticleValid = false
    private var trackedPlanes = 0
    @Volatile private var anchorDistanceM = -1f

    companion object {
        private const val TAG = "ARPlacementActivity"
        private const val CAMERA_PERMISSION_CODE = 101
        private const val MIN_SCALE_MULTIPLIER = 0.05f
        private const val MAX_SCALE_MULTIPLIER = 12f
        /** Above this frame time (ms) the splat budget is cut. */
        private const val LOD_SLOW_MS = 24f
        /** Below this frame time (ms) the budget may be restored. */
        private const val LOD_FAST_MS = 13f
        private const val MAX_LOD_STRIDE = 8

        /**
         * Set once an ARCore session in this process has been left unclosable.
         * ARCore allows one session per process and a broken one can never be
         * released, so every later attempt in the same process is doomed.
         */
        @Volatile private var arcoreSessionPoisoned = false
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_ar_placement)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        surfaceView = findViewById(R.id.surfaceview_ar)
        tvStatus = findViewById(R.id.tv_ar_status)
        btnClose = findViewById(R.id.btn_close_ar)
        btnScaleToggle = findViewById(R.id.btn_scale_toggle)
        btnResetAnchor = findViewById(R.id.btn_reset_anchor)

        val view = surfaceView
        if (view == null) {
            Toast.makeText(this, "AR view unavailable", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        touchSlopPx = android.view.ViewConfiguration.get(this).scaledTouchSlop.toFloat()
        modelName = intent.getStringExtra("MODEL_NAME") ?: "Model"
        modelPath = intent.getStringExtra("MODEL_PATH")

        applyLightOverlayTheme()

        btnClose?.setOnClickListener { finish() }
        btnScaleToggle?.setOnClickListener { cyclePresetScale() }
        btnResetAnchor?.setOnClickListener { clearPlacement() }

        scaleDetector = ScaleGestureDetector(
            this,
            object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
                override fun onScale(detector: ScaleGestureDetector): Boolean {
                    val f = detector.scaleFactor
                    if (f.isFinite() && f > 0f) {
                        scaleMultiplier = (scaleMultiplier * f)
                            .coerceIn(MIN_SCALE_MULTIPLIER, MAX_SCALE_MULTIPLIER)
                        draggedSinceDown = true
                    }
                    return true
                }
            }
        )

        view.preserveEGLContextOnPause = true
        view.setEGLContextClientVersion(3)
        view.setEGLConfigChooser(8, 8, 8, 8, 16, 0)
        view.setRenderer(this)
        view.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        view.setOnTouchListener { _, event -> handleTouch(event); true }

        splatBudget = budgetForDevice()
        startModelLoad()
        checkCameraPermission()
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            return
        }
        if (resumeFailed) return

        if (session == null) {
            if (sessionAttempted) return
            if (!createSessionOnce()) return
        }

        val s = session ?: return
        if (!sessionResumed) {
            try {
                s.resume()
                sessionResumed = true
                sessionEverResumed = true
            } catch (e: CameraNotAvailableException) {
                // The camera is held by something else (or ARCore never got it).
                // Do NOT retry and do NOT close -- either aborts the process.
                markSessionPoisoned()
                failOut(
                    "Camera unavailable",
                    "Another part of the app is still using the camera. " +
                        "Close the app completely and reopen it to use AR."
                )
                return
            } catch (t: Throwable) {
                Log.e(TAG, "session.resume() failed", t)
                markSessionPoisoned()
                failOut("AR could not start", "ARCore failed to start the camera on this device.")
                return
            }
        }
        surfaceView?.onResume()
    }

    override fun onPause() {
        super.onPause()
        surfaceView?.onPause()
        // Pausing a session that never resumed is not safe either.
        if (sessionResumed) {
            try {
                session?.pause()
            } catch (t: Throwable) {
                Log.w(TAG, "session.pause() failed: ${t.message}")
            }
            sessionResumed = false
        }
    }

    override fun onDestroy() {
        try {
            activeAnchor?.detach()
        } catch (t: Throwable) {
            Log.w(TAG, "anchor detach: ${t.message}")
        }
        activeAnchor = null
        sorter?.release()
        sorter = null

        // Closing a session whose camera never came up aborts the whole process
        // from inside native ARCore, so only a cleanly resumed session is closed.
        val s = session
        session = null
        if (s != null && sessionEverResumed && !resumeFailed) {
            try {
                s.close()
            } catch (t: Throwable) {
                Log.w(TAG, "session.close() failed: ${t.message}")
            }
        } else if (s != null) {
            Log.w(TAG, "Leaving ARCore session open: it never resumed cleanly")
        }
        super.onDestroy()
    }

    private fun checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_CODE
            )
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.isEmpty() || grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(
                    this, "Camera permission is required for AR", Toast.LENGTH_LONG
                ).show()
                finish()
            }
        }
    }

    /**
     * Create and configure the ARCore session. Exactly one attempt per launch.
     * Returns false if the activity is finishing (or waiting on an install).
     */
    private fun createSessionOnce(): Boolean {
        sessionAttempted = true

        // A session whose camera failed can never be closed (closing it aborts the
        // process), so it stays alive and holds ARCore for the rest of the process
        // lifetime. Stacking more dead sessions on top of it helps nobody.
        if (arcoreSessionPoisoned) {
            failOut(
                "AR unavailable",
                "The AR camera could not be released after an earlier failure. " +
                    "Close the app completely and reopen it to place the model."
            )
            return false
        }

        // Play Services for AR installs on any device, and requestInstall() will
        // happily report INSTALLED on hardware ARCore cannot actually track on.
        // Ask about capability explicitly so the user gets a real explanation.
        val availability = try {
            ArCoreApk.getInstance().checkAvailability(this)
        } catch (t: Throwable) {
            Log.w(TAG, "checkAvailability failed: ${t.message}")
            null
        }
        Log.i(TAG, "ARCore availability = $availability")
        if (availability == ArCoreApk.Availability.UNSUPPORTED_DEVICE_NOT_CAPABLE) {
            failOut(
                "AR not supported",
                "This device is not ARCore-certified, so the model cannot be placed " +
                    "in the real world. You can still view it in the 3D viewer."
            )
            return false
        }

        try {
            when (ArCoreApk.getInstance().requestInstall(this, userRequestedInstall)) {
                ArCoreApk.InstallStatus.INSTALL_REQUESTED -> {
                    userRequestedInstall = false
                    sessionAttempted = false   // the install flow re-enters onResume
                    return false
                }
                else -> { /* installed; continue */ }
            }
        } catch (e: UnavailableUserDeclinedInstallationException) {
            failOut("AR unavailable", "Google Play Services for AR is required to place models.")
            return false
        } catch (e: UnavailableArcoreNotInstalledException) {
            failOut("AR unavailable", "Please install Google Play Services for AR.")
            return false
        } catch (e: UnavailableDeviceNotCompatibleException) {
            failOut("AR not supported", "This device does not support ARCore.")
            return false
        } catch (e: UnavailableApkTooOldException) {
            failOut("AR needs an update", "Please update Google Play Services for AR.")
            return false
        } catch (e: UnavailableSdkTooOldException) {
            failOut("AR unavailable", "This app is out of date for ARCore.")
            return false
        } catch (t: Throwable) {
            Log.e(TAG, "requestInstall failed", t)
            failOut("AR unavailable", "ARCore could not be initialised: ${t.localizedMessage}")
            return false
        }

        val created = try {
            Session(this)
        } catch (t: Throwable) {
            // Most commonly: another Session already exists in this process.
            Log.e(TAG, "Session creation failed", t)
            failOut(
                "AR unavailable",
                "An AR session is already running in this app. Close the app " +
                    "completely and reopen it to place the model."
            )
            return false
        }

        try {
            val config = Config(created).apply {
                planeFindingMode = Config.PlaneFindingMode.HORIZONTAL_AND_VERTICAL
                updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                // Splat colour is baked in at training time and is never relit, so
                // light estimation would cost CPU every frame for nothing.
                lightEstimationMode = Config.LightEstimationMode.DISABLED
                focusMode = Config.FocusMode.AUTO
            }
            created.configure(config)
        } catch (t: Throwable) {
            Log.e(TAG, "Session configure failed", t)
            // Never closed: the camera may already be half-started.
            session = created
            markSessionPoisoned()
            failOut("AR unavailable", "ARCore rejected the AR configuration on this device.")
            return false
        }

        session = created
        return true
    }

    /**
     * Record that this process's ARCore session can no longer be trusted or
     * released. [resumeFailed] keeps onDestroy from closing it (which would abort
     * the process); the companion flag keeps a later AR launch from piling
     * another dead session on top of it.
     */
    private fun markSessionPoisoned() {
        resumeFailed = true
        arcoreSessionPoisoned = true
    }

    /** Show a dialog and leave. Never throws, never closes a broken session. */
    private fun failOut(title: String, message: String) {
        if (isFinishing || isDestroyed) return
        runOnUiThread {
            if (isFinishing || isDestroyed) return@runOnUiThread
            try {
                AlertDialog.Builder(this)
                    .setTitle(title)
                    .setMessage(message)
                    .setPositiveButton("OK") { _, _ -> finish() }
                    .setOnDismissListener { finish() }
                    .setCancelable(true)
                    .show()
            } catch (t: Throwable) {
                Toast.makeText(this, message, Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }

    // =========================================================================
    // Model loading
    // =========================================================================

    /**
     * Splat budget for this phone. Gaussians are cheap to store but expensive to
     * blend: at AR frame rates the fill cost dominates, so a budget device gets a
     * much smaller working set. Dropping splats is always preferable to dropping
     * frames -- a sparser model still tracks the room; a stuttering one is unusable.
     */
    private fun budgetForDevice(): Int {
        return try {
            when (DeviceCapabilityManager.getDeviceProfile(this).tier) {
                HardwareTier.TIER_1_FLAGSHIP -> 400_000
                HardwareTier.TIER_2_BALANCED -> 250_000
                HardwareTier.TIER_3_STANDALONE -> 120_000
            }
        } catch (t: Throwable) {
            Log.w(TAG, "Device profile unavailable: ${t.message}")
            180_000
        }
    }

    private fun startModelLoad() {
        val path = modelPath
        if (path.isNullOrEmpty()) {
            modelError = "No model was supplied to AR mode."
            failOut("No model", "AR mode was opened without a model file.")
            return
        }
        Thread({
            try {
                val loaded = SplatCloud.load(File(path), splatBudget)
                pendingCloud = loaded
                Log.i(TAG, "Model ready: ${loaded.count} of ${loaded.sourceCount} splats")
            } catch (t: Throwable) {
                Log.e(TAG, "Model load failed", t)
                modelError = t.message ?: "The model file could not be read."
                failOut("Model unavailable", modelError!!)
            }
        }, "splat-loader").start()
    }

    /** Runs on the GL thread once the loader has produced a cloud. */
    private fun consumePendingCloud() {
        val loaded = pendingCloud ?: return
        pendingCloud = null
        try {
            if (!splatRenderer.uploadModel(loaded)) {
                failOut("Model too large", "This model does not fit in GPU memory on this device.")
                return
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Splat upload failed", t)
            failOut("Rendering unavailable", "The 3D renderer could not start: ${t.message}")
            return
        }
        cloud = loaded
        sorter = SplatDepthSorter(loaded.positions, loaded.count)

        // A model trained from ARCore poses is already metric. Anything wildly off
        // that came from elsewhere gets auto-fitted to a sane tabletop size.
        baseScale = if (loaded.looksNonMetric()) loaded.autoFitScale() else 1.0f
        if (loaded.looksNonMetric()) {
            presetLabels = arrayOf("Auto fit", "50%", "25%")
            runOnUiThread {
                runCatching { btnScaleToggle?.text = "Scale: ${presetLabels[0]}" }
            }
        }
    }

    // =========================================================================
    // Interaction
    // =========================================================================

    private fun handleTouch(event: MotionEvent) {
        scaleDetector?.onTouchEvent(event)

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                downX = event.x
                downY = event.y
                downTime = System.currentTimeMillis()
                draggedSinceDown = false
                multiTouchSinceDown = false
                twistLastAngle = Float.NaN
            }

            MotionEvent.ACTION_POINTER_DOWN -> {
                multiTouchSinceDown = true
                if (event.pointerCount >= 2) twistLastAngle = pointerAngle(event)
            }

            MotionEvent.ACTION_MOVE -> {
                if (event.pointerCount >= 2) {
                    // Two-finger twist -> yaw about the model's vertical axis.
                    val angle = pointerAngle(event)
                    if (!twistLastAngle.isNaN() && angle.isFinite()) {
                        var delta = angle - twistLastAngle
                        while (delta > 180f) delta -= 360f
                        while (delta < -180f) delta += 360f
                        if (abs(delta) > 0.05f) {
                            yawDegrees = normaliseDegrees(yawDegrees - delta)
                            draggedSinceDown = true
                        }
                    }
                    twistLastAngle = angle
                } else if (!multiTouchSinceDown) {
                    // One-finger drag -> the same yaw, for people who never twist.
                    val dx = event.x - downX
                    val dy = event.y - downY
                    if (!draggedSinceDown && hypot(dx, dy) > touchSlopPx) {
                        draggedSinceDown = true
                        downX = event.x
                        downY = event.y
                    } else if (draggedSinceDown) {
                        yawDegrees = normaliseDegrees(yawDegrees + (event.x - downX) * 0.35f)
                        downX = event.x
                        downY = event.y
                    }
                }
            }

            MotionEvent.ACTION_POINTER_UP -> {
                twistLastAngle = Float.NaN
            }

            MotionEvent.ACTION_UP -> {
                val quick = System.currentTimeMillis() - downTime < 450
                if (!draggedSinceDown && !multiTouchSinceDown && quick) {
                    // Hit tests must run against the frame the GL thread is holding,
                    // so the tap is queued rather than calling session.update() here.
                    pendingTapX = event.x
                    pendingTapY = event.y
                    hasPendingTap = true
                }
                twistLastAngle = Float.NaN
            }

            MotionEvent.ACTION_CANCEL -> {
                twistLastAngle = Float.NaN
            }
        }
    }

    private fun pointerAngle(event: MotionEvent): Float {
        if (event.pointerCount < 2) return Float.NaN
        val dx = event.getX(1) - event.getX(0)
        val dy = event.getY(1) - event.getY(0)
        return Math.toDegrees(atan2(dy.toDouble(), dx.toDouble())).toFloat()
    }

    private fun normaliseDegrees(d: Float): Float {
        var v = d % 360f
        if (v < 0f) v += 360f
        return v
    }

    private fun cyclePresetScale() {
        currentPresetIndex = (currentPresetIndex + 1) % scalePresets.size
        scaleMultiplier = scalePresets[currentPresetIndex]
        btnScaleToggle?.text = "Scale: ${presetLabels[currentPresetIndex]}"
    }

    private fun clearPlacement() {
        try {
            activeAnchor?.detach()
        } catch (t: Throwable) {
            Log.w(TAG, "detach failed: ${t.message}")
        }
        activeAnchor = null
        anchorDistanceM = -1f
        yawDegrees = 0f
        scaleMultiplier = scalePresets[currentPresetIndex]
        tvStatus?.text = "Placement cleared. Point at a surface and tap to place."
    }

    // =========================================================================
    // GLSurfaceView.Renderer
    // =========================================================================

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0.94f, 0.95f, 0.97f, 1.0f)
        glReady = false

        // A recreated EGL context has thrown away the splat texture and buffers,
        // so an already-loaded model has to be queued for re-upload -- otherwise
        // the user comes back from the home screen to an empty scene.
        cloud?.let { existing ->
            cloud = null
            sorter?.release()
            sorter = null
            pendingCloud = existing
        }

        try {
            cameraRenderer.createOnGlThread()
            splatRenderer.createOnGlThread()
            reticleRenderer.createOnGlThread()
            glReady = true
        } catch (t: Throwable) {
            Log.e(TAG, "GL init failed", t)
            failOut("Rendering unavailable", "The AR renderer could not start: ${t.message}")
        }
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        viewportWidth = width
        viewportHeight = height
        viewportChanged = true
        cameraRenderer.invalidateGeometry()
    }

    override fun onDrawFrame(gl: GL10?) {
        val now = System.nanoTime()
        if (lastFrameNs != 0L) {
            val ms = (now - lastFrameNs) / 1_000_000f
            if (ms in 0.5f..500f) frameMsEma = frameMsEma * 0.9f + ms * 0.1f
        }
        lastFrameNs = now

        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        if (!glReady) return

        val s = session ?: return
        if (!sessionResumed) return
        if (cameraRenderer.textureId == -1) return

        val frame: Frame = try {
            s.setCameraTextureName(cameraRenderer.textureId)
            if (viewportChanged) {
                s.setDisplayGeometry(displayRotation(), viewportWidth, viewportHeight)
                viewportChanged = false
            }
            s.update()
        } catch (e: CameraNotAvailableException) {
            Log.e(TAG, "Camera lost during draw", e)
            markSessionPoisoned()
            failOut("Camera lost", "The camera stopped being available. AR has been closed.")
            return
        } catch (t: Throwable) {
            Log.w(TAG, "session.update() failed: ${t.message}")
            return
        }

        // 1. Camera feed.
        try {
            cameraRenderer.draw(frame)
        } catch (t: Throwable) {
            Log.w(TAG, "camera draw failed: ${t.message}")
        }

        consumePendingCloud()

        val camera = frame.camera
        val tracking = camera.trackingState == TrackingState.TRACKING
        if (!tracking) {
            publishStatus(tracking = false)
            return
        }

        camera.getViewMatrix(viewMatrix, 0)
        camera.getProjectionMatrix(projMatrix, 0, 0.05f, 100f)
        Matrix.multiplyMM(viewProjMatrix, 0, projMatrix, 0, viewMatrix, 0)

        // getAllTrackables allocates, so it runs at status cadence, not frame cadence.
        val nowMs = System.currentTimeMillis()
        if (nowMs - lastPlaneCountMs > 500) {
            lastPlaneCountMs = nowMs
            trackedPlanes = try {
                s.getAllTrackables(Plane::class.java)
                    .count { it.trackingState == TrackingState.TRACKING }
            } catch (t: Throwable) {
                trackedPlanes
            }
        }

        // 2. Placement: consume a queued tap against this frame.
        if (hasPendingTap) {
            hasPendingTap = false
            placeAt(frame, pendingTapX, pendingTapY)
        }

        // 3. Reticle under the crosshair (or on the anchor once placed).
        drawReticle(frame)

        // 4. The model itself.
        val anchor = activeAnchor
        if (anchor != null && anchor.trackingState == TrackingState.TRACKING) {
            drawModel(anchor, camera.pose.tx(), camera.pose.ty(), camera.pose.tz())
        }

        updateLod()
        publishStatus(tracking = true)
    }

    private fun displayRotation(): Int {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            display?.rotation ?: 0
        } else {
            @Suppress("DEPRECATION") windowManager.defaultDisplay.rotation
        }
    }

    private fun placeAt(frame: Frame, x: Float, y: Float) {
        if (x < 0f || y < 0f) return
        val hit = bestHit(frame.hitTest(x, y)) ?: run {
            runOnUiThread {
                Toast.makeText(
                    this, "No surface there yet - move the phone to scan the area",
                    Toast.LENGTH_SHORT
                ).show()
            }
            return
        }
        try {
            activeAnchor?.detach()
            activeAnchor = hit.createAnchor()
            val hp = hit.hitPose
            val cp = frame.camera.pose
            anchorDistanceM = sqrt(
                (hp.tx() - cp.tx()) * (hp.tx() - cp.tx()) +
                    (hp.ty() - cp.ty()) * (hp.ty() - cp.ty()) +
                    (hp.tz() - cp.tz()) * (hp.tz() - cp.tz())
            )
        } catch (t: Throwable) {
            Log.w(TAG, "createAnchor failed: ${t.message}")
        }
    }

    /** Prefer a real plane hit; fall back to a feature point with a normal. */
    private fun bestHit(hits: List<HitResult>): HitResult? {
        for (hit in hits) {
            val trackable = hit.trackable
            if (trackable is Plane &&
                trackable.isPoseInPolygon(hit.hitPose) &&
                trackable.trackingState == TrackingState.TRACKING
            ) {
                return hit
            }
        }
        for (hit in hits) {
            val trackable = hit.trackable
            if (trackable is Point &&
                trackable.orientationMode == Point.OrientationMode.ESTIMATED_SURFACE_NORMAL
            ) {
                return hit
            }
        }
        return null
    }

    private fun drawReticle(frame: Frame) {
        val anchor = activeAnchor
        if (anchor != null && anchor.trackingState == TrackingState.TRACKING) {
            anchor.pose.toMatrix(reticleMatrix, 0)
            Matrix.setIdentityM(scratchMatrix, 0)
            Matrix.scaleM(scratchMatrix, 0, 0.09f, 0.09f, 0.09f)
            Matrix.multiplyMM(scratchMatrix, 0, reticleMatrix, 0, scratchMatrix, 0)
            reticleRenderer.draw(scratchMatrix, viewProjMatrix, 0.42f, 0.47f, 0.98f, 0.55f)
            return
        }
        if (viewportWidth == 0 || viewportHeight == 0) return

        // Each hitTest allocates a result list, so the crosshair probe runs at
        // ~15 Hz and the last good pose is reused in between.
        val nowMs = System.currentTimeMillis()
        if (nowMs - lastReticleProbeMs > 66) {
            lastReticleProbeMs = nowMs
            val hit = bestHit(
                try {
                    frame.hitTest(viewportWidth * 0.5f, viewportHeight * 0.5f)
                } catch (t: Throwable) {
                    emptyList()
                }
            )
            if (hit != null) {
                hit.hitPose.toMatrix(reticleMatrix, 0)
                reticleValid = true
            } else {
                reticleValid = false
            }
        }
        if (!reticleValid) return

        Matrix.setIdentityM(scratchMatrix, 0)
        Matrix.scaleM(scratchMatrix, 0, 0.075f, 0.075f, 0.075f)
        Matrix.multiplyMM(scratchMatrix, 0, reticleMatrix, 0, scratchMatrix, 0)
        reticleRenderer.draw(scratchMatrix, viewProjMatrix, 1f, 1f, 1f, 0.75f)
    }

    private fun drawModel(anchor: Anchor, camX: Float, camY: Float, camZ: Float) {
        val model = cloud ?: return
        if (!splatRenderer.isReady) return

        anchor.pose.toMatrix(anchorMatrix, 0)
        val s = baseScale * scaleMultiplier

        // anchor * lift * yaw * scale * recentre
        //
        // The recentre puts the model's robust centroid on the anchor, and the
        // lift raises it by half its height so it rests on the surface instead of
        // being buried halfway into it. Both are derived from percentile bounds so
        // a few stray "floater" gaussians cannot throw the placement off.
        Matrix.setIdentityM(localMatrix, 0)
        Matrix.translateM(localMatrix, 0, 0f, model.halfHeight * s, 0f)
        Matrix.rotateM(localMatrix, 0, yawDegrees, 0f, 1f, 0f)
        Matrix.scaleM(localMatrix, 0, s, s, s)
        Matrix.translateM(localMatrix, 0, -model.centerX, -model.centerY, -model.centerZ)
        Matrix.multiplyMM(modelMatrix, 0, anchorMatrix, 0, localMatrix, 0)
        Matrix.multiplyMM(modelViewMatrix, 0, viewMatrix, 0, modelMatrix, 0)

        anchorDistanceM = sqrt(
            (anchor.pose.tx() - camX) * (anchor.pose.tx() - camX) +
                (anchor.pose.ty() - camY) * (anchor.pose.ty() - camY) +
                (anchor.pose.tz() - camZ) * (anchor.pose.tz() - camZ)
        )

        // Depth order. The sorter runs on its own thread and is skipped entirely
        // when the viewpoint has not moved enough to change the ordering.
        val depthSorter = sorter
        if (depthSorter != null) {
            depthSorter.takeOrder()?.let { (order, count) ->
                splatRenderer.setDrawOrder(order, count)
            }
            depthSorter.requestSort(modelViewMatrix, lodStride, System.currentTimeMillis())
        }

        splatRenderer.draw(modelViewMatrix, projMatrix, viewportWidth, viewportHeight, 1.0f)
    }

    /**
     * Adaptive level of detail. The stride is applied inside the depth sort, so
     * raising it decimates uniformly across the whole model rather than clipping
     * away its far half.
     */
    private fun updateLod() {
        val now = System.currentTimeMillis()
        if (now - lastLodChangeMs < 1200) return
        if (frameMsEma > LOD_SLOW_MS && lodStride < MAX_LOD_STRIDE) {
            lodStride++
            lastLodChangeMs = now
            Log.i(TAG, "LOD down: stride=$lodStride (frame ${"%.1f".format(frameMsEma)} ms)")
        } else if (frameMsEma < LOD_FAST_MS && lodStride > 1) {
            lodStride--
            lastLodChangeMs = now
            Log.i(TAG, "LOD up: stride=$lodStride (frame ${"%.1f".format(frameMsEma)} ms)")
        }
    }

    private fun publishStatus(tracking: Boolean) {
        val now = System.currentTimeMillis()
        if (now - lastStatusMs < 280) return
        lastStatusMs = now

        val model = cloud
        val placed = activeAnchor != null
        val fps = if (frameMsEma > 0.1f) (1000f / frameMsEma) else 0f
        val text = when {
            model == null && modelError == null -> "Loading model..."
            !tracking -> "Move the phone slowly to start tracking..."
            !placed && trackedPlanes == 0 -> "Scanning for surfaces - pan across a floor or table"
            !placed -> "Tap a surface to place · $trackedPlanes surface(s) found"
            else -> {
                val shown = splatRenderer.visibleSplats
                val total = model?.sourceCount ?: 0
                val dist = if (anchorDistanceM > 0f) "%.2f m".format(anchorDistanceM) else "--"
                "$dist · ${compact(shown)}/${compact(total)} splats · " +
                    "${"%.0f".format(scaleMultiplier * 100)}% · " +
                    "${"%.0f".format(yawDegrees)}° · ${"%.0f".format(fps)} fps"
            }
        }
        runOnUiThread {
            runCatching { tvStatus?.text = text }
        }
    }

    private fun compact(n: Int): String = when {
        n >= 1_000_000 -> "%.1fM".format(n / 1_000_000f)
        n >= 1_000 -> "${n / 1000}k"
        else -> n.toString()
    }

    // =========================================================================
    // Overlay styling (light / soft, to match the rest of the app)
    // =========================================================================

    private fun dp(value: Float): Float = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP, value, resources.displayMetrics
    )

    private fun softSurface(fill: Int, radiusDp: Float, strokeColor: Int): GradientDrawable =
        GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            cornerRadius = dp(radiusDp)
            setColor(fill)
            setStroke(dp(1f).toInt().coerceAtLeast(1), strokeColor)
        }

    /**
     * The AR overlay floats over a live camera image, so it uses the same soft,
     * light, high-radius surfaces as the rest of the app rather than the dark
     * chrome the placeholder layout shipped with.
     */
    private fun applyLightOverlayTheme() = runCatching {
        val ink = Color.parseColor("#1B2430")
        val surface = Color.parseColor("#F2FFFFFF")
        val hairline = Color.parseColor("#1F0B1B33")

        tvStatus?.apply {
            background = softSurface(surface, 16f, hairline)
            setTextColor(ink)
            setPadding(dp(14f).toInt(), dp(9f).toInt(), dp(14f).toInt(), dp(9f).toInt())
            elevation = dp(4f)
            textSize = 13f
        }

        // The placeholder header is a dark scrim; let the camera through instead
        // so the floating light chips read as the only chrome on screen.
        (tvStatus?.parent as? ViewGroup)?.setBackgroundColor(Color.TRANSPARENT)

        (findViewById<TextView>(R.id.tv_ar_title))?.apply {
            background = softSurface(surface, 14f, hairline)
            setTextColor(ink)
            setPadding(dp(12f).toInt(), dp(6f).toInt(), dp(12f).toInt(), dp(6f).toInt())
            elevation = dp(4f)
        }

        btnClose?.apply {
            background = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(surface)
                setStroke(dp(1f).toInt().coerceAtLeast(1), hairline)
            }
            elevation = dp(4f)
            imageTintList = android.content.res.ColorStateList.valueOf(ink)
        }

        btnScaleToggle?.apply {
            background = softSurface(surface, 22f, hairline)
            setTextColor(ink)
            isAllCaps = false
            elevation = dp(5f)
            stateListAnimator = null
            text = "Scale: ${presetLabels[currentPresetIndex]}"
        }

        btnResetAnchor?.apply {
            background = softSurface(
                Color.parseColor("#F0E9EDFF"), 22f, Color.parseColor("#33515BD6")
            )
            setTextColor(Color.parseColor("#3B4BD8"))
            isAllCaps = false
            elevation = dp(5f)
            stateListAnimator = null
            text = "Clear"
        }

        tvStatus?.text = "Point the camera at a floor or table..."
    }.onFailure { Log.w(TAG, "Overlay styling skipped: ${it.message}") }
}
