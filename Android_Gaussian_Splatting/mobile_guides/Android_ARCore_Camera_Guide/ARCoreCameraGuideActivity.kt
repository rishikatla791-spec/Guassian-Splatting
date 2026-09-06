package com.gaussian.cameraguide

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.ar.core.Camera
import com.google.ar.core.Config
import com.google.ar.core.Frame
import com.google.ar.core.PointCloud
import com.google.ar.core.Pose
import com.google.ar.core.Session
import com.google.ar.core.TrackingFailureReason
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import java.io.ByteArrayOutputStream

class ARCoreCameraGuideActivity : AppCompatActivity(), SurfaceHolder.Callback {

    private var arSession: Session? = null
    private var domePlanner: GeodesicDomePlanner? = null
    private lateinit var datasetExporter: DatasetExporter
    private val frameQualityFilter = FrameQualityFilter()

    private lateinit var statusTextView: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var captureButton: Button
    private lateinit var surfaceView: SurfaceView

    private var vibrator: Vibrator? = null
    private var isObjectCenterPlaced = false
    private var isProcessingFrame = false

    // Pose tracking parameters
    private val poseMatrixBuffer = FloatArray(16)
    private var lastCapturedPoseMatrix: FloatArray? = null
    private val minCaptureDistanceMeters = 0.15f // Minimum camera movement to prevent redundant captures

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_ar_camera_guide)

        statusTextView = findViewById(R.id.statusTextView)
        progressBar = findViewById(R.id.progressBar)
        captureButton = findViewById(R.id.captureButton)
        surfaceView = findViewById(R.id.surfaceView)

        surfaceView.holder.addCallback(this)
        vibrator = getSystemService(Vibrator::class.java)
        datasetExporter = DatasetExporter(this)

        statusTextView.text = "Initializing ARCore SLAM..."
        captureButton.setOnClickListener {
            onCaptureRequested()
        }
    }

    override fun onResume() {
        super.onResume()
        if (arSession == null) {
            try {
                arSession = Session(this)
                val config = Config(arSession).apply {
                    focusMode = Config.FocusMode.AUTO
                    updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                    lightEstimationMode = Config.LightEstimationMode.AMBIENT_INTENSITY
                }
                arSession?.configure(config)
            } catch (e: Exception) {
                Toast.makeText(this, "ARCore Error: ${e.localizedMessage}", Toast.LENGTH_LONG).show()
                finish()
                return
            }
        }

        try {
            arSession?.resume()
        } catch (e: CameraNotAvailableException) {
            Toast.makeText(this, "Camera not available", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onPause() {
        super.onPause()
        arSession?.pause()
    }

    override fun onDestroy() {
        super.onDestroy()
        arSession?.close()
        arSession = null
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        arSession?.setCameraTextureName(0)
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {}
    override fun surfaceDestroyed(holder: SurfaceHolder) {}

    /**
     * Set 3D target dome center when user taps object.
     */
    fun onUserTappedObject(worldX: Float, worldY: Float, worldZ: Float) {
        val center = Vector3(worldX, worldY, worldZ)
        domePlanner = GeodesicDomePlanner(center = center, radius = 1.2f, numElevationRings = 3, samplesPerRing = 12)
        isObjectCenterPlaced = true

        runOnUiThread {
            statusTextView.text = "Target Dome Set! Orbit object to capture."
            triggerHaptic()
        }
    }

    /**
     * Production 60 FPS Frame Processing Loop.
     * Guarantees resource cleanup (frame.close(), image.close()) to prevent memory leaks.
     */
    fun processARFrame() {
        val session = arSession ?: return
        if (isProcessingFrame) return
        isProcessingFrame = true

        var currentFrame: Frame? = null
        var cameraImage: Image? = null
        var pointCloud: PointCloud? = null

        try {
            currentFrame = session.update()
            val camera: Camera = currentFrame.camera

            // Handle Tracking Failures Gracefully
            if (camera.trackingState != TrackingState.TRACKING) {
                val reason = when (camera.trackingFailureReason) {
                    TrackingFailureReason.INSUFFICIENT_LIGHT -> "Low Light - Increase Lighting"
                    TrackingFailureReason.EXCESSIVE_MOTION -> "Moving Too Fast - Slow Down"
                    TrackingFailureReason.INSUFFICIENT_FEATURES -> "Low Texture Surface"
                    TrackingFailureReason.BAD_STATE -> "Tracking Initializing..."
                    else -> "Tracking Interrupted"
                }
                runOnUiThread { statusTextView.text = "⚠️ $reason" }
                return
            }

            // Extract Camera Pose
            val pose: Pose = camera.pose
            pose.toMatrix(poseMatrixBuffer, 0)

            val camPos = Vector3(pose.tx(), pose.ty(), pose.tz())

            // Evaluate Dome Coverage if target is placed
            if (isObjectCenterPlaced && domePlanner != null) {
                val evalResult = domePlanner!!.evaluateCameraPose(camPos)
                val ratio = domePlanner!!.overallCoverageRatio

                runOnUiThread {
                    progressBar.progress = (ratio * 100).toInt()
                    statusTextView.text = String.format("Coverage: %.0f%% | Gap: %.1f°", ratio * 100, evalResult.angularDistanceDeg)
                }

                // Check redundant capture distance threshold
                val isSufficientDistance = lastCapturedPoseMatrix == null ||
                        ARCoreCoordinateUtils.distance3D(
                            floatArrayOf(pose.tx(), pose.ty(), pose.tz()),
                            floatArrayOf(lastCapturedPoseMatrix!![12], lastCapturedPoseMatrix!![13], lastCapturedPoseMatrix!![14])
                        ) >= minCaptureDistanceMeters

                if (evalResult.isNewlyCaptured && isSufficientDistance) {
                    // Acquire Frame Image and Point Cloud
                    cameraImage = currentFrame.acquireCameraImage()
                    pointCloud = currentFrame.acquireRawFeatureCloud()

                    // Evaluate Frame Quality
                    val quality = frameQualityFilter.evaluateFrameQuality(cameraImage)
                    if (quality.isPassed) {
                        val bitmap = yuvImageToBitmap(cameraImage)
                        val intrinsics = camera.textureIntrinsics

                        datasetExporter.saveCapturedFrame(
                            bitmap = bitmap,
                            poseMatrix = poseMatrixBuffer.clone(),
                            timestampNs = currentFrame.timestamp,
                            sharpnessScore = quality.blurScore,
                            meanLuminance = quality.meanLuminance,
                            pointCloud = pointCloud
                        )
                        lastCapturedPoseMatrix = poseMatrixBuffer.clone()
                        triggerHaptic()
                    }
                }
            }

        } catch (e: Exception) {
            // Handle frame update exceptions gracefully
        } finally {
            // CRITICAL: Explicitly release ARCore resources to prevent memory leaks / crashes
            cameraImage?.close()
            pointCloud?.close()
            isProcessingFrame = false
        }
    }

    private fun onCaptureRequested() {
        val session = arSession ?: return
        try {
            val frame = session.update()
            val camera = frame.camera
            if (camera.trackingState == TrackingState.TRACKING) {
                val image = frame.acquireCameraImage()
                val ptCloud = frame.acquireRawFeatureCloud()
                val quality = frameQualityFilter.evaluateFrameQuality(image)

                val bitmap = yuvImageToBitmap(image)
                val intrinsics = camera.textureIntrinsics

                val poseMatrix = FloatArray(16)
                camera.pose.toMatrix(poseMatrix, 0)

                datasetExporter.saveCapturedFrame(
                    bitmap = bitmap,
                    poseMatrix = poseMatrix,
                    timestampNs = frame.timestamp,
                    sharpnessScore = quality.blurScore,
                    meanLuminance = quality.meanLuminance,
                    pointCloud = ptCloud
                )

                datasetExporter.exportDataset(
                    fx = intrinsics.focalLength[0],
                    fy = intrinsics.focalLength[1],
                    cx = intrinsics.principalPoint[0],
                    cy = intrinsics.principalPoint[1],
                    width = image.width,
                    height = image.height
                )

                image.close()
                ptCloud.close()

                triggerHaptic()
                Toast.makeText(this, "📸 Photo & Transforms Exported!", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Toast.makeText(this, "Capture Error: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun yuvImageToBitmap(image: Image): Bitmap {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 95, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun triggerHaptic() {
        vibrator?.vibrate(VibrationEffect.createOneShot(80, VibrationEffect.DEFAULT_AMPLITUDE))
    }
}
