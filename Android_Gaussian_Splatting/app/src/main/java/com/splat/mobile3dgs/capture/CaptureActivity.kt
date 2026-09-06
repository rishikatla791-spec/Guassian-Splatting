package com.splat.mobile3dgs.capture

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.gson.Gson
import com.splat.mobile3dgs.R
import com.splat.mobile3dgs.engine.NativeBrushEngine
import com.splat.mobile3dgs.model.CameraPose
import com.splat.mobile3dgs.model.NerfstudioDataset
import com.splat.mobile3dgs.model.NerfstudioFrame
import com.splat.mobile3dgs.network.ApiClient
import com.splat.mobile3dgs.viewer.ViewerActivity
import kotlinx.coroutines.*
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class CaptureActivity : AppCompatActivity(), SensorEventListener {
    private lateinit var previewView: PreviewView
    private lateinit var btnRecord: Button
    private lateinit var tvStatus: TextView
    private lateinit var tvFrameCount: TextView
    private lateinit var progressBar: ProgressBar

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var sensorManager: SensorManager
    private var rotationVectorSensor: Sensor? = null

    private var imageCapture: ImageCapture? = null
    private var isRecording = false
    private var frameIndex = 0
    private val recordedPoses = mutableListOf<CameraPose>()
    private var currentQuaternion = floatArrayOf(0f, 0f, 0f, 1f)

    // Tracks asynchronous image saves in flight
    private val inFlightCaptures = AtomicInteger(0)

    // Dynamic camera intrinsics: [fx, fy, cx, cy]
    private var cameraIntrinsics = floatArrayOf(1000f, 1000f, 540f, 960f)
    private var detectedImageWidth = 1080
    private var detectedImageHeight = 1920

    private lateinit var currentSessionDir: File
    private val apiClient = ApiClient()
    private val handler = Handler(Looper.getMainLooper())
    private var captureRunnable: Runnable? = null

    companion object {
        private const val TAG = "CaptureActivity"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_capture)

        previewView = findViewById(R.id.viewFinder)
        btnRecord = findViewById(R.id.btn_record_scan)
        tvStatus = findViewById(R.id.tv_capture_status)
        tvFrameCount = findViewById(R.id.tv_frame_count)
        progressBar = findViewById(R.id.progress_upload)

        val prefs = getSharedPreferences("Mobile3DGS_Prefs", Context.MODE_PRIVATE)
        val savedServerUrl = prefs.getString("SERVER_URL", "http://10.0.2.2:8000") ?: "http://10.0.2.2:8000"
        apiClient.setServerUrl(savedServerUrl)

        cameraExecutor = Executors.newSingleThreadExecutor()
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        rotationVectorSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

        initCameraIntrinsics()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        btnRecord.setOnClickListener {
            if (!isRecording) {
                startRecordingSession()
            } else {
                stopRecordingSession()
            }
        }
    }

    private fun initCameraIntrinsics() {
        try {
            val cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
            for (id in cameraManager.cameraIdList) {
                val chars = cameraManager.getCameraCharacteristics(id)
                val facing = chars.get(CameraCharacteristics.LENS_FACING)
                if (facing == CameraCharacteristics.LENS_FACING_BACK) {
                    val focalLengths = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                    val sensorSize = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                    val activeArray = chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
                    if (focalLengths != null && focalLengths.isNotEmpty() && sensorSize != null && activeArray != null && sensorSize.width > 0f) {
                        val fMm = focalLengths[0]
                        detectedImageWidth = activeArray.width()
                        detectedImageHeight = activeArray.height()
                        val flX = (fMm / sensorSize.width) * detectedImageWidth
                        val flY = (fMm / sensorSize.height) * detectedImageHeight
                        val cx = detectedImageWidth / 2.0f
                        val cy = detectedImageHeight / 2.0f
                        cameraIntrinsics = floatArrayOf(flX, flY, cx, cy)
                        Log.i(TAG, "Dynamic camera intrinsics: fx=$flX, fy=$flY, cx=$cx, cy=$cy")
                        return
                    }
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Could not query camera characteristics: ${e.message}")
        }
        val f = (detectedImageWidth / 2.0f) / Math.tan(Math.toRadians(65.0 / 2.0)).toFloat()
        cameraIntrinsics = floatArrayOf(f, f, detectedImageWidth / 2f, detectedImageHeight / 2f)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Camera permission required for 3D scan.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture
                )
            } catch (exc: Exception) {
                Toast.makeText(this, "Camera start error: ${exc.message}", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun startRecordingSession() {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        currentSessionDir = File(getExternalFilesDir(null), "scan_$timeStamp")
        File(currentSessionDir, "images").mkdirs()

        frameIndex = 0
        inFlightCaptures.set(0)
        recordedPoses.clear()
        isRecording = true

        btnRecord.text = "Stop & Auto-Train 3DGS"
        btnRecord.setBackgroundColor(ContextCompat.getColor(this, android.R.color.holo_red_dark))
        tvStatus.text = "Scanning (Move slowly around object)..."

        scheduleNextCapture(100)
    }

    private fun scheduleNextCapture(delayMs: Long) {
        if (!isRecording) return
        captureRunnable = Runnable {
            if (isRecording) {
                captureFrame()
            }
        }
        handler.postDelayed(captureRunnable!!, delayMs)
    }

    private fun captureFrame() {
        val imgCapture = imageCapture ?: return
        val formattedIndex = String.format("%04d", frameIndex)
        val imgFile = File(currentSessionDir, "images/frame_${formattedIndex}.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(imgFile).build()

        val q = currentQuaternion.clone()
        val currentIdx = frameIndex
        frameIndex++
        tvFrameCount.text = "Frames: $frameIndex"

        inFlightCaptures.incrementAndGet()

        imgCapture.takePicture(
            outputOptions,
            cameraExecutor,
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    inFlightCaptures.decrementAndGet()

                    // Calculate real 3D translation along orbital look vector
                    val qx = q[0]; val qy = q[1]; val qz = q[2]; val qw = q[3]
                    val r02 = 2f * (qx * qz + qy * qw)
                    val r12 = 2f * (qy * qz - qx * qw)
                    val r22 = 1f - 2f * (qx * qx + qy * qy)

                    val tx = -r02 * 1.0f
                    val ty = -r12 * 1.0f
                    val tz = -r22 * 1.0f

                    val pose = CameraPose(
                        frameIndex = currentIdx,
                        timestamp = System.currentTimeMillis(),
                        rotation = listOf(qx, qy, qz, qw),
                        position = listOf(tx, ty, tz),
                        focalLengthX = cameraIntrinsics[0],
                        focalLengthY = cameraIntrinsics[1],
                        principalPointX = cameraIntrinsics[2],
                        principalPointY = cameraIntrinsics[3]
                    )
                    synchronized(recordedPoses) {
                        recordedPoses.add(pose)
                    }

                    if (isRecording) {
                        scheduleNextCapture(350)
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    inFlightCaptures.decrementAndGet()
                    Log.e(TAG, "Capture error: ${exception.message}")
                    if (isRecording) {
                        scheduleNextCapture(350)
                    }
                }
            }
        )
    }

    private fun stopRecordingSession() {
        isRecording = false
        captureRunnable?.let { handler.removeCallbacks(it) }
        btnRecord.text = "Saving Photos..."
        btnRecord.isEnabled = false
        progressBar.visibility = View.VISIBLE
        progressBar.isIndeterminate = true
        tvStatus.text = "Finalizing photo capture..."

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Wait for any remaining in-flight camera writes to complete
                var waitCycles = 0
                while (inFlightCaptures.get() > 0 && waitCycles < 30) {
                    withContext(Dispatchers.Main) {
                        tvStatus.text = "Saving photos (${inFlightCaptures.get()} in queue)..."
                    }
                    delay(200)
                    waitCycles++
                }

                val posesSnapshot = synchronized(recordedPoses) {
                    recordedPoses.sortedBy { it.frameIndex }
                }

                if (posesSnapshot.isEmpty()) {
                    withContext(Dispatchers.Main) {
                        progressBar.visibility = View.GONE
                        btnRecord.isEnabled = true
                        btnRecord.text = "Start 3D Scan"
                        tvStatus.text = "No frames saved. Try again."
                        Toast.makeText(this@CaptureActivity, "No frames captured. Please try again.", Toast.LENGTH_SHORT).show()
                    }
                    return@launch
                }

                withContext(Dispatchers.Main) {
                    tvStatus.text = "Generating 3D camera trajectory (${posesSnapshot.size} frames)..."
                }

                // 1. Generate standard Nerfstudio transforms.json
                generateTransformsJson(currentSessionDir, posesSnapshot)

                // 2. Check native engine availability
                if (!NativeBrushEngine.isNativeEngineAvailable()) {
                    val loadErr = NativeBrushEngine.getLoadError() ?: "libbrush_c.so or libbrush_bridge.so failed to load"
                    withContext(Dispatchers.Main) {
                        progressBar.visibility = View.GONE
                        btnRecord.isEnabled = true
                        btnRecord.text = "Start 3D Scan"
                        tvStatus.text = "Native Engine Error"

                        AlertDialog.Builder(this@CaptureActivity)
                            .setTitle("Vulkan Engine Load Error")
                            .setMessage("The on-device 3DGS engine could not be initialized:\n\n$loadErr\n\nPlease check device Vulkan support.")
                            .setPositiveButton("OK", null)
                            .show()
                    }
                    return@launch
                }

                val prefs = getSharedPreferences("Mobile3DGS_Prefs", Context.MODE_PRIVATE)
                val targetIterations = prefs.getInt("PREF_TRAINING_STEPS", 30000)
                val targetResolution = prefs.getInt("PREF_TRAINING_RES", 1080)

                val nativeEngine = NativeBrushEngine()
                withContext(Dispatchers.Main) {
                    tvStatus.text = "⚡ Snapdragon 8 Gen 2: Training 3DGS on Vulkan GPU (0% / 30K Steps)..."
                    Toast.makeText(this@CaptureActivity, "Starting 30K Studio 3DGS training on Adreno GPU...", Toast.LENGTH_SHORT).show()
                }

                val outputSplat = File(filesDir, "${currentSessionDir.name}.splat")
                val success = nativeEngine.startOnDeviceTraining(
                    datasetPath = currentSessionDir.absolutePath,
                    outputPath = outputSplat.absolutePath,
                    iterations = targetIterations,
                    maxResolution = targetResolution
                ) { step: Int, progress: Float ->
                    runOnUiThread {
                        val pct = (progress * 100f).toInt().coerceIn(0, 100)
                        tvStatus.text = "⚡ Adreno 740: Step $step / $targetIterations ($pct%)"
                        progressBar.isIndeterminate = false
                        progressBar.progress = pct
                    }
                }

                withContext(Dispatchers.Main) {
                    progressBar.visibility = View.GONE
                    if (success && outputSplat.exists() && outputSplat.length() > 0) {
                        Toast.makeText(this@CaptureActivity, "🎉 3D Model Generated On-Device!", Toast.LENGTH_LONG).show()
                        val intent = Intent(this@CaptureActivity, ViewerActivity::class.java).apply {
                            putExtra("MODEL_NAME", "On-Device 3D Scan ($targetIterations steps)")
                            putExtra("MODEL_PATH", outputSplat.absolutePath)
                        }
                        startActivity(intent)
                        finish()
                    } else {
                        btnRecord.isEnabled = true
                        btnRecord.text = "Start 3D Scan"
                        tvStatus.text = "Training finished (No output generated)"
                        AlertDialog.Builder(this@CaptureActivity)
                            .setTitle("Training Notice")
                            .setMessage("On-device optimization completed, but the output model file was not found.\n\nPlease check Logcat for tag 'BrushBridge'.")
                            .setPositiveButton("OK", null)
                            .show()
                    }
                }
            } catch (e: Throwable) {
                Log.e(TAG, "Unhandled exception in scan pipeline", e)
                withContext(Dispatchers.Main) {
                    progressBar.visibility = View.GONE
                    btnRecord.isEnabled = true
                    btnRecord.text = "Start 3D Scan"
                    tvStatus.text = "Error: ${e.message}"
                    AlertDialog.Builder(this@CaptureActivity)
                        .setTitle("Pipeline Error")
                        .setMessage("An unexpected error occurred:\n\n${e.message}")
                        .setPositiveButton("OK", null)
                        .show()
                }
            }
        }
    }

    private fun generateTransformsJson(sessionDir: File, poses: List<CameraPose>) {
        val imagesDir = File(sessionDir, "images")
        val sampleImg = imagesDir.listFiles()?.firstOrNull { it.isFile && it.length() > 0 }
        if (sampleImg != null) {
            val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFile(sampleImg.absolutePath, opts)
            if (opts.outWidth > 0 && opts.outHeight > 0) {
                detectedImageWidth = opts.outWidth
                detectedImageHeight = opts.outHeight
                cameraIntrinsics[2] = detectedImageWidth / 2.0f
                cameraIntrinsics[3] = detectedImageHeight / 2.0f
            }
        }

        val frames = mutableListOf<NerfstudioFrame>()
        for (pose in poses) {
            val formattedIndex = String.format("%04d", pose.frameIndex)
            val imageRelPath = "images/frame_${formattedIndex}.jpg"
            val imageFile = File(sessionDir, imageRelPath)

            // Verify that the image was actually flushed to disk
            if (!imageFile.exists() || imageFile.length() == 0L) {
                Log.w(TAG, "Skipping frame $imageRelPath because file does not exist on disk")
                continue
            }

            val q = pose.rotation
            val qx = if (q.isNotEmpty()) q[0] else 0f
            val qy = if (q.size > 1) q[1] else 0f
            val qz = if (q.size > 2) q[2] else 0f
            val qw = if (q.size > 3) q[3] else 1f

            val r00 = 1f - 2f * (qy * qy + qz * qz)
            val r01 = 2f * (qx * qy - qz * qw)
            val r02 = 2f * (qx * qz + qy * qw)

            val r10 = 2f * (qx * qy + qz * qw)
            val r11 = 1f - 2f * (qx * qx + qz * qz)
            val r12 = 2f * (qy * qz - qx * qw)

            val r20 = 2f * (qx * qz - qy * qw)
            val r21 = 2f * (qy * qz + qx * qw)
            val r22 = 1f - 2f * (qx * qx + qy * qy)

            val pos = pose.position
            val tx = if (pos.isNotEmpty() && (pos[0] != 0f || pos[1] != 0f || pos[2] != 0f)) pos[0] else -r02 * 1.0f
            val ty = if (pos.size > 1 && (pos[0] != 0f || pos[1] != 0f || pos[2] != 0f)) pos[1] else -r12 * 1.0f
            val tz = if (pos.size > 2 && (pos[0] != 0f || pos[1] != 0f || pos[2] != 0f)) pos[2] else -r22 * 1.0f

            val transformMatrix = listOf(
                listOf(r00, r01, r02, tx),
                listOf(r10, r11, r12, ty),
                listOf(r20, r21, r22, tz),
                listOf(0.0f, 0.0f, 0.0f, 1.0f)
            )

            frames.add(NerfstudioFrame(
                filePath = imageRelPath,
                transformMatrix = transformMatrix
            ))
        }

        val dataset = NerfstudioDataset(
            flX = cameraIntrinsics[0],
            flY = cameraIntrinsics[1],
            cx = cameraIntrinsics[2],
            cy = cameraIntrinsics[3],
            w = detectedImageWidth,
            h = detectedImageHeight,
            cameraModel = "OPENCV",
            frames = frames
        )

        val transformsFile = File(sessionDir, "transforms.json")
        transformsFile.writeText(Gson().toJson(dataset))
        Log.i(TAG, "Generated transforms.json with ${frames.size} valid frames (${detectedImageWidth}x${detectedImageHeight})")
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type == Sensor.TYPE_ROTATION_VECTOR) {
            val q = FloatArray(4)
            SensorManager.getQuaternionFromVector(q, event.values)
            currentQuaternion = q
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onResume() {
        super.onResume()
        rotationVectorSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }
}
