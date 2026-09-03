package com.splat.mobile3dgs.engine

import android.util.Log

interface TrainingProgressListener {
    fun onProgress(step: Int, progress: Float)
}

class NativeBrushEngine {
    companion object {
        private const val TAG = "NativeBrushEngine"
        private var isLoaded = false
        private var loadError: String? = null

        init {
            try {
                // 1. Load the core Vulkan compute engine (Rust + Burn + wgpu)
                System.loadLibrary("brush_c")
                Log.i(TAG, "Successfully loaded libbrush_c.so")

                // 2. Load the JNI bridge that interfaces with Kotlin
                System.loadLibrary("brush_bridge")
                Log.i(TAG, "Successfully loaded libbrush_bridge.so")

                isLoaded = true
                Log.i(TAG, "All native 3DGS Vulkan libraries successfully linked and ready!")
            } catch (e: Throwable) {
                loadError = e.message ?: e.toString()
                Log.e(TAG, "FATAL: Native library load failed: $loadError", e)
                isLoaded = false
            }
        }

        fun isNativeEngineAvailable(): Boolean = isLoaded
        fun getLoadError(): String? = loadError
    }

    /**
     * Executes on-device non-CUDA 3D Gaussian Splatting optimization via Vulkan compute shaders.
     */
    fun startOnDeviceTraining(
        datasetPath: String,
        outputPath: String,
        iterations: Int = 7000,
        maxResolution: Int = 720,
        onProgress: (Int, Float) -> Unit = { _, _ -> }
    ): Boolean {
        if (!isLoaded) {
            Log.e(TAG, "Native engine not loaded: $loadError")
            return false
        }
        val listener = object : TrainingProgressListener {
            override fun onProgress(step: Int, progress: Float) {
                onProgress(step, progress)
            }
        }
        return try {
            val code = nativeTrainAndSave(datasetPath, outputPath, iterations, maxResolution, listener)
            code == 0
        } catch (e: Throwable) {
            Log.e(TAG, "On-device training error: ", e)
            false
        }
    }

    private external fun nativeTrainAndSave(
        datasetPath: String,
        outputPath: String,
        iterations: Int,
        maxResolution: Int,
        listener: TrainingProgressListener?
    ): Int
}
