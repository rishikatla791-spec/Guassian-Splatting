package com.splat.mobile3dgs.capture

import android.content.Context
import android.graphics.Bitmap
import com.google.ar.core.PointCloud
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer

data class FrameMetaData(
    val frameId: Int,
    val filePath: String,
    val timestampNs: Long,
    val transformMatrix: Array<FloatArray>, // 4x4 homogenous matrix
    val sharpnessScore: Float,
    val meanLuminance: Float
)

data class FeaturePoint3D(
    val id: Int,
    val x: Float,
    val y: Float,
    val z: Float,
    val confidence: Float
)

class DatasetExporter(context: Context, sessionName: String = "3dgs_arcore_${System.currentTimeMillis() / 1000}") {

    val outputDir: File = File(context.getExternalFilesDir(null), sessionName).apply { mkdirs() }
    private val imagesDir: File = File(outputDir, "images").apply { mkdirs() }
    val capturedFrames = mutableListOf<FrameMetaData>()
    val accumulatedFeaturePoints = mutableListOf<FeaturePoint3D>()

    /**
     * Save a frame that has already been JPEG-encoded (e.g. from an ARCore CPU image).
     * [poseMatrix] is the 16-element column-major camera-to-world matrix from
     * `pose.toMatrix()`. [pointsXyzc] is an optional flat array of ARCore point-cloud
     * entries as [x, y, z, confidence, ...] copied out before the frame was released.
     */
    fun saveCapturedFrameJpeg(
        jpegBytes: ByteArray,
        poseMatrix: FloatArray,
        timestampNs: Long,
        sharpnessScore: Float,
        meanLuminance: Float,
        pointsXyzc: FloatArray? = null
    ): String {
        val frameId = capturedFrames.size
        val fileName = String.format("frame_%04d.jpg", frameId)
        val imageFile = File(imagesDir, fileName)
        FileOutputStream(imageFile).use { out -> out.write(jpegBytes) }

        val c2wMatrix = ARCoreCoordinateUtils.arcoreMatrixToC2W(poseMatrix)
        capturedFrames.add(
            FrameMetaData(
                frameId = frameId,
                filePath = "images/$fileName",
                timestampNs = timestampNs,
                transformMatrix = c2wMatrix,
                sharpnessScore = sharpnessScore,
                meanLuminance = meanLuminance
            )
        )

        if (pointsXyzc != null) {
            val numPoints = pointsXyzc.size / 4
            var ptId = accumulatedFeaturePoints.size
            for (i in 0 until numPoints) {
                val confidence = pointsXyzc[i * 4 + 3]
                if (confidence > 0.1f) {
                    accumulatedFeaturePoints.add(
                        FeaturePoint3D(
                            ptId++,
                            pointsXyzc[i * 4 + 0],
                            pointsXyzc[i * 4 + 1],
                            pointsXyzc[i * 4 + 2],
                            confidence
                        )
                    )
                }
            }
        }
        return imageFile.absolutePath
    }

    fun saveCapturedFrame(
        bitmap: Bitmap,
        poseMatrix: FloatArray,
        timestampNs: Long,
        sharpnessScore: Float,
        meanLuminance: Float,
        pointCloud: PointCloud? = null
    ): String {
        val frameId = capturedFrames.size
        val fileName = String.format("frame_%04d.jpg", frameId)
        val imageFile = File(imagesDir, fileName)

        FileOutputStream(imageFile).use { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
        }

        // Convert ARCore 16-element column-major matrix to 4x4 row-major array
        val c2wMatrix = ARCoreCoordinateUtils.arcoreMatrixToC2W(poseMatrix)

        val metadata = FrameMetaData(
            frameId = frameId,
            filePath = "images/$fileName",
            timestampNs = timestampNs,
            transformMatrix = c2wMatrix,
            sharpnessScore = sharpnessScore,
            meanLuminance = meanLuminance
        )
        capturedFrames.add(metadata)

        // Accumulate 3D Feature Points from ARCore raw point cloud
        if (pointCloud != null) {
            val pointBuffer: FloatBuffer = pointCloud.points
            val numPoints = pointBuffer.remaining() / 4
            var ptId = accumulatedFeaturePoints.size
            for (i in 0 until numPoints) {
                val px = pointBuffer.get(i * 4 + 0)
                val py = pointBuffer.get(i * 4 + 1)
                val pz = pointBuffer.get(i * 4 + 2)
                val confidence = pointBuffer.get(i * 4 + 3)
                if (confidence > 0.1f) {
                    accumulatedFeaturePoints.add(FeaturePoint3D(ptId++, px, py, pz, confidence))
                }
            }
        }

        return imageFile.absolutePath
    }

    /**
     * Export complete NeRF / 3DGS compliant dataset files:
     * 1. transforms.json
     * 2. points3D_initial.json
     */
    fun exportDataset(fx: Float, fy: Float, cx: Float, cy: Float, width: Int, height: Int): File {
        val rootJson = JSONObject()
        val fovX = 2.0 * Math.atan((width / (2.0 * fx)).toDouble())
        val fovY = 2.0 * Math.atan((height / (2.0 * fy)).toDouble())

        rootJson.put("camera_angle_x", fovX)
        rootJson.put("camera_angle_y", fovY)
        rootJson.put("fl_x", fx)
        rootJson.put("fl_y", fy)
        rootJson.put("cx", cx)
        rootJson.put("cy", cy)
        rootJson.put("w", width)
        rootJson.put("h", height)
        rootJson.put("camera_model", "OPENCV")
        rootJson.put("system_source", "ARCore_6DoF_SLAM")

        val framesArray = JSONArray()
        for (frame in capturedFrames) {
            val frameObj = JSONObject()
            frameObj.put("file_path", frame.filePath)
            frameObj.put("timestamp_ns", frame.timestampNs)
            frameObj.put("sharpness_score", frame.sharpnessScore)
            frameObj.put("mean_luminance", frame.meanLuminance)

            val matrixJson = JSONArray()
            for (row in frame.transformMatrix) {
                val rowJson = JSONArray()
                for (valItem in row) {
                    rowJson.put(valItem.toDouble())
                }
                matrixJson.put(rowJson)
            }
            frameObj.put("transform_matrix", matrixJson)
            framesArray.put(frameObj)
        }
        rootJson.put("frames", framesArray)

        // Seed the optimizer with real ARCore surface geometry. Without a
        // ply_file_path the engine initialises from random points inside the
        // camera frustums, which produces a formless blob on short runs.
        val plyFile = File(outputDir, "points3d.ply")
        val seededPoints = com.splat.mobile3dgs.engine.GaussianInitializer
            .writeInitialPointCloudPly(accumulatedFeaturePoints, plyFile)
        if (seededPoints > 0) {
            rootJson.put("ply_file_path", plyFile.name)
            android.util.Log.i(
                "DatasetExporter",
                "Seeded initial geometry: $seededPoints points -> ${plyFile.name}"
            )
        } else {
            android.util.Log.w(
                "DatasetExporter",
                "No ARCore feature points captured; engine will fall back to random init"
            )
        }

        val jsonFile = File(outputDir, "transforms.json")
        jsonFile.writeText(rootJson.toString(2))

        exportInitialPointCloud()

        return jsonFile
    }

    private fun exportInitialPointCloud() {
        val ptsJson = JSONObject()
        val pointsArray = JSONArray()

        val sampleStep = maxOf(1, accumulatedFeaturePoints.size / 10000)
        for (i in 0 until accumulatedFeaturePoints.size step sampleStep) {
            val pt = accumulatedFeaturePoints[i]
            val ptObj = JSONObject()
            ptObj.put("id", pt.id)
            ptObj.put("xyz", JSONArray(listOf(pt.x, pt.y, pt.z)))
            ptObj.put("confidence", pt.confidence)
            pointsArray.put(ptObj)
        }

        ptsJson.put("num_points", pointsArray.length())
        ptsJson.put("points", pointsArray)

        val ptsFile = File(outputDir, "points3D_initial.json")
        ptsFile.writeText(ptsJson.toString(2))
    }
}
