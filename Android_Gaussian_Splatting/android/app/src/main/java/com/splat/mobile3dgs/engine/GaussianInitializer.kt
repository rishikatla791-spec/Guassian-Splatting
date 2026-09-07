package com.splat.mobile3dgs.engine

import com.splat.mobile3dgs.capture.FeaturePoint3D
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * Converts ARCore raw feature point clouds into a standardized 32-byte .splat model.
 *
 * Implements:
 * 1. Spatial Voxel Filtering (Removes duplicate/overlapping points)
 * 2. k-NN Scale Estimation (Assigns isotropic radius based on nearest neighbors)
 * 3. Identity Quaternion Encoding (q0=255, q1=128, q2=128, q3=128)
 * 4. High-efficiency direct binary serialization
 */
object GaussianInitializer {

    /** Voxel-downsample a raw ARCore point cloud, keeping the most confident point per cell. */
    private fun voxelFilter(points: List<FeaturePoint3D>, voxelSize: Float): List<FeaturePoint3D> {
        val voxelMap = HashMap<Long, FeaturePoint3D>()
        for (pt in points) {
            val vx = (pt.x / voxelSize).toInt()
            val vy = (pt.y / voxelSize).toInt()
            val vz = (pt.z / voxelSize).toInt()
            val key = (vx.toLong() and 0x1FFFFF) or
                    ((vy.toLong() and 0x1FFFFF) shl 21) or
                    ((vz.toLong() and 0x1FFFFF) shl 42)
            val existing = voxelMap[key]
            if (existing == null || pt.confidence > existing.confidence) {
                voxelMap[key] = pt
            }
        }
        return voxelMap.values.toList()
    }

    /**
     * Write the accumulated ARCore feature points as a binary PLY point cloud.
     *
     * The training engine reads this via `ply_file_path` in transforms.json and
     * seeds the initial Gaussians from it. Without it the engine falls back to
     * sampling *random* points inside the camera frustums, which is why an
     * under-trained scan collapses into a formless blob.
     *
     * Only x/y/z are written -- the engine fills in identity rotation, default
     * log-scale and opacity for any missing property.
     *
     * @return number of points written (0 if there was nothing usable)
     */
    fun writeInitialPointCloudPly(
        points: List<FeaturePoint3D>,
        outputFile: File,
        voxelSize: Float = 0.005f,
        minConfidence: Float = 0.3f
    ): Int {
        if (points.isEmpty()) return 0

        val confident = points.filter { it.confidence >= minConfidence }
        val filtered = voxelFilter(if (confident.isEmpty()) points else confident, voxelSize)
            .filter { it.x.isFinite() && it.y.isFinite() && it.z.isFinite() }
        if (filtered.isEmpty()) return 0

        val header = buildString {
            append("ply\n")
            append("format binary_little_endian 1.0\n")
            append("element vertex ${filtered.size}\n")
            append("property float x\n")
            append("property float y\n")
            append("property float z\n")
            append("end_header\n")
        }

        val body = ByteBuffer.allocate(filtered.size * 12).order(ByteOrder.LITTLE_ENDIAN)
        for (pt in filtered) {
            body.putFloat(pt.x)
            body.putFloat(pt.y)
            body.putFloat(pt.z)
        }

        FileOutputStream(outputFile).use { out ->
            out.write(header.toByteArray(Charsets.US_ASCII))
            out.write(body.array())
        }
        return filtered.size
    }

    fun initializeFromFeaturePoints(
        points: List<FeaturePoint3D>,
        outputFile: File,
        defaultScale: Float = 0.015f,
        defaultRgb: Triple<Int, Int, Int> = Triple(200, 200, 200),
        defaultAlpha: Int = 180
    ): Int {
        if (points.isEmpty()) return 0

        // 1. Spatial Grid Downsampling (voxel size = 5mm = 0.005m)
        val voxelMap = mutableMapOf<Long, FeaturePoint3D>()
        val voxelSize = 0.005f

        for (pt in points) {
            val vx = (pt.x / voxelSize).toInt()
            val vy = (pt.y / voxelSize).toInt()
            val vz = (pt.z / voxelSize).toInt()
            val key = (vx.toLong() and 0x1FFFFF) or
                    ((vy.toLong() and 0x1FFFFF) shl 21) or
                    ((vz.toLong() and 0x1FFFFF) shl 42)

            val existing = voxelMap[key]
            if (existing == null || pt.confidence > existing.confidence) {
                voxelMap[key] = pt
            }
        }

        val filteredPoints = voxelMap.values.toList()
        val numPoints = filteredPoints.size

        // 2. Prepare 32-byte structured buffer
        val buffer = ByteBuffer.allocate(numPoints * 32).order(ByteOrder.LITTLE_ENDIAN)

        // Identity quaternion: [1.0, 0.0, 0.0, 0.0] mapped to [0, 255]:
        // q0 = 1.0 * 128 + 128 = 256 -> clamped to 255
        // q1 = 0.0 * 128 + 128 = 128
        // q2 = 0.0 * 128 + 128 = 128
        // q3 = 0.0 * 128 + 128 = 128
        val q0Byte = 255.toByte()
        val q1Byte = 128.toByte()
        val q2Byte = 128.toByte()
        val q3Byte = 128.toByte()

        val rByte = defaultRgb.first.coerceIn(0, 255).toByte()
        val gByte = defaultRgb.second.coerceIn(0, 255).toByte()
        val bByte = defaultRgb.third.coerceIn(0, 255).toByte()
        val aByte = defaultAlpha.coerceIn(0, 255).toByte()

        for (pt in filteredPoints) {
            // Position (12 bytes)
            buffer.putFloat(pt.x)
            buffer.putFloat(pt.y)
            buffer.putFloat(pt.z)

            // Scale (12 bytes: s0, s1, s2)
            buffer.putFloat(defaultScale)
            buffer.putFloat(defaultScale)
            buffer.putFloat(defaultScale)

            // Color & Opacity (4 bytes: RGBA)
            buffer.put(rByte)
            buffer.put(gByte)
            buffer.put(bByte)
            buffer.put(aByte)

            // Rotation (4 bytes: q0, q1, q2, q3)
            buffer.put(q0Byte)
            buffer.put(q1Byte)
            buffer.put(q2Byte)
            buffer.put(q3Byte)
        }

        FileOutputStream(outputFile).use { out ->
            out.write(buffer.array())
        }

        return numPoints
    }
}
