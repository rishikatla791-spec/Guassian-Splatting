package com.splat.mobile3dgs.capture

import android.media.Image
import com.google.ar.core.Pose

/**
 * Unprojects an ARCore depth map into world-space 3D points.
 *
 * ARCore's feature point cloud yields only a few hundred points per scan, which
 * is far too sparse to seed a Gaussian Splatting optimizer. The Depth API gives
 * a dense depth map per frame; unprojecting it produces tens of thousands of
 * real surface points, so training starts near the true geometry instead of
 * from random points inside the camera frustums.
 */
object DepthPointExtractor {

    // ARCore depth maps are tiny (e.g. 160x90 = 14k pixels), so sampling every
    // pixel costs almost nothing. The earlier 1500-sample budget was throwing away
    // most of the map for no benefit.
    private const val TARGET_SAMPLES_PER_FRAME = 20000

    /**
     * @param depthImage DEPTH16 image; each pixel is millimetres along the camera ray.
     * @param confidenceImage optional Y8 image, same dimensions, 0..255.
     * @param focal  [fx, fy] of the CPU camera image.
     * @param principal [cx, cy] of the CPU camera image.
     * @param imageDims [width, height] of the CPU camera image the intrinsics belong to.
     * @param cameraPose camera-to-world pose for this frame.
     * @return flat [x, y, z, confidence, ...] array in world space.
     */
    fun extractWorldPoints(
        depthImage: Image,
        confidenceImage: Image?,
        focal: FloatArray,
        principal: FloatArray,
        imageDims: IntArray,
        cameraPose: Pose,
        minConfidence: Float = 0.1f,
        minDepthMeters: Float = 0.15f,
        maxDepthMeters: Float = 6.0f
    ): FloatArray {
        val dw = depthImage.width
        val dh = depthImage.height
        if (dw <= 0 || dh <= 0) return FloatArray(0)

        val depthPlane = depthImage.planes[0]
        val depthBuf = depthPlane.buffer
        val depthRowStride = depthPlane.rowStride
        val depthPixStride = depthPlane.pixelStride

        var confBuf: java.nio.ByteBuffer? = null
        var confRowStride = 0
        var confPixStride = 1
        if (confidenceImage != null &&
            confidenceImage.width == dw && confidenceImage.height == dh
        ) {
            val p = confidenceImage.planes[0]
            confBuf = p.buffer
            confRowStride = p.rowStride
            confPixStride = p.pixelStride
        }

        // Scale the CPU-image intrinsics down to the (much smaller) depth image.
        val sx = dw.toFloat() / imageDims[0].toFloat()
        val sy = dh.toFloat() / imageDims[1].toFloat()
        val fx = focal[0] * sx
        val fy = focal[1] * sy
        val cx = principal[0] * sx
        val cy = principal[1] * sy
        if (fx <= 0f || fy <= 0f) return FloatArray(0)

        // Sample on a grid so cost stays flat regardless of depth resolution.
        val totalPixels = dw * dh
        var stride = 1
        if (totalPixels > TARGET_SAMPLES_PER_FRAME) {
            stride = Math.sqrt(totalPixels.toDouble() / TARGET_SAMPLES_PER_FRAME).toInt().coerceAtLeast(1)
        }

        // Camera-to-world matrix, column-major (m[col * 4 + row]).
        val m = FloatArray(16)
        cameraPose.toMatrix(m, 0)

        val estimated = ((dw / stride) + 1) * ((dh / stride) + 1)
        val out = FloatArray(estimated * 4)
        var n = 0

        var v = 0
        while (v < dh) {
            var u = 0
            while (u < dw) {
                val depthIndex = v * depthRowStride + u * depthPixStride
                if (depthIndex < 0 || depthIndex + 1 >= depthBuf.capacity()) { u += stride; continue }

                val mm = depthBuf.getShort(depthIndex).toInt() and 0xFFFF
                if (mm == 0) { u += stride; continue }
                val d = mm / 1000.0f
                if (d < minDepthMeters || d > maxDepthMeters) { u += stride; continue }

                var conf = 1.0f
                val cb = confBuf
                if (cb != null) {
                    val ci = v * confRowStride + u * confPixStride
                    if (ci >= 0 && ci < cb.capacity()) {
                        conf = (cb.get(ci).toInt() and 0xFF) / 255.0f
                    }
                }
                if (conf < minConfidence) { u += stride; continue }

                // Pinhole unprojection in the CV camera frame (x right, y down, z forward),
                // then converted to ARCore's GL convention (y up, -z forward).
                val xCv = (u - cx) * d / fx
                val yCv = (v - cy) * d / fy
                val xC = xCv
                val yC = -yCv
                val zC = -d

                // world = M * pointCamera
                val wx = m[0] * xC + m[4] * yC + m[8] * zC + m[12]
                val wy = m[1] * xC + m[5] * yC + m[9] * zC + m[13]
                val wz = m[2] * xC + m[6] * yC + m[10] * zC + m[14]

                if (wx.isFinite() && wy.isFinite() && wz.isFinite() && n + 4 <= out.size) {
                    out[n] = wx; out[n + 1] = wy; out[n + 2] = wz; out[n + 3] = conf
                    n += 4
                }
                u += stride
            }
            v += stride
        }

        return if (n == out.size) out else out.copyOf(n)
    }
}
