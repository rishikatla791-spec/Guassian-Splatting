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

    // Why samples were discarded on the most recent call. Without this a zero
    // yield looks identical to "depth works fine", which is exactly how seeding
    // silently collapsed from 90k points to 0.
    var lastSampled = 0; private set
    var lastZeroDepth = 0; private set
    var lastOutOfRange = 0; private set
    var lastLowConfidence = 0; private set
    var lastKept = 0; private set
    /** Median depth of the frame centre, i.e. how far away the subject is. */
    var lastSubjectDepth = 0f; private set

    fun lastStats(): String =
        "sampled=$lastSampled kept=$lastKept zero=$lastZeroDepth outOfRange=$lastOutOfRange lowConf=$lastLowConfidence"

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
        colorImage: Image? = null,
        minConfidence: Float = 0.1f,
        minDepthMeters: Float = 0.10f,
        maxDepthMeters: Float = 8.0f,
        /** Keep depth within this multiple of the frame's central (subject) depth. */
        subjectBandLow: Float = 0.5f,
        subjectBandHigh: Float = 1.8f
    ): FloatArray {
        lastSampled = 0; lastZeroDepth = 0; lastOutOfRange = 0; lastLowConfidence = 0; lastKept = 0
        val dw = depthImage.width
        val dh = depthImage.height
        if (dw <= 0 || dh <= 0) return FloatArray(0)

        val depthPlane = depthImage.planes[0]
        // DEPTH16 samples are 16-bit little-endian on every Android device, but the
        // byte order of the buffer handed back by Image.Plane is not specified --
        // a plain java.nio direct buffer defaults to BIG_ENDIAN. Reading with the
        // wrong order byte-swaps every sample (500 mm reads as 62 m), which does
        // not throw: the points simply fall outside the usable range and the cloud
        // silently collapses. Pin it, as the official ARCore depth samples do.
        val depthBuf = depthPlane.buffer.order(java.nio.ByteOrder.LITTLE_ENDIAN)
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

        // Find the subject's depth plane from the centre of the frame, then reject
        // anything far off it. The depth map reaches ~8 m, so scanning a small object
        // indoors otherwise fills the entire point budget with walls and floor -- a
        // plastic box produced 250k seeds of which only 0.7% were on the box. The
        // user keeps the subject centred, so the central median is a good proxy, and
        // the window is relative so a room scan keeps its own walls.
        var loD = minDepthMeters
        var hiD = maxDepthMeters
        run {
            val cx0 = (dw * 0.3f).toInt(); val cx1 = (dw * 0.7f).toInt()
            val cy0 = (dh * 0.3f).toInt(); val cy1 = (dh * 0.7f).toInt()
            val central = ArrayList<Float>(((cx1 - cx0) / 2 + 1) * ((cy1 - cy0) / 2 + 1))
            var vv = cy0
            while (vv < cy1) {
                var uu = cx0
                while (uu < cx1) {
                    val idx = vv * depthRowStride + uu * depthPixStride
                    if (idx >= 0 && idx + 1 < depthBuf.capacity()) {
                        val mmC = depthBuf.getShort(idx).toInt() and 0xFFFF
                        if (mmC != 0) {
                            val dC = mmC / 1000.0f
                            if (dC in minDepthMeters..maxDepthMeters) central.add(dC)
                        }
                    }
                    uu += 2
                }
                vv += 2
            }
            if (central.size >= 20) {
                central.sort()
                val subjectDepth = central[central.size / 2]
                lastSubjectDepth = subjectDepth
                loD = kotlin.math.max(minDepthMeters, subjectDepth * subjectBandLow)
                hiD = kotlin.math.min(maxDepthMeters, subjectDepth * subjectBandHigh)
            }
        }

        // Camera-to-world matrix, column-major (m[col * 4 + row]).
        val m = FloatArray(16)
        cameraPose.toMatrix(m, 0)

        // Sample the camera image so each seed carries its real colour, and derive a
        // real-world size from its depth. Without these the engine starts every point
        // as a 1.8 cm mid-grey blob and wastes thousands of steps relearning both.
        var yBuf: java.nio.ByteBuffer? = null; var yRow = 0
        var uBuf: java.nio.ByteBuffer? = null; var vBuf: java.nio.ByteBuffer? = null
        var uvRow = 0; var uvPix = 1
        var cw = 0; var ch = 0
        if (colorImage != null && colorImage.format == android.graphics.ImageFormat.YUV_420_888) {
            cw = colorImage.width; ch = colorImage.height
            val yp = colorImage.planes[0]; yBuf = yp.buffer; yRow = yp.rowStride
            val up = colorImage.planes[1]; uBuf = up.buffer
            val vp = colorImage.planes[2]; vBuf = vp.buffer
            uvRow = up.rowStride; uvPix = up.pixelStride
        }
        // Angular width of one depth pixel -> world size at distance d is d * pixelAngle.
        val pixelAngle = (imageDims[0].toFloat() / dw.toFloat()) / focal[0]

        val estimated = ((dw / stride) + 1) * ((dh / stride) + 1)
        val out = FloatArray(estimated * 8)
        var n = 0

        var v = 0
        while (v < dh) {
            var u = 0
            while (u < dw) {
                val depthIndex = v * depthRowStride + u * depthPixStride
                if (depthIndex < 0 || depthIndex + 1 >= depthBuf.capacity()) { u += stride; continue }

                lastSampled++
                val mm = depthBuf.getShort(depthIndex).toInt() and 0xFFFF
                if (mm == 0) { lastZeroDepth++; u += stride; continue }
                val d = mm / 1000.0f
                if (d < loD || d > hiD) { lastOutOfRange++; u += stride; continue }

                var conf = 1.0f
                val cb = confBuf
                if (cb != null) {
                    val ci = v * confRowStride + u * confPixStride
                    if (ci >= 0 && ci < cb.capacity()) {
                        conf = (cb.get(ci).toInt() and 0xFF) / 255.0f
                    }
                }
                if (conf < minConfidence) { lastLowConfidence++; u += stride; continue }

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

                if (wx.isFinite() && wy.isFinite() && wz.isFinite() && n + 8 <= out.size) {
                    var r = 0.5f; var g = 0.5f; var b = 0.5f
                    val yb = yBuf
                    if (yb != null && cw > 0) {
                        val sx2 = (u.toFloat() / dw * cw).toInt().coerceIn(0, cw - 1)
                        val sy2 = (v.toFloat() / dh * ch).toInt().coerceIn(0, ch - 1)
                        val yi = sy2 * yRow + sx2
                        val ci = (sy2 / 2) * uvRow + (sx2 / 2) * uvPix
                        if (yi < yb.capacity() && ci < (uBuf?.capacity() ?: 0) && ci < (vBuf?.capacity() ?: 0)) {
                            val yv = (yb.get(yi).toInt() and 0xFF).toFloat()
                            val uv = (uBuf!!.get(ci).toInt() and 0xFF) - 128f
                            val vv = (vBuf!!.get(ci).toInt() and 0xFF) - 128f
                            r = ((yv + 1.370705f * vv) / 255f).coerceIn(0f, 1f)
                            g = ((yv - 0.337633f * uv - 0.698001f * vv) / 255f).coerceIn(0f, 1f)
                            b = ((yv + 1.732446f * uv) / 255f).coerceIn(0f, 1f)
                        }
                    }
                    // log-space scale, as the PLY format expects
                    val worldSize = (d * pixelAngle).coerceIn(0.002f, 0.5f)
                    val logScale = kotlin.math.ln(worldSize)

                    out[n] = wx; out[n + 1] = wy; out[n + 2] = wz; out[n + 3] = conf
                    out[n + 4] = r; out[n + 5] = g; out[n + 6] = b; out[n + 7] = logScale
                    n += 8
                    lastKept++
                }
                u += stride
            }
            v += stride
        }

        return if (n == out.size) out else out.copyOf(n)
    }
}
