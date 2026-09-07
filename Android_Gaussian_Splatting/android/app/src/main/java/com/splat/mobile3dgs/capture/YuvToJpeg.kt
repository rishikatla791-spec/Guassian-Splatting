package com.splat.mobile3dgs.capture

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import java.io.ByteArrayOutputStream

/**
 * Converts an ARCore YUV_420_888 [Image] to JPEG bytes.
 *
 * The image is kept in its native (sensor / landscape) orientation so that the
 * saved pixels stay consistent with the camera intrinsics reported by
 * `frame.camera.imageIntrinsics` and the extrinsics from `frame.camera.pose`.
 * Do NOT rotate here, or the intrinsics/extrinsics would no longer match.
 */
object YuvToJpeg {

    fun toJpeg(image: Image, quality: Int = 95): ByteArray {
        require(image.format == ImageFormat.YUV_420_888) {
            "Expected YUV_420_888 but got ${image.format}"
        }
        val nv21 = yuv420888ToNv21(image)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), quality, out)
        return out.toByteArray()
    }

    /** Also returns mean luminance (0..255) computed from the Y plane, for quality metrics. */
    fun meanLuminance(image: Image): Float {
        val yPlane = image.planes[0]
        val buffer = yPlane.buffer.duplicate()
        val rowStride = yPlane.rowStride
        val width = image.width
        val height = image.height
        var sum = 0L
        var count = 0L
        // Sample every 4th pixel/row for speed.
        var row = 0
        while (row < height) {
            val base = row * rowStride
            var col = 0
            while (col < width) {
                sum += (buffer.get(base + col).toInt() and 0xFF).toLong()
                count++
                col += 4
            }
            row += 4
        }
        return if (count > 0) sum.toFloat() / count else 0f
    }

    private fun yuv420888ToNv21(image: Image): ByteArray {
        val width = image.width
        val height = image.height
        val ySize = width * height
        val nv21 = ByteArray(ySize + ySize / 2)

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        // --- Copy Y plane (respecting rowStride) ---
        val yRowStride = yPlane.rowStride
        if (yRowStride == width) {
            yBuffer.get(nv21, 0, ySize)
        } else {
            var pos = 0
            for (r in 0 until height) {
                yBuffer.position(r * yRowStride)
                yBuffer.get(nv21, pos, width)
                pos += width
            }
        }

        // --- Interleave V,U as NV21 (VU order), respecting strides ---
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride
        val chromaHeight = height / 2
        val chromaWidth = width / 2
        var offset = ySize
        for (r in 0 until chromaHeight) {
            val uRow = r * uvRowStride
            val vRow = r * uvRowStride
            for (c in 0 until chromaWidth) {
                val uIndex = uRow + c * uvPixelStride
                val vIndex = vRow + c * uvPixelStride
                nv21[offset++] = vBuffer.get(vIndex)
                nv21[offset++] = uBuffer.get(uIndex)
            }
        }
        return nv21
    }
}
