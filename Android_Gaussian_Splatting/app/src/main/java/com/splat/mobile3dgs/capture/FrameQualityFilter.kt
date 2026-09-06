package com.splat.mobile3dgs.capture

import android.media.Image
import java.nio.ByteBuffer

/**
 * Production-Grade Zero-Allocation Frame Quality & Motion Filter.
 *
 * Evaluates:
 * 1. Fast YUV_420_888 Luminance Laplacian Variance (Blur Detection)
 * 2. Exposure & Contrast (Under/Over-exposed frame rejection)
 */
class FrameQualityFilter(
    private val minBlurVarianceThreshold: Float = 80.0f,
    private val minLuminanceMean: Float = 20.0f,
    private val maxLuminanceMean: Float = 240.0f
) {

    // Reusable buffer to eliminate allocations during 60 FPS frame updates
    private var sampleBuffer = FloatArray(0)

    data class QualityReport(
        val isPassed: Boolean,
        val blurScore: Float,
        val meanLuminance: Float,
        val failureReason: String?
    )

    fun evaluateFrameQuality(image: Image): QualityReport {
        if (image.format != android.graphics.ImageFormat.YUV_420_888) {
            return QualityReport(false, 0f, 0f, "Unsupported image format")
        }

        val yPlane = image.planes[0]
        val yBuffer: ByteBuffer = yPlane.buffer
        val width = image.width
        val height = image.height
        val rowStride = yPlane.rowStride
        val pixelStride = yPlane.pixelStride

        val step = 4
        val sampleW = (width - 2) / step
        val sampleH = (height - 2) / step
        val totalSamples = sampleW * sampleH

        if (sampleBuffer.size < totalSamples) {
            sampleBuffer = FloatArray(totalSamples)
        }

        var sumY = 0.0
        var sumY2 = 0.0
        var idx = 0

        for (y in 1 until height - 1 step step) {
            val rowOffset = y * rowStride
            val prevRowOffset = (y - 1) * rowStride
            val nextRowOffset = (y + 1) * rowStride

            for (x in 1 until width - 1 step step) {
                val c = (yBuffer.get(rowOffset + x * pixelStride).toInt() and 0xFF).toFloat()
                val t = (yBuffer.get(prevRowOffset + x * pixelStride).toInt() and 0xFF).toFloat()
                val b = (yBuffer.get(nextRowOffset + x * pixelStride).toInt() and 0xFF).toFloat()
                val l = (yBuffer.get(rowOffset + (x - 1) * pixelStride).toInt() and 0xFF).toFloat()
                val r = (yBuffer.get(rowOffset + (x + 1) * pixelStride).toInt() and 0xFF).toFloat()

                val lap = t + b + l + r - (4.0f * c)
                sampleBuffer[idx++] = lap

                sumY += c
                sumY2 += (c * c)
            }
        }

        val count = idx
        if (count == 0) {
            return QualityReport(false, 0f, 0f, "Empty image buffer")
        }

        val meanLum = (sumY / count).toFloat()
        if (meanLum < minLuminanceMean) {
            return QualityReport(false, 0f, meanLum, "Frame too dark (Underexposed)")
        }
        if (meanLum > maxLuminanceMean) {
            return QualityReport(false, 0f, meanLum, "Frame overexposed")
        }

        var lapSum = 0.0
        var lapSumSq = 0.0
        for (i in 0 until count) {
            val v = sampleBuffer[i]
            lapSum += v
            lapSumSq += (v * v)
        }

        val lapMean = lapSum / count
        val blurVariance = ((lapSumSq / count) - (lapMean * lapMean)).toFloat()

        if (blurVariance < minBlurVarianceThreshold) {
            return QualityReport(false, blurVariance, meanLum, "Motion blur detected (Score: ${blurVariance.toInt()})")
        }

        return QualityReport(true, blurVariance, meanLum, null)
    }
}
