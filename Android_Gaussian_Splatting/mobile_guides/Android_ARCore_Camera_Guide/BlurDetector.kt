package com.gaussian.cameraguide

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.pow

object BlurDetector {

    /**
     * Computes Laplacian variance on a Bitmap frame.
     * Values > 100 indicate a crisp sharp image; values < 100 indicate motion blur.
     */
    fun computeLaplacianVariance(bitmap: Bitmap): Float {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // Convert to Grayscale matrix
        val gray = FloatArray(width * height)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            gray[i] = 0.299f * r + 0.587f * g + 0.114f * b
        }

        // Apply 3x3 Laplacian Kernel:
        //  0   1   0
        //  1  -4   1
        //  0   1   0
        val laplacian = FloatArray((width - 2) * (height - 2))
        var idx = 0
        var sum = 0.0
        var sumSq = 0.0

        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                val centerVal = gray[y * width + x]
                val topVal    = gray[(y - 1) * width + x]
                val bottomVal = gray[(y + 1) * width + x]
                val leftVal   = gray[y * width + (x - 1)]
                val rightVal  = gray[y * width + (x + 1)]

                val lapVal = topVal + bottomVal + leftVal + rightVal - (4.0f * centerVal)
                laplacian[idx++] = lapVal
                sum += lapVal
                sumSq += lapVal * lapVal
            }
        }

        val count = laplacian.size
        if (count == 0) return 0.0f

        val mean = sum / count
        val variance = (sumSq / count) - (mean * mean)

        return variance.toFloat()
    }
}
