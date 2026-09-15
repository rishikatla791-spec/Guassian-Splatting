package com.splat.mobile3dgs.capture

import android.media.Image
import android.util.Log
import java.nio.ByteBuffer

/**
 * Zero-allocation frame quality gate for the keyframe path.
 *
 * Blurry input produces blurry Gaussians: a motion-blurred keyframe teaches the
 * optimizer that the surface really is smeared, and no amount of training
 * recovers the detail. Rejecting the frame costs nothing (another one arrives
 * 30 ms later) so the gate runs on every keyframe candidate before the image is
 * ever JPEG-encoded.
 *
 * Two quantities are measured in a single pass over the Y plane:
 *  1. Laplacian variance    -> sharpness (low variance == blur)
 *  2. Mean luminance        -> exposure  (a black or blown-out frame carries no
 *                              usable photometric signal)
 *
 * Sharpness is scene-dependent (a textureless white wall scores low even when
 * perfectly in focus), so an absolute threshold alone either passes blur in a
 * detailed scene or starves capture in a plain one. The gate therefore combines
 * an absolute floor with a *relative* one derived from an EWMA of recently
 * observed scores, and relaxes the absolute floor if it ever starves capture.
 */
class FrameQualityFilter(
    private val absoluteBlurFloor: Float = ABSOLUTE_BLUR_FLOOR,
    private val minLuminanceMean: Float = MIN_LUMINANCE_MEAN,
    private val maxLuminanceMean: Float = MAX_LUMINANCE_MEAN,
    private val relativeSharpnessFraction: Float = RELATIVE_SHARPNESS_FRACTION
) {

    companion object {
        private const val TAG = "FrameQualityFilter"

        /**
         * Hard sharpness floor (Laplacian variance of the 8-bit Y plane).
         * ~100 is the classic OpenCV "is it blurry" threshold for photographs;
         * 80 leaves headroom for the darker, noisier frames a phone produces
         * while the user is walking.
         */
        const val ABSOLUTE_BLUR_FLOOR = 80.0f

        /** Below this mean Y the frame is essentially black. */
        const val MIN_LUMINANCE_MEAN = 20.0f

        /** Above this mean Y highlights are clipped and colour is gone. */
        const val MAX_LUMINANCE_MEAN = 240.0f

        /**
         * A frame is rejected if it is sharper-than-this fraction of the running
         * average sharpness. Catches the frames that are blurry *for this scene*
         * even when the scene as a whole scores high.
         */
        const val RELATIVE_SHARPNESS_FRACTION = 0.6f

        /** Weight of the newest sample in the running sharpness average. */
        private const val SHARPNESS_EWMA_ALPHA = 0.1f

        /**
         * If this many candidate keyframes are rejected back to back, the scene
         * is simply low-texture rather than blurry. Drop the absolute floor and
         * rely on the relative gate, so a plain-walled room still scans.
         */
        const val REJECT_STREAK_LIMIT = 24

        /**
         * Laplacian samples per frame. The kernel always reads adjacent pixels,
         * so subsampling the *grid* leaves the variance estimate unbiased while
         * keeping the cost flat across 720p..4K CPU images.
         */
        private const val TARGET_LAPLACIAN_SAMPLES = 40_000

        /** Never sample denser than this; adjacent pixels are highly correlated. */
        private const val MIN_SAMPLE_STEP = 4
    }

    enum class Verdict { PASSED, MOTION_BLUR, UNDEREXPOSED, OVEREXPOSED, UNUSABLE }

    data class QualityReport(
        val isPassed: Boolean,
        val blurScore: Float,
        val meanLuminance: Float,
        val verdict: Verdict,
        val failureReason: String?
    )

    // Reusable buffer to eliminate allocations during 60 FPS frame updates.
    private var sampleBuffer = FloatArray(0)

    // Running statistics / rejection bookkeeping.
    private var sharpnessEwma = 0.0f
    private var rejectStreak = 0
    private var floorRelaxed = false
    private var evaluated = 0
    private var passed = 0
    private var rejectedBlur = 0
    private var rejectedDark = 0
    private var rejectedBright = 0
    private var rejectedUnusable = 0
    private var measurementFailureLogged = false

    /** Number of candidate keyframes rejected so far. */
    val rejectedCount: Int
        get() = rejectedBlur + rejectedDark + rejectedBright + rejectedUnusable

    /** True once the absolute sharpness floor has been relaxed for a low-texture scene. */
    val isFloorRelaxed: Boolean get() = floorRelaxed

    /** One-line breakdown for the capture log. */
    fun summary(): String =
        "$rejectedCount/$evaluated candidate keyframes rejected " +
            "(blur=$rejectedBlur, dark=$rejectedDark, bright=$rejectedBright, " +
            "unreadable=$rejectedUnusable; kept=$passed; " +
            "avgSharpness=${"%.0f".format(sharpnessEwma)}" +
            (if (floorRelaxed) ", absolute floor relaxed for low-texture scene" else "") + ")"

    fun reset() {
        sharpnessEwma = 0.0f
        rejectStreak = 0
        floorRelaxed = false
        evaluated = 0; passed = 0
        rejectedBlur = 0; rejectedDark = 0; rejectedBright = 0; rejectedUnusable = 0
        measurementFailureLogged = false
    }

    fun evaluateFrameQuality(image: Image): QualityReport {
        evaluated++

        if (image.format != android.graphics.ImageFormat.YUV_420_888) {
            return reject(Verdict.UNUSABLE, 0f, 0f, "Unsupported image format ${image.format}")
        }
        val width = image.width
        val height = image.height
        if (width < 3 || height < 3) {
            return reject(Verdict.UNUSABLE, 0f, 0f, "Image too small (${width}x$height)")
        }

        val yPlane = image.planes[0]
        val yBuffer: ByteBuffer = yPlane.buffer
        val rowStride = yPlane.rowStride
        val pixelStride = yPlane.pixelStride

        val innerW = width - 2
        val innerH = height - 2
        val step = sampleStep(innerW, innerH)
        val sampleW = (innerW + step - 1) / step
        val sampleH = (innerH + step - 1) / step
        val totalSamples = sampleW * sampleH
        if (totalSamples <= 0) {
            return reject(Verdict.UNUSABLE, 0f, 0f, "Empty image buffer")
        }
        // Ceil-based sizing: `for (y in 1 until height - 1 step s)` runs
        // ceil(innerH / s) times, which integer division under-counts.
        if (sampleBuffer.size < totalSamples) {
            sampleBuffer = FloatArray(totalSamples)
        }

        var sumY = 0.0
        var idx = 0
        try {
            var y = 1
            while (y < height - 1) {
                val rowOffset = y * rowStride
                val prevRowOffset = (y - 1) * rowStride
                val nextRowOffset = (y + 1) * rowStride

                var x = 1
                while (x < width - 1 && idx < totalSamples) {
                    val xo = x * pixelStride
                    val c = (yBuffer.get(rowOffset + xo).toInt() and 0xFF).toFloat()
                    val t = (yBuffer.get(prevRowOffset + xo).toInt() and 0xFF).toFloat()
                    val b = (yBuffer.get(nextRowOffset + xo).toInt() and 0xFF).toFloat()
                    val l = (yBuffer.get(rowOffset + (x - 1) * pixelStride).toInt() and 0xFF).toFloat()
                    val r = (yBuffer.get(rowOffset + (x + 1) * pixelStride).toInt() and 0xFF).toFloat()

                    sampleBuffer[idx++] = t + b + l + r - (4.0f * c)
                    sumY += c
                    x += step
                }
                y += step
            }
        } catch (t: Throwable) {
            // An unexpected plane layout must never cost the user their scan:
            // stop measuring and let the frame through.
            if (!measurementFailureLogged) {
                measurementFailureLogged = true
                Log.w(TAG, "Quality measurement failed, passing frames unfiltered: ${t.message}")
            }
            passed++
            rejectStreak = 0
            return QualityReport(true, 0f, 0f, Verdict.PASSED, null)
        }

        val count = idx
        if (count == 0) {
            return reject(Verdict.UNUSABLE, 0f, 0f, "Empty image buffer")
        }

        val meanLum = (sumY / count).toFloat()
        if (meanLum < minLuminanceMean) {
            return reject(Verdict.UNDEREXPOSED, 0f, meanLum, "Too dark (mean Y ${meanLum.toInt()})")
        }
        if (meanLum > maxLuminanceMean) {
            return reject(Verdict.OVEREXPOSED, 0f, meanLum, "Overexposed (mean Y ${meanLum.toInt()})")
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

        // Update the running average BEFORE gating so a scene that legitimately
        // gets less detailed drags the relative threshold down with it.
        sharpnessEwma =
            if (evaluated == 1 || sharpnessEwma <= 0f) blurVariance
            else sharpnessEwma + SHARPNESS_EWMA_ALPHA * (blurVariance - sharpnessEwma)

        val relativeFloor = sharpnessEwma * relativeSharpnessFraction
        val effectiveFloor =
            if (floorRelaxed) relativeFloor else maxOf(absoluteBlurFloor, relativeFloor)

        if (blurVariance < effectiveFloor) {
            val report = reject(
                Verdict.MOTION_BLUR, blurVariance, meanLum,
                "Motion blur (sharpness ${blurVariance.toInt()} < ${effectiveFloor.toInt()})"
            )
            if (!floorRelaxed && rejectStreak >= REJECT_STREAK_LIMIT &&
                blurVariance >= relativeFloor
            ) {
                // Everything is failing the ABSOLUTE floor only -> low-texture
                // scene, not a shaky user. Keep scanning on the relative gate.
                floorRelaxed = true
                Log.i(
                    TAG,
                    "Low-texture scene (avg sharpness ${sharpnessEwma.toInt()}): " +
                        "relaxing absolute blur floor after $rejectStreak rejections"
                )
            }
            return report
        }

        passed++
        rejectStreak = 0
        return QualityReport(true, blurVariance, meanLum, Verdict.PASSED, null)
    }

    private fun sampleStep(innerW: Int, innerH: Int): Int {
        val pixels = innerW.toLong() * innerH.toLong()
        if (pixels <= TARGET_LAPLACIAN_SAMPLES) return MIN_SAMPLE_STEP
        val s = Math.sqrt(pixels.toDouble() / TARGET_LAPLACIAN_SAMPLES).toInt()
        return maxOf(MIN_SAMPLE_STEP, s)
    }

    private fun reject(
        verdict: Verdict, blur: Float, lum: Float, reason: String
    ): QualityReport {
        when (verdict) {
            Verdict.MOTION_BLUR -> rejectedBlur++
            Verdict.UNDEREXPOSED -> rejectedDark++
            Verdict.OVEREXPOSED -> rejectedBright++
            else -> rejectedUnusable++
        }
        rejectStreak++
        return QualityReport(false, blur, lum, verdict, reason)
    }
}
