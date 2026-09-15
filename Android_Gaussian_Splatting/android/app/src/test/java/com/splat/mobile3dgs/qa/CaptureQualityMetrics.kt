package com.splat.mobile3dgs.qa

import kotlin.math.abs
import kotlin.math.atan
import kotlin.math.sqrt

/**
 * QA reference implementation of the capture-quality metrics.
 *
 * A 3DGS reconstruction fails for geometric reasons long before it fails for
 * software reasons: a user who pans the phone on the spot produces perfectly
 * valid frames with *zero* triangulation baseline, and the optimiser then
 * collapses the scene into a flat smear. These metrics make that condition
 * detectable at the end of a capture (and, if the capture agent wires them in,
 * during it) instead of after a 20-minute training run.
 *
 * All distances are metres in the ARCore world frame; all angles are degrees.
 * Camera positions are the translation column of each keyframe's camera-to-world
 * pose, i.e. the optical centre.
 *
 * The Python analyser in `qa/analyze_scan.py` mirrors this file; keep them in step.
 */
object CaptureQualityMetrics {

    /** Below this the capture is effectively a pure rotation (a "panorama"). */
    const val MIN_BASELINE_M = 0.05f

    /** Below this, depth is not observable -- triangulation is ill-conditioned. */
    const val MIN_PARALLAX_DEG = 3.0f

    /** The training pipeline in CaptureActivity refuses to start below this. */
    const val MIN_KEYFRAMES = 8

    /** A path whose smallest principal extent is under this fraction of its largest is a straight line. */
    const val COLLINEARITY_RATIO = 0.02f

    enum class Verdict {
        OK,
        TOO_FEW_FRAMES,
        /** Camera never moved: baseline below [MIN_BASELINE_M]. */
        PURE_ROTATION,
        /** Moved, but not enough relative to scene depth: parallax below [MIN_PARALLAX_DEG]. */
        INSUFFICIENT_PARALLAX,
        /** Camera travelled along a straight line: depth is only weakly constrained. */
        COLLINEAR_PATH,
        /** No usable seed geometry, so scene depth is unknown. */
        NO_SEED_POINTS
    }

    data class Report(
        val frameCount: Int,
        val seedPointCount: Int,
        /** Largest distance between any two camera centres. */
        val maxBaselineM: Float,
        /** Median distance from the camera path centroid to a seed point. */
        val medianDepthM: Float,
        /** Triangulation angle implied by [maxBaselineM] at [medianDepthM]. */
        val parallaxDeg: Float,
        /** min/max principal extent ratio of the camera path (0 = perfectly straight). */
        val pathFlatness: Float,
        val verdict: Verdict
    ) {
        val passed: Boolean get() = verdict == Verdict.OK
    }

    fun distance(a: FloatArray, b: FloatArray): Float {
        val dx = a[0] - b[0]; val dy = a[1] - b[1]; val dz = a[2] - b[2]
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    /** Largest pairwise distance between camera centres. O(n^2); n is at most a few hundred keyframes. */
    fun maxBaselineMeters(cameraPositions: List<FloatArray>): Float {
        if (cameraPositions.size < 2) return 0f
        var best = 0f
        for (i in cameraPositions.indices) {
            for (j in i + 1 until cameraPositions.size) {
                val d = distance(cameraPositions[i], cameraPositions[j])
                if (d > best) best = d
            }
        }
        return best
    }

    fun centroid(points: List<FloatArray>): FloatArray {
        if (points.isEmpty()) return floatArrayOf(0f, 0f, 0f)
        var x = 0.0; var y = 0.0; var z = 0.0
        for (p in points) { x += p[0]; y += p[1]; z += p[2] }
        val n = points.size.toDouble()
        return floatArrayOf((x / n).toFloat(), (y / n).toFloat(), (z / n).toFloat())
    }

    /** Median distance from [from] to each seed point. Robust to the long tail of far-away outliers. */
    fun medianDepthMeters(seedPoints: List<FloatArray>, from: FloatArray): Float {
        if (seedPoints.isEmpty()) return 0f
        val d = FloatArray(seedPoints.size) { distance(from, seedPoints[it]) }
        d.sort()
        val mid = d.size / 2
        return if (d.size % 2 == 1) d[mid] else (d[mid - 1] + d[mid]) / 2f
    }

    /**
     * Triangulation (parallax) angle, in degrees, subtended at a point at
     * [depthMeters] by two views [baselineMeters] apart:
     * `theta = 2 * atan(B / (2 * Z))`.
     */
    fun parallaxAngleDeg(baselineMeters: Float, depthMeters: Float): Float {
        if (depthMeters <= 0f || baselineMeters <= 0f) return 0f
        val rad = 2.0 * atan((baselineMeters / (2.0 * depthMeters)))
        return Math.toDegrees(rad).toFloat()
    }

    /**
     * Ratio of the smallest to the largest principal extent of the camera path,
     * computed from the covariance of the camera centres via its eigenvalues.
     * A perfect straight line gives 0; a well-distributed orbit gives well above
     * [COLLINEARITY_RATIO].
     */
    fun pathFlatness(cameraPositions: List<FloatArray>): Float {
        if (cameraPositions.size < 3) return 0f
        val c = centroid(cameraPositions)
        // 3x3 covariance
        val cov = Array(3) { DoubleArray(3) }
        for (p in cameraPositions) {
            val d = doubleArrayOf(
                (p[0] - c[0]).toDouble(), (p[1] - c[1]).toDouble(), (p[2] - c[2]).toDouble()
            )
            for (i in 0..2) for (j in 0..2) cov[i][j] += d[i] * d[j]
        }
        for (i in 0..2) for (j in 0..2) cov[i][j] /= cameraPositions.size.toDouble()
        val eig = symmetricEigenvalues3x3(cov)
        val maxEv = eig.max()
        val minEv = eig.min().coerceAtLeast(0.0)
        if (maxEv <= 1e-12) return 0f
        // Eigenvalues are squared extents; compare standard deviations.
        return (sqrt(minEv) / sqrt(maxEv)).toFloat()
    }

    /**
     * Closed-form eigenvalues of a real symmetric 3x3 matrix (Smith's method).
     * Avoids pulling a linear-algebra dependency into the test classpath.
     */
    internal fun symmetricEigenvalues3x3(m: Array<DoubleArray>): DoubleArray {
        val p1 = m[0][1] * m[0][1] + m[0][2] * m[0][2] + m[1][2] * m[1][2]
        if (p1 <= 1e-300) return doubleArrayOf(m[0][0], m[1][1], m[2][2])
        val q = (m[0][0] + m[1][1] + m[2][2]) / 3.0
        val p2 = (m[0][0] - q) * (m[0][0] - q) +
            (m[1][1] - q) * (m[1][1] - q) +
            (m[2][2] - q) * (m[2][2] - q) + 2.0 * p1
        val p = sqrt(p2 / 6.0)
        if (p <= 0.0) return doubleArrayOf(q, q, q)
        val b = Array(3) { i -> DoubleArray(3) { j -> (m[i][j] - if (i == j) q else 0.0) / p } }
        val detB =
            b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1]) -
                b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0]) +
                b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0])
        val r = (detB / 2.0).coerceIn(-1.0, 1.0)
        val phi = Math.acos(r) / 3.0
        val e1 = q + 2.0 * p * Math.cos(phi)
        val e3 = q + 2.0 * p * Math.cos(phi + 2.0 * Math.PI / 3.0)
        val e2 = 3.0 * q - e1 - e3
        return doubleArrayOf(e1, e2, e3)
    }

    /**
     * Full degenerate-capture check.
     *
     * @param cameraPositions optical centres of the kept keyframes, world frame.
     * @param seedPoints the seed cloud handed to the optimiser (depth + feature points).
     */
    fun analyse(
        cameraPositions: List<FloatArray>,
        seedPoints: List<FloatArray>,
        minKeyframes: Int = MIN_KEYFRAMES,
        minBaselineM: Float = MIN_BASELINE_M,
        minParallaxDeg: Float = MIN_PARALLAX_DEG
    ): Report {
        val baseline = maxBaselineMeters(cameraPositions)
        val camCentroid = centroid(cameraPositions)
        val medianDepth = medianDepthMeters(seedPoints, camCentroid)
        val parallax = parallaxAngleDeg(baseline, medianDepth)
        val flatness = pathFlatness(cameraPositions)

        val verdict = when {
            cameraPositions.size < minKeyframes -> Verdict.TOO_FEW_FRAMES
            baseline < minBaselineM -> Verdict.PURE_ROTATION
            seedPoints.isEmpty() -> Verdict.NO_SEED_POINTS
            parallax < minParallaxDeg -> Verdict.INSUFFICIENT_PARALLAX
            flatness < COLLINEARITY_RATIO -> Verdict.COLLINEAR_PATH
            else -> Verdict.OK
        }
        return Report(
            frameCount = cameraPositions.size,
            seedPointCount = seedPoints.size,
            maxBaselineM = baseline,
            medianDepthM = medianDepth,
            parallaxDeg = parallax,
            pathFlatness = flatness,
            verdict = verdict
        )
    }

    /** Convenience: pull the optical centre out of a 16-element column-major camera-to-world matrix. */
    fun cameraCentreFromColumnMajor(m: FloatArray): FloatArray {
        require(m.size >= 16) { "expected a 16-element matrix, got ${m.size}" }
        return floatArrayOf(m[12], m[13], m[14])
    }

    /** Convenience: pull the optical centre out of a row-major 4x4 camera-to-world matrix. */
    fun cameraCentreFromRowMajor(m: Array<FloatArray>): FloatArray {
        require(m.size == 4 && m.all { it.size == 4 }) { "expected a 4x4 matrix" }
        return floatArrayOf(m[0][3], m[1][3], m[2][3])
    }

    internal fun approxEquals(a: Float, b: Float, eps: Float): Boolean = abs(a - b) <= eps
}
