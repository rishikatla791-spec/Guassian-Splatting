package com.splat.mobile3dgs

import com.splat.mobile3dgs.qa.CaptureQualityMetrics
import com.splat.mobile3dgs.qa.CaptureQualityMetrics.Verdict
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

/**
 * Capture-quality metrics and degenerate-capture detection.
 *
 * The commonest way this app produces a useless model has nothing to do with code:
 * the user stands still and pans. Every frame is sharp, tracking is perfect, 120
 * keyframes are written -- and the triangulation baseline is zero, so depth is
 * unobservable and the optimiser fits a smear. These tests pin down the arithmetic
 * that lets the app say so before it spends twenty minutes training.
 */
class CaptureQualityMetricsTest {

    private fun orbit(
        radius: Float, count: Int, centre: FloatArray = floatArrayOf(0f, 0f, 0f),
        sweepDeg: Double = 360.0
    ): List<FloatArray> = (0 until count).map { i ->
        val a = Math.toRadians(sweepDeg * i / count)
        floatArrayOf(
            centre[0] + radius * cos(a).toFloat(),
            centre[1] + 0.08f * sin(3.0 * a).toFloat(),   // slight bob, so the path is not planar-degenerate
            centre[2] + radius * sin(a).toFloat()
        )
    }

    private fun sphereOfSeedPoints(radius: Float, count: Int): List<FloatArray> =
        (0 until count).map { i ->
            val t = Math.toRadians(360.0 * i / count)
            val p = Math.toRadians(180.0 * ((i * 7) % count) / count - 90.0)
            floatArrayOf(
                (radius * cos(p) * cos(t)).toFloat(),
                (radius * sin(p)).toFloat(),
                (radius * cos(p) * sin(t)).toFloat()
            )
        }

    // -------------------------------------------------------------------------
    // Baseline
    // -------------------------------------------------------------------------

    @Test
    fun maxBaseline_isTheLargestPairwiseDistance() {
        val positions = listOf(
            floatArrayOf(0f, 0f, 0f),
            floatArrayOf(3f, 4f, 0f),     // 5 m from the first
            floatArrayOf(1f, 1f, 1f)
        )
        assertEquals(5.0f, CaptureQualityMetrics.maxBaselineMeters(positions), 1e-5f)
    }

    @Test
    fun maxBaseline_ofAnOrbitIsItsDiameter() {
        val positions = orbit(radius = 1.5f, count = 36)
        // Opposite points on a 1.5 m orbit are 3 m apart (plus the small vertical bob).
        assertEquals(3.0f, CaptureQualityMetrics.maxBaselineMeters(positions), 0.05f)
    }

    @Test
    fun maxBaseline_ofFewerThanTwoCamerasIsZero() {
        assertEquals(0f, CaptureQualityMetrics.maxBaselineMeters(emptyList()), 0f)
        assertEquals(0f, CaptureQualityMetrics.maxBaselineMeters(listOf(floatArrayOf(9f, 9f, 9f))), 0f)
    }

    // -------------------------------------------------------------------------
    // Median depth
    // -------------------------------------------------------------------------

    @Test
    fun medianDepth_isTheMiddleDistanceNotTheMean() {
        val origin = floatArrayOf(0f, 0f, 0f)
        // 1, 2, 3, 4 m and one 1000 m outlier: mean 202 m, median 3 m.
        val pts = listOf(
            floatArrayOf(1f, 0f, 0f),
            floatArrayOf(2f, 0f, 0f),
            floatArrayOf(3f, 0f, 0f),
            floatArrayOf(4f, 0f, 0f),
            floatArrayOf(1000f, 0f, 0f)
        )
        assertEquals(
            "a single far-away feature point must not move the depth estimate",
            3.0f, CaptureQualityMetrics.medianDepthMeters(pts, origin), 1e-5f
        )
    }

    @Test
    fun medianDepth_ofAnEvenCountAveragesTheMiddlePair() {
        val origin = floatArrayOf(0f, 0f, 0f)
        val pts = listOf(
            floatArrayOf(1f, 0f, 0f), floatArrayOf(2f, 0f, 0f),
            floatArrayOf(3f, 0f, 0f), floatArrayOf(6f, 0f, 0f)
        )
        assertEquals(2.5f, CaptureQualityMetrics.medianDepthMeters(pts, origin), 1e-5f)
    }

    @Test
    fun medianDepth_withNoSeedPointsIsZero() {
        assertEquals(0f, CaptureQualityMetrics.medianDepthMeters(emptyList(), floatArrayOf(0f, 0f, 0f)), 0f)
    }

    // -------------------------------------------------------------------------
    // Parallax / triangulation angle
    // -------------------------------------------------------------------------

    @Test
    fun parallaxAngle_ofEqualBaselineAndDepthIsFiftyThreeDegrees() {
        // theta = 2 * atan(B / 2Z); B == Z gives 2 * atan(0.5) = 53.13 degrees.
        assertEquals(53.1301f, CaptureQualityMetrics.parallaxAngleDeg(1.0f, 1.0f), 1e-3f)
    }

    @Test
    fun parallaxAngle_ofABaselineTwiceTheDepthIsNinetyDegrees() {
        assertEquals(90.0f, CaptureQualityMetrics.parallaxAngleDeg(2.0f, 1.0f), 1e-3f)
    }

    @Test
    fun parallaxAngle_shrinksAsTheSubjectGetsFurtherAway() {
        // 30 cm of sideways travel is plenty at 1 m and useless at 30 m.
        val near = CaptureQualityMetrics.parallaxAngleDeg(0.30f, 1.0f)
        val far = CaptureQualityMetrics.parallaxAngleDeg(0.30f, 30.0f)
        assertTrue("30 cm at 1 m should be a healthy angle, was $near deg", near > 15f)
        assertTrue("30 cm at 30 m should be negligible, was $far deg", far < 1f)
    }

    @Test
    fun parallaxAngle_isZeroForDegenerateInputs() {
        assertEquals(0f, CaptureQualityMetrics.parallaxAngleDeg(0f, 2f), 0f)
        assertEquals(0f, CaptureQualityMetrics.parallaxAngleDeg(1f, 0f), 0f)
        assertEquals(0f, CaptureQualityMetrics.parallaxAngleDeg(1f, -3f), 0f)
    }

    // -------------------------------------------------------------------------
    // Path shape
    // -------------------------------------------------------------------------

    @Test
    fun pathFlatness_ofAPerfectlyStraightWalkIsZero() {
        val line = (0 until 20).map { floatArrayOf(it * 0.05f, 0f, 0f) }
        assertEquals(0f, CaptureQualityMetrics.pathFlatness(line), 1e-4f)
    }

    @Test
    fun pathFlatness_ofAnOrbitIsWellAboveTheCollinearityThreshold() {
        val flatness = CaptureQualityMetrics.pathFlatness(orbit(radius = 1.2f, count = 40))
        assertTrue(
            "an orbit is not collinear; flatness was $flatness",
            flatness > CaptureQualityMetrics.COLLINEARITY_RATIO
        )
    }

    // -------------------------------------------------------------------------
    // Degenerate-capture detection
    // -------------------------------------------------------------------------

    @Test
    fun pureRotationInPlaceIsRejected() {
        // The classic failure: the user pivots on the spot. ARCore reports 6-DoF poses
        // the whole time, so nothing upstream notices.
        val onTheSpot = (0 until 60).map {
            floatArrayOf(0.002f * (it % 3), 0.001f * (it % 2), 0.0015f * (it % 4))
        }
        val report = CaptureQualityMetrics.analyse(onTheSpot, sphereOfSeedPoints(2.0f, 500))
        assertEquals(Verdict.PURE_ROTATION, report.verdict)
        assertTrue(report.maxBaselineM < CaptureQualityMetrics.MIN_BASELINE_M)
    }

    @Test
    fun movingButTooFarFromTheSubjectIsRejectedForParallax() {
        // 40 cm of travel while scanning a building 40 m away: real motion, no depth.
        val walk = (0 until 30).map { floatArrayOf(it * 0.0138f, 0f, 0f) }
        val faraway = (0 until 400).map { i ->
            floatArrayOf((i % 20) * 0.5f - 5f, (i / 20) * 0.5f, -40f)
        }
        val report = CaptureQualityMetrics.analyse(walk, faraway)
        assertTrue(
            "baseline should be real: ${report.maxBaselineM}",
            report.maxBaselineM > CaptureQualityMetrics.MIN_BASELINE_M
        )
        assertTrue(
            "parallax should be far below the threshold: ${report.parallaxDeg}",
            report.parallaxDeg < CaptureQualityMetrics.MIN_PARALLAX_DEG
        )
        assertEquals(Verdict.INSUFFICIENT_PARALLAX, report.verdict)
    }

    @Test
    fun tooFewKeyframesIsRejectedFirst() {
        val report = CaptureQualityMetrics.analyse(orbit(1.5f, 4), sphereOfSeedPoints(1.0f, 100))
        assertEquals(Verdict.TOO_FEW_FRAMES, report.verdict)
    }

    @Test
    fun noSeedPointsIsReportedSeparatelyFromNoParallax() {
        // A real orbit but the depth API produced nothing: the user needs to be told
        // something different from "move more".
        val report = CaptureQualityMetrics.analyse(orbit(1.5f, 40), emptyList())
        assertEquals(Verdict.NO_SEED_POINTS, report.verdict)
        assertEquals(0f, report.medianDepthM, 0f)
    }

    @Test
    fun aStraightLineDollyWithGoodParallaxIsFlaggedAsCollinear() {
        // 2 m of sideways travel past a subject 1 m away: excellent parallax, but every
        // camera centre lies on one line, so the scene is only weakly constrained.
        val dolly = (0 until 30).map { floatArrayOf(it * 0.069f - 1f, 0f, 0f) }
        val subject = (0 until 200).map { i ->
            floatArrayOf((i % 10) * 0.02f - 0.1f, (i / 10) * 0.02f - 0.2f, -1.0f)
        }
        val report = CaptureQualityMetrics.analyse(dolly, subject)
        assertTrue(
            "parallax should be healthy: ${report.parallaxDeg}",
            report.parallaxDeg > CaptureQualityMetrics.MIN_PARALLAX_DEG
        )
        assertEquals(Verdict.COLLINEAR_PATH, report.verdict)
    }

    @Test
    fun aProperOrbitPasses() {
        val report = CaptureQualityMetrics.analyse(
            orbit(radius = 1.2f, count = 48),
            sphereOfSeedPoints(0.5f, 2000)
        )
        assertEquals("a 1.2 m orbit around a 0.5 m object must pass: $report", Verdict.OK, report.verdict)
        assertTrue(report.passed)
        assertTrue("baseline ${report.maxBaselineM}", report.maxBaselineM > 2.0f)
        assertTrue("parallax ${report.parallaxDeg}", report.parallaxDeg > 45f)
    }

    @Test
    fun analyse_neverThrowsOnEmptyInput() {
        val report = CaptureQualityMetrics.analyse(emptyList(), emptyList())
        assertEquals(Verdict.TOO_FEW_FRAMES, report.verdict)
        assertEquals(0, report.frameCount)
        assertEquals(0f, report.maxBaselineM, 0f)
        assertEquals(0f, report.parallaxDeg, 0f)
    }

    // -------------------------------------------------------------------------
    // Pose plumbing
    // -------------------------------------------------------------------------

    @Test
    fun cameraCentre_isReadFromTheTranslationOfEitherMatrixConvention() {
        // ARCore's pose.toMatrix() is COLUMN-major: translation lives at 12, 13, 14.
        val columnMajor = FloatArray(16).also {
            it[0] = 1f; it[5] = 1f; it[10] = 1f; it[15] = 1f
            it[12] = 1.5f; it[13] = -0.25f; it[14] = 3f
        }
        val fromCol = CaptureQualityMetrics.cameraCentreFromColumnMajor(columnMajor)
        assertEquals(1.5f, fromCol[0], 0f)
        assertEquals(-0.25f, fromCol[1], 0f)
        assertEquals(3f, fromCol[2], 0f)

        // DatasetExporter stores the transposed 4x4, where translation is column 3.
        val rowMajor = com.splat.mobile3dgs.capture.ARCoreCoordinateUtils
            .arcoreMatrixToC2W(columnMajor)
        val fromRow = CaptureQualityMetrics.cameraCentreFromRowMajor(rowMajor)
        assertEquals(
            "the two conventions must name the same optical centre",
            fromCol.toList(), fromRow.toList()
        )
    }
}
