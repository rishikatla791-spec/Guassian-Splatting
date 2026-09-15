package com.splat.mobile3dgs

import com.google.ar.core.Pose
import com.splat.mobile3dgs.capture.ARCoreCoordinateUtils
import com.splat.mobile3dgs.capture.GeodesicDomePlanner
import com.splat.mobile3dgs.capture.Vector3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sqrt

/**
 * Pose bookkeeping between ARCore and the dataset that gets written to disk.
 *
 * `pose.toMatrix()` is COLUMN-major; `transforms.json` stores a ROW-major 4x4. A
 * transpose dropped anywhere between the two flips every camera's orientation while
 * leaving the file structurally valid, so the transposition itself is asserted here
 * against a pose whose matrix is known by hand.
 */
class PoseConventionTest {

    @Test
    fun arcoreMatrixToC2W_transposesColumnMajorIntoRowMajor() {
        val columnMajor = FloatArray(16) { it.toFloat() }
        val rowMajor = ARCoreCoordinateUtils.arcoreMatrixToC2W(columnMajor)

        for (row in 0..3) {
            for (col in 0..3) {
                assertEquals(
                    "element ($row,$col) must come from column-major index ${col * 4 + row}",
                    columnMajor[col * 4 + row], rowMajor[row][col], 0f
                )
            }
        }
        // Translation ends up in the last COLUMN of the row-major matrix.
        assertEquals(12f, rowMajor[0][3], 0f)
        assertEquals(13f, rowMajor[1][3], 0f)
        assertEquals(14f, rowMajor[2][3], 0f)
    }

    @Test
    fun aTranslationOnlyPose_roundTripsThroughTheRowMajorMatrix() {
        val pose = Pose(floatArrayOf(0.5f, -1.25f, 3.0f), floatArrayOf(0f, 0f, 0f, 1f))
        val rowMajor = ARCoreCoordinateUtils.arcoreMatrixToC2W(
            ARCoreCoordinateUtils.poseToMatrix4x4(pose)
        )
        assertEquals(pose.tx(), rowMajor[0][3], 1e-6f)
        assertEquals(pose.ty(), rowMajor[1][3], 1e-6f)
        assertEquals(pose.tz(), rowMajor[2][3], 1e-6f)
        // Rotation block is the identity.
        for (r in 0..2) for (c in 0..2) {
            assertEquals(if (r == c) 1f else 0f, rowMajor[r][c], 1e-6f)
        }
        assertEquals(1f, rowMajor[3][3], 1e-6f)
    }

    @Test
    fun aRotatedPose_producesAValidRotationBlock() {
        // 90 degrees about +Y.
        val s = (sqrt(2.0) / 2.0).toFloat()
        val pose = Pose(floatArrayOf(0f, 0f, 0f), floatArrayOf(0f, s, 0f, s))
        val rowMajor = ARCoreCoordinateUtils.arcoreMatrixToC2W(
            ARCoreCoordinateUtils.poseToMatrix4x4(pose)
        )
        assertTrue(
            "a pose straight out of ARCore must be a proper rotation",
            ARCoreCoordinateUtils.validateRotationMatrix(rowMajor)
        )
        // +X maps to -Z under a +90 degree yaw in a right-handed frame.
        val mappedX = floatArrayOf(rowMajor[0][0], rowMajor[1][0], rowMajor[2][0])
        assertEquals(0f, mappedX[0], 1e-5f)
        assertEquals(0f, mappedX[1], 1e-5f)
        assertEquals(-1f, mappedX[2], 1e-5f)
    }

    @Test
    fun validateRotationMatrix_rejectsAScaledOrMirroredBasis() {
        val scaled = arrayOf(
            floatArrayOf(2f, 0f, 0f, 0f),
            floatArrayOf(0f, 2f, 0f, 0f),
            floatArrayOf(0f, 0f, 2f, 0f),
            floatArrayOf(0f, 0f, 0f, 1f)
        )
        assertFalse("a scaled basis is not a rotation", ARCoreCoordinateUtils.validateRotationMatrix(scaled))

        val mirrored = arrayOf(
            floatArrayOf(-1f, 0f, 0f, 0f),
            floatArrayOf(0f, 1f, 0f, 0f),
            floatArrayOf(0f, 0f, 1f, 0f),
            floatArrayOf(0f, 0f, 0f, 1f)
        )
        assertFalse(
            "determinant -1 is a reflection, not a rotation",
            ARCoreCoordinateUtils.validateRotationMatrix(mirrored)
        )
    }

    @Test
    fun c2wToOpenCV_flipsYAndZOnly() {
        val c2w = Array(4) { r -> FloatArray(4) { c -> (r * 4 + c + 1).toFloat() } }
        val cv = ARCoreCoordinateUtils.c2wToOpenCV(c2w)
        for (r in 0..3) {
            assertEquals(c2w[r][0], cv[r][0], 0f)
            assertEquals(-c2w[r][1], cv[r][1], 0f)
            assertEquals(-c2w[r][2], cv[r][2], 0f)
            assertEquals("the translation column must be left alone", c2w[r][3], cv[r][3], 0f)
        }
    }

    @Test
    fun distance3D_matchesEuclidean() {
        assertEquals(
            5f,
            ARCoreCoordinateUtils.distance3D(floatArrayOf(1f, 2f, 3f), floatArrayOf(4f, 6f, 3f)),
            1e-6f
        )
    }

    // -------------------------------------------------------------------------
    // Coverage planner
    // -------------------------------------------------------------------------

    @Test
    fun domePlanner_generatesThreeRingsOfTargets() {
        val planner = GeodesicDomePlanner(Vector3(0f, 0f, 0f), radius = 1.5f, samplesPerRing = 12)
        assertEquals(36, planner.targetNodes.size)
        assertEquals(0f, planner.overallCoverageRatio, 0f)
        // Every node sits on the requested radius.
        for (node in planner.targetNodes) {
            val p = node.position
            assertEquals(1.5f, sqrt(p.x * p.x + p.y * p.y + p.z * p.z), 1e-4f)
        }
    }

    @Test
    fun domePlanner_marksNodesCoveredAndNeverExceedsOne() {
        val planner = GeodesicDomePlanner(Vector3(0f, 0f, 0f), radius = 1.5f, samplesPerRing = 12)
        for (node in planner.targetNodes.toList()) {
            planner.evaluateCameraPose(node.position)
        }
        assertEquals(
            "standing at every target must cover the whole dome",
            1.0f, planner.overallCoverageRatio, 1e-6f
        )
        planner.reset()
        assertEquals(0f, planner.overallCoverageRatio, 0f)
    }

    @Test
    fun domePlanner_handlesACameraAtTheExactCentreWithoutNaN() {
        // normalize() of a zero vector returns zero, so the dot product is 0 and acos
        // is well defined. This used to be a NaN source in coverage HUDs.
        val planner = GeodesicDomePlanner(Vector3(0f, 0f, 0f), radius = 1.5f, samplesPerRing = 12)
        val result = planner.evaluateCameraPose(Vector3(0f, 0f, 0f))
        assertFalse("angular distance must not be NaN", result.angularDistanceDeg.isNaN())
        assertEquals(90.0f, result.angularDistanceDeg, 1e-3f)
    }
}
