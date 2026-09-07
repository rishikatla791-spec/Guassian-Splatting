package com.splat.mobile3dgs.capture

import com.google.ar.core.Pose
import kotlin.math.sqrt

/**
 * Coordinate system conversions and pose utilities for ARCore and 3DGS.
 */
object ARCoreCoordinateUtils {

    fun poseToMatrix4x4(pose: Pose): FloatArray {
        val matrix = FloatArray(16)
        pose.toMatrix(matrix, 0)
        return matrix
    }

    fun arcoreMatrixToC2W(arcoreMatrix: FloatArray): Array<FloatArray> {
        val c2w = Array(4) { FloatArray(4) }
        for (row in 0..3) {
            for (col in 0..3) {
                c2w[row][col] = arcoreMatrix[col * 4 + row]
            }
        }
        return c2w
    }

    fun c2wToOpenCV(c2w: Array<FloatArray>): Array<FloatArray> {
        val opencvMat = Array(4) { FloatArray(4) }
        for (i in 0..3) {
            opencvMat[i][0] = c2w[i][0]
            opencvMat[i][1] = -c2w[i][1] // Flip Y
            opencvMat[i][2] = -c2w[i][2] // Flip Z
            opencvMat[i][3] = c2w[i][3]
        }
        return opencvMat
    }

    fun validateRotationMatrix(matrix: Array<FloatArray>, tolerance: Float = 1e-3f): Boolean {
        val r00 = matrix[0][0]; val r01 = matrix[0][1]; val r02 = matrix[0][2]
        val r10 = matrix[1][0]; val r11 = matrix[1][1]; val r12 = matrix[1][2]
        val r20 = matrix[2][0]; val r21 = matrix[2][1]; val r22 = matrix[2][2]

        val det = r00 * (r11 * r22 - r12 * r21) -
                  r01 * (r10 * r22 - r12 * r20) +
                  r02 * (r10 * r21 - r11 * r20)

        if (kotlin.math.abs(det - 1.0f) > tolerance) return false

        val lenR0 = sqrt(r00 * r00 + r01 * r01 + r02 * r02)
        if (kotlin.math.abs(lenR0 - 1.0f) > tolerance) return false

        return true
    }

    fun distance3D(p1: FloatArray, p2: FloatArray): Float {
        val dx = p1[0] - p2[0]
        val dy = p1[1] - p2[1]
        val dz = p1[2] - p2[2]
        return sqrt(dx * dx + dy * dy + dz * dz)
    }
}
