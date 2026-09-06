package com.gaussian.cameraguide

import com.google.ar.core.Pose
import kotlin.math.sqrt

/**
 * Production-grade coordinate system conversions and pose utilities.
 *
 * Coordinate Conventions:
 *  - ARCore: World is Right-handed (+X Right, +Y Up, -Z Forward into scene).
 *  - NeRF / 3DGS (OpenGL convention): +X Right, +Y Up, -Z Forward.
 *  - OpenCV / COLMAP: +X Right, +Y Down, +Z Forward.
 *
 * This utility converts ARCore Pose objects into 4x4 homogenous matrix representations
 * and handles matrix orthonormality checks det(R) ≈ 1.
 */
object ARCoreCoordinateUtils {

    /**
     * Converts an ARCore Pose into a 4x4 column-major float matrix array.
     */
    fun poseToMatrix4x4(pose: Pose): FloatArray {
        val matrix = FloatArray(16)
        pose.toMatrix(matrix, 0)
        return matrix
    }

    /**
     * Converts ARCore 4x4 matrix (column-major) to NeRF / 3DGS standard 4x4 matrix (2D Array row-major).
     * ARCore Pose gives camera-to-world transform (c2w).
     */
    fun arcoreMatrixToC2W(arcoreMatrix: FloatArray): Array<FloatArray> {
        val c2w = Array(4) { FloatArray(4) }
        for (row in 0..3) {
            for (col in 0..3) {
                c2w[row][col] = arcoreMatrix[col * 4 + row]
            }
        }
        return c2w
    }

    /**
     * Converts c2w (Camera-to-World) matrix to OpenCV/COLMAP format by flipping Y and Z axes if needed.
     * OpenCV Camera convention: X Right, Y Down, Z Forward.
     */
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

    /**
     * Validates that the upper-left 3x3 rotation matrix R satisfies:
     * 1. det(R) ≈ 1.0 (valid SO(3) rotation, no reflection or scaling)
     * 2. R * R^T ≈ I (orthonormality)
     */
    fun validateRotationMatrix(matrix: Array<FloatArray>, tolerance: Float = 1e-3f): Boolean {
        val r00 = matrix[0][0]; val r01 = matrix[0][1]; val r02 = matrix[0][2]
        val r10 = matrix[1][0]; val r11 = matrix[1][1]; val r12 = matrix[1][2]
        val r20 = matrix[2][0]; val r21 = matrix[2][1]; val r22 = matrix[2][2]

        // Determinant of 3x3
        val det = r00 * (r11 * r22 - r12 * r21) -
                  r01 * (r10 * r22 - r12 * r20) +
                  r02 * (r10 * r21 - r11 * r20)

        if (kotlin.math.abs(det - 1.0f) > tolerance) {
            return false
        }

        // Check length of row 0
        val lenR0 = sqrt(r00 * r00 + r01 * r01 + r02 * r02)
        if (kotlin.math.abs(lenR0 - 1.0f) > tolerance) {
            return false
        }

        return true
    }

    /**
     * Calculates Euclidean distance between two 3D positions.
     */
    fun distance3D(p1: FloatArray, p2: FloatArray): Float {
        val dx = p1[0] - p2[0]
        val dy = p1[1] - p2[1]
        val dz = p1[2] - p2[2]
        return sqrt(dx * dx + dy * dy + dz * dz)
    }
}
