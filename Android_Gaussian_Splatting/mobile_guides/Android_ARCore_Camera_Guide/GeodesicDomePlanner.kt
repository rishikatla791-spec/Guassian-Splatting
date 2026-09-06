package com.gaussian.cameraguide

import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.sin

data class Vector3(val x: Float, val y: Float, val z: Float) {
    fun length(): Float = kotlin.math.sqrt(x * x + y * y + z * z)
    fun normalize(): Vector3 {
        let len = length()
        return if (len > 1e-6f) Vector3(x / len, y / len, z / len) else Vector3(0f, 0f, 0f)
    }
    fun dot(other: Vector3): Float = x * other.x + y * other.y + z * other.z
}

data class TargetNode(
    val id: Int,
    val position: Vector3,
    val elevationDeg: Float,
    val azimuthDeg: Float,
    var isCaptured: Boolean = false
)

class GeodesicDomePlanner(
    val center: Vector3,
    val radius: Float = 1.5f,
    numElevationRings: Int = 3,
    samplesPerRing: Int = 12,
    private val satisfactionAngleDeg: Float = 20.0f
) {
    val targetNodes = mutableListOf<TargetNode>()

    init {
        generateTargetNodes(numElevationRings, samplesPerRing)
    }

    private fun generateTargetNodes(numRings: Int, samplesPerRing: Int) {
        targetNodes.clear()
        var nodeId = 0
        val elevations = floatArrayOf(15.0f, 45.0f, 75.0f)

        for (elev in elevations) {
            val elevRad = Math.toRadians(elev.toDouble()).toFloat()
            val azimuthOffset = if (nodeId % 2 == 0) 0.0f else (180.0f / samplesPerRing)

            for (i in 0 until samplesPerRing) {
                val azimuthDeg = (i * (360.0f / samplesPerRing) + azimuthOffset) % 360.0f
                val azimuthRad = Math.toRadians(azimuthDeg.toDouble()).toFloat()

                val x = center.x + radius * cos(elevRad) * cos(azimuthRad)
                val y = center.y + radius * sin(elevRad)
                val z = center.z + radius * cos(elevRad) * sin(azimuthRad)

                val nodePos = Vector3(x, y, z)
                targetNodes.add(TargetNode(nodeId, nodePos, elev, azimuthDeg))
                nodeId++
            }
        }
    }

    data class PoseEvalResult(
        val closestNodeIndex: Int?,
        val angularDistanceDeg: Float,
        val isNewlyCaptured: Boolean
    )

    fun evaluateCameraPose(camPos: Vector3): PoseEvalResult {
        val camDir = Vector3(camPos.x - center.x, camPos.y - center.y, camPos.z - center.z).normalize()

        var minAngDeg = 180.0f
        var closestIdx: Int? = null
        var newlyCaptured = false

        for (i in targetNodes.indices) {
            val targetPos = targetNodes[i].position
            val targetDir = Vector3(targetPos.x - center.x, targetPos.y - center.y, targetPos.z - center.z).normalize()

            val dotProduct = (camDir.dot(targetDir)).coerceIn(-1.0f, 1.0f)
            val angRad = acos(dotProduct)
            val angDeg = Math.toDegrees(angRad.toDouble()).toFloat()

            if (angDeg < minAngDeg) {
                minAngDeg = angDeg
                closestIdx = i
            }

            if (angDeg <= satisfactionAngleDeg && !targetNodes[i].isCaptured) {
                targetNodes[i].isCaptured = true
                newlyCaptured = true
            }
        }

        return PoseEvalResult(closestIdx, minAngDeg, newlyCaptured)
    }

    val overallCoverageRatio: Float
        get() {
            if (targetNodes.isEmpty()) return 0.0f
            val captured = targetNodes.count { it.isCaptured }
            return captured.toFloat() / targetNodes.size
        }
}
