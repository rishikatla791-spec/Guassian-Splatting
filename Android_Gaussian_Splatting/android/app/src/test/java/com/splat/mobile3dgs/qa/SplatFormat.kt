package com.splat.mobile3dgs.qa

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * QA-owned reference decoder for the 32-byte `.splat` record, and a validator that
 * the device-verification harness and the instrumented tests both use.
 *
 * This is the single written-down definition of the on-disk contract. Every
 * producer in the app must agree with it:
 *  - [com.splat.mobile3dgs.engine.GaussianInitializer.initializeFromFeaturePoints]
 *  - [com.splat.mobile3dgs.engine.GaussianInitializer.generatePhotometricSplatModel]
 *  - `convert_ply_to_splat()` in `app/src/main/cpp/brush_bridge.cpp`
 *  - the WebGL2 viewer in `app/src/main/assets/viewer/viewer.js`
 *
 * Layout (little-endian, 32 bytes, no padding, no header):
 * ```
 *   offset  0 : float32 x
 *   offset  4 : float32 y
 *   offset  8 : float32 z
 *   offset 12 : float32 scale_0   (linear metres, NOT log-scale)
 *   offset 16 : float32 scale_1
 *   offset 20 : float32 scale_2
 *   offset 24 : uint8   r
 *   offset 25 : uint8   g
 *   offset 26 : uint8   b
 *   offset 27 : uint8   a         (opacity, 0..255)
 *   offset 28 : uint8   q0        quaternion, SCALAR FIRST: (w, x, y, z)
 *   offset 29 : uint8   q1        each packed as  byte = clamp(component * 128 + 128, 0, 255)
 *   offset 30 : uint8   q2
 *   offset 31 : uint8   q3
 * ```
 */
object SplatFormat {

    const val RECORD_BYTES = 32

    const val OFF_X = 0
    const val OFF_Y = 4
    const val OFF_Z = 8
    const val OFF_SCALE_0 = 12
    const val OFF_SCALE_1 = 16
    const val OFF_SCALE_2 = 20
    const val OFF_R = 24
    const val OFF_G = 25
    const val OFF_B = 26
    const val OFF_A = 27
    /** Quaternion is scalar-first: q0=w, q1=x, q2=y, q3=z. */
    const val OFF_QW = 28
    const val OFF_QX = 29
    const val OFF_QY = 30
    const val OFF_QZ = 31

    /** `byte = clamp(round-toward-zero(component * 128 + 128), 0, 255)`. */
    fun packQuatComponent(component: Float): Int {
        val v = component * 128.0f + 128.0f
        return v.coerceIn(0.0f, 255.0f).toInt()
    }

    /** Inverse of [packQuatComponent]; exact only for components representable on the 1/128 grid. */
    fun unpackQuatComponent(byteValue: Int): Float = (byteValue - 128) / 128.0f

    data class Splat(
        val x: Float, val y: Float, val z: Float,
        val s0: Float, val s1: Float, val s2: Float,
        val r: Int, val g: Int, val b: Int, val a: Int,
        /** Raw packed bytes, scalar-first. */
        val qw: Int, val qx: Int, val qy: Int, val qz: Int
    ) {
        val position: FloatArray get() = floatArrayOf(x, y, z)
    }

    fun decode(bytes: ByteArray): List<Splat> {
        require(bytes.size % RECORD_BYTES == 0) {
            "A .splat file must be a whole number of $RECORD_BYTES-byte records, got ${bytes.size}"
        }
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val out = ArrayList<Splat>(bytes.size / RECORD_BYTES)
        var base = 0
        while (base < bytes.size) {
            out.add(
                Splat(
                    x = bb.getFloat(base + OFF_X),
                    y = bb.getFloat(base + OFF_Y),
                    z = bb.getFloat(base + OFF_Z),
                    s0 = bb.getFloat(base + OFF_SCALE_0),
                    s1 = bb.getFloat(base + OFF_SCALE_1),
                    s2 = bb.getFloat(base + OFF_SCALE_2),
                    r = bytes[base + OFF_R].toInt() and 0xFF,
                    g = bytes[base + OFF_G].toInt() and 0xFF,
                    b = bytes[base + OFF_B].toInt() and 0xFF,
                    a = bytes[base + OFF_A].toInt() and 0xFF,
                    qw = bytes[base + OFF_QW].toInt() and 0xFF,
                    qx = bytes[base + OFF_QX].toInt() and 0xFF,
                    qy = bytes[base + OFF_QY].toInt() and 0xFF,
                    qz = bytes[base + OFF_QZ].toInt() and 0xFF
                )
            )
            base += RECORD_BYTES
        }
        return out
    }

    fun decode(file: File): List<Splat> = decode(file.readBytes())

    data class ValidationResult(
        val splatCount: Int,
        val problems: List<String>,
        val bboxMin: FloatArray,
        val bboxMax: FloatArray
    ) {
        val isValid: Boolean get() = problems.isEmpty()
        val extentMeters: Float
            get() = maxOf(
                bboxMax[0] - bboxMin[0],
                bboxMax[1] - bboxMin[1],
                bboxMax[2] - bboxMin[2]
            )
    }

    /**
     * Structural validation used by the device-verification harness. Deliberately
     * tolerant about content and strict about anything that would make the WebGL2
     * viewer produce NaNs or an empty screen.
     */
    fun validate(
        bytes: ByteArray,
        maxPlausibleExtentMeters: Float = 2_000f,
        maxPlausibleScaleMeters: Float = 50f
    ): ValidationResult {
        val problems = mutableListOf<String>()
        if (bytes.isEmpty()) {
            problems += "file is empty"
            return ValidationResult(0, problems, FloatArray(3), FloatArray(3))
        }
        if (bytes.size % RECORD_BYTES != 0) {
            problems += "size ${bytes.size} is not a multiple of $RECORD_BYTES"
            return ValidationResult(0, problems, FloatArray(3), FloatArray(3))
        }
        val splats = decode(bytes)
        val min = floatArrayOf(Float.MAX_VALUE, Float.MAX_VALUE, Float.MAX_VALUE)
        val max = floatArrayOf(-Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE)
        var nonFinite = 0
        var badScale = 0
        var fullyTransparent = 0
        for (s in splats) {
            val p = s.position
            if (!p[0].isFinite() || !p[1].isFinite() || !p[2].isFinite()) { nonFinite++; continue }
            for (i in 0..2) {
                if (p[i] < min[i]) min[i] = p[i]
                if (p[i] > max[i]) max[i] = p[i]
            }
            if (!s.s0.isFinite() || !s.s1.isFinite() || !s.s2.isFinite() ||
                s.s0 <= 0f || s.s1 <= 0f || s.s2 <= 0f ||
                s.s0 > maxPlausibleScaleMeters || s.s1 > maxPlausibleScaleMeters ||
                s.s2 > maxPlausibleScaleMeters
            ) badScale++
            if (s.a == 0) fullyTransparent++
        }
        if (nonFinite > 0) problems += "$nonFinite/${splats.size} splats have non-finite positions"
        if (badScale > splats.size / 10) {
            problems += "$badScale/${splats.size} splats have implausible scales " +
                "(expected linear metres in (0, $maxPlausibleScaleMeters]) - " +
                "a log-scale value written without exp() looks exactly like this"
        }
        if (fullyTransparent == splats.size) problems += "every splat has alpha 0; the viewer would render nothing"
        if (nonFinite < splats.size) {
            val extent = maxOf(max[0] - min[0], max[1] - min[1], max[2] - min[2])
            if (extent > maxPlausibleExtentMeters) {
                problems += "bounding box extent ${extent}m exceeds $maxPlausibleExtentMeters m"
            }
            if (extent == 0f && splats.size > 1) problems += "all splats occupy a single point"
        }
        return ValidationResult(splats.size, problems, min, max)
    }
}
