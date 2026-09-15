package com.splat.mobile3dgs

import com.splat.mobile3dgs.capture.FeaturePoint3D
import com.splat.mobile3dgs.engine.GaussianInitializer
import com.splat.mobile3dgs.qa.SplatFormat
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The `.splat` binary contract: N * 32 bytes, fixed field offsets, little-endian
 * floats and a scalar-first quaternion packed as `byte = component * 128 + 128`.
 *
 * Three independent producers write this format (Kotlin `GaussianInitializer`, the
 * C++ `convert_ply_to_splat`, and any future exporter) and one consumer reads it
 * (the WebGL2 viewer). A one-byte disagreement shows up as a scene of random
 * colours or invisible geometry, never as an exception -- so it has to be asserted.
 */
class SplatBinaryLayoutTest {

    @get:Rule
    val tmp = TemporaryFolder()

    /** Points far enough apart that the 5 mm voxel filter keeps every one of them. */
    private fun spacedPoints(n: Int): List<FeaturePoint3D> =
        (0 until n).map { i ->
            FeaturePoint3D(
                id = i,
                x = i * 0.5f,
                y = i * 0.25f + 1.0f,
                z = -i * 0.75f - 2.0f,
                confidence = 0.9f
            )
        }

    @Test
    fun splatFile_isExactlyThirtyTwoBytesPerGaussian() {
        val n = 37
        val out = tmp.newFile("layout.splat")
        val written = GaussianInitializer.initializeFromFeaturePoints(spacedPoints(n), out)

        assertEquals("voxel filter must not merge points 0.5 m apart", n, written)
        assertEquals(
            "a .splat file is exactly count * ${SplatFormat.RECORD_BYTES} bytes with no header",
            (n * SplatFormat.RECORD_BYTES).toLong(), out.length()
        )
        assertEquals(0L, out.length() % SplatFormat.RECORD_BYTES)
    }

    @Test
    fun splatFile_fieldOffsetsAndEndiannessRoundTrip() {
        val points = spacedPoints(16)
        val out = tmp.newFile("offsets.splat")
        val scale = 0.0125f
        GaussianInitializer.initializeFromFeaturePoints(
            points = points,
            outputFile = out,
            defaultScale = scale,
            defaultRgb = Triple(12, 200, 77),
            defaultAlpha = 180
        )

        val decoded = SplatFormat.decode(out)
        assertEquals(points.size, decoded.size)

        // Voxel-filter output order is a HashMap iteration order, so compare as sets.
        val expected = points.map { Triple(it.x, it.y, it.z) }.toSet()
        val actual = decoded.map { Triple(it.x, it.y, it.z) }.toSet()
        assertEquals(
            "positions must decode as little-endian float32 at offsets 0/4/8",
            expected, actual
        )

        for (s in decoded) {
            assertEquals("scale_0 at offset ${SplatFormat.OFF_SCALE_0}", scale, s.s0, 0f)
            assertEquals("scale_1 at offset ${SplatFormat.OFF_SCALE_1}", scale, s.s1, 0f)
            assertEquals("scale_2 at offset ${SplatFormat.OFF_SCALE_2}", scale, s.s2, 0f)
            assertEquals("r at offset ${SplatFormat.OFF_R}", 12, s.r)
            assertEquals("g at offset ${SplatFormat.OFF_G}", 200, s.g)
            assertEquals("b at offset ${SplatFormat.OFF_B}", 77, s.b)
            assertEquals("a at offset ${SplatFormat.OFF_A}", 180, s.a)
        }
    }

    @Test
    fun quaternionPacking_isScalarFirstTimes128Plus128() {
        // The packing rule shared with brush_bridge.cpp:
        //   byte = clamp(component * 128 + 128, 0, 255)
        assertEquals(255, SplatFormat.packQuatComponent(1.0f))  // 256 clamps to 255
        assertEquals(128, SplatFormat.packQuatComponent(0.0f))
        assertEquals(0, SplatFormat.packQuatComponent(-1.0f))
        assertEquals(192, SplatFormat.packQuatComponent(0.5f))
        assertEquals(64, SplatFormat.packQuatComponent(-0.5f))

        // And the identity rotation, scalar first: (w=1, x=0, y=0, z=0).
        val out = tmp.newFile("quat.splat")
        GaussianInitializer.initializeFromFeaturePoints(spacedPoints(4), out)
        for (s in SplatFormat.decode(out)) {
            assertEquals("q0 is w and must be 1.0 -> 255", 255, s.qw)
            assertEquals("q1 is x and must be 0.0 -> 128", 128, s.qx)
            assertEquals("q2 is y and must be 0.0 -> 128", 128, s.qy)
            assertEquals("q3 is z and must be 0.0 -> 128", 128, s.qz)
        }
    }

    @Test
    fun quaternionPacking_roundTripsOnTheOneOver128Grid() {
        // Every value the decoder can produce must re-encode to the same byte, so
        // a viewer that unpacks and an exporter that repacks stay in agreement.
        for (b in 0..255) {
            val component = SplatFormat.unpackQuatComponent(b)
            // 255 is the clamped image of both 1.0 and anything above it.
            val repacked = SplatFormat.packQuatComponent(component)
            assertEquals("byte $b must survive unpack/pack", b, repacked)
        }
    }

    @Test
    fun validate_rejectsFileWhoseSizeIsNotAMultipleOfThirtyTwo() {
        val result = SplatFormat.validate(ByteArray(32 * 3 + 7))
        assertTrue(result.problems.any { it.contains("not a multiple of 32") })
    }

    @Test
    fun validate_flagsLogScaleWrittenWithoutExp() {
        // brush_bridge.cpp applies exp() to PLY scale_* because 3DGS stores log-scale.
        // Dropping that exp() yields negative "scales"; the viewer then draws nothing.
        val bad = ByteBuffer.allocate(SplatFormat.RECORD_BYTES * 4).order(ByteOrder.LITTLE_ENDIAN)
        repeat(4) { i ->
            bad.putFloat(i.toFloat()); bad.putFloat(0f); bad.putFloat(0f)
            bad.putFloat(-5.3f); bad.putFloat(-5.3f); bad.putFloat(-5.3f)  // raw log-scale
            bad.put(200.toByte()); bad.put(200.toByte()); bad.put(200.toByte()); bad.put(255.toByte())
            bad.put(255.toByte()); bad.put(128.toByte()); bad.put(128.toByte()); bad.put(128.toByte())
        }
        val result = SplatFormat.validate(bad.array())
        assertTrue(
            "negative scales must be reported: ${result.problems}",
            result.problems.any { it.contains("implausible scales") }
        )
    }

    @Test
    fun validate_acceptsAWellFormedModelAndReportsItsBoundingBox() {
        val out = tmp.newFile("bbox.splat")
        GaussianInitializer.initializeFromFeaturePoints(spacedPoints(10), out)
        val result = SplatFormat.validate(out.readBytes())
        assertTrue("well-formed model rejected: ${result.problems}", result.isValid)
        assertEquals(10, result.splatCount)
        assertEquals(0.0f, result.bboxMin[0], 1e-6f)
        assertEquals(4.5f, result.bboxMax[0], 1e-6f)   // 9 * 0.5
        assertEquals(-8.75f, result.bboxMin[2], 1e-6f) // -9 * 0.75 - 2
    }

    @Test
    fun emptyPointList_producesNoFileContentAndDoesNotThrow() {
        val out = tmp.newFile("empty.splat")
        val written = GaussianInitializer.initializeFromFeaturePoints(emptyList(), out)
        assertEquals(0, written)
        assertEquals("an empty scan must not leave a partially written model", 0L, out.length())
    }
}
