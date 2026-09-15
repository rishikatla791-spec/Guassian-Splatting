package com.splat.mobile3dgs

import com.splat.mobile3dgs.capture.FeaturePoint3D
import com.splat.mobile3dgs.engine.GaussianInitializer
import com.splat.mobile3dgs.qa.PlyHeader
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The seed point cloud is the single most load-bearing file in the pipeline: with a
 * good one the optimiser starts on the real surface, without one it samples random
 * points inside the camera frustums and produces the formless blob users report.
 *
 * The failure mode these tests exist for is silent: the ASCII header declares one
 * property list while the binary body is written with a different stride. Nothing
 * throws -- the trainer just reads garbage coordinates. So the invariant asserted
 * here is *header/body agreement*, expressed in terms of the header itself rather
 * than a hard-coded byte count, so it keeps holding when the capture agent adds
 * colour or normals to the seed cloud.
 */
class SeedPlyLayoutTest {

    @get:Rule
    val tmp = TemporaryFolder()

    @Test
    fun writePlyFromXyz_headerAndBodyAgreeOnBytesPerVertex() {
        val count = 64
        val xyz = FloatArray(count * 3) { it * 0.013f - 0.4f }
        val ply = tmp.newFile("dense.ply")

        val written = GaussianInitializer.writePlyFromXyz(xyz, count, ply)
        assertEquals(count, written)

        val header = PlyHeader.parse(ply)
        assertEquals("the trainer only reads binary little-endian PLY",
            "binary_little_endian 1.0", header.format)
        assertEquals(count, header.vertexCount)
        assertEquals(
            "file size must be header + vertexCount * bytesPerVertex; a mismatch here " +
                "means the declared properties do not describe the bytes that follow",
            header.expectedFileBytes, ply.length()
        )
    }

    @Test
    fun writePlyFromXyz_propertyOrderMatchesTheBinaryBody() {
        val count = 5
        val xyz = floatArrayOf(
            1f, 2f, 3f,
            -4f, 5.5f, -6.25f,
            0f, 0f, 0f,
            100f, -200f, 300f,
            0.001f, -0.002f, 0.003f
        )
        val ply = tmp.newFile("order.ply")
        GaussianInitializer.writePlyFromXyz(xyz, count, ply)

        val bytes = ply.readBytes()
        val header = PlyHeader.parse(bytes)
        val body = ByteBuffer.wrap(bytes, header.headerBytes, bytes.size - header.headerBytes)
            .slice().order(ByteOrder.LITTLE_ENDIAN)

        val offX = header.offsetOf("x")
        val offY = header.offsetOf("y")
        val offZ = header.offsetOf("z")
        assertTrue("seed PLY must declare x, y and z", offX >= 0 && offY >= 0 && offZ >= 0)

        for (i in 0 until count) {
            val base = i * header.bytesPerVertex
            assertEquals("x of vertex $i", xyz[i * 3], body.getFloat(base + offX), 0f)
            assertEquals("y of vertex $i", xyz[i * 3 + 1], body.getFloat(base + offY), 0f)
            assertEquals("z of vertex $i", xyz[i * 3 + 2], body.getFloat(base + offZ), 0f)
        }
    }

    @Test
    fun writeInitialPointCloudPly_headerAndBodyAgree() {
        val points = (0 until 40).map {
            FeaturePoint3D(it, it * 0.2f, it * 0.1f, -it * 0.3f, confidence = 0.8f)
        }
        val ply = tmp.newFile("features.ply")
        val written = GaussianInitializer.writeInitialPointCloudPly(points, ply)
        assertEquals(points.size, written)

        val header = PlyHeader.parse(ply)
        assertEquals(written, header.vertexCount)
        assertEquals(header.expectedFileBytes, ply.length())
        assertTrue("header must end with end_header + newline",
            String(ply.readBytes(), 0, header.headerBytes, Charsets.US_ASCII)
                .endsWith("end_header\n"))
    }

    @Test
    fun writeInitialPointCloudPly_dropsNonFinitePointsAndStillKeepsHeaderConsistent() {
        // A single NaN from a failed depth unprojection used to be enough to make
        // the trainer's PLY reader produce an all-NaN scene.
        val points = listOf(
            FeaturePoint3D(0, 0f, 0f, 0f, 0.9f),
            FeaturePoint3D(1, Float.NaN, 1f, 1f, 0.9f),
            FeaturePoint3D(2, 1f, Float.POSITIVE_INFINITY, 1f, 0.9f),
            FeaturePoint3D(3, 2f, 2f, 2f, 0.9f)
        )
        val ply = tmp.newFile("nonfinite.ply")
        val written = GaussianInitializer.writeInitialPointCloudPly(points, ply)

        assertEquals("non-finite points must be dropped", 2, written)
        val header = PlyHeader.parse(ply)
        assertEquals(2, header.vertexCount)
        assertEquals(
            "the declared vertex count must match what was actually written",
            header.expectedFileBytes, ply.length()
        )
    }

    @Test
    fun writePlyFromXyz_withZeroCountWritesNothing() {
        val ply = tmp.newFile("zero.ply")
        assertEquals(0, GaussianInitializer.writePlyFromXyz(FloatArray(0), 0, ply))
        assertEquals(
            "a zero-vertex PLY must not be produced; the exporter must omit ply_file_path instead",
            0L, ply.length()
        )
    }

    // -------------------------------------------------------------------------
    // Header-parser contract, matched to convert_ply_to_splat() in brush_bridge.cpp
    // -------------------------------------------------------------------------

    @Test
    fun bytesPerVertex_ofTheColouredSeedLayout_is27() {
        // 3 float positions + 3 uchar colours + 3 float normals = 12 + 3 + 12 = 27.
        // This is the layout the seed writer moves to once colours are seeded; if the
        // header gains a property and the writer does not, this catches it.
        val headerText = buildString {
            append("ply\n")
            append("format binary_little_endian 1.0\n")
            append("element vertex 3\n")
            append("property float x\n")
            append("property float y\n")
            append("property float z\n")
            append("property uchar red\n")
            append("property uchar green\n")
            append("property uchar blue\n")
            append("property float nx\n")
            append("property float ny\n")
            append("property float nz\n")
            append("end_header\n")
        }
        val header = PlyHeader.parse(headerText.toByteArray(Charsets.US_ASCII) + ByteArray(3 * 27))

        assertEquals(27, header.bytesPerVertex)
        assertEquals(
            listOf("x", "y", "z", "red", "green", "blue", "nx", "ny", "nz"),
            header.propertyNames
        )
        assertEquals(0, header.offsetOf("x"))
        assertEquals(12, header.offsetOf("red"))
        assertEquals(15, header.offsetOf("nx"))
        assertEquals(-1, header.offsetOf("opacity"))
        assertEquals(headerText.length.toLong() + 3L * 27L, header.expectedFileBytes)
    }

    @Test
    fun bytesPerVertex_ofTheFullGaussianPlyLayout_matchesTheCppParser() {
        // The layout brush_bridge.cpp expects back from the trainer.
        val props = listOf(
            "x", "y", "z",
            "f_dc_0", "f_dc_1", "f_dc_2",
            "opacity",
            "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3"
        )
        val headerText = buildString {
            append("ply\n")
            append("format binary_little_endian 1.0\n")
            append("element vertex 2\n")
            props.forEach { append("property float $it\n") }
            append("end_header\n")
        }
        val header = PlyHeader.parse(headerText.toByteArray(Charsets.US_ASCII) + ByteArray(2 * 56))

        assertEquals(56, header.bytesPerVertex)
        assertEquals(0, header.offsetOf("x"))
        assertEquals(12, header.offsetOf("f_dc_0"))
        assertEquals(24, header.offsetOf("opacity"))
        assertEquals(28, header.offsetOf("scale_0"))
        assertEquals(40, header.offsetOf("rot_0"))
        assertFalse("rot_* must be present or the C++ converter silently writes identity rotations",
            header.offsetOf("rot_3") < 0)
    }

    @Test
    fun plyParser_toleratesCrLfLineEndings() {
        val headerText = "ply\r\nformat binary_little_endian 1.0\r\nelement vertex 1\r\n" +
            "property float x\r\nproperty float y\r\nproperty float z\r\nend_header\r\n"
        val header = PlyHeader.parse(headerText.toByteArray(Charsets.US_ASCII) + ByteArray(12))
        assertEquals(1, header.vertexCount)
        assertEquals(12, header.bytesPerVertex)
    }
}
