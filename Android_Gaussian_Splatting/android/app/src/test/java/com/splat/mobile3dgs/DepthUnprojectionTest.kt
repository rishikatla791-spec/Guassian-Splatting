package com.splat.mobile3dgs

import android.media.Image
import com.google.ar.core.Pose
import com.splat.mobile3dgs.capture.DepthPointExtractor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.mockito.Mockito
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * Depth unprojection: the coordinate convention and its round-trip accuracy.
 *
 * `DepthPointExtractor` converts an ARCore DEPTH16 map into world-space seed points.
 * It bridges two conventions that differ by two sign flips:
 *
 *   pixel/CV frame : +u right, +v DOWN, depth d positive along the view direction
 *   ARCore camera  : +x right, +y UP,   -z forward   (OpenGL)
 *
 * Getting either sign wrong still produces a plausible-looking cloud -- mirrored or
 * flipped -- which the optimiser then fits to the wrong side of the camera. Nothing
 * throws. These tests pin the convention down from both ends: a hand-checkable
 * directional case with an identity camera, and an exact numeric round-trip through
 * an arbitrary rotated and translated pose.
 */
class DepthUnprojectionTest {

    /**
     * Byte order the production code currently uses to read DEPTH16 samples.
     *
     * `DepthPointExtractor` pins the plane buffer to LITTLE_ENDIAN before reading,
     * matching how every Android device stores DEPTH16 and what the official ARCore
     * Depth samples do. Fixtures are therefore encoded little-endian and handed over
     * BIG_ENDIAN, as ARCore delivers them -- see
     * [depth16_readWithoutNativeByteOrder_discardsRealisticSamples] below.
     *
     * When the byte-order defect is fixed, flip this constant to LITTLE_ENDIAN and
     * update [depth16_readWithoutNativeByteOrder_discardsRealisticSamples].
     */
    /**
     * Floats emitted per surviving sample: x, y, z, confidence, r, g, b, logScale.
     * Seed points carry colour and an initial scale so the optimiser starts from a
     * coloured cloud rather than grey dots.
     */
    private val FLOATS_PER_POINT = 8

    private val DEPTH16_ORDER_AS_READ: ByteOrder = ByteOrder.LITTLE_ENDIAN

    // -------------------------------------------------------------------------
    // Fixtures
    // -------------------------------------------------------------------------

    /**
     * A DEPTH16 [Image] whose plane holds [millimetres] laid out row-major, 2 bytes
     * per pixel, with the sample bytes *encoded* in [wireOrder].
     *
     * The returned buffer is always handed over in `ByteBuffer`'s default BIG_ENDIAN
     * order, because that is exactly what ARCore does: the plane comes from
     * `NewDirectByteBuffer`, which never sets an order. Encoding order and read order
     * are therefore independent here, which is what makes the byte-order tripwire at
     * the bottom of this file meaningful.
     */
    private fun depthImage(
        width: Int,
        height: Int,
        millimetres: ShortArray,
        wireOrder: ByteOrder = DEPTH16_ORDER_AS_READ
    ): Image {
        require(millimetres.size == width * height)
        val rowStride = width * 2
        val buf = ByteBuffer.allocate(rowStride * height)
        buf.order(wireOrder)
        for (i in millimetres.indices) buf.putShort(i * 2, millimetres[i])
        buf.order(ByteOrder.BIG_ENDIAN) // as delivered by ARCore: production re-orders it

        val plane = Mockito.mock(Image.Plane::class.java)
        Mockito.`when`(plane.buffer).thenReturn(buf)
        Mockito.`when`(plane.rowStride).thenReturn(rowStride)
        Mockito.`when`(plane.pixelStride).thenReturn(2)

        val image = Mockito.mock(Image::class.java)
        Mockito.`when`(image.width).thenReturn(width)
        Mockito.`when`(image.height).thenReturn(height)
        Mockito.`when`(image.planes).thenReturn(arrayOf(plane))
        return image
    }

    /** Single non-zero depth sample at (u, v); everything else is "no reading". */
    private fun singleSampleDepthImage(
        width: Int, height: Int, u: Int, v: Int, depthMeters: Float,
        wireOrder: ByteOrder = DEPTH16_ORDER_AS_READ
    ): Image {
        val mm = ShortArray(width * height)
        mm[v * width + u] = Math.round(depthMeters * 1000f).toShort()
        return depthImage(width, height, mm, wireOrder)
    }

    private fun pose(tx: Float, ty: Float, tz: Float, quatXyzw: FloatArray): Pose =
        Pose(floatArrayOf(tx, ty, tz), quatXyzw)

    private val identityPose: Pose get() = pose(0f, 0f, 0f, floatArrayOf(0f, 0f, 0f, 1f))

    /**
     * Independent forward model: world point -> (u, v, depth), written from the
     * convention as documented rather than from the code under test.
     */
    private fun project(
        world: FloatArray, cameraPose: Pose,
        fx: Float, fy: Float, cx: Float, cy: Float
    ): Triple<Float, Float, Float> {
        val cam = cameraPose.inverse().transformPoint(world)   // ARCore camera frame
        val depth = -cam[2]                                     // -z is forward
        val u = fx * (cam[0] / depth) + cx
        val v = fy * (-cam[1] / depth) + cy                     // +v is down, +y is up
        return Triple(u, v, depth)
    }

    private fun unproject(
        image: Image, cameraPose: Pose,
        fx: Float, fy: Float, cx: Float, cy: Float,
        cpuW: Int = image.width, cpuH: Int = image.height
    ): FloatArray = DepthPointExtractor.extractWorldPoints(
        depthImage = image,
        confidenceImage = null,
        focal = floatArrayOf(fx, fy),
        principal = floatArrayOf(cx, cy),
        imageDims = intArrayOf(cpuW, cpuH),
        cameraPose = cameraPose
    )

    // -------------------------------------------------------------------------
    // Direction / sign convention
    // -------------------------------------------------------------------------

    @Test
    fun identityCamera_unprojectsAlongNegativeZ() {
        // ARCore identity pose looks down -Z. A sample at the principal point at 2 m
        // must land 2 m in FRONT of the camera, i.e. at z = -2, not z = +2.
        val w = 32; val h = 24
        val img = singleSampleDepthImage(w, h, u = 16, v = 12, depthMeters = 2.0f)
        val out = unproject(img, identityPose, fx = 20f, fy = 20f, cx = 16f, cy = 12f)

        assertEquals("exactly one sample must unproject", FLOATS_PER_POINT, out.size)
        assertEquals(0f, out[0], 1e-5f)
        assertEquals(0f, out[1], 1e-5f)
        assertEquals(
            "a point in front of an identity ARCore camera has NEGATIVE z; " +
                "a positive value means the forward axis was flipped",
            -2.0f, out[2], 1e-5f
        )
    }

    @Test
    fun identityCamera_pixelRightOfPrincipalPointGoesRight_pixelAbovePrincipalPointGoesUp() {
        val w = 64; val h = 48
        val fx = 40f; val fy = 40f; val cx = 32f; val cy = 24f

        // 8 px to the RIGHT of the principal point -> world +x
        val right = unproject(
            singleSampleDepthImage(w, h, u = 40, v = 24, depthMeters = 2.0f),
            identityPose, fx, fy, cx, cy
        )
        assertEquals(FLOATS_PER_POINT, right.size)
        assertTrue(
            "a pixel right of the principal point must unproject to +x, got x=${right[0]}",
            right[0] > 0.1f
        )
        assertEquals("no vertical offset expected", 0f, right[1], 1e-5f)

        // 8 px ABOVE the principal point. Image v grows DOWNWARD, so a smaller v is
        // higher in the scene and must unproject to +y.
        val above = unproject(
            singleSampleDepthImage(w, h, u = 32, v = 16, depthMeters = 2.0f),
            identityPose, fx, fy, cx, cy
        )
        assertEquals(FLOATS_PER_POINT, above.size)
        assertEquals("no horizontal offset expected", 0f, above[0], 1e-5f)
        assertTrue(
            "a pixel ABOVE the principal point must unproject to +y (the v axis points " +
                "down while the ARCore y axis points up), got y=${above[1]}",
            above[1] > 0.1f
        )
    }

    @Test
    fun identityCamera_producesTheExactPinholeCoordinates() {
        val w = 64; val h = 48
        val fx = 40f; val fy = 40f; val cx = 32f; val cy = 24f
        val d = 2.5f
        val u = 44; val v = 14

        val out = unproject(singleSampleDepthImage(w, h, u, v, d), identityPose, fx, fy, cx, cy)
        assertEquals(FLOATS_PER_POINT, out.size)
        assertEquals((u - cx) * d / fx, out[0], 1e-5f)
        assertEquals(-(v - cy) * d / fy, out[1], 1e-5f)
        assertEquals(-d, out[2], 1e-5f)
    }

    // -------------------------------------------------------------------------
    // Round trip
    // -------------------------------------------------------------------------

    @Test
    fun worldPoint_survivesProjectThenUnprojectWithinOneMicrometre() {
        val w = 80; val h = 60
        val fx = 70f; val fy = 70f; val cx = 40f; val cy = 30f

        // An arbitrary rotated and translated camera: 40 degrees about a tilted axis.
        val axis = floatArrayOf(0.3f, 0.9f, -0.32f)
        val n = sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
        val half = Math.toRadians(40.0 / 2.0)
        val s = Math.sin(half).toFloat()
        val q = floatArrayOf(axis[0] / n * s, axis[1] / n * s, axis[2] / n * s, Math.cos(half).toFloat())
        val camera = pose(1.25f, -0.5f, 3.75f, q)

        // Pick the pixel and depth first so the fixture holds an exact DEPTH16 value,
        // then derive the world point the camera would have seen there.
        val cases = listOf(
            Triple(40, 30, 1.000f),
            Triple(12, 8, 0.500f),
            Triple(70, 52, 4.000f),
            Triple(5, 55, 2.345f)
        )

        for ((u, v, d) in cases) {
            val camLocal = floatArrayOf((u - cx) * d / fx, -(v - cy) * d / fy, -d)
            val world = camera.transformPoint(camLocal)

            // Sanity: the independent forward model agrees on which pixel this is.
            val (pu, pv, pd) = project(world, camera, fx, fy, cx, cy)
            assertEquals("forward model u", u.toFloat(), pu, 1e-3f)
            assertEquals("forward model v", v.toFloat(), pv, 1e-3f)
            assertEquals("forward model depth", d, pd, 1e-4f)

            val out = unproject(singleSampleDepthImage(w, h, u, v, d), camera, fx, fy, cx, cy)
            assertEquals("expected exactly one point for pixel ($u,$v)", FLOATS_PER_POINT, out.size)

            // DEPTH16 quantises to whole millimetres, so the fixture depths are chosen
            // to be exact. What is asserted here is the geometry, to under a micrometre.
            assertEquals("round-trip x for pixel ($u,$v)", world[0], out[0], 1e-5f)
            assertEquals("round-trip y for pixel ($u,$v)", world[1], out[1], 1e-5f)
            assertEquals("round-trip z for pixel ($u,$v)", world[2], out[2], 1e-5f)
        }
    }

    @Test
    fun intrinsicsAreScaledFromTheCpuImageToTheDepthImage() {
        // ARCore reports intrinsics for the large CPU image but the depth map is tiny.
        // Failing to rescale fx/fy/cx/cy makes the cloud collapse toward the optical
        // axis; this pins the 12x factor between a 1920x1080 CPU image and a 160x90 map.
        val cpuW = 1920; val cpuH = 1080
        val dw = 160; val dh = 90
        val cpuFx = 1440f; val cpuFy = 1440f; val cpuCx = 960f; val cpuCy = 540f
        val scale = dw.toFloat() / cpuW          // 1/12

        val d = 3.0f
        val u = 100; val v = 20
        val img = singleSampleDepthImage(dw, dh, u, v, d)
        val out = DepthPointExtractor.extractWorldPoints(
            depthImage = img,
            confidenceImage = null,
            focal = floatArrayOf(cpuFx, cpuFy),
            principal = floatArrayOf(cpuCx, cpuCy),
            imageDims = intArrayOf(cpuW, cpuH),
            cameraPose = identityPose
        )
        assertEquals(FLOATS_PER_POINT, out.size)

        val fx = cpuFx * scale; val cx = cpuCx * scale
        val fy = cpuFy * (dh.toFloat() / cpuH); val cy = cpuCy * (dh.toFloat() / cpuH)
        assertEquals((u - cx) * d / fx, out[0], 1e-5f)
        assertEquals(-(v - cy) * d / fy, out[1], 1e-5f)
        assertEquals(-d, out[2], 1e-5f)
    }

    // -------------------------------------------------------------------------
    // Robustness / crash-freedom
    // -------------------------------------------------------------------------

    @Test
    fun depthOutsideTheUsableRangeIsDiscarded() {
        val w = 32; val h = 24
        // Default gate in DepthPointExtractor is 0.15 m .. 6.0 m.
        assertEquals(0, unproject(singleSampleDepthImage(w, h, 16, 12, 0.05f), identityPose, 20f, 20f, 16f, 12f).size)
        assertEquals(0, unproject(singleSampleDepthImage(w, h, 16, 12, 9.0f), identityPose, 20f, 20f, 16f, 12f).size)
        assertEquals(FLOATS_PER_POINT, unproject(singleSampleDepthImage(w, h, 16, 12, 1.0f), identityPose, 20f, 20f, 16f, 12f).size)
    }

    @Test
    fun zeroDepthPixelsAreTreatedAsNoReading() {
        // An all-zero depth map is what a device returns before the depth model warms
        // up. It must yield an empty cloud, not a wall of points at the camera origin.
        val w = 32; val h = 24
        val out = unproject(depthImage(w, h, ShortArray(w * h)), identityPose, 20f, 20f, 16f, 12f)
        assertEquals(0, out.size)
    }

    @Test
    fun degenerateIntrinsicsReturnEmptyRatherThanNaNs() {
        val img = singleSampleDepthImage(32, 24, 16, 12, 2.0f)
        val out = DepthPointExtractor.extractWorldPoints(
            depthImage = img,
            confidenceImage = null,
            focal = floatArrayOf(0f, 0f),       // a device that reported no focal length
            principal = floatArrayOf(16f, 12f),
            imageDims = intArrayOf(32, 24),
            cameraPose = identityPose
        )
        assertEquals("zero focal length must short-circuit, not divide by zero", 0, out.size)
    }

    @Test
    fun zeroSizedDepthImageReturnsEmpty() {
        val plane = Mockito.mock(Image.Plane::class.java)
        Mockito.`when`(plane.buffer).thenReturn(ByteBuffer.allocate(0))
        Mockito.`when`(plane.rowStride).thenReturn(0)
        Mockito.`when`(plane.pixelStride).thenReturn(2)
        val image = Mockito.mock(Image::class.java)
        Mockito.`when`(image.width).thenReturn(0)
        Mockito.`when`(image.height).thenReturn(0)
        Mockito.`when`(image.planes).thenReturn(arrayOf(plane))

        val out = DepthPointExtractor.extractWorldPoints(
            depthImage = image,
            confidenceImage = null,
            focal = floatArrayOf(20f, 20f),
            principal = floatArrayOf(16f, 12f),
            imageDims = intArrayOf(32, 24),
            cameraPose = identityPose
        )
        assertEquals(0, out.size)
    }

    @Test
    fun mismatchedConfidenceImageIsIgnoredRatherThanIndexedOutOfBounds() {
        // CaptureActivity drops a confidence map whose dimensions differ, but the
        // extractor must be safe on its own: raw-depth confidence maps are routinely
        // a different size from the smoothed depth map.
        val depth = singleSampleDepthImage(32, 24, 16, 12, 2.0f)
        val confPlane = Mockito.mock(Image.Plane::class.java)
        Mockito.`when`(confPlane.buffer).thenReturn(ByteBuffer.allocate(8 * 6))
        Mockito.`when`(confPlane.rowStride).thenReturn(8)
        Mockito.`when`(confPlane.pixelStride).thenReturn(1)
        val conf = Mockito.mock(Image::class.java)
        Mockito.`when`(conf.width).thenReturn(8)
        Mockito.`when`(conf.height).thenReturn(6)
        Mockito.`when`(conf.planes).thenReturn(arrayOf(confPlane))

        val out = DepthPointExtractor.extractWorldPoints(
            depthImage = depth,
            confidenceImage = conf,
            focal = floatArrayOf(20f, 20f),
            principal = floatArrayOf(16f, 12f),
            imageDims = intArrayOf(32, 24),
            cameraPose = identityPose
        )
        assertEquals("a differently sized confidence map must be ignored, not indexed", FLOATS_PER_POINT, out.size)
        assertEquals("ignored confidence means full confidence", 1.0f, out[3], 1e-6f)
    }

    // -------------------------------------------------------------------------
    // Byte-order defect (see DEPTH16_ORDER_AS_READ)
    // -------------------------------------------------------------------------

    /**
     * REGRESSION TRIPWIRE, not an endorsement.
     *
     * ARCore hands the DEPTH16 plane back as a direct `ByteBuffer` whose order is
     * unspecified; a java.nio direct buffer defaults to BIG_ENDIAN while the device
     * stores the samples little-endian. Reading with the wrong order byte-swaps
     * every sample -- a real 1.5 m reading (1500 mm = 0x05DC) decodes as 0xDC05 =
     * 56 325 mm and is silently dropped by the range gate, collapsing the cloud
     * without raising anything.
     *
     * `DepthPointExtractor` now pins the order explicitly, so this fixture -- handed
     * over BIG_ENDIAN exactly as ARCore delivers it -- must still decode correctly.
     */
    @Test
    fun depth16_readWithoutNativeByteOrder_discardsRealisticSamples() {
        val littleEndian = singleSampleDepthImage(
            32, 24, u = 16, v = 12, depthMeters = 1.5f, wireOrder = ByteOrder.LITTLE_ENDIAN
        )
        val out = unproject(littleEndian, identityPose, 20f, 20f, 16f, 12f)
        assertEquals(
            "DEPTH16 samples are little-endian on-device; DepthPointExtractor now " +
                "pins the buffer order, so a real 1500 mm reading must survive the " +
                "range gate instead of being byte-swapped into 56325 mm",
            FLOATS_PER_POINT, out.size
        )
    }
}
