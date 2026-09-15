package com.splat.mobile3dgs.ar

import android.opengl.GLES20
import android.opengl.GLES30
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.IntBuffer

/**
 * Real 3D Gaussian Splatting rasteriser for AR, on OpenGL ES 3.0.
 *
 * Each splat is one instanced quad. The vertex shader rebuilds the 3D covariance
 * from the stored rotation and scale, pushes it through the perspective Jacobian
 * into a 2x2 screen-space covariance, eigen-decomposes that to get the major and
 * minor screen axes, and sizes the quad along those axes. The fragment shader
 * evaluates the gaussian.
 *
 * Two details matter a great deal for image quality:
 *
 *  * The screen-space extents come from an eigen-decomposition of the projected
 *    2D covariance, not from the columns of `view * R * S`. Those columns are the
 *    projected *3D* axes; when a splat is close to edge-on, one of them nearly
 *    vanishes while another blows up, producing the classic needle-spike
 *    artefacts. The eigenvectors of the 2D covariance are always a sane, stable
 *    screen-space frame.
 *  * A small low-pass term is added to the 2D covariance diagonal. Without it,
 *    splats that project to sub-pixel size alias violently as the phone moves.
 *
 * The payload lives in an RGBA32UI texture (2 texels per splat, byte-identical to
 * the `.splat` record), and the draw order lives in a small per-instance index
 * buffer. Re-sorting therefore only re-uploads 4 bytes per splat, never the
 * 32-byte payload.
 */
class SplatRenderer {

    private var program = 0
    private var vao = 0
    private var quadVbo = 0
    private var indexVbo = 0
    private var dataTexture = 0

    private var uModelView = 0
    private var uProjection = 0
    private var uViewport = 0
    private var uFocal = 0
    private var uSplats = 0
    private var uAlphaScale = 0

    private var indexCapacity = 0
    private var drawCount = 0
    private var stagingIndices: IntBuffer? = null

    var isReady = false
        private set

    /** Number of splats currently being drawn (after LOD decimation). */
    val visibleSplats: Int get() = drawCount

    private val quadCorners = floatArrayOf(
        -2f, -2f,
        +2f, -2f,
        -2f, +2f,
        +2f, +2f
    )

    fun createOnGlThread() {
        // The EGL context can be recreated (background/foreground on some
        // drivers); every id from the old context is dead, so reset all state
        // rather than half-reusing it.
        isReady = false
        drawCount = 0
        indexCapacity = 0
        stagingIndices = null
        dataTexture = 0

        val vs = GlUtil.compileShader(GLES20.GL_VERTEX_SHADER, VERTEX_SHADER)
        val fs = GlUtil.compileShader(GLES20.GL_FRAGMENT_SHADER, FRAGMENT_SHADER)
        program = GlUtil.linkProgram(vs, fs)

        uModelView = GLES20.glGetUniformLocation(program, "u_modelView")
        uProjection = GLES20.glGetUniformLocation(program, "u_projection")
        uViewport = GLES20.glGetUniformLocation(program, "u_viewport")
        uFocal = GLES20.glGetUniformLocation(program, "u_focal")
        uSplats = GLES20.glGetUniformLocation(program, "u_splats")
        uAlphaScale = GLES20.glGetUniformLocation(program, "u_alphaScale")

        val buffers = IntArray(2)
        GLES20.glGenBuffers(2, buffers, 0)
        quadVbo = buffers[0]
        indexVbo = buffers[1]

        val cornerBuffer = ByteBuffer.allocateDirect(quadCorners.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        cornerBuffer.put(quadCorners).position(0)
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, quadVbo)
        GLES20.glBufferData(
            GLES20.GL_ARRAY_BUFFER, quadCorners.size * 4, cornerBuffer, GLES20.GL_STATIC_DRAW
        )

        val vaos = IntArray(1)
        GLES30.glGenVertexArrays(1, vaos, 0)
        vao = vaos[0]
        GLES30.glBindVertexArray(vao)

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, quadVbo)
        GLES20.glEnableVertexAttribArray(ATTR_CORNER)
        GLES20.glVertexAttribPointer(ATTR_CORNER, 2, GLES20.GL_FLOAT, false, 0, 0)
        GLES30.glVertexAttribDivisor(ATTR_CORNER, 0)

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, indexVbo)
        GLES20.glEnableVertexAttribArray(ATTR_INDEX)
        GLES30.glVertexAttribIPointer(ATTR_INDEX, 1, GLES20.GL_UNSIGNED_INT, 0, 0)
        GLES30.glVertexAttribDivisor(ATTR_INDEX, 1)

        GLES30.glBindVertexArray(0)
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)
        GlUtil.checkGlError("SplatRenderer.create")
    }

    /**
     * Upload a model's payload. Returns false if the model needs a texture taller
     * than the driver allows (the caller should retry with a smaller budget).
     */
    fun uploadModel(cloud: SplatCloud): Boolean {
        val rows = cloud.textureRows
        val maxSize = IntArray(1)
        GLES20.glGetIntegerv(GLES20.GL_MAX_TEXTURE_SIZE, maxSize, 0)
        if (rows > maxSize[0] || SPLAT_TEXTURE_WIDTH > maxSize[0]) {
            Log.e(TAG, "Model needs a ${SPLAT_TEXTURE_WIDTH}x$rows texture, max is ${maxSize[0]}")
            return false
        }

        if (dataTexture != 0) {
            GLES20.glDeleteTextures(1, intArrayOf(dataTexture), 0)
            dataTexture = 0
        }
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        dataTexture = textures[0]

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, dataTexture)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        cloud.payload.position(0)
        GLES20.glPixelStorei(GLES20.GL_UNPACK_ALIGNMENT, 4)
        GLES20.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES30.GL_RGBA32UI,
            SPLAT_TEXTURE_WIDTH, rows, 0,
            GLES30.GL_RGBA_INTEGER, GLES20.GL_UNSIGNED_INT, cloud.payload
        )
        val err = GLES20.glGetError()
        if (err != GLES20.GL_NO_ERROR) {
            Log.e(TAG, "Splat texture upload failed: 0x${Integer.toHexString(err)}")
            return false
        }

        // Start with an identity ordering so something is on screen before the
        // first depth sort lands.
        val identity = IntArray(cloud.count) { it }
        setDrawOrder(identity, cloud.count)

        isReady = true
        Log.i(TAG, "Uploaded ${cloud.count} splats as a ${SPLAT_TEXTURE_WIDTH}x$rows RGBA32UI texture")
        return true
    }

    /** Replace the per-instance draw order. GL thread only. */
    fun setDrawOrder(order: IntArray, count: Int) {
        if (count <= 0) {
            drawCount = 0
            return
        }
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, indexVbo)
        val bytes = count * 4
        // Re-sorting happens several times a second; a fresh direct buffer each
        // time would be pure GC pressure, so the staging buffer is reused.
        val existing = stagingIndices
        val buffer: IntBuffer = if (existing != null && existing.capacity() >= count) {
            existing
        } else {
            ByteBuffer.allocateDirect(bytes)
                .order(ByteOrder.nativeOrder()).asIntBuffer()
                .also { stagingIndices = it }
        }
        buffer.position(0)
        buffer.put(order, 0, count)
        buffer.position(0)
        if (bytes > indexCapacity) {
            GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, bytes, buffer, GLES20.GL_DYNAMIC_DRAW)
            indexCapacity = bytes
        } else {
            GLES20.glBufferSubData(GLES20.GL_ARRAY_BUFFER, 0, bytes, buffer)
        }
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)
        drawCount = count
    }

    /**
     * Draw the model.
     *
     * @param modelView column-major model-view (anchor pose folded into the model).
     * @param projection column-major projection straight from ARCore.
     */
    fun draw(
        modelView: FloatArray,
        projection: FloatArray,
        viewportWidth: Int,
        viewportHeight: Int,
        alphaScale: Float
    ) {
        if (!isReady || drawCount == 0 || program == 0) return

        // Focal length in pixels, recovered from the projection matrix so the
        // Jacobian matches whatever intrinsics ARCore handed us this frame.
        val fx = projection[0] * viewportWidth * 0.5f
        val fy = projection[5] * viewportHeight * 0.5f

        GLES20.glUseProgram(program)

        // Splats are pre-sorted back to front, so they composite with plain "over"
        // blending and must not depth-test against each other.
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glDepthMask(false)
        GLES20.glDisable(GLES20.GL_CULL_FACE)
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFuncSeparate(
            GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA,
            GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA
        )

        GLES20.glUniformMatrix4fv(uModelView, 1, false, modelView, 0)
        GLES20.glUniformMatrix4fv(uProjection, 1, false, projection, 0)
        GLES20.glUniform2f(uViewport, viewportWidth.toFloat(), viewportHeight.toFloat())
        GLES20.glUniform2f(uFocal, fx, fy)
        GLES20.glUniform1f(uAlphaScale, alphaScale)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, dataTexture)
        GLES20.glUniform1i(uSplats, 0)

        GLES30.glBindVertexArray(vao)
        GLES30.glDrawArraysInstanced(GLES20.GL_TRIANGLE_STRIP, 0, 4, drawCount)
        GLES30.glBindVertexArray(0)

        GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glDepthMask(true)
    }

    fun release() {
        if (program != 0) { GLES20.glDeleteProgram(program); program = 0 }
        if (dataTexture != 0) { GLES20.glDeleteTextures(1, intArrayOf(dataTexture), 0); dataTexture = 0 }
        if (quadVbo != 0 || indexVbo != 0) {
            GLES20.glDeleteBuffers(2, intArrayOf(quadVbo, indexVbo), 0)
            quadVbo = 0; indexVbo = 0
        }
        if (vao != 0) { GLES30.glDeleteVertexArrays(1, intArrayOf(vao), 0); vao = 0 }
        isReady = false
        drawCount = 0
    }

    companion object {
        private const val TAG = "SplatRenderer"
        private const val ATTR_CORNER = 0
        private const val ATTR_INDEX = 1

        private val VERTEX_SHADER = """
            #version 300 es
            precision highp float;
            precision highp int;

            layout(location = 0) in vec2 a_corner;   // quad corner, +/-2 sigma-ish
            layout(location = 1) in uint a_index;    // splat id, back-to-front order

            uniform highp usampler2D u_splats;
            uniform mat4 u_modelView;
            uniform mat4 u_projection;
            uniform vec2 u_viewport;   // pixels
            uniform vec2 u_focal;      // pixels
            uniform float u_alphaScale;

            out vec4 v_color;
            out vec2 v_quad;

            vec4 unpackBytes(uint v) {
                return vec4(
                    float( v         & 255u),
                    float((v >>  8u) & 255u),
                    float((v >> 16u) & 255u),
                    float((v >> 24u) & 255u)
                );
            }

            void main() {
                int texWidth = textureSize(u_splats, 0).x;
                int base = int(a_index) * 2;
                ivec2 t0 = ivec2(base % texWidth, base / texWidth);
                ivec2 t1 = ivec2((base + 1) % texWidth, (base + 1) / texWidth);
                uvec4 d0 = texelFetch(u_splats, t0, 0);
                uvec4 d1 = texelFetch(u_splats, t1, 0);

                vec3 center = vec3(
                    uintBitsToFloat(d0.x), uintBitsToFloat(d0.y), uintBitsToFloat(d0.z)
                );
                vec3 scale = vec3(
                    uintBitsToFloat(d0.w), uintBitsToFloat(d1.x), uintBitsToFloat(d1.y)
                );
                vec4 rgba = unpackBytes(d1.z) / 255.0;

                // Rotation bytes are (w, x, y, z) with b = r * 128 + 128.
                vec4 qb = (unpackBytes(d1.w) - 128.0) / 128.0;
                vec4 q = vec4(qb.y, qb.z, qb.w, qb.x);   // -> (x, y, z, w), w scalar
                float qn = length(q);
                q = (qn > 1e-6) ? q / qn : vec4(0.0, 0.0, 0.0, 1.0);

                vec4 camPos = u_modelView * vec4(center, 1.0);
                float depth = -camPos.z;              // ARCore/GL: camera looks down -Z
                if (depth < 0.05 || rgba.a <= 0.0) {
                    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);   // off-screen, clipped
                    v_color = vec4(0.0);
                    v_quad = vec2(0.0);
                    return;
                }

                // Build the 3D covariance in camera space: Sigma = (T R S)(T R S)^T,
                // where T is the model-view basis (so the model's own rotation and
                // user scale are already folded in).
                mat3 R = mat3(
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z),
                    2.0 * (q.x * q.y + q.w * q.z),
                    2.0 * (q.x * q.z - q.w * q.y),

                    2.0 * (q.x * q.y - q.w * q.z),
                    1.0 - 2.0 * (q.x * q.x + q.z * q.z),
                    2.0 * (q.y * q.z + q.w * q.x),

                    2.0 * (q.x * q.z + q.w * q.y),
                    2.0 * (q.y * q.z - q.w * q.x),
                    1.0 - 2.0 * (q.x * q.x + q.y * q.y)
                );
                mat3 S = mat3(
                    scale.x, 0.0, 0.0,
                    0.0, scale.y, 0.0,
                    0.0, 0.0, scale.z
                );
                mat3 T = mat3(u_modelView) * R * S;
                mat3 sigma = T * transpose(T);

                // Perspective Jacobian at the splat centre. The tangents are clamped
                // so splats far outside the frustum do not produce absurd screen
                // covariances near the singularity.
                float invDepth = 1.0 / depth;
                float limX = 1.3 * (u_viewport.x * 0.5) / u_focal.x;
                float limY = 1.3 * (u_viewport.y * 0.5) / u_focal.y;
                float tx = clamp(camPos.x * invDepth, -limX, limX) * depth;
                float ty = clamp(camPos.y * invDepth, -limY, limY) * depth;

                mat3x2 J = mat3x2(
                    u_focal.x * invDepth, 0.0,
                    0.0, u_focal.y * invDepth,
                    u_focal.x * tx * invDepth * invDepth,
                    u_focal.y * ty * invDepth * invDepth
                );
                mat2 cov = J * sigma * transpose(J);

                // Low-pass: guarantee at least ~half a pixel of support so tiny
                // splats stop flickering.
                cov[0][0] += 0.3;
                cov[1][1] += 0.3;

                float a = cov[0][0];
                float b = cov[0][1];
                float c = cov[1][1];
                float mid = 0.5 * (a + c);
                float disc = sqrt(max(mid * mid - (a * c - b * b), 0.0));
                float l1 = mid + disc;
                float l2 = max(mid - disc, 0.1);
                if (l1 < 0.1) {
                    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                    v_color = vec4(0.0);
                    v_quad = vec2(0.0);
                    return;
                }

                vec2 major;
                if (abs(b) > 1e-6) {
                    major = normalize(vec2(b, l1 - a));
                } else {
                    major = (a >= c) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
                }
                vec2 minor = vec2(-major.y, major.x);

                float capPx = 0.35 * max(u_viewport.x, u_viewport.y);
                vec2 majorAxis = min(sqrt(2.0 * l1), capPx) * major;
                vec2 minorAxis = min(sqrt(2.0 * l2), capPx) * minor;

                vec4 clip = u_projection * camPos;
                vec2 ndc = clip.xy / clip.w;
                vec2 offset = (a_corner.x * majorAxis + a_corner.y * minorAxis)
                              / u_viewport * 2.0;

                gl_Position = vec4(ndc + offset, 0.0, 1.0);
                v_color = vec4(rgba.rgb, rgba.a * u_alphaScale);
                v_quad = a_corner;
            }
        """.trimIndent()

        private val FRAGMENT_SHADER = """
            #version 300 es
            precision mediump float;

            in vec4 v_color;
            in vec2 v_quad;
            out vec4 fragColor;

            void main() {
                float r2 = dot(v_quad, v_quad);
                if (r2 > 4.0) discard;
                float alpha = exp(-r2) * v_color.a;
                if (alpha < 0.004) discard;
                fragColor = vec4(v_color.rgb, alpha);
            }
        """.trimIndent()
    }
}
