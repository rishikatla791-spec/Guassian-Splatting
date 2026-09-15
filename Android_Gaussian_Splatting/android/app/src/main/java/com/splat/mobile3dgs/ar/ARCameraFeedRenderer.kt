package com.splat.mobile3dgs.ar

import android.opengl.GLES11Ext
import android.opengl.GLES20
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Draws the ARCore camera image as a full-screen background quad.
 *
 * The camera image arrives as an OES external texture whose id must be handed to
 * the session via `Session.setCameraTextureName` before every `Session.update`.
 * The texture is not axis-aligned with the display -- the sensor orientation,
 * display rotation and aspect-fill crop all differ -- so the texture coordinates
 * are derived from ARCore itself with `Frame.transformCoordinates2d` rather than
 * being hard coded.
 *
 * Shaders are written in GLSL ES 1.00, which an ES 3.0 context still accepts;
 * `GL_OES_EGL_image_external` has no ESSL 3.00 form on every driver.
 */
class ARCameraFeedRenderer {

    var textureId: Int = -1
        private set

    private var program = 0
    private var positionAttrib = 0
    private var texCoordAttrib = 0
    private var textureUniform = 0

    private var needsTexCoordUpdate = true

    private lateinit var quadVertices: FloatBuffer
    private lateinit var quadTexCoords: FloatBuffer

    private val ndcQuad = floatArrayOf(
        -1f, -1f,
        +1f, -1f,
        -1f, +1f,
        +1f, +1f
    )

    private val vertexShaderSource = """
        attribute vec4 a_Position;
        attribute vec2 a_TexCoord;
        varying vec2 v_TexCoord;
        void main() {
            gl_Position = a_Position;
            v_TexCoord = a_TexCoord;
        }
    """.trimIndent()

    private val fragmentShaderSource = """
        #extension GL_OES_EGL_image_external : require
        precision mediump float;
        varying vec2 v_TexCoord;
        uniform samplerExternalOES sTexture;
        void main() {
            gl_FragColor = texture2D(sTexture, v_TexCoord);
        }
    """.trimIndent()

    /** Must run on the GL thread (onSurfaceCreated). */
    fun createOnGlThread() {
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        textureId = textures[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES20.glTexParameteri(
            GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE
        )
        GLES20.glTexParameteri(
            GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE
        )
        GLES20.glTexParameteri(
            GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR
        )
        GLES20.glTexParameteri(
            GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR
        )

        quadVertices = ByteBuffer.allocateDirect(ndcQuad.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        quadVertices.put(ndcQuad).position(0)

        quadTexCoords = ByteBuffer.allocateDirect(ndcQuad.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        quadTexCoords.put(floatArrayOf(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f)).position(0)

        val vs = GlUtil.compileShader(GLES20.GL_VERTEX_SHADER, vertexShaderSource)
        val fs = GlUtil.compileShader(GLES20.GL_FRAGMENT_SHADER, fragmentShaderSource)
        program = GlUtil.linkProgram(vs, fs)
        positionAttrib = GLES20.glGetAttribLocation(program, "a_Position")
        texCoordAttrib = GLES20.glGetAttribLocation(program, "a_TexCoord")
        textureUniform = GLES20.glGetUniformLocation(program, "sTexture")
        needsTexCoordUpdate = true
    }

    /** Force a texture-coordinate refresh (after a viewport/rotation change). */
    fun invalidateGeometry() {
        needsTexCoordUpdate = true
    }

    /** Draw the camera background for [frame]. GL thread only. */
    fun draw(frame: Frame) {
        if (textureId == -1 || program == 0) return

        if (needsTexCoordUpdate || frame.hasDisplayGeometryChanged()) {
            quadVertices.position(0)
            quadTexCoords.position(0)
            frame.transformCoordinates2d(
                Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,
                quadVertices,
                Coordinates2d.TEXTURE_NORMALIZED,
                quadTexCoords
            )
            needsTexCoordUpdate = false
        }

        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glDepthMask(false)

        GLES20.glUseProgram(program)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES20.glUniform1i(textureUniform, 0)

        quadVertices.position(0)
        GLES20.glVertexAttribPointer(positionAttrib, 2, GLES20.GL_FLOAT, false, 0, quadVertices)
        GLES20.glEnableVertexAttribArray(positionAttrib)

        quadTexCoords.position(0)
        GLES20.glVertexAttribPointer(texCoordAttrib, 2, GLES20.GL_FLOAT, false, 0, quadTexCoords)
        GLES20.glEnableVertexAttribArray(texCoordAttrib)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        GLES20.glDisableVertexAttribArray(positionAttrib)
        GLES20.glDisableVertexAttribArray(texCoordAttrib)

        GLES20.glDepthMask(true)
    }

    fun release() {
        if (program != 0) {
            GLES20.glDeleteProgram(program)
            program = 0
        }
        if (textureId != -1) {
            GLES20.glDeleteTextures(1, intArrayOf(textureId), 0)
            textureId = -1
        }
    }
}
