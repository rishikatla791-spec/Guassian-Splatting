package com.splat.mobile3dgs.capture

import android.opengl.GLES11Ext
import android.opengl.GLES20
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Draws the ARCore camera feed as a full-screen background quad using an
 * OES external texture. Also owns the GL texture id that must be handed to
 * the ARCore session via [com.google.ar.core.Session.setCameraTextureName]
 * before every [com.google.ar.core.Session.update] call.
 */
class ARBackgroundRenderer {

    var textureId: Int = -1
        private set

    private var program = 0
    private var positionAttrib = 0
    private var texCoordAttrib = 0

    private lateinit var quadVertices: FloatBuffer
    private lateinit var quadTexCoords: FloatBuffer

    // Full-screen quad in normalized device coordinates (triangle strip).
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

    /** Must be called on the GL thread (onSurfaceCreated). */
    fun createOnGlThread() {
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        textureId = textures[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)

        quadVertices = ByteBuffer.allocateDirect(ndcQuad.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        quadVertices.put(ndcQuad).position(0)

        quadTexCoords = ByteBuffer.allocateDirect(ndcQuad.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        // Placeholder; recomputed from ARCore each frame if display geometry changes.
        quadTexCoords.put(floatArrayOf(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f)).position(0)

        val vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexShaderSource)
        val fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentShaderSource)
        program = GLES20.glCreateProgram()
        GLES20.glAttachShader(program, vertexShader)
        GLES20.glAttachShader(program, fragmentShader)
        GLES20.glLinkProgram(program)
        positionAttrib = GLES20.glGetAttribLocation(program, "a_Position")
        texCoordAttrib = GLES20.glGetAttribLocation(program, "a_TexCoord")
    }

    /** Draw the camera background for the given frame. Call on GL thread. */
    fun draw(frame: Frame) {
        if (textureId == -1 || program == 0) return

        if (frame.hasDisplayGeometryChanged()) {
            frame.transformCoordinates2d(
                Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,
                quadVertices,
                Coordinates2d.TEXTURE_NORMALIZED,
                quadTexCoords
            )
        }

        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glDepthMask(false)

        GLES20.glUseProgram(program)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)

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
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
    }

    private fun loadShader(type: Int, source: String): Int {
        val shader = GLES20.glCreateShader(type)
        GLES20.glShaderSource(shader, source)
        GLES20.glCompileShader(shader)
        val compiled = IntArray(1)
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0)
        if (compiled[0] == 0) {
            val log = GLES20.glGetShaderInfoLog(shader)
            GLES20.glDeleteShader(shader)
            throw RuntimeException("Shader compile failed: $log")
        }
        return shader
    }
}
