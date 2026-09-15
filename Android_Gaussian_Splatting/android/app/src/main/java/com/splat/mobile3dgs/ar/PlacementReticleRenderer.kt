package com.splat.mobile3dgs.ar

import android.opengl.GLES20
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.cos
import kotlin.math.sin

/**
 * A soft white ring drawn flat on the plane under the crosshair, so the user can
 * see where a tap will land before committing. Also doubles as the "model is
 * here" marker once something is placed.
 *
 * Geometry is a unit-radius annulus in the local XZ plane; the caller supplies a
 * model matrix built from the ARCore hit pose, so the ring lies exactly on the
 * detected surface.
 */
class PlacementReticleRenderer {

    private var program = 0
    private var uMvp = 0
    private var uColor = 0
    private var aPosition = 0
    private var vertexBuffer: FloatBuffer? = null
    private var vertexCount = 0

    private val mvp = FloatArray(16)

    fun createOnGlThread() {
        val vs = GlUtil.compileShader(GLES20.GL_VERTEX_SHADER, VERTEX_SHADER)
        val fs = GlUtil.compileShader(GLES20.GL_FRAGMENT_SHADER, FRAGMENT_SHADER)
        program = GlUtil.linkProgram(vs, fs)
        aPosition = GLES20.glGetAttribLocation(program, "a_position")
        uMvp = GLES20.glGetUniformLocation(program, "u_mvp")
        uColor = GLES20.glGetUniformLocation(program, "u_color")

        val segments = 64
        val inner = 0.86f
        val outer = 1.0f
        val verts = FloatArray((segments + 1) * 2 * 3)
        var i = 0
        for (s in 0..segments) {
            val a = (s.toDouble() / segments) * Math.PI * 2.0
            val cx = cos(a).toFloat()
            val cz = sin(a).toFloat()
            verts[i++] = cx * inner; verts[i++] = 0f; verts[i++] = cz * inner
            verts[i++] = cx * outer; verts[i++] = 0f; verts[i++] = cz * outer
        }
        vertexCount = (segments + 1) * 2
        vertexBuffer = ByteBuffer.allocateDirect(verts.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
            .also { it.put(verts).position(0) }
    }

    /**
     * @param model column-major model matrix (pose * radius scale).
     * @param viewProjection column-major projection * view.
     */
    fun draw(
        model: FloatArray,
        viewProjection: FloatArray,
        r: Float, g: Float, b: Float, alpha: Float
    ) {
        val buffer = vertexBuffer ?: return
        if (program == 0) return
        android.opengl.Matrix.multiplyMM(mvp, 0, viewProjection, 0, model, 0)

        GLES20.glUseProgram(program)
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glDepthMask(false)
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)

        GLES20.glUniformMatrix4fv(uMvp, 1, false, mvp, 0)
        GLES20.glUniform4f(uColor, r, g, b, alpha)

        buffer.position(0)
        GLES20.glVertexAttribPointer(aPosition, 3, GLES20.GL_FLOAT, false, 0, buffer)
        GLES20.glEnableVertexAttribArray(aPosition)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, vertexCount)
        GLES20.glDisableVertexAttribArray(aPosition)

        GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glDepthMask(true)
    }

    fun release() {
        if (program != 0) {
            GLES20.glDeleteProgram(program)
            program = 0
        }
        vertexBuffer = null
    }

    companion object {
        private val VERTEX_SHADER = """
            attribute vec4 a_position;
            uniform mat4 u_mvp;
            void main() {
                gl_Position = u_mvp * a_position;
            }
        """.trimIndent()

        private val FRAGMENT_SHADER = """
            precision mediump float;
            uniform vec4 u_color;
            void main() {
                gl_FragColor = u_color;
            }
        """.trimIndent()
    }
}
