package com.splat.mobile3dgs.ar

import android.util.Log
import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Width (in texels) of the RGBA32UI texture the splat payload is uploaded into.
 * Two texels (32 bytes) hold one splat, and the width is even so a splat never
 * straddles a row -- the vertex shader can therefore fetch texel `2*i` and
 * `2*i+1` with a single row division.
 */
const val SPLAT_TEXTURE_WIDTH = 2048

/** Bytes on disk (and in the GPU texture) for a single splat. */
const val SPLAT_STRIDE_BYTES = 32

/**
 * A `.splat` model loaded into a GPU-ready buffer.
 *
 * The on-disk record layout is used verbatim as the texture payload, so loading
 * is a straight copy with no per-splat repacking:
 *
 * ```
 *  0..11   position   3 x float32 (little endian)
 * 12..23   scale      3 x float32
 * 24..27   colour     4 x uint8  (r, g, b, a)
 * 28..31   rotation   4 x uint8  packed as b = r * 128 + 128, in (w, x, y, z)
 *                     order -- i.e. the SCALAR comes first
 * ```
 *
 * [payload] is padded with zeroed splats up to a whole number of texture rows;
 * padding splats have zero scale and zero alpha and are never indexed anyway.
 */
class SplatCloud private constructor(
    /** Number of splats actually retained (after load-time decimation). */
    val count: Int,
    /** Number of splats in the source file before decimation. */
    val sourceCount: Int,
    /** Raw 32-byte records, row-padded, direct + native order. */
    val payload: ByteBuffer,
    /** Interleaved xyz per retained splat, kept on the CPU for depth sorting. */
    val positions: FloatArray,
    /** Robust centre of the model in its own coordinate frame. */
    val centerX: Float,
    val centerY: Float,
    val centerZ: Float,
    /** Robust half-extent on the model's Y axis (used to sit it on the plane). */
    val halfHeight: Float,
    /** Robust largest dimension of the model, in model units. */
    val extent: Float
) {

    /** Rows of [SPLAT_TEXTURE_WIDTH] texels needed to hold [payload]. */
    val textureRows: Int
        get() {
            val texels = count * 2
            return (texels + SPLAT_TEXTURE_WIDTH - 1) / SPLAT_TEXTURE_WIDTH
        }

    companion object {
        private const val TAG = "SplatCloud"

        /**
         * Load [file], keeping at most [budget] splats.
         *
         * Decimation is a uniform stride over the file rather than a truncation:
         * `.splat` files are usually written in densification order, so taking the
         * first N would keep only one region of the model. Striding keeps the
         * whole shape, just sparser.
         *
         * Throws [java.io.IOException] / [IllegalArgumentException] on a missing,
         * empty or malformed file -- the caller is expected to report and exit.
         */
        fun load(file: File, budget: Int): SplatCloud {
            require(file.exists()) { "Model file does not exist" }
            val length = file.length()
            require(length >= SPLAT_STRIDE_BYTES) { "Model file is empty" }
            require(length % SPLAT_STRIDE_BYTES == 0L) {
                "Not a .splat file (size $length is not a multiple of $SPLAT_STRIDE_BYTES)"
            }

            val sourceCount = (length / SPLAT_STRIDE_BYTES).toInt()
            val safeBudget = max(1, budget)
            val stride = max(1, (sourceCount + safeBudget - 1) / safeBudget)
            val kept = (sourceCount + stride - 1) / stride

            // Pad up to whole texture rows so the payload can be handed to
            // glTexImage2D in one call.
            val texels = kept * 2
            val paddedTexels = ((texels + SPLAT_TEXTURE_WIDTH - 1) / SPLAT_TEXTURE_WIDTH) *
                SPLAT_TEXTURE_WIDTH
            val payload = ByteBuffer.allocateDirect(paddedTexels * 16).order(ByteOrder.nativeOrder())

            val positions = FloatArray(kept * 3)
            val record = ByteArray(SPLAT_STRIDE_BYTES)
            val recordView = ByteBuffer.wrap(record).order(ByteOrder.LITTLE_ENDIAN)
            var written = 0

            BufferedInputStream(FileInputStream(file), 1 shl 16).use { input ->
                while (written < kept) {
                    if (!readFully(input, record)) break
                    payload.put(record)
                    positions[written * 3] = recordView.getFloat(0)
                    positions[written * 3 + 1] = recordView.getFloat(4)
                    positions[written * 3 + 2] = recordView.getFloat(8)
                    written++
                    if (stride > 1 && written < kept) {
                        if (!skipFully(input, (stride - 1).toLong() * SPLAT_STRIDE_BYTES)) break
                    }
                }
            }

            require(written > 0) { "Model file contained no readable splats" }
            payload.position(0)
            payload.limit(payload.capacity())

            val bounds = robustBounds(positions, written)
            val cx = (bounds[0] + bounds[3]) * 0.5f
            val cy = (bounds[1] + bounds[4]) * 0.5f
            val cz = (bounds[2] + bounds[5]) * 0.5f
            val ex = bounds[3] - bounds[0]
            val ey = bounds[4] - bounds[1]
            val ez = bounds[5] - bounds[2]
            val extent = max(ex, max(ey, ez)).coerceAtLeast(1e-4f)

            Log.i(
                TAG,
                "Loaded ${written}/$sourceCount splats (stride $stride) " +
                    "extent=%.3f centre=(%.3f, %.3f, %.3f)".format(extent, cx, cy, cz)
            )

            return SplatCloud(
                count = written,
                sourceCount = sourceCount,
                payload = payload,
                positions = positions,
                centerX = cx,
                centerY = cy,
                centerZ = cz,
                halfHeight = (ey * 0.5f).coerceAtLeast(1e-4f),
                extent = extent
            )
        }

        /**
         * 2nd/98th percentile AABB, returned as [minX, minY, minZ, maxX, maxY, maxZ].
         *
         * Trained splat clouds nearly always carry a handful of "floater" gaussians
         * far outside the real object; a raw min/max would blow the extent up by
         * orders of magnitude and place the model kilometres above the floor.
         * Percentiles are taken over a bounded subsample so this stays O(1) in
         * model size.
         */
        private fun robustBounds(positions: FloatArray, count: Int): FloatArray {
            val sampleTarget = min(count, 20000)
            val step = max(1, count / sampleTarget)
            val n = (count + step - 1) / step
            val xs = FloatArray(n)
            val ys = FloatArray(n)
            val zs = FloatArray(n)
            var j = 0
            var i = 0
            while (i < count && j < n) {
                val x = positions[i * 3]
                val y = positions[i * 3 + 1]
                val z = positions[i * 3 + 2]
                // Drop non-finite values outright; they would poison the sort keys.
                if (x.isFinite() && y.isFinite() && z.isFinite()) {
                    xs[j] = x; ys[j] = y; zs[j] = z
                    j++
                }
                i += step
            }
            if (j == 0) return floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f)
            val xa = xs.copyOf(j); val ya = ys.copyOf(j); val za = zs.copyOf(j)
            xa.sort(); ya.sort(); za.sort()
            val lo = (j * 2) / 100
            val hi = (j - 1 - lo).coerceAtLeast(lo)
            return floatArrayOf(xa[lo], ya[lo], za[lo], xa[hi], ya[hi], za[hi])
        }

        private fun readFully(input: java.io.InputStream, into: ByteArray): Boolean {
            var read = 0
            while (read < into.size) {
                val n = input.read(into, read, into.size - read)
                if (n < 0) return false
                read += n
            }
            return true
        }

        private fun skipFully(input: java.io.InputStream, bytes: Long): Boolean {
            var remaining = bytes
            while (remaining > 0) {
                val skipped = input.skip(remaining)
                if (skipped <= 0) {
                    // skip() may legitimately return 0; fall back to a read.
                    if (input.read() < 0) return false
                    remaining--
                } else {
                    remaining -= skipped
                }
            }
            return true
        }
    }

    /** True when the model's size looks nothing like a real-world metric object. */
    fun looksNonMetric(): Boolean = extent > 25f || extent < 0.02f || !extent.isFinite()

    /** Scale that brings a non-metric model to a sane on-table size. */
    fun autoFitScale(targetMeters: Float = 0.45f): Float {
        if (!extent.isFinite() || abs(extent) < 1e-6f) return 1f
        return targetMeters / extent
    }
}
