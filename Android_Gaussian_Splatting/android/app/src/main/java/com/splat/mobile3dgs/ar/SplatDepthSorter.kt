package com.splat.mobile3dgs.ar

import android.util.Log
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Produces a back-to-front draw order for a [SplatCloud].
 *
 * Alpha compositing of gaussians is order dependent: drawn in file order the
 * model looks like a soup of the wrong colours. A full comparison sort of a few
 * hundred thousand splats every frame is far too slow on a phone, so this uses
 * an O(n) counting sort over a quantised view-space depth, run on a worker
 * thread and only re-run when the viewpoint has actually changed enough to
 * matter.
 *
 * The sorter also applies the level-of-detail stride: every `stride`-th splat is
 * emitted, which decimates uniformly across the whole model instead of lopping
 * off the far half.
 */
class SplatDepthSorter(
    private val positions: FloatArray,
    private val count: Int
) {

    /** Depth quantisation buckets. 16 bits is well past what 8-bit alpha can show. */
    private val buckets = 1 shl 16

    private val counts = IntArray(buckets)

    // Double buffered: the GL thread may still be uploading the previous
    // ordering when the next sort starts writing.
    private var bufA = IntArray(count)
    private var bufB = IntArray(count)
    private var useA = true

    private val busy = AtomicBoolean(false)
    private var worker: Thread? = null

    @Volatile private var released = false

    /** Latest finished ordering, or null when nothing new is ready. */
    @Volatile private var pending: IntArray? = null
    @Volatile private var pendingCount = 0

    // View state the pending/last ordering was computed for.
    private var lastDirX = 0f
    private var lastDirY = 0f
    private var lastDirZ = 0f
    private var lastOffset = Float.NaN
    private var lastStride = -1
    private var lastSortAtMs = 0L

    /**
     * Ask for a new ordering if the view has moved enough.
     *
     * @param mv column-major model-view matrix; only its third row is needed,
     *           since view-space depth is `-(row2 . p)`.
     * @param stride LOD decimation factor (1 = keep everything).
     * @return true if a sort was started.
     */
    fun requestSort(mv: FloatArray, stride: Int, nowMs: Long): Boolean {
        if (released || count == 0) return false
        if (busy.get()) return false

        val dx = mv[2]
        val dy = mv[6]
        val dz = mv[10]
        val offset = mv[14]

        // A re-sort is only worth its cost when the depth ordering can plausibly
        // have changed: a rotation of the model-view basis, a translation along
        // the view axis, or a change of LOD level.
        val rotated = abs(dx - lastDirX) + abs(dy - lastDirY) + abs(dz - lastDirZ) > 0.02f
        val moved = !lastOffset.isFinite() || abs(offset - lastOffset) > 0.02f
        val lodChanged = stride != lastStride
        val elapsed = nowMs - lastSortAtMs
        if (!lodChanged && !rotated && !moved) return false
        if (!lodChanged && elapsed < MIN_INTERVAL_MS) return false

        lastDirX = dx; lastDirY = dy; lastDirZ = dz
        lastOffset = offset
        lastStride = stride
        lastSortAtMs = nowMs

        if (!busy.compareAndSet(false, true)) return false
        val t = Thread({
            try {
                sortInto(dx, dy, dz, offset, max(1, stride))
            } catch (t: Throwable) {
                Log.w(TAG, "Depth sort failed: ${t.message}")
            } finally {
                busy.set(false)
            }
        }, "splat-depth-sort")
        t.priority = Thread.NORM_PRIORITY - 1
        worker = t
        t.start()
        return true
    }

    /**
     * Hand the newest finished ordering to the caller, or null if none is ready.
     * The returned array is owned by the caller until the next call.
     */
    fun takeOrder(): Pair<IntArray, Int>? {
        val order = pending ?: return null
        pending = null
        return order to pendingCount
    }

    fun release() {
        released = true
        worker?.let { if (it.isAlive) it.interrupt() }
        worker = null
        pending = null
    }

    private fun sortInto(dx: Float, dy: Float, dz: Float, offset: Float, stride: Int) {
        val n = count
        val kept = (n + stride - 1) / stride

        val out = if (useA) {
            if (bufA.size < kept) bufA = IntArray(kept)
            bufA
        } else {
            if (bufB.size < kept) bufB = IntArray(kept)
            bufB
        }
        useA = !useA

        // Pass 1: depth range over the retained set.
        var minD = Float.MAX_VALUE
        var maxD = -Float.MAX_VALUE
        var i = 0
        while (i < n) {
            val b = i * 3
            val d = -(dx * positions[b] + dy * positions[b + 1] + dz * positions[b + 2] + offset)
            if (d.isFinite()) {
                if (d < minD) minD = d
                if (d > maxD) maxD = d
            }
            i += stride
        }
        if (minD > maxD) return
        val span = max(maxD - minD, 1e-5f)
        val scale = (buckets - 1) / span

        // Pass 2: histogram of the *reversed* bucket, so ascending bucket order
        // means descending depth -- i.e. far splats first.
        java.util.Arrays.fill(counts, 0)
        i = 0
        while (i < n) {
            val b = i * 3
            val d = -(dx * positions[b] + dy * positions[b + 1] + dz * positions[b + 2] + offset)
            val key = if (d.isFinite()) {
                buckets - 1 - min(buckets - 1, max(0, ((d - minD) * scale).toInt()))
            } else {
                0
            }
            counts[key]++
            i += stride
        }

        // Prefix sum.
        var running = 0
        var k = 0
        while (k < buckets) {
            val c = counts[k]
            counts[k] = running
            running += c
            k++
        }

        // Pass 3: scatter.
        var written = 0
        i = 0
        while (i < n) {
            val b = i * 3
            val d = -(dx * positions[b] + dy * positions[b + 1] + dz * positions[b + 2] + offset)
            val key = if (d.isFinite()) {
                buckets - 1 - min(buckets - 1, max(0, ((d - minD) * scale).toInt()))
            } else {
                0
            }
            val slot = counts[key]
            counts[key] = slot + 1
            if (slot < out.size) out[slot] = i
            written++
            i += stride
        }

        if (released) return
        pendingCount = min(written, out.size)
        pending = out
    }

    companion object {
        private const val TAG = "SplatDepthSorter"
        /** Floor on re-sort frequency; sorting harder than this buys nothing visible. */
        private const val MIN_INTERVAL_MS = 110L
    }
}
