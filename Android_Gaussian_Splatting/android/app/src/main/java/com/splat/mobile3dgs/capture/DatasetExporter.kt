package com.splat.mobile3dgs.capture

import android.content.Context
import android.graphics.Bitmap
import com.google.ar.core.Anchor
import com.google.ar.core.PointCloud
import com.google.ar.core.Pose
import com.google.ar.core.TrackingState
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer

/**
 * Binds a keyframe to a nearby ARCore [Anchor] plus the pose it had RELATIVE to
 * that anchor at capture time.
 *
 * A raw VIO pose is frozen the instant it is read, but ARCore keeps refining its
 * map afterwards. Storing the relative pose lets the absolute one be recomposed
 * at export time against the anchor's corrected pose, so the dataset inherits
 * every correction ARCore made after the shot was taken.
 */
class AnchorPoseRef(
    val anchor: Anchor,
    val relativePose: Pose
)

data class FrameMetaData(
    val frameId: Int,
    val filePath: String,
    val timestampNs: Long,
    /**
     * 4x4 camera-to-world matrix. Rewritten in place by
     * [DatasetExporter.applyAnchorCorrections] with the drift-corrected pose,
     * hence `var`.
     */
    var transformMatrix: Array<FloatArray>,
    val sharpnessScore: Float,
    val meanLuminance: Float,
    /** Anchor binding, or null when anchors were unavailable for this frame. */
    val anchorRef: AnchorPoseRef? = null,
    /** The raw ARCore pose as recorded, needed to compute the correction delta. */
    val capturePose: Pose? = null
)

data class FeaturePoint3D(
    val id: Int,
    val x: Float,
    val y: Float,
    val z: Float,
    val confidence: Float
)

class DatasetExporter(context: Context, sessionName: String = "3dgs_arcore_${System.currentTimeMillis() / 1000}") {

    val outputDir: File = File(context.getExternalFilesDir(null), sessionName).apply { mkdirs() }
    private val imagesDir: File = File(outputDir, "images").apply { mkdirs() }
    val capturedFrames = mutableListOf<FrameMetaData>()
    val accumulatedFeaturePoints = mutableListOf<FeaturePoint3D>()

    // Dense geometry unprojected from the ARCore depth map. Deduplicated into a
    // voxel grid as points arrive so memory stays bounded by scene volume rather
    // than by (frames x depth pixels).
    private val depthVoxels = HashMap<Long, FloatArray>()
    // Grows coarser rather than truncating -- see addDepthPoints.
    private var depthVoxelSize = 0.01f
    private val maxDepthPoints = 900_000

    /**
     * How usable this capture actually is for 3D reconstruction.
     *
     * 3DGS recovers depth from parallax, so what matters is not how many frames
     * were taken but how far the camera *moved* relative to how far away the
     * scene is. Panning on the spot produces a triangulation angle near zero,
     * where depth is mathematically unrecoverable and extra iterations only
     * overfit -- the optimizer densifies into billboard soup that looks right
     * from the capture position and wrong from anywhere else.
     */
    data class CaptureQuality(
        val frames: Int,
        val baselineM: Float,
        val medianDepthM: Float,
        val parallaxRatio: Float,
        val triangulationDeg: Float,
        val rotationOnlyPct: Int,
        /** False when no geometry was available, so the ratio means nothing. */
        val measured: Boolean = true
    ) {
        // Only claim a capture is degenerate when scene depth was actually
        // measurable; otherwise an empty seed cloud reads as "0 parallax" and
        // produces a false warning on a perfectly good capture.
        val isDegenerate: Boolean get() = measured && parallaxRatio < 0.15f
    }

    @Synchronized
    fun computeQuality(): CaptureQuality {
        val cams = capturedFrames.map {
            floatArrayOf(it.transformMatrix[0][3], it.transformMatrix[1][3], it.transformMatrix[2][3])
        }
        if (cams.size < 2) return CaptureQuality(cams.size, 0f, 0f, 0f, 0f, 0)

        // Widest separation between any two camera positions.
        var baseline = 0f
        for (i in cams.indices) for (j in i + 1 until cams.size) {
            val dx = cams[i][0] - cams[j][0]
            val dy = cams[i][1] - cams[j][1]
            val dz = cams[i][2] - cams[j][2]
            val dist = kotlin.math.sqrt(dx * dx + dy * dy + dz * dz)
            if (dist > baseline) baseline = dist
        }

        // Median distance from the seed geometry to the nearest camera. Depth points
        // are preferred, but fall back to ARCore feature points so the check still
        // works when the depth map yields nothing.
        val sample = if (depthVoxels.isNotEmpty()) depthVoxels.values.toList()
        else accumulatedFeaturePoints.map { floatArrayOf(it.x, it.y, it.z) }
        val depths = ArrayList<Float>(minOf(sample.size, 2000))
        val stride = maxOf(1, sample.size / 2000)
        var i = 0
        while (i < sample.size) {
            val v = sample[i]
            var best = Float.MAX_VALUE
            for (c in cams) {
                val dx = v[0] - c[0]; val dy = v[1] - c[1]; val dz = v[2] - c[2]
                val dd = dx * dx + dy * dy + dz * dz
                if (dd < best) best = dd
            }
            if (best < Float.MAX_VALUE) depths.add(kotlin.math.sqrt(best))
            i += stride
        }
        depths.sort()
        val medianDepth = if (depths.isEmpty()) 0f else depths[depths.size / 2]

        val ratio = if (medianDepth > 0.01f) baseline / medianDepth else 0f
        val triDeg = (2.0 * kotlin.math.atan((ratio / 2.0)) * 180.0 / Math.PI).toFloat()

        // Fraction of keyframes that added rotation but essentially no translation.
        var rotOnly = 0
        for (k in 0 until cams.size - 1) {
            val dx = cams[k][0] - cams[k + 1][0]
            val dy = cams[k][1] - cams[k + 1][1]
            val dz = cams[k][2] - cams[k + 1][2]
            if (kotlin.math.sqrt(dx * dx + dy * dy + dz * dz) < 0.02f) rotOnly++
        }
        val rotPct = if (cams.size > 1) (100 * rotOnly / (cams.size - 1)) else 0

        return CaptureQuality(cams.size, baseline, medianDepth, ratio, triDeg, rotPct, medianDepth > 0.01f)
    }

    /** @param pts flat [x, y, z, confidence, r, g, b, logScale, ...] in world space. */
    @Synchronized
    fun addDepthPoints(pts: FloatArray, frameIndex: Int = -1) {
        var i = 0
        while (i + 7 < pts.size) {
            // Previously this returned outright once the budget filled, so every
            // later frame contributed NOTHING and the seed cloud described only
            // whatever the scan happened to start on -- measured: the cap was hit
            // exactly, on a 101-frame scan. Coarsen the grid instead, which keeps
            // coverage global and bounded rather than first-come-first-served.
            if (depthVoxels.size >= maxDepthPoints) coarsenVoxelGrid()
            val x = pts[i]; val y = pts[i + 1]; val z = pts[i + 2]; val c = pts[i + 3]
            val vx = kotlin.math.floor(x / depthVoxelSize).toInt()
            val vy = kotlin.math.floor(y / depthVoxelSize).toInt()
            val vz = kotlin.math.floor(z / depthVoxelSize).toInt()
            val key = (vx.toLong() and 0x1FFFFF) or
                    ((vy.toLong() and 0x1FFFFF) shl 21) or
                    ((vz.toLong() and 0x1FFFFF) shl 42)
            val existing = depthVoxels[key]
            if (existing == null || c > existing[3]) {
                // [8] is the source keyframe, so anchor drift corrections can move
                // each point by exactly the delta its own camera received.
                depthVoxels[key] = floatArrayOf(
                    x, y, z, c, pts[i + 4], pts[i + 5], pts[i + 6], pts[i + 7],
                    frameIndex.toFloat()
                )
            }
            i += 8
        }
    }

    /**
     * Halve the resolution of the seed grid, merging occupied voxels.
     *
     * Surfaces are effectively 2D, so doubling the voxel edge drops the count by
     * roughly 4x -- enough headroom to keep ingesting the rest of the scan. The
     * cloud stays a uniform sample of everything seen, just slightly coarser,
     * which is far better for reconstruction than a dense sample of the first
     * few frames and nothing after.
     */
    private fun coarsenVoxelGrid() {
        val before = depthVoxels.size
        depthVoxelSize *= 2f
        val merged = HashMap<Long, FloatArray>(before / 2)
        for (v in depthVoxels.values) {
            val vx = kotlin.math.floor(v[0] / depthVoxelSize).toInt()
            val vy = kotlin.math.floor(v[1] / depthVoxelSize).toInt()
            val vz = kotlin.math.floor(v[2] / depthVoxelSize).toInt()
            val key = (vx.toLong() and 0x1FFFFF) or
                    ((vy.toLong() and 0x1FFFFF) shl 21) or
                    ((vz.toLong() and 0x1FFFFF) shl 42)
            val existing = merged[key]
            if (existing == null || v[3] > existing[3]) merged[key] = v
        }
        depthVoxels.clear()
        depthVoxels.putAll(merged)
        android.util.Log.i(
            "DatasetExporter",
            "Seed grid coarsened to ${"%.0f".format(depthVoxelSize * 1000f)}mm: " +
                "$before -> ${depthVoxels.size} points (still ingesting all frames)"
        )
    }

    @Synchronized
    fun depthPointCount(): Int = depthVoxels.size

    /**
     * The dense depth seed cloud as feature points, for the photometric fallback.
     *
     * The fallback was being fed only [accumulatedFeaturePoints] -- a few hundred
     * sparse ARCore features -- while tens of thousands of dense, coloured depth
     * points sat unused in this same object. That is why a failed training run
     * showed the user ~150 scattered dots instead of a recognisable preview.
     */
    @Synchronized
    fun depthPointsAsFeatures(): List<FeaturePoint3D> {
        var id = accumulatedFeaturePoints.size
        return depthVoxels.values.map { v -> FeaturePoint3D(id++, v[0], v[1], v[2], v[3]) }
    }

    /** Guards against a second correction pass re-applying deltas. */
    private var anchorCorrectionsApplied = false

    /**
     * Replace every anchor-backed keyframe pose with `anchor.pose * relativePose`,
     * reading each anchor's CURRENT (ARCore-corrected) pose.
     *
     * MUST run on the ARCore/GL thread while the session is still resumed, and
     * before the dataset is written or the photometric model is built -- both
     * read [FrameMetaData.transformMatrix].
     *
     * Every failure path leaves the original absolute pose in place, so the worst
     * case is exactly the previous behaviour.
     *
     * @return number of frames whose pose was corrected.
     */
    @Synchronized
    fun applyAnchorCorrections(): Int {
        if (anchorCorrectionsApplied) return 0
        anchorCorrectionsApplied = true

        var corrected = 0
        var skipped = 0
        var maxShiftM = 0f
        val scratch = FloatArray(16)
        // Per-frame 4x4 "corrected world <- recorded world" delta, column-major.
        val deltas = arrayOfNulls<FloatArray>(capturedFrames.size)

        for ((i, frame) in capturedFrames.withIndex()) {
            val ref = frame.anchorRef ?: continue
            try {
                if (ref.anchor.trackingState != TrackingState.TRACKING) { skipped++; continue }
                val world = ref.anchor.pose.compose(ref.relativePose)
                world.toMatrix(scratch, 0)
                if (!scratch.all { it.isFinite() }) { skipped++; continue }

                val recorded = frame.capturePose
                if (recorded != null) {
                    val dx = world.tx() - recorded.tx()
                    val dy = world.ty() - recorded.ty()
                    val dz = world.tz() - recorded.tz()
                    val shift = kotlin.math.sqrt(dx * dx + dy * dy + dz * dz)
                    if (shift.isFinite() && shift > maxShiftM) maxShiftM = shift

                    val d = FloatArray(16)
                    world.compose(recorded.inverse()).toMatrix(d, 0)
                    if (d.all { it.isFinite() }) deltas[i] = d
                }

                frame.transformMatrix = ARCoreCoordinateUtils.arcoreMatrixToC2W(scratch)
                corrected++
            } catch (t: Throwable) {
                skipped++
            }
        }

        if (corrected > 0) migrateSeedCloud(deltas)

        android.util.Log.i(
            "DatasetExporter",
            "Anchor-relative poses: $corrected/${capturedFrames.size} frames re-composed from " +
                "corrected anchors ($skipped kept absolute), max drift correction " +
                "${"%.3f".format(maxShiftM)} m"
        )
        return corrected
    }

    /**
     * Move each seed point by the same correction its source camera received.
     *
     * Correcting the cameras but not the points they were unprojected from would
     * leave the two in different frames, so the optimizer would start from
     * geometry that no longer lines up with any view -- worse than not
     * correcting at all.
     */
    private fun migrateSeedCloud(deltas: Array<FloatArray?>) {
        if (depthVoxels.isEmpty()) return
        var moved = 0
        for (v in depthVoxels.values) {
            if (v.size < 9) continue
            val fi = v[8].toInt()
            if (fi < 0 || fi >= deltas.size) continue
            val d = deltas[fi] ?: continue
            val x = v[0]; val y = v[1]; val z = v[2]
            // Column-major 4x4 applied to a point.
            val nx = d[0] * x + d[4] * y + d[8] * z + d[12]
            val ny = d[1] * x + d[5] * y + d[9] * z + d[13]
            val nz = d[2] * x + d[6] * y + d[10] * z + d[14]
            if (nx.isFinite() && ny.isFinite() && nz.isFinite()) {
                v[0] = nx; v[1] = ny; v[2] = nz
                moved++
            }
        }
        android.util.Log.i("DatasetExporter", "Seed cloud migrated with its cameras: $moved points")
    }

    /**
     * Save a frame that has already been JPEG-encoded (e.g. from an ARCore CPU image).
     * [poseMatrix] is the 16-element column-major camera-to-world matrix from
     * `pose.toMatrix()`. [pointsXyzc] is an optional flat array of ARCore point-cloud
     * entries as [x, y, z, confidence, ...] copied out before the frame was released.
     */
    fun saveCapturedFrameJpeg(
        jpegBytes: ByteArray,
        poseMatrix: FloatArray,
        timestampNs: Long,
        sharpnessScore: Float,
        meanLuminance: Float,
        pointsXyzc: FloatArray? = null,
        anchorRef: AnchorPoseRef? = null,
        capturePose: Pose? = null
    ): String {
        val frameId = capturedFrames.size
        val fileName = String.format("frame_%04d.jpg", frameId)
        val imageFile = File(imagesDir, fileName)
        FileOutputStream(imageFile).use { out -> out.write(jpegBytes) }

        val c2wMatrix = ARCoreCoordinateUtils.arcoreMatrixToC2W(poseMatrix)
        capturedFrames.add(
            FrameMetaData(
                frameId = frameId,
                filePath = "images/$fileName",
                timestampNs = timestampNs,
                transformMatrix = c2wMatrix,
                sharpnessScore = sharpnessScore,
                meanLuminance = meanLuminance,
                anchorRef = anchorRef,
                capturePose = capturePose
            )
        )

        if (pointsXyzc != null) {
            val numPoints = pointsXyzc.size / 4
            var ptId = accumulatedFeaturePoints.size
            for (i in 0 until numPoints) {
                val confidence = pointsXyzc[i * 4 + 3]
                if (confidence > 0.1f) {
                    accumulatedFeaturePoints.add(
                        FeaturePoint3D(
                            ptId++,
                            pointsXyzc[i * 4 + 0],
                            pointsXyzc[i * 4 + 1],
                            pointsXyzc[i * 4 + 2],
                            confidence
                        )
                    )
                }
            }
        }
        return imageFile.absolutePath
    }

    fun saveCapturedFrame(
        bitmap: Bitmap,
        poseMatrix: FloatArray,
        timestampNs: Long,
        sharpnessScore: Float,
        meanLuminance: Float,
        pointCloud: PointCloud? = null
    ): String {
        val frameId = capturedFrames.size
        val fileName = String.format("frame_%04d.jpg", frameId)
        val imageFile = File(imagesDir, fileName)

        FileOutputStream(imageFile).use { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
        }

        // Convert ARCore 16-element column-major matrix to 4x4 row-major array
        val c2wMatrix = ARCoreCoordinateUtils.arcoreMatrixToC2W(poseMatrix)

        val metadata = FrameMetaData(
            frameId = frameId,
            filePath = "images/$fileName",
            timestampNs = timestampNs,
            transformMatrix = c2wMatrix,
            sharpnessScore = sharpnessScore,
            meanLuminance = meanLuminance
        )
        capturedFrames.add(metadata)

        // Accumulate 3D Feature Points from ARCore raw point cloud
        if (pointCloud != null) {
            val pointBuffer: FloatBuffer = pointCloud.points
            val numPoints = pointBuffer.remaining() / 4
            var ptId = accumulatedFeaturePoints.size
            for (i in 0 until numPoints) {
                val px = pointBuffer.get(i * 4 + 0)
                val py = pointBuffer.get(i * 4 + 1)
                val pz = pointBuffer.get(i * 4 + 2)
                val confidence = pointBuffer.get(i * 4 + 3)
                if (confidence > 0.1f) {
                    accumulatedFeaturePoints.add(FeaturePoint3D(ptId++, px, py, pz, confidence))
                }
            }
        }

        return imageFile.absolutePath
    }

    /**
     * Export complete NeRF / 3DGS compliant dataset files:
     * 1. transforms.json
     * 2. points3D_initial.json
     */
    /**
     * @param distortion optional OpenCV coefficients [k1, k2, p1, p2]. Declaring
     *   `camera_model: "OPENCV"` while supplying none makes brush build a
     *   RadialTangential8 model with all-zero terms -- i.e. a perfect pinhole --
     *   so real barrel distortion is left baked into every image and shows up as
     *   a radially growing error towards the frame border.
     */
    fun exportDataset(
        fx: Float, fy: Float, cx: Float, cy: Float, width: Int, height: Int,
        distortion: FloatArray? = null
    ): File {
        val rootJson = JSONObject()
        val fovX = 2.0 * Math.atan((width / (2.0 * fx)).toDouble())
        val fovY = 2.0 * Math.atan((height / (2.0 * fy)).toDouble())

        rootJson.put("camera_angle_x", fovX)
        rootJson.put("camera_angle_y", fovY)
        rootJson.put("fl_x", fx)
        rootJson.put("fl_y", fy)
        rootJson.put("cx", cx)
        rootJson.put("cy", cy)
        rootJson.put("w", width)
        rootJson.put("h", height)
        rootJson.put("camera_model", "OPENCV")
        if (distortion != null && distortion.size >= 4 && distortion.all { it.isFinite() }) {
            rootJson.put("k1", distortion[0].toDouble())
            rootJson.put("k2", distortion[1].toDouble())
            rootJson.put("p1", distortion[2].toDouble())
            rootJson.put("p2", distortion[3].toDouble())
            android.util.Log.i("DatasetExporter", "Lens distortion: k1=${distortion[0]} k2=${distortion[1]} " +
                    "p1=${distortion[2]} p2=${distortion[3]}")
        } else {
            android.util.Log.i("DatasetExporter", "No lens distortion reported by the device; training as a pinhole camera")
        }
        rootJson.put("system_source", "ARCore_6DoF_SLAM")

        val framesArray = JSONArray()
        for (frame in capturedFrames) {
            val frameObj = JSONObject()
            frameObj.put("file_path", frame.filePath)
            frameObj.put("timestamp_ns", frame.timestampNs)
            frameObj.put("sharpness_score", frame.sharpnessScore)
            frameObj.put("mean_luminance", frame.meanLuminance)

            val matrixJson = JSONArray()
            for (row in frame.transformMatrix) {
                val rowJson = JSONArray()
                for (valItem in row) {
                    rowJson.put(valItem.toDouble())
                }
                matrixJson.put(rowJson)
            }
            frameObj.put("transform_matrix", matrixJson)
            framesArray.put(frameObj)
        }
        rootJson.put("frames", framesArray)

        // Seed the optimizer with real ARCore surface geometry. Without a
        // ply_file_path the engine initialises from random points inside the
        // camera frustums, which produces a formless blob on short runs.
        val plyFile = File(outputDir, "points3d.ply")
        val seededPoints = writeSeedCloud(plyFile)
        if (seededPoints > 0) {
            rootJson.put("ply_file_path", plyFile.name)
            android.util.Log.i(
                "DatasetExporter",
                "Seeded initial geometry: $seededPoints points -> ${plyFile.name}"
            )
        } else {
            android.util.Log.w(
                "DatasetExporter",
                "No ARCore feature points captured; engine will fall back to random init"
            )
        }

        val jsonFile = File(outputDir, "transforms.json")
        jsonFile.writeText(rootJson.toString(2))

        exportInitialPointCloud()

        return jsonFile
    }

    /**
     * Least-squares point where the camera view rays converge -- i.e. what the user
     * was orbiting. Solves A x = b with A = sum(I - d*d^T), b = sum((I - d*d^T) p).
     * Returns null when the rays are too parallel to define a subject (e.g. the user
     * walked down a corridor rather than around something).
     */
    private fun subjectCenter(cams: List<FloatArray>, fwds: List<FloatArray>): FloatArray? {
        if (cams.size < 8) return null
        val a = FloatArray(9)
        val b = FloatArray(3)
        for (i in cams.indices) {
            val d = fwds[i]; val p = cams[i]
            // M = I - d d^T
            val m = floatArrayOf(
                1f - d[0] * d[0], -d[0] * d[1], -d[0] * d[2],
                -d[1] * d[0], 1f - d[1] * d[1], -d[1] * d[2],
                -d[2] * d[0], -d[2] * d[1], 1f - d[2] * d[2]
            )
            for (k in 0..8) a[k] += m[k]
            for (r in 0..2) b[r] += m[r * 3] * p[0] + m[r * 3 + 1] * p[1] + m[r * 3 + 2] * p[2]
        }
        val det = a[0] * (a[4] * a[8] - a[5] * a[7]) -
                  a[1] * (a[3] * a[8] - a[5] * a[6]) +
                  a[2] * (a[3] * a[7] - a[4] * a[6])
        if (kotlin.math.abs(det) < 1e-4f) return null
        val inv = floatArrayOf(
            (a[4] * a[8] - a[5] * a[7]), (a[2] * a[7] - a[1] * a[8]), (a[1] * a[5] - a[2] * a[4]),
            (a[5] * a[6] - a[3] * a[8]), (a[0] * a[8] - a[2] * a[6]), (a[2] * a[3] - a[0] * a[5]),
            (a[3] * a[7] - a[4] * a[6]), (a[1] * a[6] - a[0] * a[7]), (a[0] * a[4] - a[1] * a[3])
        )
        val c = FloatArray(3)
        for (r in 0..2) {
            c[r] = (inv[r * 3] * b[0] + inv[r * 3 + 1] * b[1] + inv[r * 3 + 2] * b[2]) / det
        }
        return if (c.all { it.isFinite() }) c else null
    }

    /**
     * Drop seed points that belong to the surroundings rather than the subject.
     *
     * The depth map reaches ~8 m, so scanning a small object indoors fills the entire
     * point budget with walls, floor and furniture -- a plastic box produced a
     * 15 x 9.5 x 15 m cloud, leaving the actual subject a tiny fraction of the seeds
     * and of the optimizer's capacity. Keeping only what surrounds the point the
     * cameras converge on spends the budget on the thing being scanned.
     *
     * Falls back to the unfiltered cloud whenever a subject cannot be identified
     * confidently, so a room or corridor scan is never mangled.
     */
    @Synchronized
    /**
     * @param subjectDistHint measured distance to whatever was in the centre of
     *   frame, from the depth map. Ray convergence alone is unreliable: a scan
     *   with a lot of rotation "converges" on a point in mid-air, and the sphere
     *   around it then contains no geometry at all (observed: kept 0 of 250000).
     *   Anchoring the centre to measured depth instead puts it on the object.
     */
    fun isolateSubject(subjectDistHint: Float = 0f, radiusFactor: Float = 0.6f): Int {
        if (depthVoxels.size < 2000 || capturedFrames.size < 8) return 0
        val cams = capturedFrames.map {
            floatArrayOf(it.transformMatrix[0][3], it.transformMatrix[1][3], it.transformMatrix[2][3])
        }
        val fwds = capturedFrames.map {
            val m = it.transformMatrix
            val f = floatArrayOf(-m[0][2], -m[1][2], -m[2][2])
            val n = kotlin.math.sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2])
            if (n > 1e-6f) floatArrayOf(f[0] / n, f[1] / n, f[2] / n) else f
        }
        // Candidate 1: where the view rays converge (good for a clean orbit).
        val cRay = subjectCenter(cams, fwds)

        // Candidate 2: the median point the cameras were actually LOOKING at, using
        // measured centre-of-frame depth. Survives rotation-heavy capture, because
        // it never relies on rays intersecting.
        val cDepth: FloatArray? = if (subjectDistHint > 0.05f) {
            val xs = FloatArray(cams.size); val ys = FloatArray(cams.size); val zs = FloatArray(cams.size)
            for (i in cams.indices) {
                xs[i] = cams[i][0] + fwds[i][0] * subjectDistHint
                ys[i] = cams[i][1] + fwds[i][1] * subjectDistHint
                zs[i] = cams[i][2] + fwds[i][2] * subjectDistHint
            }
            xs.sort(); ys.sort(); zs.sort()
            floatArrayOf(xs[xs.size / 2], ys[ys.size / 2], zs[zs.size / 2])
        } else null

        // Pick whichever candidate actually has geometry around it -- the failure
        // mode being guarded against is a centre floating in empty space.
        fun pointsNear(centre: FloatArray, r: Float): Int {
            var n = 0
            val r2 = r * r
            for (v in depthVoxels.values) {
                val dx = v[0] - centre[0]; val dy = v[1] - centre[1]; val dz = v[2] - centre[2]
                if (dx * dx + dy * dy + dz * dz <= r2) n++
            }
            return n
        }

        val c = when {
            cRay == null && cDepth == null -> {
                android.util.Log.i("DatasetExporter", "Subject isolation skipped: no usable subject centre")
                return 0
            }
            cRay == null -> cDepth!!
            cDepth == null -> cRay
            else -> {
                // Probe both at a generous radius before committing.
                val probe = if (subjectDistHint > 0.05f) subjectDistHint * radiusFactor else 0.5f
                val nRay = pointsNear(cRay, probe)
                val nDepth = pointsNear(cDepth, probe)
                android.util.Log.i(
                    "DatasetExporter",
                    "Subject centre probe: rayConvergence=$nRay points, depthGrounded=$nDepth points"
                )
                if (nDepth > nRay) cDepth else cRay
            }
        }
        val dists = cams.map {
            kotlin.math.sqrt(
                (it[0] - c[0]) * (it[0] - c[0]) +
                (it[1] - c[1]) * (it[1] - c[1]) +
                (it[2] - c[2]) * (it[2] - c[2])
            )
        }.sorted()
        val subjectDist = dists[dists.size / 2]
        if (!subjectDist.isFinite() || subjectDist < 0.05f) return 0

        // Only isolate when the surroundings clearly dominate the subject. A room
        // scanned from its perimeter also has converging rays, but there the cloud
        // IS the subject -- cropping it would destroy the capture.
        var mnX = Float.MAX_VALUE; var mxX = -Float.MAX_VALUE
        var mnY = Float.MAX_VALUE; var mxY = -Float.MAX_VALUE
        var mnZ = Float.MAX_VALUE; var mxZ = -Float.MAX_VALUE
        for (v in depthVoxels.values) {
            if (v[0] < mnX) mnX = v[0]; if (v[0] > mxX) mxX = v[0]
            if (v[1] < mnY) mnY = v[1]; if (v[1] > mxY) mxY = v[1]
            if (v[2] < mnZ) mnZ = v[2]; if (v[2] > mxZ) mxZ = v[2]
        }
        val diag = kotlin.math.sqrt(
            (mxX - mnX) * (mxX - mnX) + (mxY - mnY) * (mxY - mnY) + (mxZ - mnZ) * (mxZ - mnZ)
        )
        if (diag < subjectDist * 5f) {
            android.util.Log.i(
                "DatasetExporter",
                "Subject isolation skipped: cloud (%.1fm) is not much larger than subject distance (%.2fm)"
                    .format(diag, subjectDist)
            )
            return 0
        }

        val keepR = subjectDist * radiusFactor

        val before = depthVoxels.size
        val it2 = depthVoxels.entries.iterator()
        var kept = 0
        val survivors = HashMap<Long, FloatArray>()
        while (it2.hasNext()) {
            val e = it2.next()
            val v = e.value
            val dx = v[0] - c[0]; val dy = v[1] - c[1]; val dz = v[2] - c[2]
            if (dx * dx + dy * dy + dz * dz <= keepR * keepR) {
                survivors[e.key] = v; kept++
            }
        }
        // Too aggressive means we misidentified the subject -- keep everything.
        if (kept < 1500 || kept < before / 50) {
            android.util.Log.i(
                "DatasetExporter",
                "Subject isolation rejected: would keep only $kept of $before points"
            )
            return 0
        }
        depthVoxels.clear()
        depthVoxels.putAll(survivors)
        android.util.Log.i(
            "DatasetExporter",
            "Subject isolation: centre=(%.2f, %.2f, %.2f) dist=%.2fm radius=%.2fm -> kept %d of %d points"
                .format(c[0], c[1], c[2], subjectDist, keepR, kept, before)
        )
        return kept
    }

    /**
     * Combine the dense depth cloud with the sparse ARCore feature points and
     * write the seed PLY the trainer initialises from.
     */
    @Synchronized
    private fun writeSeedCloud(plyFile: File): Int {
        val total = depthVoxels.size + accumulatedFeaturePoints.size
        if (total == 0) return 0
        val xyz = FloatArray(total * 3)
        val rgb = FloatArray(total * 3)
        val logScale = FloatArray(total)
        var n = 0
        for (v in depthVoxels.values) {
            if (v[0].isFinite() && v[1].isFinite() && v[2].isFinite()) {
                rgb[n] = v[4]; rgb[n + 1] = v[5]; rgb[n + 2] = v[6]
                logScale[n / 3] = v[7]
                xyz[n++] = v[0]; xyz[n++] = v[1]; xyz[n++] = v[2]
            }
        }
        // Feature points carry no colour or size; fall back to neutral defaults.
        for (p in accumulatedFeaturePoints) {
            if (p.x.isFinite() && p.y.isFinite() && p.z.isFinite()) {
                rgb[n] = 0.5f; rgb[n + 1] = 0.5f; rgb[n + 2] = 0.5f
                logScale[n / 3] = -4.0f
                xyz[n++] = p.x; xyz[n++] = p.y; xyz[n++] = p.z
            }
        }
        android.util.Log.i(
            "DatasetExporter",
            "Seed cloud: ${depthVoxels.size} depth points + ${accumulatedFeaturePoints.size} feature points"
        )
        return com.splat.mobile3dgs.engine.GaussianInitializer
            .writeSeedPly(xyz, rgb, logScale, n / 3, plyFile)
    }

    private fun exportInitialPointCloud() {
        val ptsJson = JSONObject()
        val pointsArray = JSONArray()

        val sampleStep = maxOf(1, accumulatedFeaturePoints.size / 10000)
        for (i in 0 until accumulatedFeaturePoints.size step sampleStep) {
            val pt = accumulatedFeaturePoints[i]
            val ptObj = JSONObject()
            ptObj.put("id", pt.id)
            ptObj.put("xyz", JSONArray(listOf(pt.x, pt.y, pt.z)))
            ptObj.put("confidence", pt.confidence)
            pointsArray.put(ptObj)
        }

        ptsJson.put("num_points", pointsArray.length())
        ptsJson.put("points", pointsArray)

        val ptsFile = File(outputDir, "points3D_initial.json")
        ptsFile.writeText(ptsJson.toString(2))
    }
}
