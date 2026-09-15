package com.splat.mobile3dgs.engine

import android.util.Log
import org.json.JSONObject
import java.io.File

/**
 * Callbacks from the native training thread.
 *
 * `onState` carries out-of-band transitions (resume, pause, cancel) so a stalled
 * progress bar can be explained instead of looking like a freeze.
 */
interface TrainingProgressListener {
    fun onProgress(step: Int, progress: Float)
    fun onState(state: String, detail: String) {}
}

/** Structural/numeric verdict on a produced `.splat`. */
data class SplatValidation(
    val ok: Boolean,
    val bytes: Long,
    val count: Long,
    val nonFinite: Long,
    val badQuat: Long,
    val badScale: Long,
    val opaque: Long,
    val message: String
) {
    companion object {
        fun parse(json: JSONObject?): SplatValidation = SplatValidation(
            ok = json?.optBoolean("ok", false) ?: false,
            bytes = json?.optLong("bytes", 0L) ?: 0L,
            count = json?.optLong("count", 0L) ?: 0L,
            nonFinite = json?.optLong("nonFinite", 0L) ?: 0L,
            badQuat = json?.optLong("badQuat", 0L) ?: 0L,
            badScale = json?.optLong("badScale", 0L) ?: 0L,
            opaque = json?.optLong("opaque", 0L) ?: 0L,
            message = json?.optString("message", "") ?: ""
        )
    }
}

/**
 * Outcome of a training run.
 *
 * The engine's C ABI returns a bare `0`/`1` with no error channel at all, so
 * [errorCode] is attributed by the bridge from how far the run got, and
 * [nativeLog] is the tail of the engine's stdout/stderr (the only place a Rust
 * panic message ever surfaces).
 */
data class TrainingResult(
    val ok: Boolean,
    val errorCode: String,
    val message: String,
    val exitCode: Int,
    /** 0 nothing started, 1 process created, 2 training, 3 training finished. */
    val phase: Int,
    val lastIter: Int,
    val splatCount: Long,
    val outputBytes: Long,
    val capped: Boolean,
    val resumedFrom: String,
    val validation: SplatValidation,
    val nativeLog: String
) {
    /** One-line summary safe to show in a notification. */
    fun shortReason(): String = when {
        ok -> "Trained $splatCount Gaussians"
        else -> "$errorCode: ${message.lineSequence().firstOrNull().orEmpty()}"
    }

    companion object {
        fun failure(code: String, msg: String) = TrainingResult(
            ok = false, errorCode = code, message = msg, exitCode = -1, phase = 0,
            lastIter = 0, splatCount = 0, outputBytes = 0, capped = false,
            resumedFrom = "",
            validation = SplatValidation(false, 0, 0, 0, 0, 0, 0, ""),
            nativeLog = ""
        )

        fun parse(raw: String): TrainingResult = try {
            val o = JSONObject(raw)
            TrainingResult(
                ok = o.optBoolean("ok", false),
                errorCode = o.optString("errorCode", "UNKNOWN"),
                message = o.optString("message", ""),
                exitCode = o.optInt("exitCode", -1),
                phase = o.optInt("phase", 0),
                lastIter = o.optInt("lastIter", 0),
                splatCount = o.optLong("splatCount", 0L),
                outputBytes = o.optLong("outputBytes", 0L),
                capped = o.optBoolean("capped", false),
                resumedFrom = o.optString("resumedFrom", ""),
                validation = SplatValidation.parse(o.optJSONObject("validation")),
                nativeLog = o.optString("nativeLog", "")
            )
        } catch (e: Throwable) {
            failure("BAD_NATIVE_RESULT", "Could not parse native result: ${e.message}\n$raw")
        }
    }
}

/** What [NativeBrushEngine.requestCancel] actually managed to do. */
enum class CancelOutcome {
    /** No run was in flight. */
    NOTHING_RUNNING,

    /**
     * The training thread will park at the next progress tick: the GPU goes
     * idle and the device stops heating, the result is discarded, but the run
     * cannot be unwound and its memory stays held until the app restarts.
     */
    STOPPED_ENGINE_POISONED,

    /** The native bridge is not loaded. */
    UNAVAILABLE
}

/**
 * Kotlin face of the prebuilt Rust/wgpu engine (`libbrush_c.so`).
 *
 * ## What the engine's ABI does and does not offer
 *
 * `train_and_save(dataset, TrainOptions*, callback, user_data) -> int` is the
 * whole thing (verified against `brush/apps/brush-c/src/lib.rs` and against the
 * shipped .so's dynamic symbol table, which exports exactly that one function).
 * `TrainOptions` carries only `total_train_steps`, `refine_every`,
 * `max_resolution`, `export_every` and `output_path`.
 *
 * Consequences that leak into this API:
 *
 *  * **No error channel.** Internal failures are `Err(_) => 1`. [TrainingResult]
 *    is therefore assembled by the bridge from the phase the run died in plus
 *    whatever the engine printed on stderr (panics only -- brush logs through
 *    the Rust `log` facade but `brush-c` installs no logger, and the .so does
 *    not even link liblog, so none of its `log::info!` output can reach logcat
 *    no matter what we do from this side).
 *  * **No cancellation.** See [CancelOutcome].
 *  * **No Gaussian cap.** `brush_train::config::TrainConfig::max_splats` exists
 *    (default 10,000,000) but is not reachable: it is not in `TrainOptions`, it
 *    has no env-var binding, and although brush can read an `args.txt` out of
 *    the dataset, `brush-c` throws that parsed config away and substitutes its
 *    own defaults. So densification really is unbounded, and `maxGaussians` is
 *    enforced the only two ways left to us -- by holding down the refinement
 *    cadence, and by decimating the finished model before it is published.
 */
class NativeBrushEngine {

    companion object {
        private const val TAG = "NativeBrushEngine"

        private var engineLoaded = false
        private var bridgeLoaded = false
        private var loadError: String? = null

        init {
            // Load the two libraries independently: the engine is arm64-v8a
            // only, and on any other ABI the bridge must still load so
            // diagnostics keep working instead of throwing UnsatisfiedLinkError
            // from every native method.
            try {
                System.loadLibrary("brush_c")
                engineLoaded = true
                Log.i(TAG, "Loaded libbrush_c.so")
            } catch (e: Throwable) {
                loadError = "libbrush_c.so: ${e.message ?: e.toString()}"
                Log.e(TAG, "Training engine unavailable on this ABI: $loadError")
            }
            try {
                System.loadLibrary("brush_bridge")
                bridgeLoaded = true
                Log.i(TAG, "Loaded libbrush_bridge.so")
            } catch (e: Throwable) {
                val msg = "libbrush_bridge.so: ${e.message ?: e.toString()}"
                loadError = if (loadError == null) msg else "$loadError; $msg"
                Log.e(TAG, "JNI bridge unavailable: $msg")
            }
        }

        /** True only when a real training run is possible. */
        fun isNativeEngineAvailable(): Boolean = engineLoaded && bridgeLoaded

        fun getLoadError(): String? = loadError

        /**
         * False once a cancelled run has parked the engine thread. Training in
         * this process is over until the app is restarted.
         */
        fun isEngineUsable(): Boolean =
            if (!bridgeLoaded) false else shared.nativeIsEngineUsable()

        /**
         * Requests cancellation. This is as close to a real cancel as the ABI
         * permits -- see [CancelOutcome.STOPPED_ENGINE_POISONED]; it is not a
         * no-op, but it is also not a clean abort.
         */
        fun requestCancel(): CancelOutcome {
            if (!bridgeLoaded) return CancelOutcome.UNAVAILABLE
            return when (shared.nativeRequestCancel()) {
                1 -> CancelOutcome.STOPPED_ENGINE_POISONED
                else -> CancelOutcome.NOTHING_RUNNING
            }
        }

        /**
         * Duty-cycles the training thread. [throttleMs] is slept at each
         * progress tick (~every 5 steps); [paused] holds it indefinitely.
         * Called from the service's thermal watchdog.
         */
        fun setDutyCycle(throttleMs: Int, paused: Boolean) {
            if (!bridgeLoaded) return
            try {
                shared.nativeSetDutyCycle(throttleMs, paused)
            } catch (e: Throwable) {
                Log.w(TAG, "setDutyCycle failed: ${e.message}")
            }
        }

        /** Newest periodic checkpoint for a dataset, or null. */
        fun findLatestCheckpoint(datasetPath: String): Checkpoint? {
            if (!bridgeLoaded) return null
            val raw = try {
                shared.nativeFindLatestCheckpoint(datasetPath)
            } catch (e: Throwable) {
                Log.w(TAG, "findLatestCheckpoint failed: ${e.message}"); return null
            }
            if (raw.isEmpty()) return null
            val parts = raw.split("|")
            if (parts.size < 3) return null
            return Checkpoint(parts[0], parts[1].toIntOrNull() ?: 0, parts[2].toLongOrNull() ?: 0L)
        }

        /**
         * Validates a finished `.splat` without training: exact 32-byte record
         * size, finite floats, normalised quaternions, some non-zero opacity,
         * plausible count.
         */
        fun validateSplat(path: String, maxGaussians: Int = 0): SplatValidation {
            if (!bridgeLoaded) {
                // Structural fallback so a corrupt file is still caught on an
                // ABI where the bridge could not load.
                val len = File(path).let { if (it.isFile) it.length() else -1L }
                return when {
                    len < 0 -> SplatValidation(false, 0, 0, 0, 0, 0, 0, "Missing: $path")
                    len == 0L -> SplatValidation(false, 0, 0, 0, 0, 0, 0, "Empty file")
                    len % 32L != 0L -> SplatValidation(false, len, 0, 0, 0, 0, 0,
                        "$len bytes is not a multiple of the 32-byte record size")
                    else -> SplatValidation(true, len, len / 32, 0, 0, 0, 0, "")
                }
            }
            return SplatValidation.parse(
                try {
                    JSONObject(shared.nativeValidateSplat(path, maxGaussians))
                } catch (e: Throwable) {
                    Log.w(TAG, "validateSplat failed: ${e.message}"); null
                }
            )
        }

        /** Engine resolution status plus a full Vulkan capability dump. */
        fun engineDiagnostics(): String =
            if (bridgeLoaded) shared.nativeCheckVulkanEngine()
            else "NOT_LOADED: $loadError"

        fun vulkanReport(): String =
            if (bridgeLoaded) shared.nativeVulkanReport() else "NOT_LOADED: $loadError"

        /**
         * Every native entry point is an instance method -- an `external` in a
         * companion object is emitted under a `$Companion` JNI name -- so the
         * process-wide helpers go through one shared instance.
         */
        private val shared: NativeBrushEngine by lazy { NativeBrushEngine() }
    }

    data class Checkpoint(val path: String, val iter: Int, val bytes: Long)

    /**
     * Runs on-device 3DGS optimization. Blocks the calling thread for the whole
     * run (minutes) -- call it from a service worker thread, never the main one.
     *
     * The result file is only written once the trained model has passed
     * validation, so a failed run can never leave a half-written or stale model
     * behind for the UI to present.
     *
     * @param resume warm-restart from the newest `exports/export_NNNN.ply`.
     *   Geometry is carried over; the learning-rate schedule still restarts from
     *   step 0 because the ABI has no `start_iter`.
     */
    fun startOnDeviceTraining(
        datasetPath: String,
        outputPath: String,
        iterations: Int = 7000,
        maxResolution: Int = 720,
        refineEvery: Int = 100,
        exportEvery: Int = 0,
        maxGaussians: Int = 0,
        resume: Boolean = false,
        // Deliberately NOT named `onState`/`onProgress`: inside the listener
        // object below, a call to a name that matches one of the interface's own
        // methods resolves to that member, not to the parameter, and the
        // override calls itself forever.
        stateCallback: (String, String) -> Unit = { _, _ -> },
        progressCallback: (Int, Float) -> Unit = { _, _ -> }
    ): TrainingResult {
        if (!bridgeLoaded) {
            return TrainingResult.failure(
                "BRIDGE_MISSING",
                "The JNI bridge is not loaded: ${loadError ?: "unknown"}"
            )
        }
        if (!engineLoaded) {
            return TrainingResult.failure(
                "ENGINE_MISSING",
                "libbrush_c.so is not available on this ABI (the engine ships for " +
                    "arm64-v8a only): ${loadError ?: "unknown"}"
            )
        }

        val listener = object : TrainingProgressListener {
            override fun onProgress(step: Int, progress: Float) = progressCallback(step, progress)
            override fun onState(state: String, detail: String) = stateCallback(state, detail)
        }

        return try {
            val raw = nativeTrainAndSave(
                datasetPath, outputPath, iterations, maxResolution,
                refineEvery, exportEvery, maxGaussians, resume, listener
            )
            TrainingResult.parse(raw).also {
                if (it.ok) Log.i(TAG, "Training OK: ${it.message}")
                else Log.e(TAG, "Training failed [${it.errorCode}] ${it.message}")
            }
        } catch (e: Throwable) {
            Log.e(TAG, "On-device training threw", e)
            TrainingResult.failure("NATIVE_EXCEPTION", e.toString())
        }
    }

    private external fun nativeRequestCancel(): Int
    private external fun nativeIsEngineUsable(): Boolean
    private external fun nativeSetDutyCycle(throttleMs: Int, paused: Boolean)
    private external fun nativeCheckVulkanEngine(): String
    private external fun nativeVulkanReport(): String
    private external fun nativeFindLatestCheckpoint(datasetPath: String): String
    private external fun nativeValidateSplat(path: String, maxGaussians: Int): String

    private external fun nativeTrainAndSave(
        datasetPath: String,
        outputPath: String,
        iterations: Int,
        maxResolution: Int,
        refineEvery: Int,
        exportEvery: Int,
        maxGaussians: Int,
        resume: Boolean,
        listener: TrainingProgressListener?
    ): String
}
