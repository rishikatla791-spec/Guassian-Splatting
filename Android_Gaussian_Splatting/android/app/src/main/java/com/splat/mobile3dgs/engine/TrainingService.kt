package com.splat.mobile3dgs.engine

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import com.splat.mobile3dgs.R
import com.splat.mobile3dgs.hardware.DeviceCapabilityManager
import com.splat.mobile3dgs.viewer.ViewerActivity
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Runs on-device 3DGS optimization as a foreground service.
 *
 * Training takes minutes, and an Activity-scoped coroutine is killed as soon as
 * the user locks the screen or switches apps -- which silently destroys the run
 * with no output and no error. A foreground service plus a partial wakelock
 * keeps the process alive and shows live progress in the notification shade.
 *
 * The service also owns the two things the native engine cannot decide for
 * itself: whether it is thermally and electrically sane to keep training (see
 * [thermalWatchdog]) and whether the model that came out is fit to show.
 */
class TrainingService : Service() {

    private var workerThread: Thread? = null
    private var watchdogThread: Thread? = null
    private var wakeLock: PowerManager.WakeLock? = null

    @Volatile private var lastNotifiedPct = -1
    @Volatile private var currentStatusText = "Preparing 3DGS optimization..."
    @Volatile private var currentStep = 0
    @Volatile private var totalSteps = 0
    private val terminated = AtomicBoolean(false)
    private val watchdogRunning = AtomicBoolean(false)

    companion object {
        private const val TAG = "TrainingService"
        private const val CHANNEL_ID = "3dgs_training"
        private const val NOTIF_ID = 4201

        const val EXTRA_DATASET = "dataset_path"
        const val EXTRA_OUTPUT = "output_path"
        const val EXTRA_STEPS = "steps"
        const val EXTRA_RESOLUTION = "resolution"
        const val EXTRA_NAME = "model_name"
        const val EXTRA_RESUME = "resume"

        const val ACTION_CANCEL = "com.splat.mobile3dgs.action.CANCEL_TRAINING"

        /** Optional UI hooks. Set by a visible Activity, cleared when it goes away. */
        @Volatile var progressListener: ((step: Int, pct: Int) -> Unit)? = null
        @Volatile var doneListener: ((success: Boolean, outputPath: String) -> Unit)? = null

        /**
         * Full outcome, including the real reason a run failed. The engine's C
         * ABI only returns 0/1, so this is assembled by the JNI bridge; see
         * [NativeBrushEngine].
         */
        @Volatile var resultListener: ((TrainingResult) -> Unit)? = null

        /** Human-readable state changes (resumed, throttled, paused, cancelled). */
        @Volatile var statusListener: ((String) -> Unit)? = null

        /** Outcome of the most recent run in this process. */
        @Volatile var lastResult: TrainingResult? = null
            private set

        fun start(
            context: Context,
            datasetPath: String,
            outputPath: String,
            steps: Int,
            resolution: Int,
            modelName: String,
            /**
             * Warm-restart from the newest checkpoint in `<dataset>/exports/`
             * when one exists, instead of starting over. Defaults on, so a run
             * that was killed, throttled out or OOM'd is continued.
             */
            resume: Boolean = true
        ) {
            val intent = Intent(context, TrainingService::class.java).apply {
                putExtra(EXTRA_DATASET, datasetPath)
                putExtra(EXTRA_OUTPUT, outputPath)
                putExtra(EXTRA_STEPS, steps)
                putExtra(EXTRA_RESOLUTION, resolution)
                putExtra(EXTRA_NAME, modelName)
                putExtra(EXTRA_RESUME, resume)
            }
            androidx.core.content.ContextCompat.startForegroundService(context, intent)
        }

        /**
         * Asks the running job to stop.
         *
         * This is a real stop -- the GPU goes idle and the result is discarded --
         * but it is not a clean abort, because the engine's C ABI has no
         * cancellation token: `train_and_save` only returns when the whole run
         * is over. The native thread is parked instead, so its memory stays held
         * and no further training is possible until the app restarts. Callers
         * should say so rather than implying a tidy cancel.
         */
        fun cancel(context: Context) {
            val intent = Intent(context, TrainingService::class.java).apply {
                action = ACTION_CANCEL
            }
            try {
                androidx.core.content.ContextCompat.startForegroundService(context, intent)
            } catch (e: Exception) {
                Log.w(TAG, "Cancel dispatch failed: ${e.message}")
            }
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent == null) {
            stopSelf()
            return START_NOT_STICKY
        }

        if (intent.action == ACTION_CANCEL) {
            handleCancel()
            return START_NOT_STICKY
        }

        if (workerThread != null) {
            Log.w(TAG, "Training already running; ignoring duplicate start request")
            return START_NOT_STICKY
        }

        val dataset = intent.getStringExtra(EXTRA_DATASET) ?: run { stopSelf(); return START_NOT_STICKY }
        val output = intent.getStringExtra(EXTRA_OUTPUT) ?: run { stopSelf(); return START_NOT_STICKY }
        val steps = intent.getIntExtra(EXTRA_STEPS, 3000)
        val resolution = intent.getIntExtra(EXTRA_RESOLUTION, 720)
        val modelName = intent.getStringExtra(EXTRA_NAME) ?: "On-Device Scan"
        val resume = intent.getBooleanExtra(EXTRA_RESUME, true)

        createChannel()
        startForeground(NOTIF_ID, buildNotification("Preparing 3DGS optimization...", 0, true))
        // Reset before the precondition checks: they report through finishWith too.
        terminated.set(false)

        // --- refuse to start rather than fail halfway -----------------------
        if (!NativeBrushEngine.isNativeEngineAvailable()) {
            finishWith(
                TrainingResult.failure(
                    "ENGINE_UNAVAILABLE",
                    "On-device training is unavailable on this device: " +
                        (NativeBrushEngine.getLoadError() ?: "native libraries did not load")
                ),
                output, modelName
            )
            return START_NOT_STICKY
        }
        if (!NativeBrushEngine.isEngineUsable()) {
            finishWith(
                TrainingResult.failure(
                    "ENGINE_POISONED",
                    "A cancelled run parked the native training thread. The engine cannot " +
                        "abort a run, so the app has to be restarted before training again."
                ),
                output, modelName
            )
            return START_NOT_STICKY
        }
        val verdict = DeviceCapabilityManager.canStartTraining(this)
        if (!verdict.allowed) {
            finishWith(TrainingResult.failure("PRECONDITION", verdict.reason), output, modelName)
            return START_NOT_STICKY
        }

        acquireWakeLock()

        val profile = DeviceCapabilityManager.getDeviceProfile(this)
        val refineEvery = DeviceCapabilityManager.refineEveryFor(profile)
        val exportEvery = DeviceCapabilityManager.exportEveryFor(profile, steps)
        totalSteps = steps
        lastNotifiedPct = -1

        val checkpoint = if (resume) NativeBrushEngine.findLatestCheckpoint(dataset) else null
        if (checkpoint != null) {
            Log.i(TAG, "Resuming from checkpoint ${checkpoint.path} (step ${checkpoint.iter})")
        }

        startThermalWatchdog()

        workerThread = Thread {
            val result = try {
                Log.i(
                    TAG,
                    "Training start: dataset=$dataset steps=$steps res=$resolution " +
                        "refineEvery=$refineEvery exportEvery=$exportEvery " +
                        "maxGaussians=${profile.maxGaussians} resume=$resume"
                )
                val engine = NativeBrushEngine()
                engine.startOnDeviceTraining(
                    datasetPath = dataset,
                    outputPath = output,
                    iterations = steps,
                    maxResolution = resolution,
                    refineEvery = refineEvery,
                    exportEvery = exportEvery,
                    maxGaussians = profile.maxGaussians,
                    resume = resume,
                    stateCallback = { state, detail -> onEngineState(state, detail) },
                    progressCallback = { step, progress -> onEngineProgress(step, progress, steps) }
                )
            } catch (t: Throwable) {
                Log.e(TAG, "Training failed", t)
                TrainingResult.failure("SERVICE_EXCEPTION", t.toString())
            }

            // A cancelled run never reaches here -- the native thread parks
            // inside train_and_save -- so this only runs for a real outcome.
            finishWith(verifyOutput(result, output, profile.maxGaussians), output, modelName)
        }.also { it.start() }

        return START_NOT_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        watchdogRunning.set(false)
        releaseWakeLock()
    }

    // ---------------------------------------------------------------------
    // Engine callbacks
    // ---------------------------------------------------------------------

    private fun onEngineProgress(step: Int, progress: Float, steps: Int) {
        currentStep = step
        val pct = (progress * 100f).toInt().coerceIn(0, 100)
        if (pct != lastNotifiedPct) {
            lastNotifiedPct = pct
            currentStatusText = "Training: step $step / $steps"
            notify(buildNotification(currentStatusText, pct, false))
        }
        progressListener?.invoke(step, pct)
    }

    private fun onEngineState(state: String, detail: String) {
        Log.i(TAG, "Engine state: $state -- $detail")
        currentStatusText = when (state) {
            "RESUMING" -> "Resuming: $detail"
            "CANCELLED" -> "Cancelled: $detail"
            else -> detail.ifEmpty { state }
        }
        statusListener?.invoke(currentStatusText)
        notify(buildNotification(currentStatusText, lastNotifiedPct.coerceAtLeast(0), false))
    }

    // ---------------------------------------------------------------------
    // Thermal / battery duty cycling
    // ---------------------------------------------------------------------

    /**
     * Polls thermal status and battery and tells the native side how hard to
     * back off. Sustained training on a mid-range SoC halves its own throughput
     * purely from heat, so easing off early beats being throttled by the kernel;
     * and a pause has to be visible in the notification or it looks like the app
     * has frozen.
     */
    private fun startThermalWatchdog() {
        if (watchdogRunning.getAndSet(true)) return
        watchdogThread = Thread {
            var lastReason = ""
            while (watchdogRunning.get()) {
                try {
                    val duty = DeviceCapabilityManager.currentDutyCycle(this)
                    NativeBrushEngine.setDutyCycle(duty.throttleMs, duty.paused)

                    if (duty.reason != lastReason) {
                        lastReason = duty.reason
                        Log.i(
                            TAG,
                            "Duty cycle -> throttle=${duty.throttleMs}ms paused=${duty.paused} (${duty.reason})"
                        )
                        statusListener?.invoke(duty.reason)
                        if (!duty.isFullSpeed) {
                            // Make a pause look deliberate instead of frozen.
                            notify(
                                buildNotification(
                                    "${duty.reason} — step $currentStep / $totalSteps",
                                    lastNotifiedPct.coerceAtLeast(0),
                                    false
                                )
                            )
                        } else if (currentStep > 0) {
                            notify(
                                buildNotification(
                                    "Training: step $currentStep / $totalSteps",
                                    lastNotifiedPct.coerceAtLeast(0),
                                    false
                                )
                            )
                        }
                    }
                } catch (e: Throwable) {
                    Log.w(TAG, "Thermal watchdog error: ${e.message}")
                }
                try {
                    Thread.sleep(5000)
                } catch (e: InterruptedException) {
                    return@Thread
                }
            }
        }.also { it.isDaemon = true; it.start() }
    }

    // ---------------------------------------------------------------------
    // Cancellation
    // ---------------------------------------------------------------------

    private fun handleCancel() {
        val outcome = NativeBrushEngine.requestCancel()
        Log.w(TAG, "Cancel requested -> $outcome")
        watchdogRunning.set(false)
        watchdogThread?.interrupt()

        when (outcome) {
            CancelOutcome.NOTHING_RUNNING, CancelOutcome.UNAVAILABLE -> {
                releaseWakeLock()
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
            CancelOutcome.STOPPED_ENGINE_POISONED -> {
                // The worker thread is parked inside train_and_save and will
                // never return, so tear the service down without it.
                val result = TrainingResult.failure(
                    "CANCELLED",
                    "Training stopped at step $currentStep. The engine cannot abort a run, " +
                        "so its memory stays held until the app is restarted."
                )
                lastResult = result
                if (terminated.compareAndSet(false, true)) {
                    resultListener?.invoke(result)
                    statusListener?.invoke("Cancelled — restart the app to train again")
                    doneListener?.invoke(false, "")
                    showTerminalNotification(result, "", "Cancelled")
                }
                releaseWakeLock()
                workerThread = null
                stopForeground(STOP_FOREGROUND_DETACH)
                stopSelf()
            }
        }
    }

    // ---------------------------------------------------------------------
    // Result handling
    // ---------------------------------------------------------------------

    /**
     * Second opinion on the model before anything presents it as a result.
     *
     * The bridge already validates and only publishes a model that passed, but
     * this catches the other half of the historical trap: an output file that
     * exists because something *else* wrote it (the direct photometric model is
     * written to the same path before training starts), so "the file is there"
     * has never meant "training produced it".
     */
    private fun verifyOutput(result: TrainingResult, output: String, maxGaussians: Int): TrainingResult {
        if (!result.ok) return result

        val f = File(output)
        if (!f.isFile || f.length() == 0L) {
            return TrainingResult.failure(
                "OUTPUT_MISSING",
                "Training reported success but $output is missing or empty."
            )
        }
        if (output.endsWith(".splat")) {
            val v = NativeBrushEngine.validateSplat(output, maxGaussians)
            if (!v.ok) {
                return TrainingResult.failure(
                    "VALIDATION_FAILED",
                    "The produced model did not validate and will not be shown: ${v.message}"
                )
            }
            if (result.splatCount > 0 && v.count != result.splatCount) {
                return TrainingResult.failure(
                    "VALIDATION_FAILED",
                    "Model holds ${v.count} Gaussians but training reported ${result.splatCount}; " +
                        "the file on disk is not the one this run produced."
                )
            }
        }
        return result
    }

    private fun finishWith(result: TrainingResult, output: String, modelName: String) {
        if (!terminated.compareAndSet(false, true)) return

        watchdogRunning.set(false)
        watchdogThread?.interrupt()
        NativeBrushEngine.setDutyCycle(0, false)

        lastResult = result
        Log.i(
            TAG,
            "Training finished: ok=${result.ok} code=${result.errorCode} " +
                "exit=${result.exitCode} phase=${result.phase} lastIter=${result.lastIter} " +
                "splats=${result.splatCount} bytes=${result.outputBytes}"
        )
        if (!result.ok) {
            Log.e(TAG, "Training failure detail:\n${result.message}")
            if (result.nativeLog.isNotEmpty()) Log.e(TAG, "Engine output tail:\n${result.nativeLog}")
        }
        if (result.capped) {
            Log.w(TAG, "Model was decimated to fit this device's Gaussian budget")
        }

        // A real reconstruction replaced the pre-written preview, so clear its
        // marker. Only a genuine success does this -- a failed or cancelled run
        // leaves the file flagged as untrained so the gallery can say so.
        if (result.ok) {
            runCatching {
                java.io.File(output + com.splat.mobile3dgs.engine.GaussianInitializer.PREVIEW_MARKER_SUFFIX).delete()
            }
        }

        resultListener?.invoke(result)
        doneListener?.invoke(result.ok, output)
        showTerminalNotification(result, output, modelName)

        releaseWakeLock()
        workerThread = null
        stopForeground(STOP_FOREGROUND_DETACH)
        stopSelf()
    }

    // ---------------------------------------------------------------------

    private fun acquireWakeLock() {
        try {
            val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
            wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "Mobile3DGS::Training").apply {
                setReferenceCounted(false)
                acquire(2 * 60 * 60 * 1000L) // hard cap: 2 hours
            }
        } catch (e: Exception) {
            Log.w(TAG, "Could not acquire wakelock: ${e.message}")
        }
    }

    private fun releaseWakeLock() {
        try {
            if (wakeLock?.isHeld == true) wakeLock?.release()
        } catch (e: Exception) {
            Log.w(TAG, "Wakelock release failed: ${e.message}")
        }
        wakeLock = null
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "3DGS Training",
                NotificationManager.IMPORTANCE_LOW
            ).apply { description = "On-device Gaussian Splatting optimization progress" }
            (getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager)
                .createNotificationChannel(channel)
        }
    }

    private fun buildNotification(text: String, pct: Int, indeterminate: Boolean): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Building 3D model")
            .setContentText(text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(text))
            .setSmallIcon(R.drawable.ic_launcher)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setProgress(100, pct, indeterminate)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            // Training is a long background job, so it has to be stoppable from
            // the shade -- otherwise the only way out is force-stopping the app.
            .addAction(
                0,
                "Stop",
                PendingIntent.getService(
                    this, 1,
                    Intent(this, TrainingService::class.java).setAction(ACTION_CANCEL),
                    PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
                )
            )
            .setContentIntent(
                PendingIntent.getActivity(
                    this, 2,
                    Intent(this, com.splat.mobile3dgs.MainActivity::class.java)
                        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP),
                    PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
                )
            )
            .build()
    }

    private fun notify(n: Notification) {
        try {
            (getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager).notify(NOTIF_ID, n)
        } catch (e: Exception) {
            Log.w(TAG, "notify failed: ${e.message}")
        }
    }

    private fun showTerminalNotification(result: TrainingResult, outputPath: String, modelName: String) {
        val builder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_launcher)
            .setOngoing(false)
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)

        if (result.ok) {
            val viewIntent = Intent(this, ViewerActivity::class.java).apply {
                putExtra("MODEL_PATH", outputPath)
                putExtra("MODEL_NAME", modelName)
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
            }
            val pi = PendingIntent.getActivity(
                this, 0, viewIntent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )
            builder.setContentTitle("3D model ready")
                .setContentText("Tap to open $modelName (${result.splatCount} Gaussians)")
                .setContentIntent(pi)
        } else {
            val reason = result.shortReason()
            builder.setContentTitle(
                if (result.errorCode == "CANCELLED") "Training cancelled" else "Training failed"
            )
                .setContentText(reason)
                .setStyle(NotificationCompat.BigTextStyle().bigText(result.message))
        }
        notify(builder.build())
    }
}
