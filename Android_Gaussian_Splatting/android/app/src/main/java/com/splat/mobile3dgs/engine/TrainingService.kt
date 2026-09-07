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
import com.splat.mobile3dgs.viewer.ViewerActivity
import java.io.File

/**
 * Runs on-device 3DGS optimization as a foreground service.
 *
 * Training takes minutes, and an Activity-scoped coroutine is killed as soon as
 * the user locks the screen or switches apps -- which silently destroys the run
 * with no output and no error. A foreground service plus a partial wakelock
 * keeps the process alive and shows live progress in the notification shade.
 */
class TrainingService : Service() {

    private var workerThread: Thread? = null
    private var wakeLock: PowerManager.WakeLock? = null
    @Volatile private var lastNotifiedPct = -1

    companion object {
        private const val TAG = "TrainingService"
        private const val CHANNEL_ID = "3dgs_training"
        private const val NOTIF_ID = 4201

        const val EXTRA_DATASET = "dataset_path"
        const val EXTRA_OUTPUT = "output_path"
        const val EXTRA_STEPS = "steps"
        const val EXTRA_RESOLUTION = "resolution"
        const val EXTRA_NAME = "model_name"

        /** Optional UI hooks. Set by a visible Activity, cleared when it goes away. */
        @Volatile var progressListener: ((step: Int, pct: Int) -> Unit)? = null
        @Volatile var doneListener: ((success: Boolean, outputPath: String) -> Unit)? = null

        fun start(
            context: Context,
            datasetPath: String,
            outputPath: String,
            steps: Int,
            resolution: Int,
            modelName: String
        ) {
            val intent = Intent(context, TrainingService::class.java).apply {
                putExtra(EXTRA_DATASET, datasetPath)
                putExtra(EXTRA_OUTPUT, outputPath)
                putExtra(EXTRA_STEPS, steps)
                putExtra(EXTRA_RESOLUTION, resolution)
                putExtra(EXTRA_NAME, modelName)
            }
            androidx.core.content.ContextCompat.startForegroundService(context, intent)
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent == null) {
            stopSelf()
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

        createChannel()
        startForeground(NOTIF_ID, buildNotification("Preparing 3DGS optimization...", 0, true))
        acquireWakeLock()

        workerThread = Thread {
            var success = false
            try {
                Log.i(TAG, "Training start: dataset=$dataset steps=$steps res=$resolution")
                val engine = NativeBrushEngine()
                success = engine.startOnDeviceTraining(
                    datasetPath = dataset,
                    outputPath = output,
                    iterations = steps,
                    maxResolution = resolution
                ) { step, progress ->
                    val pct = (progress * 100f).toInt().coerceIn(0, 100)
                    if (pct != lastNotifiedPct) {
                        lastNotifiedPct = pct
                        notify(buildNotification("Training: step $step / $steps", pct, false))
                    }
                    progressListener?.invoke(step, pct)
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Training failed", t)
                success = false
            }

            val outFile = File(output)
            val ok = success && outFile.exists() && outFile.length() > 0
            Log.i(TAG, "Training finished. success=$success outputExists=${outFile.exists()} size=${outFile.length()}")

            doneListener?.invoke(ok, output)
            showTerminalNotification(ok, output, modelName)

            releaseWakeLock()
            workerThread = null
            stopForeground(STOP_FOREGROUND_DETACH)
            stopSelf()
        }.also { it.start() }

        return START_NOT_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        releaseWakeLock()
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
            .setSmallIcon(R.drawable.ic_launcher)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setProgress(100, pct, indeterminate)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun notify(n: Notification) {
        try {
            (getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager).notify(NOTIF_ID, n)
        } catch (e: Exception) {
            Log.w(TAG, "notify failed: ${e.message}")
        }
    }

    private fun showTerminalNotification(ok: Boolean, outputPath: String, modelName: String) {
        val builder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_launcher)
            .setOngoing(false)
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)

        if (ok) {
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
                .setContentText("Tap to open $modelName")
                .setContentIntent(pi)
        } else {
            builder.setContentTitle("Training failed")
                .setContentText("No model was produced. Check Logcat tag 'BrushBridge'.")
        }
        notify(builder.build())
    }
}
