package com.splat.mobile3dgs

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.splat.mobile3dgs.capture.CaptureActivity
import com.splat.mobile3dgs.engine.TrainingService
import com.splat.mobile3dgs.hardware.DeviceCapabilityManager
import com.splat.mobile3dgs.viewer.ViewerActivity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.text.DateFormat
import java.util.Date

/**
 * Home screen: start a capture, browse finished models, and watch training.
 *
 * There is no server, cloud or desktop component -- capture, optimization and
 * viewing all happen on this device -- so nothing here touches the network.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var rvModels: RecyclerView
    private lateinit var progressBar: ProgressBar
    private lateinit var layoutEmpty: View
    private lateinit var tvModelCount: TextView
    private lateinit var tvDeviceStatus: TextView

    private lateinit var cardTraining: View
    private lateinit var tvTrainingPct: TextView
    private lateinit var tvTrainingState: TextView
    private lateinit var progressTraining: ProgressBar

    private val models = mutableListOf<LocalModel>()
    private lateinit var adapter: ModelAdapter

    /** A finished .splat on disk, plus the capture it came from if still present. */
    data class LocalModel(
        val file: File,
        val name: String,
        val sizeMb: Double,
        val gaussians: Long,
        val modifiedAt: Long,
        val datasetDir: File?
    )

    private val pickFileLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let { handleImportedFile(it) }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        rvModels = findViewById(R.id.rv_models)
        progressBar = findViewById(R.id.pb_loading_models)
        layoutEmpty = findViewById(R.id.layout_empty)
        tvModelCount = findViewById(R.id.tv_model_count)
        tvDeviceStatus = findViewById(R.id.tv_device_status)
        cardTraining = findViewById(R.id.card_training)
        findViewById<View>(R.id.btn_cancel_training).setOnClickListener { confirmStopTraining() }
        tvTrainingPct = findViewById(R.id.tv_training_pct)
        tvTrainingState = findViewById(R.id.tv_training_state)
        progressTraining = findViewById(R.id.progress_training)

        adapter = ModelAdapter(models) { openModel(it) }
        rvModels.layoutManager = LinearLayoutManager(this)
        rvModels.adapter = adapter
        rvModels.isNestedScrollingEnabled = false

        findViewById<View>(R.id.card_capture_hero).setOnClickListener { startCapture() }
        findViewById<View>(R.id.btn_empty_capture).setOnClickListener { startCapture() }
        findViewById<View>(R.id.card_import).setOnClickListener { pickFileLauncher.launch("*/*") }
        findViewById<View>(R.id.card_quality).setOnClickListener { showQualityDialog() }
        findViewById<View>(R.id.btn_quality_settings).setOnClickListener { showQualityDialog() }

        cardTraining.visibility = View.GONE
        showDeviceProfile()
    }

    override fun onResume() {
        super.onResume()
        // Training runs in a foreground service, so it can still be going while
        // this screen is reopened -- keep the progress card live either way.
        TrainingService.progressListener = { step, pct ->
            runOnUiThread { showTrainingProgress(step, pct) }
        }
        TrainingService.doneListener = { ok, _ ->
            runOnUiThread { showTrainingDone(ok) }
        }
        loadModels()
    }

    override fun onPause() {
        super.onPause()
        TrainingService.progressListener = null
        TrainingService.doneListener = null
    }

    private fun startCapture() {
        startActivity(Intent(this, CaptureActivity::class.java))
    }

    private fun showDeviceProfile() {
        val profile = DeviceCapabilityManager.getDeviceProfile(this)
        tvDeviceStatus.text = getString(
            R.string.device_profile,
            profile.tier.tierName,
            "%.1f".format(profile.totalRamGb)
        )
    }

    // ---------------------------------------------------------------- models

    private fun loadModels() {
        progressBar.visibility = View.VISIBLE
        lifecycleScope.launch(Dispatchers.IO) {
            val external = getExternalFilesDir(null)

            // Models placed beside the app (for example pushed over adb) are adopted
            // into internal storage so the gallery has a single source of truth.
            external?.listFiles { f -> f.isFile && f.name.endsWith(".splat") && f.length() > 0 }
                ?.forEach { ext ->
                    val dest = File(filesDir, ext.name)
                    if (!dest.exists() || dest.length() != ext.length()) {
                        runCatching { ext.copyTo(dest, overwrite = true) }
                    }
                }

            val found = (filesDir.listFiles { f ->
                f.isFile && f.name.endsWith(".splat") && f.length() > 0
            } ?: emptyArray())
                .map { f ->
                    val base = f.nameWithoutExtension
                    val dataset = external?.let { File(it, base) }
                        ?.takeIf { File(it, "transforms.json").exists() }
                    LocalModel(
                        file = f,
                        name = base,
                        sizeMb = f.length() / (1024.0 * 1024.0),
                        gaussians = f.length() / SPLAT_STRIDE,
                        modifiedAt = f.lastModified(),
                        datasetDir = dataset
                    )
                }
                .sortedByDescending { it.modifiedAt }

            withContext(Dispatchers.Main) {
                models.clear()
                models.addAll(found)
                adapter.notifyDataSetChanged()
                progressBar.visibility = View.GONE
                layoutEmpty.visibility = if (found.isEmpty()) View.VISIBLE else View.GONE
                rvModels.visibility = if (found.isEmpty()) View.GONE else View.VISIBLE
                tvModelCount.text = found.size.toString()
            }
        }
    }

    private fun handleImportedFile(uri: Uri) {
        progressBar.visibility = View.VISIBLE
        lifecycleScope.launch(Dispatchers.IO) {
            val result = runCatching {
                val dest = File(filesDir, "imported_" + System.currentTimeMillis() + ".splat")
                contentResolver.openInputStream(uri)?.use { input ->
                    FileOutputStream(dest).use { output -> input.copyTo(output) }
                } ?: error("could not open the selected file")
                dest
            }
            withContext(Dispatchers.Main) {
                progressBar.visibility = View.GONE
                result.onSuccess { dest ->
                    Toast.makeText(this@MainActivity, R.string.toast_import_success, Toast.LENGTH_SHORT).show()
                    openViewer(getString(R.string.imported_model_name), dest)
                    loadModels()
                }.onFailure {
                    Toast.makeText(this@MainActivity, R.string.toast_import_failed, Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun openModel(model: LocalModel) {
        if (!model.file.exists() || model.file.length() == 0L) {
            Toast.makeText(this, R.string.toast_model_missing, Toast.LENGTH_SHORT).show()
            loadModels()
            return
        }
        val canRetrain = model.datasetDir != null
        val options = if (canRetrain) {
            arrayOf(
                getString(R.string.action_view_3d),
                getString(R.string.action_place_ar),
                getString(R.string.action_retrain)
            )
        } else {
            arrayOf(
                getString(R.string.action_view_3d),
                getString(R.string.action_place_ar)
            )
        }

        AlertDialog.Builder(this)
            .setTitle(model.name)
            .setItems(options) { _, which ->
                when (which) {
                    0 -> openViewer(model.name, model.file)
                    1 -> startActivity(
                        Intent(this, com.splat.mobile3dgs.ar.ARPlacementActivity::class.java).apply {
                            putExtra("MODEL_NAME", model.name)
                            putExtra("MODEL_PATH", model.file.absolutePath)
                        }
                    )
                    2 -> model.datasetDir?.let { showRetrainDialog(model, it) }
                }
            }
            .setNegativeButton(R.string.action_cancel, null)
            .show()
    }

    private fun openViewer(name: String, file: File) {
        startActivity(Intent(this, ViewerActivity::class.java).apply {
            putExtra("MODEL_NAME", name)
            putExtra("MODEL_PATH", file.absolutePath)
        })
    }

    // --------------------------------------------------------------- quality

    private fun showQualityDialog() {
        val prefs = getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val stepValues = intArrayOf(0, 300, 1000, 3000, 7000, 15000)
        val current = prefs.getInt(PREF_STEPS, 3000)
        // Never land on the instant photometric option by default: it disables
        // training, so pressing Save without looking would silently switch it off.
        var selected = stepValues.indexOf(current).let { if (it >= 0) it else 3 }

        AlertDialog.Builder(this)
            .setTitle(R.string.dialog_quality_title)
            .setSingleChoiceItems(R.array.quality_options, selected) { _, which -> selected = which }
            .setPositiveButton(R.string.action_save) { dialog, _ ->
                val steps = stepValues[selected]
                val res = when {
                    steps >= 7000 -> 1080
                    steps >= 1000 -> 720
                    else -> 360
                }
                prefs.edit().putInt(PREF_STEPS, steps).putInt(PREF_RES, res).apply()
                val summary = if (steps == 0) {
                    getString(R.string.quality_summary_instant)
                } else {
                    getString(R.string.quality_summary_steps, steps, res)
                }
                Toast.makeText(this, getString(R.string.toast_quality_saved, summary), Toast.LENGTH_SHORT).show()
                dialog.dismiss()
            }
            .setNegativeButton(R.string.action_cancel, null)
            .show()
    }

    /** Re-optimize an existing capture; its frames and poses are still on disk. */
    private fun showRetrainDialog(model: LocalModel, datasetDir: File) {
        val steps = intArrayOf(1500, 3000, 7000, 7000)
        val res = intArrayOf(720, 720, 720, 1080)
        var choice = 1

        AlertDialog.Builder(this)
            .setTitle(R.string.dialog_retrain_title)
            .setMessage(R.string.dialog_retrain_message)
            .setSingleChoiceItems(R.array.retrain_options, choice) { _, w -> choice = w }
            .setPositiveButton(R.string.action_start) { dialog, _ ->
                showTrainingProgress(0, 0)
                TrainingService.start(
                    context = this,
                    datasetPath = datasetDir.absolutePath,
                    outputPath = model.file.absolutePath,
                    steps = steps[choice],
                    resolution = res[choice],
                    modelName = getString(R.string.retrain_model_name, model.name, steps[choice])
                )
                Toast.makeText(this, R.string.toast_training_started, Toast.LENGTH_LONG).show()
                dialog.dismiss()
            }
            .setNegativeButton(R.string.action_cancel, null)
            .show()
    }

    // -------------------------------------------------------------- training

    /**
     * Stop a run in flight.
     *
     * brush's C ABI has no abort: `train_and_save` owns the thread until the run
     * finishes, so the most a cancel can do in-process is park the training
     * thread -- the GPU goes idle and the device stops heating, but the run's
     * memory stays held for the life of the process. Rather than leave the app in
     * that poisoned state (where the next training attempt would fail until the
     * user manually force-stopped it), park the engine and then restart the
     * process cleanly. Checkpoints already written survive on disk.
     */
    private fun confirmStopTraining() {
        AlertDialog.Builder(this)
            .setTitle(R.string.training_cancel_title)
            .setMessage(R.string.training_cancel_body)
            .setPositiveButton(R.string.training_cancel_confirm) { _, _ -> stopTrainingAndRestart() }
            .setNegativeButton(R.string.training_cancel_keep, null)
            .show()
    }

    private fun stopTrainingAndRestart() {
        // Parks the training thread immediately so the GPU stops before the
        // process goes down, and lets the service tear its notification down.
        try {
            TrainingService.cancel(this)
        } catch (t: Throwable) {
            android.util.Log.w("MainActivity", "Cancel dispatch failed: ${t.message}")
        }
        Toast.makeText(this, R.string.training_cancel_title, Toast.LENGTH_SHORT).show()

        // Queue a relaunch, then end the process: that is the only thing that
        // actually reclaims what the engine is holding.
        val restart = Intent(this, MainActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
        val pending = android.app.PendingIntent.getActivity(
            this, 0, restart,
            android.app.PendingIntent.FLAG_CANCEL_CURRENT or android.app.PendingIntent.FLAG_IMMUTABLE
        )
        val am = getSystemService(Context.ALARM_SERVICE) as android.app.AlarmManager
        am.set(android.app.AlarmManager.RTC, System.currentTimeMillis() + 400, pending)

        finishAffinity()
        android.os.Process.killProcess(android.os.Process.myPid())
    }

    private fun showTrainingProgress(step: Int, pct: Int) {
        if (cardTraining.visibility != View.VISIBLE) {
            cardTraining.alpha = 0f
            cardTraining.visibility = View.VISIBLE
            cardTraining.animate().alpha(1f).setDuration(220).start()
        }
        tvTrainingPct.text = pct.toString() + "%"
        progressTraining.isIndeterminate = false
        progressTraining.progress = pct
        tvTrainingState.text = if (step <= 0) {
            getString(R.string.training_preparing)
        } else {
            getString(R.string.training_progress, step, pct)
        }
    }

    private fun showTrainingDone(ok: Boolean) {
        tvTrainingState.setText(if (ok) R.string.training_done else R.string.training_failed)
        if (ok) progressTraining.progress = 100
        cardTraining.animate().alpha(0f).setStartDelay(2500).setDuration(300)
            .withEndAction {
                cardTraining.visibility = View.GONE
                cardTraining.alpha = 1f
                loadModels()
            }.start()
    }

    // --------------------------------------------------------------- adapter

    private class ModelAdapter(
        private val items: List<LocalModel>,
        private val onClick: (LocalModel) -> Unit
    ) : RecyclerView.Adapter<ModelAdapter.VH>() {

        private val thumbs = intArrayOf(
            R.drawable.clay_thumb_mint,
            R.drawable.clay_thumb_violet,
            R.drawable.clay_thumb_coral
        )
        private var lastAnimated = -1

        class VH(v: View) : RecyclerView.ViewHolder(v) {
            val card: View = v.findViewById(R.id.card_model)
            // thumb_model is the claymorphism tile (a FrameLayout holding an icon),
            // not an ImageView -- the clay_thumb_* drawables are its BACKGROUND.
            val thumb: View = v.findViewById(R.id.thumb_model)
            val name: TextView = v.findViewById(R.id.tv_model_name)
            val gaussians: TextView = v.findViewById(R.id.tv_model_gaussians)
            val size: TextView = v.findViewById(R.id.tv_model_size)
            val date: TextView = v.findViewById(R.id.tv_model_date)
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH = VH(
            LayoutInflater.from(parent.context).inflate(R.layout.item_model, parent, false)
        )

        override fun getItemCount(): Int = items.size

        override fun onBindViewHolder(holder: VH, position: Int) {
            val m = items[position]
            val ctx = holder.itemView.context
            holder.name.text = m.name
            holder.gaussians.text = ctx.getString(
                R.string.model_meta_gaussians, String.format("%,d", m.gaussians)
            )
            holder.size.text = ctx.getString(R.string.model_meta_size, "%.1f".format(m.sizeMb))
            holder.date.text = DateFormat.getDateInstance(DateFormat.MEDIUM).format(Date(m.modifiedAt))
            holder.thumb.setBackgroundResource(thumbs[position % thumbs.size])
            holder.card.setOnClickListener { onClick(m) }

            // Stagger cards in on first appearance only, so scrolling stays still.
            if (position > lastAnimated) {
                lastAnimated = position
                holder.itemView.alpha = 0f
                holder.itemView.translationY = 24f
                holder.itemView.animate()
                    .alpha(1f)
                    .translationY(0f)
                    .setStartDelay((position % 6) * 40L)
                    .setDuration(260)
                    .start()
            }
        }
    }

    companion object {
        private const val PREFS = "Mobile3DGS_Prefs"
        private const val PREF_STEPS = "PREF_TRAINING_STEPS"
        private const val PREF_RES = "PREF_TRAINING_RES"
        private const val SPLAT_STRIDE = 32L
    }
}
