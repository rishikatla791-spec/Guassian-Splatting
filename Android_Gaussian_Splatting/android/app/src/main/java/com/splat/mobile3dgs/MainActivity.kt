package com.splat.mobile3dgs

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.splat.mobile3dgs.capture.CaptureActivity
import com.splat.mobile3dgs.model.RemoteModel
import com.splat.mobile3dgs.network.ApiClient
import com.splat.mobile3dgs.viewer.ViewerActivity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {
    private lateinit var btnImportCustomDataset: Button
    private lateinit var btnScan: Button
    private lateinit var btnServerSettings: Button
    private lateinit var tvServerStatus: TextView
    private lateinit var listViewModels: ListView
    private lateinit var progressBar: ProgressBar

    private val apiClient = ApiClient()
    private val modelList = mutableListOf<RemoteModel>()
    private lateinit var adapter: ArrayAdapter<String>

    // SAF Document Picker for .splat files
    private val pickFileLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { handleImportedFile(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnImportCustomDataset = findViewById(R.id.btn_import_custom_dataset)
        btnScan = findViewById(R.id.btn_new_scan)
        btnServerSettings = findViewById(R.id.btn_server_settings)
        tvServerStatus = findViewById(R.id.tv_server_status)
        listViewModels = findViewById(R.id.lv_models)
        progressBar = findViewById(R.id.pb_loading_models)

        adapter = ArrayAdapter(this, android.R.layout.simple_list_item_1, mutableListOf())
        listViewModels.adapter = adapter

        // 1. Primary: Start Real-Time 3D Camera Scan
        btnScan.setOnClickListener {
            startActivity(Intent(this, CaptureActivity::class.java))
        }

        // 2. Secondary: Import local .splat file
        btnImportCustomDataset.setOnClickListener {
            pickFileLauncher.launch("*/*")
        }

        // 3. Configure Quality and Training Steps
        btnServerSettings.setOnClickListener {
            showQualityDialog()
        }

        listViewModels.setOnItemClickListener { _, _, position, _ ->
            if (position < modelList.size) {
                val model = modelList[position]
                openModel(model)
            }
        }

        loadScansAndStatus()
    }

    override fun onResume() {
        super.onResume()
        loadScansAndStatus()
    }

    private fun handleImportedFile(uri: Uri) {
        progressBar.visibility = View.VISIBLE
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val destFile = File(filesDir, "imported_${System.currentTimeMillis()}.splat")
                contentResolver.openInputStream(uri)?.use { input ->
                    FileOutputStream(destFile).use { output ->
                        input.copyTo(output)
                    }
                }
                withContext(Dispatchers.Main) {
                    progressBar.visibility = View.GONE
                    Toast.makeText(this@MainActivity, "Imported successfully!", Toast.LENGTH_SHORT).show()
                    val intent = Intent(this@MainActivity, ViewerActivity::class.java).apply {
                        putExtra("MODEL_NAME", "Imported 3D Model")
                        putExtra("MODEL_PATH", destFile.absolutePath)
                    }
                    startActivity(intent)
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    progressBar.visibility = View.GONE
                    Toast.makeText(this@MainActivity, "Import error: " + (e.message ?: ""), Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun showQualityDialog() {
        val prefs = getSharedPreferences("Mobile3DGS_Prefs", Context.MODE_PRIVATE)
        val currentSteps = prefs.getInt("PREF_TRAINING_STEPS", 3000)

        // Timings measured on a Snapdragon 8 Gen 2 (SM8550): ~8 steps/s initially,
        // degrading to ~6 steps/s once densification and thermal throttling kick in.
        val stepOptions = arrayOf(
            "Direct Photometric (Instant - ~2s, Zero GPU)",
            "300 Steps (Fast Standalone - ~45s, 360p)",
            "1,000 Steps (Quick test - ~3 min, 720p)",
            "3,000 Steps (Preview - ~8 min, 720p)",
            "7,000 Steps (Balanced - ~20 min, 1080p)",
            "15,000 Steps (High quality - ~45 min, 1080p)"
        )
        val stepValues = intArrayOf(0, 300, 1000, 3000, 7000, 15000)
        // Default to a real training run, not the instant photometric placeholder --
        // landing on index 0 made "Save" silently disable training.
        var selectedIndex = stepValues.indexOf(currentSteps).let { if (it >= 0) it else 3 }

        AlertDialog.Builder(this)
            .setTitle("⚙️ 3DGS Quality & Iterations")
            .setSingleChoiceItems(stepOptions, selectedIndex) { _, which ->
                selectedIndex = which
            }
            .setPositiveButton("Save") { dialog, _ ->
                val chosenSteps = stepValues[selectedIndex]
                val chosenRes = when {
                    chosenSteps >= 7000 -> 1080
                    chosenSteps >= 1000 -> 720
                    else -> 360
                }
                prefs.edit()
                    .putInt("PREF_TRAINING_STEPS", chosenSteps)
                    .putInt("PREF_TRAINING_RES", chosenRes)
                    .apply()
                val desc = if (chosenSteps == 0) "Direct Photometric (~2s)" else "$chosenSteps steps (${chosenRes}p)"
                Toast.makeText(this, "Target: $desc", Toast.LENGTH_SHORT).show()
                dialog.dismiss()
            }
            .setNeutralButton("Cloud Server IP") { dialog, _ ->
                dialog.dismiss()
                showCloudIpDialog()
            }
            .setNegativeButton("Cancel") { dialog, _ -> dialog.dismiss() }
            .show()
    }

    private fun showCloudIpDialog() {
        val dialogView = LayoutInflater.from(this).inflate(R.layout.dialog_server_config, null)
        val etServerIp = dialogView.findViewById<EditText>(R.id.et_server_ip)
        val btnCancel = dialogView.findViewById<Button>(R.id.btn_dialog_cancel)
        val btnConnect = dialogView.findViewById<Button>(R.id.btn_dialog_connect)

        etServerIp.setText(apiClient.getServerUrl())

        val dialog = AlertDialog.Builder(this)
            .setView(dialogView)
            .create()

        dialog.window?.setBackgroundDrawableResource(android.R.color.transparent)

        btnCancel.setOnClickListener {
            dialog.dismiss()
        }

        btnConnect.setOnClickListener {
            val url = etServerIp.text.toString().trim()
            if (url.isNotEmpty()) {
                apiClient.setServerUrl(url)
                loadScansAndStatus()
            }
            dialog.dismiss()
        }

        dialog.show()
    }

    /** Re-run optimization on an existing capture at a chosen quality. */
    private fun showRetrainDialog(model: RemoteModel, datasetDir: File) {
        val labels = arrayOf(
            "Fast  - 1500 steps @ 720p  (~4 min)",
            "Normal - 3000 steps @ 720p  (~8 min)",
            "High  - 7000 steps @ 720p  (~18 min)",
            "Max   - 7000 steps @ 1080p (~40 min, hot)"
        )
        val steps = intArrayOf(1500, 3000, 7000, 7000)
        val res = intArrayOf(720, 720, 720, 1080)
        var choice = 1

        AlertDialog.Builder(this)
            .setTitle("Re-train " + model.name)
            .setSingleChoiceItems(labels, choice) { _, w -> choice = w }
            .setPositiveButton("Start") { d, _ ->
                val out = File(filesDir, model.filename)
                com.splat.mobile3dgs.engine.TrainingService.start(
                    context = this,
                    datasetPath = datasetDir.absolutePath,
                    outputPath = out.absolutePath,
                    steps = steps[choice],
                    resolution = res[choice],
                    modelName = model.name + " (" + steps[choice] + " steps)"
                )
                Toast.makeText(
                    this,
                    "Training started - progress is in the notification shade",
                    Toast.LENGTH_LONG
                ).show()
                d.dismiss()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun loadScansAndStatus() {
        progressBar.visibility = View.VISIBLE
        val profile = com.splat.mobile3dgs.hardware.DeviceCapabilityManager.getDeviceProfile(this)
        tvServerStatus.text = "${profile.tier.tierName} • ${"%.1f".format(profile.totalRamGb)}GB"
        tvServerStatus.setTextColor(getColor(R.color.accent_green))

        lifecycleScope.launch(Dispatchers.IO) {
            // Automatically unpack default 3D models from assets on first run or if missing
            val bundledModels = listOf(
                "truck.splat" to "viewer/truck.splat",
                "train.splat" to "viewer/train.splat",
                "room.splat" to "viewer/room.splat",
                "photoreal_demo.splat" to "viewer/demo.splat"
            )
            for ((filename, assetPath) in bundledModels) {
                val dest = File(filesDir, filename)
                if (!dest.exists() || dest.length() == 0L) {
                    try {
                        assets.open(assetPath).use { input ->
                            FileOutputStream(dest).use { output ->
                                input.copyTo(output)
                            }
                        }
                    } catch (e: Exception) {
                        // Optional asset
                    }
                }
            }

            // Also import any .splat files placed in external app storage (e.g. via USB MTP)
            getExternalFilesDir(null)?.let { extDir ->
                extDir.listFiles { f -> f.isFile && f.name.endsWith(".splat") && f.length() > 0 }?.forEach { extFile ->
                    val target = File(filesDir, extFile.name)
                    if (!target.exists() || target.length() != extFile.length()) {
                        try {
                            extFile.copyTo(target, overwrite = true)
                        } catch (_: Exception) {}
                    }
                }
            }

            // Find all local captured and imported models
            val localSplats = filesDir.listFiles { file ->
                file.isFile && file.name.endsWith(".splat") && file.length() > 0
            } ?: emptyArray()

            val localModels = localSplats.map { file ->
                val sizeMb = file.length() / (1024.0 * 1024.0)
                RemoteModel(name = file.nameWithoutExtension, filename = file.name, sizeMb = sizeMb)
            }.sortedByDescending { it.filename }

            withContext(Dispatchers.Main) {
                modelList.clear()
                modelList.addAll(localModels)

                val names = localModels.map { "📱 ${it.name} (${String.format("%.1f", it.sizeMb)} MB)" }
                adapter.clear()
                adapter.addAll(names)
                adapter.notifyDataSetChanged()
                progressBar.visibility = View.GONE
            }
        }
    }

    private fun openModel(model: RemoteModel) {
        val localFile = File(filesDir, model.filename)
        if (localFile.exists() && localFile.length() > 0) {
            // A capture keeps its images, poses and seed cloud on disk, so it can be
            // re-optimized with different settings without walking around the subject
            // again -- a failed or over-long run no longer costs a rescan.
            val datasetDir = File(getExternalFilesDir(null), model.name)
            val canRetrain = File(datasetDir, "transforms.json").exists()

            val options = if (canRetrain) arrayOf(
                "🎮 Open 3D Viewport (Orbit, Measure & Crop)",
                "👓 Place Model in Real World (AR Placement)",
                "⚡ Re-train this capture (no rescan)"
            ) else arrayOf(
                "🎮 Open 3D Viewport (Orbit, Measure & Crop)",
                "👓 Place Model in Real World (AR Placement)"
            )
            AlertDialog.Builder(this)
                .setTitle(model.name)
                .setItems(options) { _, which ->
                    when (which) {
                        0 -> {
                            val intent = Intent(this, ViewerActivity::class.java).apply {
                                putExtra("MODEL_NAME", model.name)
                                putExtra("MODEL_PATH", localFile.absolutePath)
                            }
                            startActivity(intent)
                        }
                        1 -> {
                            val intent = Intent(this, com.splat.mobile3dgs.ar.ARPlacementActivity::class.java).apply {
                                putExtra("MODEL_NAME", model.name)
                                putExtra("MODEL_PATH", localFile.absolutePath)
                            }
                            startActivity(intent)
                        }
                        2 -> showRetrainDialog(model, datasetDir)
                    }
                }
                .setNegativeButton("Cancel", null)
                .show()
        } else {
            Toast.makeText(this, "Model file not found.", Toast.LENGTH_SHORT).show()
        }
    }
}
