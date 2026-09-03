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
        val currentSteps = prefs.getInt("PREF_TRAINING_STEPS", 30000)

        val stepOptions = arrayOf(
            "3,000 Steps (Fast Preview - ~3 mins)",
            "7,000 Steps (Balanced - ~8 mins)",
            "15,000 Steps (High Quality - ~18 mins)",
            "30,000 Steps (Studio Pro - Sharpest 1080p)"
        )
        val stepValues = intArrayOf(3000, 7000, 15000, 30000)
        var selectedIndex = stepValues.indexOf(currentSteps).let { if (it >= 0) it else 3 }

        AlertDialog.Builder(this)
            .setTitle("⚙️ 3DGS Quality & Iterations")
            .setSingleChoiceItems(stepOptions, selectedIndex) { _, which ->
                selectedIndex = which
            }
            .setPositiveButton("Save") { dialog, _ ->
                val chosenSteps = stepValues[selectedIndex]
                val chosenRes = if (chosenSteps >= 15000) 1080 else 720
                prefs.edit()
                    .putInt("PREF_TRAINING_STEPS", chosenSteps)
                    .putInt("PREF_TRAINING_RES", chosenRes)
                    .apply()
                Toast.makeText(this, "Target: $chosenSteps steps (${chosenRes}p)", Toast.LENGTH_SHORT).show()
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

    private fun loadScansAndStatus() {
        progressBar.visibility = View.VISIBLE
        tvServerStatus.text = "Vulkan Ready (Snapdragon 8 Gen 2)"
        tvServerStatus.setTextColor(getColor(R.color.accent_green))

        lifecycleScope.launch(Dispatchers.IO) {
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
            val intent = Intent(this, ViewerActivity::class.java).apply {
                putExtra("MODEL_NAME", model.name)
                putExtra("MODEL_PATH", localFile.absolutePath)
            }
            startActivity(intent)
        } else {
            Toast.makeText(this, "Model file not found.", Toast.LENGTH_SHORT).show()
        }
    }
}
