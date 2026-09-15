package com.splat.mobile3dgs.viewer

import android.annotation.SuppressLint
import android.app.ActivityManager
import android.content.Context
import android.graphics.Color
import android.os.Bundle
import android.util.Base64
import android.util.Log
import android.view.WindowManager
import android.webkit.ConsoleMessage
import android.webkit.JavascriptInterface
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream

class ViewerActivity : AppCompatActivity() {
    private lateinit var webView: WebView
    private lateinit var tvTitle: TextView
    private lateinit var btnClose: ImageButton

    /** Streaming target for the chunked save bridge. Guarded by [saveLock]. */
    private val saveLock = Any()
    private var saveStream: BufferedOutputStream? = null
    private var saveFile: File? = null
    private var saveBytes: Long = 0

    companion object {
        private const val TAG = "ViewerActivity"
        /** Virtual origin used to serve bundled assets and the model file. */
        private const val ASSET_HOST = "appassets.androidplatform.net"
        private const val SPLAT_STRIDE = 32
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(com.splat.mobile3dgs.R.layout.activity_viewer)

        webView = findViewById(com.splat.mobile3dgs.R.id.webview_splat)
        tvTitle = findViewById(com.splat.mobile3dgs.R.id.tv_viewer_title)
        btnClose = findViewById(com.splat.mobile3dgs.R.id.btn_close_viewer)

        val modelName = intent.getStringExtra("MODEL_NAME") ?: "Captured 3D Model"
        val modelPath = intent.getStringExtra("MODEL_PATH")
        tvTitle.text = modelName

        btnClose.setOnClickListener { finish() }

        // The page defaults to the light theme; keep the native chrome in step so
        // there is never a dark flash or an invisible title behind the WebView.
        applyChrome(dark = false)
        // Orbiting a model is a "watching" interaction -- do not dim mid-gesture.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            allowFileAccess = false
            allowContentAccess = false
            // The page and the model are both served from a virtual https origin
            // via shouldInterceptRequest, so file:// access is not needed and the
            // dangerous cross-origin file flags stay off.
            allowFileAccessFromFileURLs = false
            allowUniversalAccessFromFileURLs = false
            loadWithOverviewMode = true
            useWideViewPort = true
            cacheMode = WebSettings.LOAD_NO_CACHE
        }

        webView.webChromeClient = object : WebChromeClient() {
            override fun onConsoleMessage(consoleMessage: ConsoleMessage?): Boolean {
                Log.d("WEBVIEW_CONSOLE", "[${consoleMessage?.messageLevel()}] ${consoleMessage?.message()} (${consoleMessage?.sourceId()}:${consoleMessage?.lineNumber()})")
                return true
            }
        }

        webView.webViewClient = object : WebViewClient() {
            override fun shouldInterceptRequest(
                view: WebView?,
                request: android.webkit.WebResourceRequest?
            ): android.webkit.WebResourceResponse? {
                val uri = request?.url ?: return null
                if (uri.host != ASSET_HOST) return null
                val path = uri.path ?: return null
                return try {
                    when {
                        // Stream the model straight off disk -- never load it all
                        // into RAM and never Base64 it through the JS bridge.
                        path == "/model.splat" -> {
                            val f = modelPath?.let { File(it) }
                            if (f != null && f.exists() && f.length() > 0) {
                                val headers = HashMap<String, String>()
                                headers["Content-Length"] = f.length().toString()
                                headers["Cache-Control"] = "no-store"
                                android.webkit.WebResourceResponse(
                                    "application/octet-stream", null, 200, "OK",
                                    headers, java.io.FileInputStream(f)
                                )
                            } else null
                        }
                        path.startsWith("/viewer/") -> {
                            val assetPath = path.removePrefix("/")
                            val mime = when {
                                assetPath.endsWith(".html") -> "text/html"
                                assetPath.endsWith(".js") -> "application/javascript"
                                assetPath.endsWith(".css") -> "text/css"
                                else -> "application/octet-stream"
                            }
                            android.webkit.WebResourceResponse(mime, "utf-8", assets.open(assetPath))
                        }
                        else -> null
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Intercept failed for $path: ${e.message}")
                    null
                }
            }

            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                Log.d("WEBVIEW_CONSOLE", "Page loaded: $url")
            }
        }

        // Validate the model without reading it into memory. The WebView streams
        // it from the virtual https origin, so a 500 MB model costs no app heap.
        val modelFile = modelPath?.let { File(it) }
        val hasModel = modelFile != null && modelFile.exists() && modelFile.length() > 0
        if (hasModel) {
            Log.i(TAG, "Model available: $modelPath (${modelFile!!.length()} bytes, ${modelFile.length() / SPLAT_STRIDE} Gaussians)")
        } else if (modelPath != null) {
            Log.e(TAG, "Model file not found or empty at: $modelPath")
            Toast.makeText(this, "Model file does not exist at: $modelPath", Toast.LENGTH_LONG).show()
        } else {
            Toast.makeText(this, "No model path provided.", Toast.LENGTH_SHORT).show()
        }

        webView.addJavascriptInterface(AndroidBridge(hasModel, modelName, modelPath), "AndroidBridge")

        // Served through shouldInterceptRequest so the page and the model share
        // one origin (no file:// access, no Base64 bridge).
        webView.loadUrl("https://$ASSET_HOST/viewer/index.html")
    }

    /** Keep the native header legible against whichever background the page uses. */
    private fun applyChrome(dark: Boolean) {
        val bg = if (dark) Color.parseColor("#0B0E14") else Color.parseColor("#EEF1F6")
        val fg = if (dark) Color.WHITE else Color.parseColor("#17202E")
        webView.setBackgroundColor(bg)
        window.decorView.setBackgroundColor(bg)
        tvTitle.setTextColor(fg)
        btnClose.imageTintList = android.content.res.ColorStateList.valueOf(fg)
    }

    /** Where edited models are written: next to the source model when possible. */
    private fun saveDirectory(modelPath: String?): File {
        val parent = modelPath?.let { File(it).parentFile }
        return if (parent != null && parent.isDirectory && parent.canWrite()) parent else filesDir
    }

    private fun closeSaveStream() {
        synchronized(saveLock) {
            try { saveStream?.flush() } catch (_: Exception) { }
            try { saveStream?.close() } catch (_: Exception) { }
            saveStream = null
        }
    }

    inner class AndroidBridge(
        private val modelAvailable: Boolean,
        private val name: String,
        private val modelPath: String?
    ) {
        @JavascriptInterface
        fun hasCustomModel(): Boolean = modelAvailable

        @JavascriptInterface
        fun getModelUrl(): String = "https://$ASSET_HOST/model.splat"

        @JavascriptInterface
        fun getModelName(): String = name

        /** Lets the renderer size its splat budget to what this phone can hold. */
        @JavascriptInterface
        fun getDeviceInfo(): String {
            return try {
                val am = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                val mi = ActivityManager.MemoryInfo()
                am.getMemoryInfo(mi)
                val totalMb = mi.totalMem / (1024L * 1024L)
                val availMb = mi.availMem / (1024L * 1024L)
                val lowRam = am.isLowRamDevice || totalMb < 2600
                """{"totalMemMb":$totalMb,"availMemMb":$availMb,"lowRam":$lowRam}"""
            } catch (e: Exception) {
                """{"totalMemMb":0,"availMemMb":0,"lowRam":false}"""
            }
        }

        @JavascriptInterface
        fun onThemeChanged(dark: Boolean) {
            runOnUiThread { applyChrome(dark) }
        }

        @JavascriptInterface
        fun launchAR() {
            runOnUiThread {
                if (modelPath != null) {
                    try {
                        val intent = android.content.Intent(this@ViewerActivity, com.splat.mobile3dgs.ar.ARPlacementActivity::class.java).apply {
                            putExtra("MODEL_PATH", modelPath)
                            putExtra("MODEL_NAME", name)
                        }
                        startActivity(intent)
                    } catch (e: Exception) {
                        Toast.makeText(this@ViewerActivity, "AR unavailable: ${e.message}", Toast.LENGTH_SHORT).show()
                    }
                } else {
                    Toast.makeText(this@ViewerActivity, "Cannot launch AR: No model file found", Toast.LENGTH_SHORT).show()
                }
            }
        }

        /**
         * Distance produced by a real raycast into the splat cloud (see
         * viewer.js raycast()). The capture is anchored by ARCore, so the units
         * are real-world metres.
         */
        @JavascriptInterface
        fun showMeasurementToast(meters: Float) {
            if (!meters.isFinite() || meters <= 0f) return
            runOnUiThread {
                val msg = if (meters >= 1f) {
                    "Distance: %.2f m (%.1f in)".format(meters, meters * 39.3701f)
                } else {
                    "Distance: %.1f cm (%.1f in)".format(meters * 100f, meters * 39.3701f)
                }
                Toast.makeText(this@ViewerActivity, msg, Toast.LENGTH_LONG).show()
            }
        }

        /* ---------------- chunked .splat export ----------------
         * The page streams the edited buffer in ~384 KB base64 chunks, so a
         * 60 MB model never needs an 80 MB string in the WebView heap (the old
         * single-shot base64 path reliably OOM'd on low-RAM phones).
         */

        @JavascriptInterface
        fun beginSave(splatCount: Int): Boolean {
            synchronized(saveLock) {
                closeSaveStream()
                return try {
                    val safe = name.replace(Regex("[^A-Za-z0-9._-]"), "_").take(40).ifEmpty { "model" }
                    val out = File(saveDirectory(modelPath), "${safe}_edited_${System.currentTimeMillis()}.splat")
                    saveFile = out
                    saveBytes = 0
                    saveStream = BufferedOutputStream(FileOutputStream(out), 1 shl 16)
                    Log.i(TAG, "Save started: ${out.absolutePath} ($splatCount splats expected)")
                    true
                } catch (e: Exception) {
                    Log.e(TAG, "beginSave failed: ${e.message}")
                    saveStream = null
                    saveFile = null
                    false
                }
            }
        }

        @JavascriptInterface
        fun appendSaveChunk(base64Chunk: String): Boolean {
            synchronized(saveLock) {
                val s = saveStream ?: return false
                return try {
                    val bytes = Base64.decode(base64Chunk, Base64.DEFAULT)
                    s.write(bytes)
                    saveBytes += bytes.size
                    true
                } catch (e: Throwable) {
                    Log.e(TAG, "appendSaveChunk failed: ${e.message}")
                    closeSaveStream()
                    saveFile?.delete()
                    saveFile = null
                    false
                }
            }
        }

        /** Returns the written file name, or an empty string on failure. */
        @JavascriptInterface
        fun endSave(): String {
            synchronized(saveLock) {
                val f = saveFile
                closeSaveStream()
                saveFile = null
                if (f == null) return ""
                // A .splat is only valid when it is a whole number of 32-byte records.
                if (saveBytes <= 0L || saveBytes % SPLAT_STRIDE != 0L || f.length() != saveBytes) {
                    Log.e(TAG, "Discarding malformed save: wrote=$saveBytes onDisk=${f.length()}")
                    f.delete()
                    runOnUiThread {
                        Toast.makeText(this@ViewerActivity, "Save failed: incomplete file discarded", Toast.LENGTH_LONG).show()
                    }
                    return ""
                }
                val splats = saveBytes / SPLAT_STRIDE
                Log.i(TAG, "Saved ${f.absolutePath} ($saveBytes bytes, $splats splats)")
                runOnUiThread {
                    Toast.makeText(
                        this@ViewerActivity,
                        "Saved ${f.name} ($splats splats)",
                        Toast.LENGTH_LONG
                    ).show()
                }
                return f.name
            }
        }

        @JavascriptInterface
        fun abortSave() {
            synchronized(saveLock) {
                closeSaveStream()
                saveFile?.delete()
                saveFile = null
                saveBytes = 0
            }
        }
    }

    override fun onDestroy() {
        synchronized(saveLock) {
            if (saveStream != null) {
                closeSaveStream()
                saveFile?.delete()
                saveFile = null
            }
        }
        try {
            webView.loadUrl("about:blank")
            (webView.parent as? android.view.ViewGroup)?.removeView(webView)
            webView.destroy()
        } catch (e: Exception) {
            Log.w(TAG, "WebView teardown: ${e.message}")
        }
        super.onDestroy()
    }
}
