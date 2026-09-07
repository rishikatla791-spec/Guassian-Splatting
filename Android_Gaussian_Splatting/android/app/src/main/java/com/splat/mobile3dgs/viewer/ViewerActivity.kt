package com.splat.mobile3dgs.viewer

import android.annotation.SuppressLint
import android.graphics.Color
import android.os.Bundle
import android.util.Base64
import android.util.Log
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
import androidx.lifecycle.lifecycleScope
import com.splat.mobile3dgs.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class ViewerActivity : AppCompatActivity() {
    private lateinit var webView: WebView
    private lateinit var tvTitle: TextView
    private lateinit var btnClose: ImageButton

    companion object {
        private const val TAG = "ViewerActivity"
        /** Virtual origin used to serve bundled assets and the model file. */
        private const val ASSET_HOST = "appassets.androidplatform.net"
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_viewer)

        webView = findViewById(R.id.webview_splat)
        tvTitle = findViewById(R.id.tv_viewer_title)
        btnClose = findViewById(R.id.btn_close_viewer)

        val modelName = intent.getStringExtra("MODEL_NAME") ?: "Captured 3D Model"
        val modelPath = intent.getStringExtra("MODEL_PATH")
        tvTitle.text = modelName

        btnClose.setOnClickListener { finish() }

        webView.setBackgroundColor(Color.parseColor("#0B0E14"))

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
                                android.webkit.WebResourceResponse(
                                    "application/octet-stream", null, java.io.FileInputStream(f)
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
            Log.i(TAG, "Model available: $modelPath (${modelFile!!.length()} bytes, ${modelFile.length() / 32} Gaussians)")
        } else if (modelPath != null) {
            Log.e(TAG, "Model file not found or empty at: $modelPath")
            Toast.makeText(this, "Model file does not exist at: $modelPath", Toast.LENGTH_LONG).show()
        } else {
            Toast.makeText(this, "No model path provided.", Toast.LENGTH_SHORT).show()
        }

        class AndroidBridge(private val modelAvailable: Boolean, private val name: String) {
            @JavascriptInterface
            fun hasCustomModel(): Boolean = modelAvailable

            @JavascriptInterface
            fun getModelUrl(): String = "https://$ASSET_HOST/model.splat"

            @JavascriptInterface
            fun getModelName(): String = name

            @JavascriptInterface
            fun launchAR() {
                runOnUiThread {
                    if (modelPath != null) {
                        val intent = android.content.Intent(this@ViewerActivity, com.splat.mobile3dgs.ar.ARPlacementActivity::class.java).apply {
                            putExtra("MODEL_PATH", modelPath)
                            putExtra("MODEL_NAME", name)
                        }
                        startActivity(intent)
                    } else {
                        Toast.makeText(this@ViewerActivity, "Cannot launch AR: No model file found", Toast.LENGTH_SHORT).show()
                    }
                }
            }

            @JavascriptInterface
            fun showMeasurementToast(meters: Float) {
                runOnUiThread {
                    val cm = meters * 100.0f
                    val inches = meters * 39.3701f
                    val msg = "📏 Real-World Distance: ${"%.1f".format(cm)} cm (${"%.1f".format(inches)} in)"
                    Toast.makeText(this@ViewerActivity, msg, Toast.LENGTH_LONG).show()
                }
            }

            @JavascriptInterface
            fun saveCroppedModel(base64Data: String) {
                lifecycleScope.launch(Dispatchers.IO) {
                    try {
                        val bytes = Base64.decode(base64Data, Base64.DEFAULT)
                        val outFile = File(filesDir, "cropped_${System.currentTimeMillis()}.splat")
                        outFile.writeBytes(bytes)
                        withContext(Dispatchers.Main) {
                            Toast.makeText(this@ViewerActivity, "Cropped model saved: ${outFile.name} (${bytes.size / 32} splats)", Toast.LENGTH_LONG).show()
                        }
                    } catch (e: Exception) {
                        withContext(Dispatchers.Main) {
                            Toast.makeText(this@ViewerActivity, "Failed to save: ${e.message}", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
            }
        }

        webView.addJavascriptInterface(AndroidBridge(hasModel, modelName), "AndroidBridge")

        // Served through shouldInterceptRequest so the page and the model share
        // one origin (no file:// access, no Base64 bridge).
        webView.loadUrl("https://$ASSET_HOST/viewer/index.html")
    }

    override fun onDestroy() {
        webView.destroy()
        super.onDestroy()
    }
}
