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
import com.splat.mobile3dgs.R
import java.io.File

class ViewerActivity : AppCompatActivity() {
    private lateinit var webView: WebView
    private lateinit var tvTitle: TextView
    private lateinit var btnClose: ImageButton

    companion object {
        private const val TAG = "ViewerActivity"
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
            allowFileAccess = true
            allowContentAccess = true
            allowFileAccessFromFileURLs = true
            allowUniversalAccessFromFileURLs = true
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
            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                Log.d("WEBVIEW_CONSOLE", "Page loaded: $url")
            }
        }

        // Read the local splat model file directly from disk
        var modelBytes: ByteArray? = null
        if (modelPath != null) {
            val file = File(modelPath)
            if (file.exists() && file.length() > 0) {
                try {
                    modelBytes = file.readBytes()
                    Log.i(TAG, "Loaded captured model from $modelPath (${modelBytes.size} bytes, ${modelBytes.size / 32} Gaussians)")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to read model file: ${e.message}", e)
                    Toast.makeText(this, "Error reading model file: ${e.message}", Toast.LENGTH_LONG).show()
                }
            } else {
                Log.e(TAG, "Model file not found or empty at: $modelPath")
                Toast.makeText(this, "Model file does not exist at: $modelPath", Toast.LENGTH_LONG).show()
            }
        } else {
            Toast.makeText(this, "No model path provided.", Toast.LENGTH_SHORT).show()
        }

        class AndroidBridge(private val bytes: ByteArray?, private val name: String) {
            @JavascriptInterface
            fun hasCustomModel(): Boolean = bytes != null && bytes.isNotEmpty()

            @JavascriptInterface
            fun getSplatData(): String {
                return if (bytes != null && bytes.isNotEmpty()) {
                    Base64.encodeToString(bytes, Base64.NO_WRAP)
                } else ""
            }

            @JavascriptInterface
            fun getModelName(): String = name
        }

        webView.addJavascriptInterface(AndroidBridge(modelBytes, modelName), "AndroidBridge")

        // Load the WebGL2 viewport without any hardcoded demo query parameters
        webView.loadUrl("file:///android_asset/viewer/index.html")
    }

    override fun onDestroy() {
        webView.destroy()
        super.onDestroy()
    }
}
