package com.splat.mobile3dgs.network

import com.google.gson.Gson
import com.splat.mobile3dgs.model.ModelListResponse
import com.splat.mobile3dgs.model.TrainingJobStatus
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.io.FileOutputStream

class ApiClient(private var baseUrl: String = "http://10.0.2.2:8000") {
    private val client = OkHttpClient.Builder().build()
    private val gson = Gson()

    fun setServerUrl(url: String) {
        this.baseUrl = url.trimEnd('/')
    }

    fun getServerUrl(): String = baseUrl

    suspend fun checkHealth(): Boolean = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder().url("${baseUrl}/health").build()
            client.newCall(request).execute().use { response ->
                response.isSuccessful
            }
        } catch (e: Exception) {
            false
        }
    }

    suspend fun getModels(): List<com.splat.mobile3dgs.model.RemoteModel> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder().url("${baseUrl}/models").build()
            client.newCall(request).execute().use { response ->
                if (response.isSuccessful && response.body != null) {
                    val json = response.body!!.string()
                    val result = gson.fromJson(json, ModelListResponse::class.java)
                    result.models
                } else {
                    emptyList()
                }
            }
        } catch (e: Exception) {
            emptyList()
        }
    }

    suspend fun uploadScan(zipFile: File): String? = withContext(Dispatchers.IO) {
        try {
            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "file",
                    zipFile.name,
                    zipFile.asRequestBody("application/zip".toMediaTypeOrNull())
                )
                .build()

            val request = Request.Builder()
                .url("${baseUrl}/upload")
                .post(requestBody)
                .build()

            client.newCall(request).execute().use { response ->
                if (response.isSuccessful && response.body != null) {
                    val json = response.body!!.string()
                    val map = gson.fromJson(json, Map::class.java)
                    map["scan_id"] as? String
                } else {
                    null
                }
            }
        } catch (e: Exception) {
            null
        }
    }

    suspend fun getStatus(scanId: String): TrainingJobStatus? = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder().url("${baseUrl}/status/${scanId}").build()
            client.newCall(request).execute().use { response ->
                if (response.isSuccessful && response.body != null) {
                    val json = response.body!!.string()
                    gson.fromJson(json, TrainingJobStatus::class.java)
                } else {
                    null
                }
            }
        } catch (e: Exception) {
            null
        }
    }

    suspend fun downloadModel(filename: String, destination: File): Boolean = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder().url("${baseUrl}/download/${filename}").build()
            client.newCall(request).execute().use { response ->
                if (response.isSuccessful && response.body != null) {
                    val inputStream = response.body!!.byteStream()
                    val outputStream = FileOutputStream(destination)
                    inputStream.copyTo(outputStream)
                    outputStream.close()
                    inputStream.close()
                    true
                } else {
                    false
                }
            }
        } catch (e: Exception) {
            false
        }
    }
}
