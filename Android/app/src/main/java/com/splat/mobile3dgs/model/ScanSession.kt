package com.splat.mobile3dgs.model

import com.google.gson.annotations.SerializedName
import java.io.File

data class CameraPose(
    val frameIndex: Int,
    val timestamp: Long,
    val rotation: List<Float>,    // [qx, qy, qz, qw]
    val position: List<Float>,    // [x, y, z]
    val focalLengthX: Float,
    val focalLengthY: Float,
    val principalPointX: Float,
    val principalPointY: Float
)

data class ScanSession(
    val id: String,
    val name: String,
    val timestamp: Long,
    val directory: File,
    val poses: MutableList<CameraPose> = mutableListOf()
)

data class NerfstudioFrame(
    @SerializedName("file_path")
    val filePath: String,
    @SerializedName("transform_matrix")
    val transformMatrix: List<List<Float>>
)

data class NerfstudioDataset(
    @SerializedName("fl_x")
    val flX: Float,
    @SerializedName("fl_y")
    val flY: Float,
    @SerializedName("cx")
    val cx: Float,
    @SerializedName("cy")
    val cy: Float,
    @SerializedName("w")
    val w: Int,
    @SerializedName("h")
    val h: Int,
    @SerializedName("camera_model")
    val cameraModel: String = "OPENCV",
    @SerializedName("frames")
    val frames: List<NerfstudioFrame>
)

data class TrainingJobStatus(
    val id: String,
    val status: String, // "pending", "training", "completed", "failed"
    val progress: Float,
    val iteration: Int,
    @SerializedName("total_iterations")
    val totalIterations: Int,
    val message: String,
    @SerializedName("splat_url")
    val splatUrl: String? = null,
    @SerializedName("num_gaussians")
    val numGaussians: Int? = null
)

data class RemoteModel(
    val name: String,
    val filename: String,
    @SerializedName("size_mb")
    val sizeMb: Double,
    @SerializedName("download_url")
    val downloadUrl: String = ""
)

data class ModelListResponse(
    val models: List<RemoteModel>
)
