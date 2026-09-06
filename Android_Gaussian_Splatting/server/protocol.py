"""
Protocol models and schemas for Android-to-Laptop 3DGS Task Coordination.
Milestone 1: Inter-Agent Communication & Task Handshake Protocol.
"""

from enum import Enum
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class JobStage(str, Enum):
    """9-stage job lifecycle state machine."""
    IDLE = "IDLE"
    UPLOADING = "UPLOADING"
    INGESTING = "INGESTING"
    QUEUED = "QUEUED"
    TRAINING = "TRAINING"
    PRUNING = "PRUNING"
    CONVERTING = "CONVERTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    @classmethod
    def from_string(cls, value: str) -> "JobStage":
        """Convert string to JobStage with fallback/alias support."""
        val = value.strip().upper()
        if val in cls.__members__:
            return cls[val]
        alias_map = {
            "PENDING": cls.QUEUED,
            "EXTRACTING": cls.INGESTING,
            "PREPROCESSING": cls.INGESTING,
            "SFM": cls.INGESTING,
            "OPTIMIZING": cls.TRAINING,
            "CANCELLED": cls.FAILED,
            "CANCELED": cls.FAILED,
            "DONE": cls.COMPLETED,
            "SUCCESS": cls.COMPLETED,
            "ERROR": cls.FAILED,
        }
        return alias_map.get(val, cls.FAILED)

    @property
    def is_terminal(self) -> bool:
        """Return True if this stage is terminal (COMPLETED or FAILED)."""
        return self in (JobStage.COMPLETED, JobStage.FAILED)

    @property
    def is_active(self) -> bool:
        """Return True if job is currently running or queued."""
        return self in (
            JobStage.UPLOADING,
            JobStage.INGESTING,
            JobStage.QUEUED,
            JobStage.TRAINING,
            JobStage.PRUNING,
            JobStage.CONVERTING,
        )


class ClientCapabilities(BaseModel):
    """Hardware and software capabilities reported by the Android client."""
    sensor_fusion: bool = Field(default=False, description="Whether ARCore sensor fusion / 6DOF is active")
    vulkan_version: Optional[str] = Field(default=None, description="Vulkan API version if supported")
    depth_sensor: bool = Field(default=False, description="Whether hardware ToF/LiDAR depth sensor is present")
    camera_fps: int = Field(default=30, description="Capture frame rate")
    gyroscope: bool = Field(default=True, description="Whether IMU gyroscope is available")
    battery_level: Optional[float] = Field(default=None, description="Battery percentage (0.0 - 1.0)")


class HandshakeRequest(BaseModel):
    """Initial handshake request from Android client to Desktop coordinator."""
    client_id: str = Field(..., description="Unique client instance or device identifier")
    device_model: str = Field(default="Android Device", description="Model name (e.g., Pixel 8, Galaxy S24)")
    app_version: str = Field(default="1.0.0", description="Android 3DGS app build version")
    protocol_version: str = Field(default="1.0.0", description="Protocol version requested")
    capabilities: Optional[ClientCapabilities] = Field(
        default_factory=ClientCapabilities, description="Client hardware/software capabilities"
    )


class HandshakeResponse(BaseModel):
    """Handshake response from Desktop coordinator establishing session & limits."""
    session_id: str = Field(..., description="Unique session ID assigned to the client")
    server_version: str = Field(default="1.0.0", description="Server build version")
    protocol_version: str = Field(default="1.0.0", description="Protocol version supported")
    status: str = Field(default="ready", description="Server operational status ('ready', 'busy', 'degraded')")
    supported_iterations: List[int] = Field(
        default_factory=lambda: [1000, 2000, 3000, 7000, 15000, 30000],
        description="List of valid training iteration milestones"
    )
    default_iterations: int = Field(default=2000, description="Default iteration count")
    max_resolution: str = Field(default="1080p", description="Maximum supported training resolution")
    supported_resolutions: List[str] = Field(
        default_factory=lambda: ["720p", "1080p", "1440p", "4k"],
        description="Supported camera resolutions"
    )
    prune_floaters_supported: bool = Field(default=True, description="Whether floater pruning hook is available")
    sh_degrees_supported: List[int] = Field(
        default_factory=lambda: [0, 1, 2, 3],
        description="Supported spherical harmonics degrees"
    )
    cuda_available: bool = Field(default=True, description="Whether CUDA GPU acceleration is online")
    gpu_device_name: Optional[str] = Field(default=None, description="GPU model name (e.g. RTX 3050)")
    active_jobs: int = Field(default=0, description="Current number of active jobs in queue/training")
    message: str = Field(default="Handshake successful. Ready for job coordination.", description="Status message")


class JobConfig(BaseModel):
    """Full execution configuration for a training job."""
    iterations: int = Field(default=2000, ge=50, le=30000, description="Training iterations (50-30000)")
    resolution: str = Field(default="1080p", description="Target resolution ('720p', '1080p', '1440p', '4k')")
    resolution_scale: int = Field(default=2, ge=1, le=8, description="Downscale factor (1=full, 2=half, 4=quarter)")
    prune_floaters: bool = Field(default=True, description="Whether to apply post-training floater pruning")
    min_opacity: float = Field(default=0.04, ge=0.001, le=0.5, description="Floater pruning minimum opacity")
    max_scale: float = Field(default=0.15, ge=0.01, le=1.0, description="Floater pruning maximum scale threshold")
    sh_degree: int = Field(default=3, ge=0, le=3, description="Spherical harmonics degree (0-3)")
    white_background: bool = Field(default=False, description="Train with white background")
    fast_test: bool = Field(default=False, description="Fast test mode bypassing full CUDA run")


class NegotiationRequest(BaseModel):
    """Client-proposed training parameters for negotiation."""
    session_id: Optional[str] = Field(default=None, description="Session ID from handshake")
    client_id: Optional[str] = Field(default=None, description="Client ID")
    iterations: Optional[int] = Field(default=None, description="Requested iteration count")
    resolution: Optional[str] = Field(default=None, description="Requested resolution")
    resolution_scale: Optional[int] = Field(default=None, description="Requested resolution downscale factor")
    prune_floaters: Optional[bool] = Field(default=None, description="Enable/disable floater pruning")
    min_opacity: Optional[float] = Field(default=None, description="Custom min opacity threshold")
    max_scale: Optional[float] = Field(default=None, description="Custom max scale threshold")
    sh_degree: Optional[int] = Field(default=None, description="Custom SH degree")
    white_background: Optional[bool] = Field(default=None, description="White background flag")
    fast_test: Optional[bool] = Field(default=None, description="Fast test execution flag")


class NegotiationResponse(BaseModel):
    """Server-accepted parameters and adjustments after negotiation."""
    accepted: bool = Field(default=True, description="Whether the negotiated config is accepted")
    session_id: Optional[str] = Field(default=None, description="Associated session ID")
    config: JobConfig = Field(..., description="Final resolved job configuration")
    negotiated_params: Dict[str, Any] = Field(default_factory=dict, description="Dictionary of resolved parameters")
    reasons: List[str] = Field(default_factory=list, description="Explanations for any clamped or adjusted values")
    message: str = Field(default="Parameters negotiated successfully", description="Status message")


class TrainingJobStatus(BaseModel):
    """Rich training job status matching both Android ApiClient and coordinator state."""
    id: str = Field(..., description="Unique scan / job ID")
    status: str = Field(default="pending", description="Legacy status string for Android ApiClient compatibility")
    stage: JobStage = Field(default=JobStage.QUEUED, description="9-stage state machine stage")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalized progress [0.0 - 1.0]")
    iteration: int = Field(default=0, ge=0, description="Current training iteration")
    total_iterations: int = Field(default=2000, ge=1, description="Total target iterations")
    message: str = Field(default="", description="Human-readable status or diagnostic message")
    splat_url: Optional[str] = Field(default=None, description="Relative URL to download completed .splat file")
    num_gaussians: Optional[int] = Field(default=None, description="Current count of 3D Gaussian splats")
    loss: Optional[float] = Field(default=None, description="Current photometric / L1+D-SSIM training loss")
    eta_seconds: Optional[float] = Field(default=None, description="Estimated remaining seconds until completion")
    created_at: float = Field(default_factory=time.time, description="Timestamp job was created")
    updated_at: float = Field(default_factory=time.time, description="Timestamp of last status change")
    error: Optional[str] = Field(default=None, description="Error detail if stage is FAILED")
    config: Optional[JobConfig] = Field(default=None, description="Active job configuration")


class TelemetryFrame(BaseModel):
    """Real-time streaming telemetry frame pushed over WebSocket."""
    scan_id: str = Field(..., description="Job / scan ID")
    stage: JobStage = Field(default=JobStage.TRAINING, description="Current stage")
    status: str = Field(default="training", description="Status label")
    iteration: int = Field(default=0, description="Current training iteration")
    total_iterations: int = Field(default=2000, description="Target iterations")
    progress: float = Field(default=0.0, description="Normalized progress [0.0 - 1.0]")
    loss: Optional[float] = Field(default=None, description="Latest training loss")
    eta_seconds: Optional[float] = Field(default=None, description="Estimated seconds remaining")
    num_gaussians: Optional[int] = Field(default=None, description="Total Gaussian count")
    vram_allocated_mb: Optional[float] = Field(default=None, description="GPU VRAM allocated in MB")
    fps: Optional[float] = Field(default=None, description="Current training step iterations/sec")
    message: str = Field(default="", description="Status detail message")
    timestamp: float = Field(default_factory=time.time, description="Epoch timestamp")


class ModelInfo(BaseModel):
    """Metadata for a ready .splat model in storage."""
    name: str = Field(..., description="Model stem name")
    filename: str = Field(..., description="Full filename with .splat extension")
    size_mb: float = Field(..., description="File size in megabytes")
    download_url: str = Field(..., description="Relative download path")


class ModelListResponse(BaseModel):
    """List of available trained models."""
    models: List[ModelInfo] = Field(default_factory=list)


class UploadResponse(BaseModel):
    """Response returned upon accepting a dataset upload."""
    scan_id: str
    status: str = "QUEUED"
    stage: JobStage = JobStage.QUEUED
    message: str = "Upload accepted and training queued"
    session_id: Optional[str] = None


class CancelResponse(BaseModel):
    """Response returned upon cancelling a job."""
    scan_id: str
    cancelled: bool = True
    message: str = "Job cancelled successfully"
