"""
Task Coordinator and State Machine for Android-to-Laptop 3DGS System.
Milestone 1: Inter-Agent Communication & Task Handshake Protocol.

Manages:
- 9-stage job lifecycle state machine
- Active client sessions and parameter negotiation
- Concurrent WebSocket telemetry broadcasting
- Thread-safe job state updates and transition history
- Subprocess lifecycle and job cancellation
"""

import asyncio
import logging
import math
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import WebSocket
import torch

from protocol import (
    ClientCapabilities,
    HandshakeRequest,
    HandshakeResponse,
    JobConfig,
    JobStage,
    NegotiationRequest,
    NegotiationResponse,
    TelemetryFrame,
    TrainingJobStatus,
)

logger = logging.getLogger("coordinator")
logger.setLevel(logging.INFO)


# Valid state machine transitions
ALLOWED_TRANSITIONS: Dict[JobStage, Set[JobStage]] = {
    JobStage.IDLE: {JobStage.UPLOADING, JobStage.INGESTING, JobStage.QUEUED, JobStage.FAILED},
    JobStage.UPLOADING: {JobStage.INGESTING, JobStage.QUEUED, JobStage.FAILED},
    JobStage.INGESTING: {JobStage.QUEUED, JobStage.TRAINING, JobStage.FAILED},
    JobStage.QUEUED: {JobStage.INGESTING, JobStage.TRAINING, JobStage.FAILED},
    JobStage.TRAINING: {JobStage.PRUNING, JobStage.CONVERTING, JobStage.COMPLETED, JobStage.FAILED},
    JobStage.PRUNING: {JobStage.CONVERTING, JobStage.COMPLETED, JobStage.FAILED},
    JobStage.CONVERTING: {JobStage.COMPLETED, JobStage.FAILED},
    JobStage.COMPLETED: set(),  # Terminal
    JobStage.FAILED: set(),     # Terminal
}


def stage_to_legacy_status(stage: JobStage) -> str:
    """Map rich 9-stage enum to legacy status string for Android ApiClient."""
    mapping = {
        JobStage.IDLE: "pending",
        JobStage.UPLOADING: "pending",
        JobStage.INGESTING: "pending",
        JobStage.QUEUED: "pending",
        JobStage.TRAINING: "training",
        JobStage.PRUNING: "training",
        JobStage.CONVERTING: "converting",
        JobStage.COMPLETED: "completed",
        JobStage.FAILED: "failed",
    }
    return mapping.get(stage, "pending")


@dataclass
class SessionInfo:
    """Active client session metadata."""
    session_id: str
    client_id: str
    device_model: str
    app_version: str
    capabilities: ClientCapabilities
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    negotiated_config: Optional[JobConfig] = None


@dataclass
class JobRecord:
    """Internal coordinator record for a training job."""
    scan_id: str
    status: TrainingJobStatus
    config: JobConfig
    session_id: Optional[str] = None
    process: Optional[subprocess.Popen] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    history: List[Dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


class TelemetryHub:
    """Manages concurrent WebSocket connections and broadcast streaming per job."""

    def __init__(self):
        self._connections: Dict[str, Set[WebSocket]] = {}
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the active asyncio event loop for thread-safe cross-thread calls."""
        self._loop = loop

    async def connect(self, scan_id: str, websocket: WebSocket):
        """Register a new WebSocket subscriber for a scan_id."""
        with self._lock:
            if scan_id not in self._connections:
                self._connections[scan_id] = set()
            self._connections[scan_id].add(websocket)
        logger.info(f"[TelemetryHub] Client connected to scan {scan_id}. Total: {len(self._connections[scan_id])}")

    async def disconnect(self, scan_id: str, websocket: WebSocket):
        """Unregister a WebSocket subscriber."""
        with self._lock:
            if scan_id in self._connections:
                self._connections[scan_id].discard(websocket)
                if not self._connections[scan_id]:
                    del self._connections[scan_id]
        logger.info(f"[TelemetryHub] Client disconnected from scan {scan_id}")

    def get_subscriber_count(self, scan_id: str) -> int:
        """Get the count of active WebSocket subscribers for a job."""
        with self._lock:
            return len(self._connections.get(scan_id, set()))

    async def broadcast(self, scan_id: str, frame: TelemetryFrame):
        """Broadcast a telemetry frame to all subscribers of scan_id asynchronously."""
        with self._lock:
            sockets = list(self._connections.get(scan_id, set()))

        if not sockets:
            return

        payload = frame.model_dump_json()
        dead_sockets = []

        for ws in sockets:
            try:
                await ws.send_text(payload)
            except Exception as e:
                logger.debug(f"[TelemetryHub] Failed to send to websocket: {e}")
                dead_sockets.append(ws)

        if dead_sockets:
            with self._lock:
                if scan_id in self._connections:
                    for ws in dead_sockets:
                        self._connections[scan_id].discard(ws)

    def broadcast_threadsafe(self, scan_id: str, frame: TelemetryFrame):
        """Thread-safe telemetry broadcast from background worker threads."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(scan_id, frame), self._loop)
        else:
            # Fallback when running in synchronous scripts/tests without active event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(self.broadcast(scan_id, frame), loop)
                else:
                    loop.run_until_complete(self.broadcast(scan_id, frame))
            except Exception:
                pass


class TrainingCoordinator:
    """Central state machine and orchestrator for 3DGS training jobs."""

    def __init__(self):
        self._jobs: Dict[str, JobRecord] = {}
        self._sessions: Dict[str, SessionInfo] = {}
        self._global_lock = threading.RLock()
        self.telemetry_hub = TelemetryHub()
        self._server_version = "1.0.0"
        self._protocol_version = "1.0.0"

    # --- Session Management ---

    def register_session(self, request: HandshakeRequest) -> HandshakeResponse:
        """Create or update a client session and return initial capabilities."""
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        caps = request.capabilities or ClientCapabilities()

        session = SessionInfo(
            session_id=session_id,
            client_id=request.client_id,
            device_model=request.device_model,
            app_version=request.app_version,
            capabilities=caps,
        )

        with self._global_lock:
            self._sessions[session_id] = session

        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else None

        active_jobs = self.get_active_job_count()

        return HandshakeResponse(
            session_id=session_id,
            server_version=self._server_version,
            protocol_version=self._protocol_version,
            status="ready" if active_jobs < 4 else "busy",
            supported_iterations=[1000, 2000, 3000, 7000, 15000, 30000],
            default_iterations=2000,
            max_resolution="1080p",
            supported_resolutions=["720p", "1080p", "1440p", "4k"],
            prune_floaters_supported=True,
            sh_degrees_supported=[0, 1, 2, 3],
            cuda_available=cuda_avail,
            gpu_device_name=gpu_name,
            active_jobs=active_jobs,
            message=f"Session established on {gpu_name or 'CPU'}. Ready for training coordination.",
        )

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Retrieve session info by ID."""
        with self._global_lock:
            return self._sessions.get(session_id)

    def negotiate_parameters(self, request: NegotiationRequest) -> NegotiationResponse:
        """Evaluate, clamp, and store negotiated parameters."""
        reasons = []

        # Start with defaults
        resolved = JobConfig()

        # Iterations negotiation
        if request.iterations is not None:
            if request.iterations < 50:
                resolved.iterations = 50
                reasons.append(f"Clamped iterations {request.iterations} to minimum 50")
            elif request.iterations > 30000:
                resolved.iterations = 30000
                reasons.append(f"Clamped iterations {request.iterations} to maximum 30000")
            else:
                resolved.iterations = request.iterations

        # Resolution negotiation
        valid_resolutions = {"720p", "1080p", "1440p", "4k"}
        if request.resolution:
            res_str = request.resolution.lower()
            if res_str in valid_resolutions:
                resolved.resolution = res_str
            else:
                resolved.resolution = "1080p"
                reasons.append(f"Unsupported resolution '{request.resolution}', defaulted to 1080p")

        # Resolution scale
        if request.resolution_scale is not None:
            if request.resolution_scale in (1, 2, 4, 8):
                resolved.resolution_scale = request.resolution_scale
            else:
                resolved.resolution_scale = 2
                reasons.append("Resolution scale must be 1, 2, 4, or 8. Defaulted to 2")

        # Floater pruning
        if request.prune_floaters is not None:
            resolved.prune_floaters = request.prune_floaters

        if request.min_opacity is not None:
            resolved.min_opacity = max(0.001, min(0.5, request.min_opacity))

        if request.max_scale is not None:
            resolved.max_scale = max(0.01, min(1.0, request.max_scale))

        # Spherical harmonics degree
        if request.sh_degree is not None:
            resolved.sh_degree = max(0, min(3, request.sh_degree))

        if request.white_background is not None:
            resolved.white_background = request.white_background

        if request.fast_test is not None:
            resolved.fast_test = request.fast_test

        # Associate with session if available
        if request.session_id:
            with self._global_lock:
                sess = self._sessions.get(request.session_id)
                if sess:
                    sess.negotiated_config = resolved
                    sess.last_seen = time.time()
                else:
                    reasons.append(f"Session '{request.session_id}' not found; parameters returned unattached")

        return NegotiationResponse(
            accepted=True,
            session_id=request.session_id,
            config=resolved,
            negotiated_params=resolved.model_dump(),
            reasons=reasons,
            message="Parameters negotiated successfully." if not reasons else "; ".join(reasons),
        )

    # --- Job Lifecycle & State Machine ---

    def create_job(
        self,
        scan_id: str,
        config: Optional[JobConfig] = None,
        session_id: Optional[str] = None,
        initial_stage: JobStage = JobStage.QUEUED,
        initial_message: str = "Job created and queued",
    ) -> TrainingJobStatus:
        """Initialize a new job in the coordinator state machine."""
        now = time.time()
        final_config = config or JobConfig()

        # If session has pre-negotiated config and none was explicitly passed, inherit it
        if session_id and config is None:
            with self._global_lock:
                sess = self._sessions.get(session_id)
                if sess and sess.negotiated_config:
                    final_config = sess.negotiated_config

        status = TrainingJobStatus(
            id=scan_id,
            status=stage_to_legacy_status(initial_stage),
            stage=initial_stage,
            progress=0.0,
            iteration=0,
            total_iterations=final_config.iterations,
            message=initial_message,
            created_at=now,
            updated_at=now,
            config=final_config,
        )

        record = JobRecord(
            scan_id=scan_id,
            status=status,
            config=final_config,
            session_id=session_id,
            history=[{"timestamp": now, "stage": initial_stage.value, "message": initial_message}],
        )

        with self._global_lock:
            self._jobs[scan_id] = record

        # Broadcast initial state snapshot
        frame = self._build_telemetry_frame(record)
        self.telemetry_hub.broadcast_threadsafe(scan_id, frame)

        logger.info(f"[Coordinator] Job '{scan_id}' created in stage {initial_stage.value}")
        return status

    def get_job_status(self, scan_id: str) -> Optional[TrainingJobStatus]:
        """Return a copy of the current public status for a job."""
        with self._global_lock:
            record = self._jobs.get(scan_id)
            if not record:
                return None
            with record.lock:
                return record.status.model_copy()

    def get_job_record(self, scan_id: str) -> Optional[JobRecord]:
        """Return the internal job record (for runner/process control)."""
        with self._global_lock:
            return self._jobs.get(scan_id)

    def list_jobs(self) -> List[TrainingJobStatus]:
        """List current status of all managed jobs."""
        with self._global_lock:
            return [rec.status.model_copy() for rec in self._jobs.values()]

    def get_active_job_count(self) -> int:
        """Count jobs currently in non-terminal active stages."""
        with self._global_lock:
            return sum(1 for rec in self._jobs.values() if rec.status.stage.is_active)

    def get_status_history(self, scan_id: str) -> List[Dict[str, Any]]:
        """Retrieve state transition history for diagnostic audits."""
        with self._global_lock:
            record = self._jobs.get(scan_id)
            if not record:
                return []
            with record.lock:
                return list(record.history)

    def transition_stage(
        self,
        scan_id: str,
        target_stage: JobStage,
        message: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        """
        Transition job to target stage adhering to the 9-stage state machine rules.
        Returns True if transition succeeded.
        """
        with self._global_lock:
            record = self._jobs.get(scan_id)
            if not record:
                logger.error(f"[Coordinator] Cannot transition non-existent job '{scan_id}'")
                return False

        with record.lock:
            current_stage = record.status.stage

            # Always allow transitioning to FAILED (abort/cancel/error)
            if target_stage == JobStage.FAILED:
                pass
            elif not force and target_stage not in ALLOWED_TRANSITIONS.get(current_stage, set()):
                logger.warning(
                    f"[Coordinator] Invalid transition for '{scan_id}': {current_stage.value} -> {target_stage.value}"
                )
                return False

            now = time.time()
            record.status.stage = target_stage
            record.status.status = stage_to_legacy_status(target_stage)
            record.status.updated_at = now
            if message:
                record.status.message = message

            record.history.append({
                "timestamp": now,
                "stage": target_stage.value,
                "message": message or f"Transitioned to {target_stage.value}",
            })

            frame = self._build_telemetry_frame(record)

        self.telemetry_hub.broadcast_threadsafe(scan_id, frame)
        logger.info(f"[Coordinator] Job '{scan_id}' transitioned to {target_stage.value}: {message}")
        return True

    def update_job_progress(
        self,
        scan_id: str,
        iteration: Optional[int] = None,
        progress: Optional[float] = None,
        loss: Optional[float] = None,
        eta_seconds: Optional[float] = None,
        num_gaussians: Optional[int] = None,
        vram_allocated_mb: Optional[float] = None,
        fps: Optional[float] = None,
        message: Optional[str] = None,
    ):
        """Update metrics for a running job and push telemetry frame."""
        with self._global_lock:
            record = self._jobs.get(scan_id)
            if not record:
                return

        with record.lock:
            now = time.time()
            record.status.updated_at = now

            if iteration is not None:
                record.status.iteration = iteration
                if record.status.total_iterations > 0 and progress is None:
                    record.status.progress = min(1.0, max(0.0, iteration / record.status.total_iterations))

            if progress is not None:
                record.status.progress = min(1.0, max(0.0, progress))

            if loss is not None:
                record.status.loss = loss

            if eta_seconds is not None:
                record.status.eta_seconds = eta_seconds

            if num_gaussians is not None:
                record.status.num_gaussians = num_gaussians

            if message is not None:
                record.status.message = message

            frame = self._build_telemetry_frame(
                record,
                vram_allocated_mb=vram_allocated_mb,
                fps=fps,
            )

        self.telemetry_hub.broadcast_threadsafe(scan_id, frame)

    def mark_completed(self, scan_id: str, splat_filename: str, num_gaussians: Optional[int] = None):
        """Mark job successfully completed with output .splat download URL."""
        with self._global_lock:
            record = self._jobs.get(scan_id)
            if not record:
                return

        with record.lock:
            now = time.time()
            record.status.stage = JobStage.COMPLETED
            record.status.status = "completed"
            record.status.progress = 1.0
            record.status.iteration = record.status.total_iterations
            record.status.splat_url = f"/download/{splat_filename}"
            if num_gaussians is not None:
                record.status.num_gaussians = num_gaussians
            record.status.message = f"Training completed successfully! Ready to download {splat_filename}."
            record.status.updated_at = now

            record.history.append({
                "timestamp": now,
                "stage": JobStage.COMPLETED.value,
                "message": record.status.message,
                "splat_url": record.status.splat_url,
            })

            frame = self._build_telemetry_frame(record)

        self.telemetry_hub.broadcast_threadsafe(scan_id, frame)
        logger.info(f"[Coordinator] Job '{scan_id}' COMPLETED: {record.status.splat_url}")

    def mark_failed(self, scan_id: str, error_message: str):
        """Mark job failed with structured error message."""
        with self._global_lock:
            record = self._jobs.get(scan_id)
            if not record:
                return

        with record.lock:
            now = time.time()
            record.status.stage = JobStage.FAILED
            record.status.status = "failed"
            record.status.error = error_message
            record.status.message = f"Failed: {error_message}"
            record.status.updated_at = now

            record.history.append({
                "timestamp": now,
                "stage": JobStage.FAILED.value,
                "error": error_message,
            })

            frame = self._build_telemetry_frame(record)

        self.telemetry_hub.broadcast_threadsafe(scan_id, frame)
        logger.error(f"[Coordinator] Job '{scan_id}' FAILED: {error_message}")

    def cancel_job(self, scan_id: str) -> bool:
        """
        Cancel an active job: signal cancel event, terminate subprocess, and mark FAILED.
        Returns True if job was cancelled, False if job not found or already terminal.
        """
        with self._global_lock:
            record = self._jobs.get(scan_id)
            if not record:
                return False

        with record.lock:
            if record.status.stage.is_terminal:
                logger.info(f"[Coordinator] Job '{scan_id}' is already terminal ({record.status.stage.value}); cannot cancel")
                return False

            record.cancel_event.set()

            # Terminate running subprocess if attached
            if record.process and record.process.poll() is None:
                try:
                    logger.info(f"[Coordinator] Terminating training subprocess for '{scan_id}' (PID: {record.process.pid})")
                    record.process.terminate()
                    try:
                        record.process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"[Coordinator] Subprocess for '{scan_id}' did not terminate cleanly. Killing...")
                        record.process.kill()
                except Exception as e:
                    logger.error(f"[Coordinator] Error terminating process for '{scan_id}': {e}")

            now = time.time()
            record.status.stage = JobStage.FAILED
            record.status.status = "failed"
            record.status.message = "Job cancelled by client."
            record.status.error = "CancelledByUser"
            record.status.updated_at = now

            record.history.append({
                "timestamp": now,
                "stage": JobStage.FAILED.value,
                "message": "Job cancelled by client",
            })

            frame = self._build_telemetry_frame(record)

        self.telemetry_hub.broadcast_threadsafe(scan_id, frame)
        logger.info(f"[Coordinator] Job '{scan_id}' cancelled successfully")
        return True

    def register_process(self, scan_id: str, process: subprocess.Popen):
        """Associate a running subprocess with a job for cancellation control."""
        with self._global_lock:
            record = self._jobs.get(scan_id)
            if record:
                with record.lock:
                    record.process = process

    def _build_telemetry_frame(
        self,
        record: JobRecord,
        vram_allocated_mb: Optional[float] = None,
        fps: Optional[float] = None,
    ) -> TelemetryFrame:
        """Construct a TelemetryFrame from a JobRecord."""
        stat = record.status
        vram = vram_allocated_mb
        if vram is None and torch.cuda.is_available():
            try:
                vram = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
            except Exception:
                pass

        return TelemetryFrame(
            scan_id=stat.id,
            stage=stat.stage,
            status=stat.status,
            iteration=stat.iteration,
            total_iterations=stat.total_iterations,
            progress=stat.progress,
            loss=stat.loss,
            eta_seconds=stat.eta_seconds,
            num_gaussians=stat.num_gaussians,
            vram_allocated_mb=vram,
            fps=fps,
            message=stat.message,
            timestamp=time.time(),
        )


# Global singleton instance
coordinator = TrainingCoordinator()
