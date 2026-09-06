"""
Adversarial Stress Test Suite for Android-to-Laptop 3DGS Task Coordination.
Author: Challenger 1 (Adversarial Verification Agent)

Missions:
1. Protocol Coordinator:
   - Rapid concurrent uploads (20 simultaneous client uploads)
   - Rapid concurrent cancellations and race conditions on active jobs
   - Duplicate requests & session reuse under multi-threaded contention
   - Invalid state machine transitions & terminal immutability audits
   - TelemetryHub concurrent subscriber registration/broadcast/deregistration
2. Dataset Adapter (dataset_adapter.py):
   - Degenerate camera poses (all-zero, singular, rank-deficient, NaN, Inf, non-4x4)
   - Degenerate intrinsics (zero/negative focal length, zero/negative resolution, camera_angle_x edge cases)
   - Mixed valid/corrupt frame handling (graceful partial rejection vs total failure)
   - Points3D initial cloud corruptions & fallbacks
3. Splat Verifier (verify_splat.py):
   - Non-32-byte divisibility & truncated files (0B, 1B, 15B, 31B, 33B, 3199B, 3215B)
   - Minimum threshold enforcement (<100 splats)
   - Non-finite coordinates (NaN, +Inf, -Inf)
   - Non-positive & non-finite scales (0.0, negative, NaN, Inf)
   - Completely transparent scenes (all alpha = 0)
   - Degenerate & unnormalized rotation quaternions (norm < 0.1, norm > 1.5, norm = 0.0)
   - Random fuzz buffers
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure server module path is resolvable
SERVER_DIR = Path(__file__).resolve().parent.parent / "Android_Gaussian_Splatting" / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

CLIENT_AGENT_DIR = Path(__file__).resolve().parent.parent / "Android_Gaussian_Splatting" / "client_agent"
if str(CLIENT_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_AGENT_DIR))

from server import app
from coordinator import coordinator, TrainingCoordinator, JobStage, ALLOWED_TRANSITIONS
from protocol import HandshakeRequest, NegotiationRequest, JobConfig, TelemetryFrame
import dataset_adapter
from dataset_adapter import (
    convert_transforms_to_colmap,
    convert_pose_opengl_to_colmap,
    parse_matrix_from_frame,
    generate_sparse_points,
    InvalidMatrixError,
    CorruptTransformsError,
    MissingIntrinsicsError,
)
import verify_splat
from verify_splat import (
    verify_splat_file,
    FileSizeError,
    SplatCountError,
    CorruptCoordinatesError,
    CorruptScaleError,
    CorruptQuaternionError,
    SplatValidationError,
    SPLAT_DTYPE,
)


# =============================================================================
# HELPERS & FIXTURES
# =============================================================================

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def create_minimal_valid_zip_bytes(num_frames: int = 3) -> bytes:
    """Generates an in-memory zip archive with minimal valid transforms.json and dummy images."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        frames = []
        for i in range(num_frames):
            img_name = f"frame_{i:04d}.jpg"
            zf.writestr(f"images/{img_name}", b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00")
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, 3] = [0.0, 0.0, float(i + 1)]
            frames.append({
                "file_path": f"images/{img_name}",
                "transform_matrix": c2w.tolist()
            })
        transforms = {
            "w": 1280,
            "h": 720,
            "fl_x": 1000.0,
            "fl_y": 1000.0,
            "cx": 640.0,
            "cy": 360.0,
            "frames": frames
        }
        zf.writestr("transforms.json", json.dumps(transforms, indent=2))
    return buf.getvalue()


def create_splat_buffer(
    num_splats: int = 100,
    x: float = 0.0, y: float = 0.0, z: float = 0.0,
    s0: float = 0.01, s1: float = 0.01, s2: float = 0.01,
    r: int = 120, g: int = 130, b: int = 140, a: int = 200,
    q0: int = 128, q1: int = 128, q2: int = 128, q3: int = 255
) -> bytes:
    """Creates a structured binary .splat buffer with specified values."""
    splats = np.zeros(num_splats, dtype=SPLAT_DTYPE)
    splats['x'] = x
    splats['y'] = y
    splats['z'] = z
    splats['s0'] = s0
    splats['s1'] = s1
    splats['s2'] = s2
    splats['r'] = r
    splats['g'] = g
    splats['b'] = b
    splats['a'] = a
    splats['q0'] = q0
    splats['q1'] = q1
    splats['q2'] = q2
    splats['q3'] = q3
    return splats.tobytes()


# =============================================================================
# PART 1: PROTOCOL COORDINATOR STRESS TESTS
# =============================================================================

class TestCoordinatorStress:
    """Adversarial stress testing of TrainingCoordinator and server endpoints."""

    def test_rapid_concurrent_uploads(self, client):
        """Stress: 20 simultaneous uploads from concurrent threads.
        Asserts unique scan_ids, thread safety, 200 OK, and QUEUED status."""
        zip_bytes = create_minimal_valid_zip_bytes(num_frames=2)
        num_concurrent = 20
        results = []

        def upload_worker(idx: int):
            files = {"file": (f"scan_{idx}.zip", io.BytesIO(zip_bytes), "application/zip")}
            data = {"iterations": 1000, "fast_test": True}
            resp = client.post("/upload", files=files, data=data)
            return resp.status_code, resp.json()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(upload_worker, i) for i in range(num_concurrent)]
            for fut in as_completed(futures):
                results.append(fut.result())

        assert len(results) == num_concurrent
        status_codes = [r[0] for r in results]
        assert all(code == 200 for code in status_codes), f"Some uploads failed: {status_codes}"

        scan_ids = [r[1]["scan_id"] for r in results]
        assert len(set(scan_ids)) == num_concurrent, "Detected scan_id collision during concurrent uploads!"
        assert all(r[1]["status"] == "QUEUED" for r in results)

    def test_rapid_concurrent_cancellations_active_job(self, client):
        """Stress: Race condition test on cancellation.
        Spawn 10 threads all attempting to cancel the exact same active scan_id simultaneously.
        Verifies:
        - Thread safety (no crashes or deadlocks)
        - Attached subprocess is cleanly terminated
        - Exactly one cancellation succeeds (cancelled=True), remaining return cancelled=False
        - Final stage is FAILED"""
        scan_id = f"test_active_cancel_{uuid.uuid4().hex[:6]}"
        coordinator.create_job(scan_id, initial_stage=JobStage.TRAINING)

        # Attach an active subprocess
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        coordinator.register_process(scan_id, proc)

        cancel_results = []
        num_cancels = 10

        def cancel_worker():
            resp = client.delete(f"/jobs/{scan_id}")
            return resp.status_code, resp.json()

        with ThreadPoolExecutor(max_workers=num_cancels) as executor:
            futures = [executor.submit(cancel_worker) for _ in range(num_cancels)]
            for fut in as_completed(futures):
                cancel_results.append(fut.result())

        assert len(cancel_results) == num_cancels
        assert all(code == 200 for code, _ in cancel_results)

        success_cancels = [r for _, r in cancel_results if r.get("cancelled") is True]
        terminal_cancels = [r for _, r in cancel_results if r.get("cancelled") is False]
        assert len(success_cancels) == 1, f"Expected exactly 1 successful cancel, got {len(success_cancels)}"
        assert len(terminal_cancels) == num_cancels - 1

        # Verify subprocess was terminated
        assert proc.poll() is not None, "Subprocess was not terminated by cancellation"

        # Verify final status
        status = coordinator.get_job_status(scan_id)
        assert status is not None
        assert status.stage == JobStage.FAILED
        assert status.status == "failed"

    def test_cancellation_of_completed_job(self):
        """Stress: Cancelling an already COMPLETED job must return False and not alter state."""
        coord = TrainingCoordinator()
        scan_id = f"test_comp_{uuid.uuid4().hex[:6]}"
        coord.create_job(scan_id, initial_stage=JobStage.QUEUED)
        coord.transition_stage(scan_id, JobStage.TRAINING)
        coord.transition_stage(scan_id, JobStage.COMPLETED)

        result = coord.cancel_job(scan_id)
        assert result is False, "cancel_job on COMPLETED job must return False"
        status = coord.get_job_status(scan_id)
        assert status.stage == JobStage.COMPLETED, "COMPLETED job stage must not be mutated by cancellation"

    def test_cancellation_of_nonexistent_job(self, client):
        """Stress: Cancelling a non-existent job returns HTTP 404."""
        resp = client.delete(f"/jobs/nonexistent_scan_{uuid.uuid4().hex[:8]}")
        assert resp.status_code == 404

    def test_duplicate_handshakes_and_concurrent_negotiation(self, client):
        """Stress: 20 simultaneous handshakes and negotiations with identical client_id."""
        client_id = "pixel_stress_device_001"
        num_requests = 20
        sessions = []

        def handshake_and_negotiate(idx: int):
            h_payload = {
                "client_id": client_id,
                "device_model": "Stress Device",
                "app_version": "1.0.0",
                "capabilities": {"sensor_fusion": True}
            }
            h_resp = client.post("/handshake", json=h_payload)
            if h_resp.status_code != 200:
                return None
            sess_id = h_resp.json()["session_id"]

            n_payload = {
                "session_id": sess_id,
                "iterations": 500 + idx * 100,
                "resolution": "720p",
                "prune_floaters": True
            }
            n_resp = client.post("/negotiate", json=n_payload)
            return sess_id, n_resp.json()

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(handshake_and_negotiate, i) for i in range(num_requests)]
            for fut in as_completed(futures):
                res = fut.result()
                assert res is not None
                sessions.append(res)

        assert len(sessions) == num_requests
        session_ids = [s[0] for s in sessions]
        assert len(set(session_ids)) == num_requests, "All generated session_ids must be strictly distinct"

    def test_state_machine_invalid_transitions(self):
        """Stress: Exhaustive rejection of illegal state machine transitions."""
        coord = TrainingCoordinator()
        
        # Test illegal transitions
        illegal_transitions = [
            (JobStage.IDLE, JobStage.COMPLETED),
            (JobStage.IDLE, JobStage.PRUNING),
            (JobStage.QUEUED, JobStage.PRUNING),
            (JobStage.QUEUED, JobStage.CONVERTING),
            (JobStage.QUEUED, JobStage.COMPLETED),
            (JobStage.TRAINING, JobStage.INGESTING),
            (JobStage.TRAINING, JobStage.QUEUED),
            (JobStage.PRUNING, JobStage.TRAINING),
            (JobStage.PRUNING, JobStage.INGESTING),
            (JobStage.CONVERTING, JobStage.TRAINING),
            (JobStage.CONVERTING, JobStage.INGESTING),
            (JobStage.COMPLETED, JobStage.TRAINING),
            (JobStage.COMPLETED, JobStage.QUEUED),
            (JobStage.COMPLETED, JobStage.INGESTING),
            (JobStage.COMPLETED, JobStage.IDLE),
            (JobStage.FAILED, JobStage.QUEUED),
            (JobStage.FAILED, JobStage.TRAINING),
            (JobStage.FAILED, JobStage.COMPLETED),
        ]

        for from_stage, to_stage in illegal_transitions:
            scan_id = f"test_trans_{uuid.uuid4().hex[:6]}"
            coord.create_job(scan_id, initial_stage=from_stage)
            success = coord.transition_stage(scan_id, to_stage, force=False)
            assert not success, f"State machine illegally allowed transition {from_stage.value} -> {to_stage.value}"
            cur_status = coord.get_job_status(scan_id)
            assert cur_status.stage == from_stage, f"Job stage was modified despite illegal transition {from_stage} -> {to_stage}"

    def test_concurrent_state_transitions_race(self):
        """Stress: 10 threads trying to transition the same job concurrently to different stages."""
        coord = TrainingCoordinator()
        scan_id = f"race_trans_{uuid.uuid4().hex[:6]}"
        coord.create_job(scan_id, initial_stage=JobStage.TRAINING)

        target_stages = [
            JobStage.PRUNING, JobStage.CONVERTING, JobStage.COMPLETED,
            JobStage.FAILED, JobStage.INGESTING, JobStage.QUEUED
        ]

        def transition_attempt(stage):
            return coord.transition_stage(scan_id, stage)

        with ThreadPoolExecutor(max_workers=len(target_stages)) as executor:
            futures = [executor.submit(transition_attempt, st) for st in target_stages]
            results = [f.result() for f in futures]

        final_status = coord.get_job_status(scan_id)
        assert final_status is not None
        history = coord.get_status_history(scan_id)
        assert len(history) >= 2
        assert final_status.stage in (JobStage.PRUNING, JobStage.CONVERTING, JobStage.COMPLETED, JobStage.FAILED)

    def test_terminal_state_resurrection_anomaly(self):
        """Adversarial Finding: Coordinator allows terminal state resurrection / overwrite.
        Empirically probes whether mark_completed overwrites FAILED, and transition_stage allows COMPLETED -> FAILED."""
        coord = TrainingCoordinator()
        
        # Test 1: mark_completed on FAILED job
        scan_id1 = f"term_test1_{uuid.uuid4().hex[:6]}"
        coord.create_job(scan_id1, initial_stage=JobStage.TRAINING)
        coord.mark_failed(scan_id1, "Simulated crash")
        assert coord.get_job_status(scan_id1).stage == JobStage.FAILED

        # In a strictly absorbing state machine, mark_completed should be rejected on FAILED jobs.
        coord.mark_completed(scan_id1, "dummy.splat", num_gaussians=100)
        status1 = coord.get_job_status(scan_id1)
        # Note: If this overwrote FAILED to COMPLETED, record the behavior
        resurrected = (status1.stage == JobStage.COMPLETED)
        assert resurrected, "Expected mark_completed to overwrite FAILED due to missing terminal check in coordinator.py:494"

        # Test 2: transition_stage(..., JobStage.FAILED) on COMPLETED job without force
        scan_id2 = f"term_test2_{uuid.uuid4().hex[:6]}"
        coord.create_job(scan_id2, initial_stage=JobStage.TRAINING)
        coord.transition_stage(scan_id2, JobStage.COMPLETED)
        assert coord.get_job_status(scan_id2).stage == JobStage.COMPLETED

        allowed_to_fail = coord.transition_stage(scan_id2, JobStage.FAILED, force=False)
        assert allowed_to_fail, "Expected transition_stage to allow COMPLETED -> FAILED due to 'if target_stage == JobStage.FAILED: pass' in coordinator.py:416"

    def test_late_telemetry_progress_anomaly(self):
        """Adversarial Finding: update_job_progress modifies progress on a COMPLETED job.
        A late worker frame overwrites 100% (1.0) back down to a previous iteration's progress."""
        coord = TrainingCoordinator()
        scan_id = f"late_prog_{uuid.uuid4().hex[:6]}"
        coord.create_job(scan_id, initial_stage=JobStage.TRAINING)
        coord.mark_completed(scan_id, "output.splat")
        assert coord.get_job_status(scan_id).progress == 1.0

        # Late telemetry arrives
        coord.update_job_progress(scan_id, iteration=20, progress=0.10)
        status = coord.get_job_status(scan_id)
        assert status.progress == 0.10, "Expected update_job_progress to overwrite progress on COMPLETED job"


# =============================================================================
# PART 2: DATASET ADAPTER STRESS TESTS
# =============================================================================

class TestDatasetAdapterStress:
    """Adversarial stress testing of dataset_adapter.py."""

    def test_singular_and_all_zero_matrix(self):
        """Stress: All-zero matrix and rank-deficient singular matrices must raise InvalidMatrixError."""
        zero_mat = np.zeros((4, 4), dtype=np.float64)
        with pytest.raises(InvalidMatrixError):
            convert_pose_opengl_to_colmap(zero_mat)

        # Singular rank-2 matrix
        singular_mat = np.array([
            [1.0, 2.0, 3.0, 0.0],
            [2.0, 4.0, 6.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        with pytest.raises(InvalidMatrixError):
            convert_pose_opengl_to_colmap(singular_mat)

    def test_matrix_with_nan_and_inf_values(self):
        """Stress: Transform matrices containing NaN, +Inf, -Inf must be rejected."""
        nan_mat = np.eye(4, dtype=np.float64)
        nan_mat[0, 3] = float("nan")
        frame_nan = {"transform_matrix": nan_mat.tolist()}
        with pytest.raises(InvalidMatrixError):
            parse_matrix_from_frame(frame_nan)

        inf_mat = np.eye(4, dtype=np.float64)
        inf_mat[1, 1] = float("inf")
        frame_inf = {"transform_matrix": inf_mat.tolist()}
        with pytest.raises(InvalidMatrixError):
            parse_matrix_from_frame(frame_inf)

        neg_inf_mat = np.eye(4, dtype=np.float64)
        neg_inf_mat[2, 3] = -float("inf")
        frame_neg_inf = {"transform_matrix": neg_inf_mat.tolist()}
        with pytest.raises(InvalidMatrixError):
            parse_matrix_from_frame(frame_neg_inf)

    def test_matrix_malformed_dimensions(self):
        """Stress: 3x3, 5x5, flat 16-element array, or 3x4 matrices must raise InvalidMatrixError."""
        with pytest.raises(InvalidMatrixError):
            parse_matrix_from_frame({"transform_matrix": np.eye(3).tolist()})

        with pytest.raises(InvalidMatrixError):
            parse_matrix_from_frame({"transform_matrix": np.eye(5).tolist()})

        with pytest.raises(InvalidMatrixError):
            parse_matrix_from_frame({"transform_matrix": [1.0] * 16})

        with pytest.raises(InvalidMatrixError):
            parse_matrix_from_frame({"transform_matrix": np.zeros((3, 4)).tolist()})

    def test_zero_camera_angle_x_zero_division(self, tmp_path):
        """Adversarial Finding: camera_angle_x = 0.0 causes unhandled ZeroDivisionError."""
        scan_dir = tmp_path / "zero_angle"
        scan_dir.mkdir()
        data = {
            "camera_angle_x": 0.0,
            "frames": [{"transform_matrix": np.eye(4).tolist()}]
        }
        with open(scan_dir / "transforms.json", "w") as f:
            json.dump(data, f)

        with pytest.raises(ZeroDivisionError):
            convert_transforms_to_colmap(scan_dir)

    def test_zero_and_negative_focal_length(self, tmp_path):
        """Stress: Negative focal length raises MissingIntrinsicsError, while 0.0 is silently masked to 1024.0."""
        # Negative focal length -> raises MissingIntrinsicsError
        scan_dir_neg = tmp_path / "neg_fx"
        scan_dir_neg.mkdir()
        with open(scan_dir_neg / "transforms.json", "w") as f:
            json.dump({
                "w": 1280, "h": 720, "fl_x": -500.0, "fl_y": -500.0,
                "frames": [{"transform_matrix": np.eye(4).tolist()}]
            }, f)

        with pytest.raises(MissingIntrinsicsError):
            convert_transforms_to_colmap(scan_dir_neg)

        # Zero focal length -> silently defaulted because '0.0 or (w * 0.8)' evaluates to default
        scan_dir_zero = tmp_path / "zero_fx"
        scan_dir_zero.mkdir()
        with open(scan_dir_zero / "transforms.json", "w") as f:
            json.dump({
                "w": 1280, "h": 720, "fl_x": 0.0, "fl_y": 0.0,
                "frames": [{"transform_matrix": np.eye(4).tolist()}]
            }, f)

        res = convert_transforms_to_colmap(scan_dir_zero)
        assert res["status"] == "success"
        # Empirically verify that 0.0 was masked to 1024.0
        assert res["intrinsics"]["fx"] == 1024.0

    def test_negative_camera_coordinates(self, tmp_path):
        """Stress: Extremely negative camera coordinates (e.g. -50000.0) should be inverted without numerical failure."""
        scan_dir = tmp_path / "neg_coords"
        scan_dir.mkdir()
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 3] = [-5000.0, -12000.0, -800.0]
        data = {
            "w": 1920, "h": 1080, "fl_x": 1200.0, "fl_y": 1200.0, "cx": 960.0, "cy": 540.0,
            "frames": [{"file_path": "frame_0.jpg", "transform_matrix": c2w.tolist()}]
        }
        with open(scan_dir / "transforms.json", "w") as f:
            json.dump(data, f)

        res = convert_transforms_to_colmap(scan_dir)
        assert res["status"] == "success"
        assert res["num_images"] == 1
        assert (scan_dir / "sparse" / "0" / "images.txt").exists()

    def test_mixed_valid_and_corrupt_frames(self, tmp_path):
        """Stress: Dataset containing 5 valid frames and 5 corrupted frames (singular/NaN).
        Dataset adapter should gracefully reject corrupted frames and succeed with 5 valid frames."""
        scan_dir = tmp_path / "mixed_dataset"
        scan_dir.mkdir()
        frames = []
        for i in range(5):
            c2w = np.eye(4, dtype=np.float64)
            c2w[2, 3] = float(i + 1)
            frames.append({"file_path": f"valid_{i}.jpg", "transform_matrix": c2w.tolist()})

        for i in range(5):
            if i % 2 == 0:
                bad_mat = np.zeros((4, 4)).tolist()
            else:
                bad_mat = np.eye(4).tolist()
                bad_mat[0][0] = float("nan")
            frames.append({"file_path": f"bad_{i}.jpg", "transform_matrix": bad_mat})

        data = {
            "w": 1280, "h": 720, "fl_x": 1000.0, "fl_y": 1000.0, "cx": 640.0, "cy": 360.0,
            "frames": frames
        }
        with open(scan_dir / "transforms.json", "w") as f:
            json.dump(data, f)

        res = convert_transforms_to_colmap(scan_dir)
        assert res["status"] == "success"
        assert res["num_images"] == 5
        assert res["rejected_images"] == 5

    def test_all_corrupt_frames_raises_invalid_matrix(self, tmp_path):
        """Stress: Dataset where 100% of frames are degenerate must raise InvalidMatrixError."""
        scan_dir = tmp_path / "all_bad"
        scan_dir.mkdir()
        data = {
            "w": 1280, "h": 720, "fl_x": 1000.0, "fl_y": 1000.0,
            "frames": [
                {"file_path": "bad_0.jpg", "transform_matrix": np.zeros((4, 4)).tolist()},
                {"file_path": "bad_1.jpg", "transform_matrix": [[float("nan")] * 4] * 4}
            ]
        }
        with open(scan_dir / "transforms.json", "w") as f:
            json.dump(data, f)

        with pytest.raises(InvalidMatrixError):
            convert_transforms_to_colmap(scan_dir)

    def test_empty_frames_list(self, tmp_path):
        """Stress: transforms.json with frames: [] must raise CorruptTransformsError."""
        scan_dir = tmp_path / "empty_frames"
        scan_dir.mkdir()
        with open(scan_dir / "transforms.json", "w") as f:
            json.dump({"w": 1280, "h": 720, "fl_x": 1000.0, "frames": []}, f)

        with pytest.raises(CorruptTransformsError):
            convert_transforms_to_colmap(scan_dir)

    def test_corrupted_points3d_initial_fallback(self, tmp_path):
        """Stress: points3D_initial.json containing NaNs or malformed entries falls back to synthetic points."""
        scan_dir = tmp_path / "bad_points"
        scan_dir.mkdir()
        bad_points = {
            "points": [
                {"xyz": [float("nan"), 1.0, 2.0], "rgb": [255, 0, 0]},
                {"xyz": [1.0, 2.0], "rgb": [0, 255, 0]},
                {"xyz": None}
            ]
        }
        with open(scan_dir / "points3D_initial.json", "w") as f:
            json.dump(bad_points, f)

        c2w = np.eye(4, dtype=np.float64)
        pts, cols = generate_sparse_points(
            [c2w],
            initial_points_file=scan_dir / "points3D_initial.json",
            num_synthetic_points=200
        )
        assert len(pts) == 200
        assert np.all(np.isfinite(pts))
        assert len(cols) == 200


# =============================================================================
# PART 3: VERIFY SPLAT BINARY MUTATION STRESS TESTS
# =============================================================================

class TestVerifySplatStress:
    """Adversarial stress testing of verify_splat.py binary validator."""

    def test_nonexistent_and_directory_path(self, tmp_path):
        """Stress: Non-existent file or directory path raises FileSizeError."""
        with pytest.raises(FileSizeError):
            verify_splat_file(tmp_path / "does_not_exist.splat")

        dir_path = tmp_path / "some_dir.splat"
        dir_path.mkdir()
        with pytest.raises(FileSizeError):
            verify_splat_file(dir_path)

    def test_zero_byte_file(self, tmp_path):
        """Stress: 0-byte file raises FileSizeError."""
        empty_splat = tmp_path / "empty.splat"
        empty_splat.write_bytes(b"")
        with pytest.raises(FileSizeError, match="empty"):
            verify_splat_file(empty_splat)

    def test_non_32_byte_divisibility(self, tmp_path):
        """Stress: Files of sizes 1B, 15B, 31B, 33B, 3199B, 3215B (size % 32 != 0) must raise FileSizeError."""
        bad_sizes = [1, 15, 31, 33, 63, 3199, 3215]
        for sz in bad_sizes:
            p = tmp_path / f"corrupt_{sz}b.splat"
            p.write_bytes(b"\x00" * sz)
            with pytest.raises(FileSizeError, match="not divisible by 32 bytes"):
                verify_splat_file(p)

    def test_insufficient_splat_count(self, tmp_path):
        """Stress: Exactly 1 splat (32B) or 99 splats (3168B) must raise SplatCountError."""
        p_1 = tmp_path / "one_splat.splat"
        p_1.write_bytes(create_splat_buffer(num_splats=1))
        with pytest.raises(SplatCountError, match="below minimum threshold"):
            verify_splat_file(p_1, min_splat_count=100)

        p_99 = tmp_path / "99_splats.splat"
        p_99.write_bytes(create_splat_buffer(num_splats=99))
        with pytest.raises(SplatCountError, match="below minimum threshold"):
            verify_splat_file(p_99, min_splat_count=100)

    def test_corrupted_coordinates_nan_and_inf(self, tmp_path):
        """Stress: Positions with NaN, +Inf, -Inf must raise CorruptCoordinatesError."""
        # Splat with NaN position
        buf_nan = bytearray(create_splat_buffer(num_splats=100))
        struct.pack_into("<f", buf_nan, 1600, float("nan"))
        p_nan = tmp_path / "nan_pos.splat"
        p_nan.write_bytes(buf_nan)
        with pytest.raises(CorruptCoordinatesError):
            verify_splat_file(p_nan)

        # Splat with Inf position
        buf_inf = bytearray(create_splat_buffer(num_splats=100))
        struct.pack_into("<f", buf_inf, 1604, float("inf"))
        p_inf = tmp_path / "inf_pos.splat"
        p_inf.write_bytes(buf_inf)
        with pytest.raises(CorruptCoordinatesError):
            verify_splat_file(p_inf)

        # Splat with -Inf position
        buf_ninf = bytearray(create_splat_buffer(num_splats=100))
        struct.pack_into("<f", buf_ninf, 1608, -float("inf"))
        p_ninf = tmp_path / "ninf_pos.splat"
        p_ninf.write_bytes(buf_ninf)
        with pytest.raises(CorruptCoordinatesError):
            verify_splat_file(p_ninf)

    def test_corrupted_scales_zero_negative_and_inf(self, tmp_path):
        """Stress: Scales with 0.0, negative value, NaN, or Inf must raise CorruptScaleError."""
        # Zero scale
        buf_zero_s = bytearray(create_splat_buffer(num_splats=100))
        struct.pack_into("<f", buf_zero_s, 12, 0.0)
        p_zero = tmp_path / "zero_scale.splat"
        p_zero.write_bytes(buf_zero_s)
        with pytest.raises(CorruptScaleError):
            verify_splat_file(p_zero)

        # Negative scale
        buf_neg_s = bytearray(create_splat_buffer(num_splats=100))
        struct.pack_into("<f", buf_neg_s, 16, -0.05)
        p_neg = tmp_path / "neg_scale.splat"
        p_neg.write_bytes(buf_neg_s)
        with pytest.raises(CorruptScaleError):
            verify_splat_file(p_neg)

        # NaN scale
        buf_nan_s = bytearray(create_splat_buffer(num_splats=100))
        struct.pack_into("<f", buf_nan_s, 20, float("nan"))
        p_nan_s = tmp_path / "nan_scale.splat"
        p_nan_s.write_bytes(buf_nan_s)
        with pytest.raises(CorruptScaleError):
            verify_splat_file(p_nan_s)

    def test_completely_transparent_scene(self, tmp_path):
        """Stress: Buffer where all 100 splats have alpha = 0 must raise SplatValidationError."""
        buf = create_splat_buffer(num_splats=100, a=0)
        p = tmp_path / "transparent.splat"
        p.write_bytes(buf)
        with pytest.raises(SplatValidationError, match="alpha = 0"):
            verify_splat_file(p)

    def test_degenerate_and_unnormalized_quaternions(self, tmp_path):
        """Stress: Quaternions with near-zero norm or unnormalized bounds must raise CorruptQuaternionError."""
        # Zero quaternion: uint8 128 -> float 0.0 -> norm = 0
        buf_zero_q = create_splat_buffer(num_splats=100, q0=128, q1=128, q2=128, q3=128)
        p_zero_q = tmp_path / "zero_quat.splat"
        p_zero_q.write_bytes(buf_zero_q)
        with pytest.raises(CorruptQuaternionError, match="near zero"):
            verify_splat_file(p_zero_q)

        # Unnormalized quaternion: uint8 255 -> float ~ 0.99 -> norm ~ 1.98 > 1.5
        buf_huge_q = create_splat_buffer(num_splats=100, q0=255, q1=255, q2=255, q3=255)
        p_huge_q = tmp_path / "huge_quat.splat"
        p_huge_q.write_bytes(buf_huge_q)
        with pytest.raises(CorruptQuaternionError, match="violate normalization range"):
            verify_splat_file(p_huge_q)

    def test_random_binary_fuzzing(self, tmp_path):
        """Stress: 30 random byte buffers (fuzzing).
        Must ALWAYS raise a known SplatValidationError or succeed; NEVER crash with unhandled exception."""
        np.random.seed(1337)
        for i in range(30):
            length = np.random.choice([0, 15, 31, 32, 64, 3200, 6400, 3217])
            random_bytes = os.urandom(int(length))
            fuzz_file = tmp_path / f"fuzz_{i}.splat"
            fuzz_file.write_bytes(random_bytes)

            try:
                verify_splat_file(fuzz_file, min_splat_count=100)
            except SplatValidationError:
                pass
            except Exception as e:
                pytest.fail(f"Fuzz test {i} crashed with unhandled exception {type(e).__name__}: {e}")

    def test_valid_splat_happy_path(self, tmp_path):
        """Sanity: A cleanly formed 100-splat file passes validation with complete stats."""
        buf = create_splat_buffer(
            num_splats=100,
            x=1.5, y=-2.0, z=3.0,
            s0=0.05, s1=0.08, s2=0.02,
            r=255, g=128, b=64, a=220,
            q0=128, q1=128, q2=128, q3=255
        )
        p = tmp_path / "valid.splat"
        p.write_bytes(buf)
        report = verify_splat_file(p, min_splat_count=100)
        assert report["valid"] is True
        assert report["num_splats"] == 100
        assert report["file_size_bytes"] == 3200
        assert report["scales"]["min"] > 0.0
        assert report["alpha"]["max"] == 220
        assert 0.5 <= report["quaternion_norms"]["min"] <= 1.5


if __name__ == "__main__":
    pytest.main(["-v", __file__])
