"""Tier 1 & Tier 2 Tests: Protocol, Handshake, Negotiation, and Lifecycle Management.

Covers:
- Feature 1: Protocol Schema & State Machine
- Feature 2: Parameter Negotiation & REST Endpoints
- Feature 9: Pipeline Resilience & Error Handling
- Feature 10: Android Capture Client Wiring & Backward Compatibility

Thresholds: >=5 tests per feature for Tier 1 (Happy Path) and Tier 2 (Boundary/Adversarial).
"""

import io
import json
import os
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure server module is discoverable
SERVER_DIR = Path(__file__).resolve().parent.parent.parent / "Android_Gaussian_Splatting" / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

CLIENT_AGENT_DIR = Path(__file__).resolve().parent.parent.parent / "Android_Gaussian_Splatting" / "client_agent"
if str(CLIENT_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_AGENT_DIR))

from server import app, coordinator
from protocol import JobStage, JobConfig, HandshakeRequest, NegotiationRequest
from mock_capture import generate_mock_dataset, create_corrupt_dataset


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


# =============================================================================
# FEATURE 1 & 2: HANDSHAKE CONTRACT TESTS
# =============================================================================

# --- Tier 1: Happy Path ---

def test_handshake_standard_client(client):
    """T1-F1-01: Standard Android client handshake returns 200 and session ID."""
    payload = {
        "client_id": f"pixel8_{uuid.uuid4().hex[:6]}",
        "device_model": "Google Pixel 8 Pro",
        "app_version": "1.0.0",
        "protocol_version": "1.0.0",
        "capabilities": {
            "sensor_fusion": True,
            "vulkan_version": "1.3",
            "camera_fps": 30,
        },
    }
    resp = client.post("/handshake", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert len(data["session_id"]) > 0
    assert data["status"] in ("ready", "busy", "degraded")
    assert "supported_iterations" in data
    assert 2000 in data["supported_iterations"]


def test_handshake_full_capabilities(client):
    """T1-F1-02: Handshake reporting full sensor array (ToF/LiDAR, gyro, battery)."""
    payload = {
        "client_id": f"samsung_s24_{uuid.uuid4().hex[:6]}",
        "device_model": "Samsung Galaxy S24 Ultra",
        "app_version": "1.2.0",
        "capabilities": {
            "sensor_fusion": True,
            "vulkan_version": "1.3",
            "depth_sensor": True,
            "camera_fps": 60,
            "gyroscope": True,
            "battery_level": 0.85,
        },
    }
    resp = client.post("/handshake", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["prune_floaters_supported"] is True
    assert "max_resolution" in data


def test_handshake_minimal_payload(client):
    """T1-F1-03: Handshake with minimal required fields uses sane defaults."""
    payload = {"client_id": f"minimal_client_{uuid.uuid4().hex[:6]}"}
    resp = client.post("/handshake", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["default_iterations"] >= 1000


def test_handshake_session_uniqueness(client):
    """T1-F1-04: Multiple handshake requests produce distinct, unique session IDs."""
    resp1 = client.post("/handshake", json={"client_id": "device_A"})
    resp2 = client.post("/handshake", json={"client_id": "device_B"})
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.json()["session_id"] != resp2.json()["session_id"]


def test_handshake_supported_resolutions(client):
    """T1-F1-05: Server advertises supported resolution tiers."""
    resp = client.post("/handshake", json={"client_id": "test_res_client"})
    assert resp.status_code == 200
    data = resp.json()
    resolutions = data.get("supported_resolutions", [])
    assert any("720p" in r.lower() for r in resolutions)
    assert any("1080p" in r.lower() for r in resolutions)


# --- Tier 2: Boundary, Edge & Negative ---

def test_handshake_missing_client_id(client):
    """T2-F1-01: Handshake missing mandatory client_id returns 422 Unprocessable."""
    resp = client.post("/handshake", json={"device_model": "Ghost Device"})
    assert resp.status_code == 422


def test_handshake_empty_payload(client):
    """T2-F1-02: Handshake with empty JSON object returns 422."""
    resp = client.post("/handshake", json={})
    assert resp.status_code == 422


def test_handshake_malformed_json_syntax(client):
    """T2-F1-03: Raw non-JSON payload returns 422."""
    resp = client.post(
        "/handshake",
        content="NOT_VALID_JSON{:::}",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


def test_handshake_extraneous_fields_tolerant(client):
    """T2-F1-04: Extraneous unexpected fields are safely ignored without crashing."""
    payload = {
        "client_id": "future_client_v99",
        "device_model": "Quantum Phone",
        "unexpected_future_key_xyz": {"some_data": 123},
        "another_bogus_field": [1, 2, 3],
    }
    resp = client.post("/handshake", json=payload)
    assert resp.status_code == 200
    assert "session_id" in resp.json()


def test_handshake_extreme_string_lengths(client):
    """T2-F1-05: Extremely long client_id string handles without buffer issues."""
    payload = {
        "client_id": "A" * 512,
        "device_model": "M" * 512,
        "app_version": "99.99.99-alpha+build.999999",
    }
    resp = client.post("/handshake", json=payload)
    assert resp.status_code == 200


# =============================================================================
# FEATURE 2: PARAMETER NEGOTIATION CONTRACT TESTS
# =============================================================================

# --- Tier 1: Happy Path ---

def test_negotiate_valid_standard_params(client):
    """T1-F2-01: Standard training parameter negotiation."""
    hs = client.post("/handshake", json={"client_id": "neg_client_1"}).json()
    sid = hs["session_id"]

    payload = {
        "session_id": sid,
        "iterations": 2000,
        "resolution": "720p",
        "prune_floaters": True,
        "sh_degree": 3,
    }
    resp = client.post("/negotiate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] is True
    cfg = data.get("config", data.get("negotiated_params", {}))
    assert cfg["iterations"] == 2000
    assert cfg["resolution"] == "720p"
    assert cfg["prune_floaters"] is True


def test_negotiate_studio_pro_params(client):
    """T1-F2-02: Studio Pro parameters (30,000 iterations, 1080p)."""
    hs = client.post("/handshake", json={"client_id": "neg_client_2"}).json()
    sid = hs["session_id"]

    payload = {
        "session_id": sid,
        "iterations": 30000,
        "resolution": "1080p",
        "prune_floaters": True,
        "sh_degree": 3,
    }
    resp = client.post("/negotiate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] is True
    cfg = data.get("config", data.get("negotiated_params", {}))
    assert cfg["iterations"] == 30000


def test_negotiate_fast_preview_params(client):
    """T1-F2-03: Fast preview parameters (1,000 iterations, sh_degree=0)."""
    hs = client.post("/handshake", json={"client_id": "neg_client_3"}).json()
    sid = hs["session_id"]

    payload = {
        "session_id": sid,
        "iterations": 1000,
        "resolution": "720p",
        "sh_degree": 0,
        "prune_floaters": False,
    }
    resp = client.post("/negotiate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] is True
    cfg = data.get("config", data.get("negotiated_params", {}))
    assert cfg["sh_degree"] == 0
    assert cfg["prune_floaters"] is False


def test_negotiate_custom_pruning_thresholds(client):
    """T1-F2-04: Negotiating custom analytical floater pruning bounds."""
    hs = client.post("/handshake", json={"client_id": "neg_client_4"}).json()
    sid = hs["session_id"]

    payload = {
        "session_id": sid,
        "iterations": 2000,
        "min_opacity": 0.04,
        "max_scale": 0.15,
    }
    resp = client.post("/negotiate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] is True
    cfg = data.get("config", data.get("negotiated_params", {}))
    assert abs(cfg["min_opacity"] - 0.04) < 1e-4
    assert abs(cfg["max_scale"] - 0.15) < 1e-4


def test_negotiate_session_persistence(client):
    """T1-F2-05: Negotiated parameters are saved to the coordinator session."""
    hs = client.post("/handshake", json={"client_id": "neg_client_5"}).json()
    sid = hs["session_id"]

    client.post("/negotiate", json={
        "session_id": sid,
        "iterations": 7000,
        "resolution": "1440p",
    })

    sess = coordinator.get_session(sid)
    assert sess is not None
    assert sess.negotiated_config is not None
    assert sess.negotiated_config.iterations == 7000


# --- Tier 2: Boundary, Edge & Negative ---

def test_negotiate_clamped_iterations_low(client):
    """T2-F2-01: Iteration count below minimum is safely clamped to at least 50."""
    hs = client.post("/handshake", json={"client_id": "clamp_client_low"}).json()
    sid = hs["session_id"]

    resp = client.post("/negotiate", json={
        "session_id": sid,
        "iterations": 5,  # Too low
    })
    assert resp.status_code == 200
    data = resp.json()
    cfg = data.get("config", data.get("negotiated_params", {}))
    assert cfg["iterations"] >= 50


def test_negotiate_clamped_iterations_high(client):
    """T2-F2-02: Iteration count above maximum (>30000) is clamped to 30000."""
    hs = client.post("/handshake", json={"client_id": "clamp_client_high"}).json()
    sid = hs["session_id"]

    resp = client.post("/negotiate", json={
        "session_id": sid,
        "iterations": 100000,  # Exceeds max
    })
    assert resp.status_code == 200
    data = resp.json()
    cfg = data.get("config", data.get("negotiated_params", {}))
    assert cfg["iterations"] <= 30000


def test_negotiate_invalid_resolution_string(client):
    """T2-F2-03: Unsupported resolution string falls back to safe default."""
    hs = client.post("/handshake", json={"client_id": "res_clamp_client"}).json()
    sid = hs["session_id"]

    resp = client.post("/negotiate", json={
        "session_id": sid,
        "resolution": "8K_CINEMATIC_IMAX",
    })
    assert resp.status_code == 200
    data = resp.json()
    cfg = data.get("config", data.get("negotiated_params", {}))
    assert cfg["resolution"] in ("720p", "1080p", "1440p", "4k")


def test_negotiate_negative_pruning_threshold(client):
    """T2-F2-04: Negative pruning opacity or scale is clamped to valid range."""
    hs = client.post("/handshake", json={"client_id": "pruning_clamp_client"}).json()
    sid = hs["session_id"]

    resp = client.post("/negotiate", json={
        "session_id": sid,
        "min_opacity": -0.5,
        "max_scale": -1.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    cfg = data.get("config", data.get("negotiated_params", {}))
    assert cfg["min_opacity"] >= 0.001
    assert cfg["max_scale"] >= 0.01


def test_negotiate_invalid_session_id(client):
    """T2-F2-05: Negotiating with unknown session_id handles gracefully and notes in reasons."""
    resp = client.post("/negotiate", json={
        "session_id": "non_existent_session_99999",
        "iterations": 2000,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] is True
    reasons_text = " ".join(data.get("reasons", [])).lower()
    assert "not found" in reasons_text or "standalone" in reasons_text


# =============================================================================
# FEATURE 1 & 9: JOB STATUS & STATE MACHINE TRANSITIONS
# =============================================================================

# --- Tier 1: Happy Path ---

def test_status_query_existing_job(client):
    """T1-F1-06: Querying GET /status/{scan_id} on existing job returns full schema."""
    scan_id = f"test_stat_{uuid.uuid4().hex[:4]}"
    coordinator.create_job(
        scan_id=scan_id,
        initial_stage=JobStage.QUEUED,
        initial_message="Test queued job",
    )

    resp = client.get(f"/status/{scan_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == scan_id
    assert data["status"] in ("QUEUED", "pending")
    assert data["stage"] == "QUEUED"
    assert "progress" in data
    assert "iteration" in data
    assert "total_iterations" in data


def test_status_query_legacy_format(client):
    """T1-F1-07: Status response conforms to legacy Android ApiClient contract."""
    scan_id = f"test_leg_{uuid.uuid4().hex[:4]}"
    coordinator.create_job(scan_id=scan_id, initial_stage=JobStage.TRAINING)

    resp = client.get(f"/status/{scan_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert "status" in data
    assert "progress" in data
    assert "iteration" in data
    assert "total_iterations" in data
    assert "message" in data


def test_status_progress_bounds(client):
    """T1-F1-08: Progress attribute is strictly bounded in [0.0, 1.0]."""
    scan_id = f"test_bnd_{uuid.uuid4().hex[:4]}"
    coordinator.create_job(scan_id=scan_id, config=JobConfig(iterations=1000))
    coordinator.update_job_progress(scan_id, iteration=500)

    resp = client.get(f"/status/{scan_id}")
    assert resp.status_code == 200
    progress = resp.json()["progress"]
    assert 0.0 <= progress <= 1.0
    assert abs(progress - 0.5) < 0.05


def test_status_stage_enum_validity(client):
    """T1-F1-09: Returned stage maps to valid JobStage enum."""
    scan_id = f"test_enum_{uuid.uuid4().hex[:4]}"
    coordinator.create_job(scan_id=scan_id, initial_stage=JobStage.CONVERTING)

    resp = client.get(f"/status/{scan_id}")
    assert resp.status_code == 200
    stage_str = resp.json()["stage"]
    assert stage_str in [s.value for s in JobStage]


def test_cancel_active_job(client):
    """T1-F1-10: DELETE /jobs/{scan_id} cancels an active job."""
    scan_id = f"test_canc_{uuid.uuid4().hex[:4]}"
    coordinator.create_job(scan_id=scan_id, initial_stage=JobStage.TRAINING)

    resp = client.delete(f"/jobs/{scan_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["cancelled"] is True
    assert data["scan_id"] == scan_id

    # Verify status is now FAILED
    status_resp = client.get(f"/status/{scan_id}").json()
    assert status_resp["stage"] == "FAILED"


# --- Tier 2: Boundary, Edge & Negative ---

def test_status_nonexistent_job(client):
    """T2-F1-06: Querying status for non-existent job returns 404."""
    resp = client.get("/status/nonexistent_scan_id_9999")
    assert resp.status_code == 404


def test_cancel_nonexistent_job(client):
    """T2-F1-07: Cancelling non-existent job returns 404."""
    resp = client.delete("/jobs/nonexistent_scan_id_9999")
    assert resp.status_code == 404


def test_cancel_already_completed_job(client):
    """T2-F1-08: Cancelling an already COMPLETED job returns cancelled=False."""
    scan_id = f"test_comp_{uuid.uuid4().hex[:4]}"
    coordinator.create_job(scan_id=scan_id, initial_stage=JobStage.COMPLETED)

    resp = client.delete(f"/jobs/{scan_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["cancelled"] is False


def test_cancel_already_failed_job(client):
    """T2-F1-09: Cancelling an already FAILED job returns cancelled=False."""
    scan_id = f"test_fail_{uuid.uuid4().hex[:4]}"
    coordinator.create_job(scan_id=scan_id, initial_stage=JobStage.FAILED)

    resp = client.delete(f"/jobs/{scan_id}")
    assert resp.status_code == 200
    assert resp.json()["cancelled"] is False


def test_double_cancellation(client):
    """T2-F1-10: Duplicate cancellation calls on same job succeed on first, false on second."""
    scan_id = f"test_dbl_{uuid.uuid4().hex[:4]}"
    coordinator.create_job(scan_id=scan_id, initial_stage=JobStage.INGESTING)

    resp1 = client.delete(f"/jobs/{scan_id}")
    assert resp1.status_code == 200
    assert resp1.json()["cancelled"] is True

    resp2 = client.delete(f"/jobs/{scan_id}")
    assert resp2.status_code == 200
    assert resp2.json()["cancelled"] is False


# =============================================================================
# FEATURE 1 & 10: DATASET UPLOAD PROTOCOL & INGESTION
# =============================================================================

# --- Tier 1: Happy Path ---

def test_upload_valid_mock_dataset(client):
    """T1-F10-01: Uploading a valid mock capture archive returns 200 with scan_id."""
    zip_path = generate_mock_dataset(num_frames=2, as_zip=True)
    try:
        with open(zip_path, "rb") as f:
            resp = client.post(
                "/upload",
                files={"file": ("mock_scan.zip", f, "application/zip")},
                data={"iterations": "1000", "fast_test": "true"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "scan_id" in data
        assert data["status"] in ("QUEUED", "pending")
    finally:
        if os.path.exists(zip_path):
            os.unlink(zip_path)


def test_upload_with_session_id(client):
    """T1-F10-02: Uploading with session_id links job to active session."""
    hs = client.post("/handshake", json={"client_id": "sess_up_client"}).json()
    sid = hs["session_id"]

    zip_path = generate_mock_dataset(num_frames=2, as_zip=True)
    try:
        with open(zip_path, "rb") as f:
            resp = client.post(
                "/upload",
                files={"file": ("mock_scan.zip", f, "application/zip")},
                data={"session_id": sid, "fast_test": "true"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == sid
    finally:
        if os.path.exists(zip_path):
            os.unlink(zip_path)


def test_upload_with_explicit_form_overrides(client):
    """T1-F10-03: Upload form fields override session-level defaults."""
    hs = client.post("/handshake", json={"client_id": "override_client"}).json()
    sid = hs["session_id"]

    zip_path = generate_mock_dataset(num_frames=2, as_zip=True)
    try:
        with open(zip_path, "rb") as f:
            resp = client.post(
                "/upload",
                files={"file": ("mock_scan.zip", f, "application/zip")},
                data={
                    "session_id": sid,
                    "iterations": "3000",
                    "resolution": "1080p",
                    "prune_floaters": "false",
                    "fast_test": "true",
                },
            )
        assert resp.status_code == 200
        scan_id = resp.json()["scan_id"]
        rec = coordinator.get_job_record(scan_id)
        assert rec is not None
        assert rec.config.iterations == 3000
        assert rec.config.prune_floaters is False
    finally:
        if os.path.exists(zip_path):
            os.unlink(zip_path)


def test_upload_creates_isolated_directory(client):
    """T1-F10-04: Upload unpacks into isolated scans/{scan_id} folder."""
    zip_path = generate_mock_dataset(num_frames=2, as_zip=True)
    try:
        with open(zip_path, "rb") as f:
            resp = client.post(
                "/upload",
                files={"file": ("mock_scan.zip", f, "application/zip")},
                data={"fast_test": "true"},
            )
        scan_id = resp.json()["scan_id"]
        extract_dir = SERVER_DIR / "scans" / scan_id
        assert extract_dir.exists()
        assert extract_dir.is_dir()
    finally:
        if os.path.exists(zip_path):
            os.unlink(zip_path)


def test_upload_generates_queued_status(client):
    """T1-F10-05: Status endpoint returns valid stage after upload."""
    zip_path = generate_mock_dataset(num_frames=2, as_zip=True)
    try:
        with open(zip_path, "rb") as f:
            resp = client.post(
                "/upload",
                files={"file": ("mock_scan.zip", f, "application/zip")},
                data={"fast_test": "true"},
            )
        scan_id = resp.json()["scan_id"]
        status_resp = client.get(f"/status/{scan_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["stage"] in ("QUEUED", "INGESTING", "TRAINING", "COMPLETED")
    finally:
        if os.path.exists(zip_path):
            os.unlink(zip_path)


# --- Tier 2: Boundary, Edge & Negative ---

def test_upload_missing_file(client):
    """T2-F10-01: POST /upload without file returns 422."""
    resp = client.post("/upload", data={"iterations": "1000"})
    assert resp.status_code == 422


def test_upload_corrupt_zip(client):
    """T2-F10-02: Uploading corrupted file returns 400 Bad Request."""
    corrupt_bytes = io.BytesIO(b"PK\x03\x04CORRUPTED_GARBAGE_PAYLOAD_NOT_A_REAL_ZIP")
    resp = client.post(
        "/upload",
        files={"file": ("corrupt.zip", corrupt_bytes, "application/zip")},
    )
    assert resp.status_code == 400
    assert "zip" in resp.json()["detail"].lower()


def test_upload_empty_file(client):
    """T2-F10-03: Uploading 0-byte file returns 400 Bad Request."""
    empty_bytes = io.BytesIO(b"")
    resp = client.post(
        "/upload",
        files={"file": ("empty.zip", empty_bytes, "application/zip")},
    )
    assert resp.status_code == 400


def test_upload_text_disguised_as_zip(client):
    """T2-F10-04: Uploading plain text file disguised as .zip returns 400."""
    fake_zip = io.BytesIO(b"Hello World. This is plain text, not a zip file.")
    resp = client.post(
        "/upload",
        files={"file": ("fake.zip", fake_zip, "application/zip")},
    )
    assert resp.status_code == 400


def test_upload_oversized_parameters(client):
    """T2-F10-05: Extreme iteration parameter via upload is safely clamped."""
    zip_path = generate_mock_dataset(num_frames=2, as_zip=True)
    try:
        with open(zip_path, "rb") as f:
            resp = client.post(
                "/upload",
                files={"file": ("mock_scan.zip", f, "application/zip")},
                data={"iterations": "999999", "fast_test": "true"},
            )
        assert resp.status_code == 200
        scan_id = resp.json()["scan_id"]
        rec = coordinator.get_job_record(scan_id)
        assert rec is not None
        assert rec.config.iterations <= 30000
    finally:
        if os.path.exists(zip_path):
            os.unlink(zip_path)


# =============================================================================
# FEATURE 10: BACKWARD COMPATIBILITY & HEALTH / DOWNLOAD
# =============================================================================

# --- Tier 1: Happy Path ---

def test_health_check_endpoint(client):
    """T1-F10-06: GET /health returns online status and GPU info."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert "gpu_available" in data
    assert "cuda_version" in data


def test_models_list_endpoint(client):
    """T1-F10-07: GET /models returns list of .splat models."""
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert isinstance(data["models"], list)


def test_download_demo_model(client):
    """T1-F10-08: GET /download/{filename} streams existing .splat model."""
    models_dir = SERVER_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    test_splat = models_dir / "test_download.splat"
    test_splat.write_bytes(b"\x00" * 320)  # 10 splats = 320 bytes

    try:
        resp = client.get("/download/test_download.splat")
        assert resp.status_code == 200
        assert len(resp.content) == 320
    finally:
        if test_splat.exists():
            test_splat.unlink()


# --- Tier 2: Boundary ---

def test_download_nonexistent_file(client):
    """T2-F10-06: GET /download/nonexistent.splat returns 404."""
    resp = client.get("/download/nonexistent_file_xyz_123.splat")
    assert resp.status_code == 404


def test_download_path_traversal_prevention(client):
    """T2-F10-07: Path traversal attempt is blocked or returned as 404."""
    resp = client.get("/download/../../server.py")
    assert resp.status_code in (400, 404)
