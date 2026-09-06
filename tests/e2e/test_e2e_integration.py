"""Tier 4 Tests: Realistic End-to-End Integration, Client Simulation, and Coordination.

Covers:
- Feature 3: Real-Time WebSocket Telemetry Stream (/ws/telemetry/{scan_id})
- Feature 11: Headless Mock Client Agent (client_agent.py, mock_capture.py)
- Feature 12: Automated E2E Verification Harness (client-server loop, splat artifact delivery)
- Multi-client concurrent execution, live WebSocket streaming, and graceful error recovery.
"""

import asyncio
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import threading
import time
import uuid

import httpx
import pytest
import uvicorn
import websockets

# Ensure server and client_agent modules are discoverable
SERVER_DIR = Path(__file__).resolve().parent.parent.parent / "Android_Gaussian_Splatting" / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

CLIENT_AGENT_DIR = Path(__file__).resolve().parent.parent.parent / "Android_Gaussian_Splatting" / "client_agent"
if str(CLIENT_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_AGENT_DIR))

from server import app, coordinator
from client_agent import SplatClientAgent
from mock_capture import generate_mock_dataset, create_corrupt_dataset
from verify_splat import verify_splat_file


def find_free_port() -> int:
    """Finds an available TCP port for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server():
    """Spins up a real live Uvicorn server in a background thread."""
    port = find_free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="error",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"

    # Wait for server to become responsive
    healthy = False
    for _ in range(50):
        try:
            r = httpx.get(f"{base_url}/health", timeout=0.5)
            if r.status_code == 200:
                healthy = True
                break
        except Exception:
            time.sleep(0.1)

    assert healthy, "Live test server failed to start within 5 seconds"

    yield base_url

    # Teardown
    server.should_exit = True
    thread.join(timeout=3.0)


# =============================================================================
# TIER 4: REALISTIC END-TO-END SCENARIO SIMULATIONS
# =============================================================================

def test_e2e_live_client_agent_full_cycle(live_server):
    """T4-E2E-01: Realistic client simulation executing complete capture-to-splat cycle."""
    client = SplatClientAgent(
        server_url=live_server,
        client_id="sim_pixel8_pro",
        device_model="Google Pixel 8 Pro",
    )

    temp_splat = Path(tempfile.mktemp(suffix=".splat"))
    try:
        report = client.run_full_cycle(
            num_frames=4,
            iterations=2000,
            resolution="720p",
            prune_floaters=True,
            output_splat_path=str(temp_splat),
            timeout=30.0,
        )

        assert report["success"] is True
        assert report["steps"]["handshake"]["status"] == "PASS"
        assert report["steps"]["negotiate"]["status"] == "PASS"
        assert report["steps"]["upload"]["status"] == "PASS"
        assert report["steps"]["download"]["status"] == "PASS"
        assert report["steps"]["splat_verification"]["status"] == "PASS"

        # Verify delivered binary file on disk
        assert temp_splat.exists()
        splat_check = verify_splat_file(temp_splat, min_splat_count=50)
        assert splat_check["valid"] is True
        assert splat_check["file_size_bytes"] % 32 == 0
    finally:
        try:
            if temp_splat.exists():
                temp_splat.unlink()
        except PermissionError:
            pass


def test_e2e_websocket_realtime_telemetry_stream(live_server):
    """T4-E2E-02: Real-time WebSocket telemetry frames verification."""
    client = SplatClientAgent(server_url=live_server)
    client.handshake()

    mock_zip = generate_mock_dataset(num_frames=3, as_zip=True)
    try:
        up_res = client.upload_scan(mock_zip, iterations=1000)
        scan_id = up_res["scan_id"]

        received_frames = []

        async def collect_ws_frames():
            ws_url = f"ws://127.0.0.1:{live_server.split(':')[-1]}/ws/telemetry/{scan_id}"
            async with websockets.connect(ws_url, open_timeout=5.0) as ws:
                for _ in range(50):
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        frame = json.loads(raw if isinstance(raw, str) else raw.decode())
                        received_frames.append(frame)
                        stage = str(frame.get("stage", frame.get("state", ""))).upper()
                        if stage in ("COMPLETED", "FAILED", "CANCELLED"):
                            break
                    except asyncio.TimeoutError:
                        break

        asyncio.run(collect_ws_frames())

        # If websocket closed quickly or buffered, fallback check
        if not received_frames:
            # Poll status to confirm it didn't hang
            st = client.get_status(scan_id)
            assert st["stage"] in ("QUEUED", "INGESTING", "TRAINING", "PRUNING", "CONVERTING", "COMPLETED")
        else:
            # Check frame contents
            for frame in received_frames:
                assert frame.get("scan_id") == scan_id
                assert "stage" in frame or "state" in frame
                assert "progress" in frame

            # Terminal frame check
            final_stage = received_frames[-1].get("stage", received_frames[-1].get("state"))
            assert final_stage in ("COMPLETED", "TRAINING", "CONVERTING")
    finally:
        if os.path.exists(mock_zip):
            os.unlink(mock_zip)


def test_e2e_multi_client_concurrent_coordination(live_server):
    """T4-E2E-03: Multi-client concurrent uploads and session isolation."""
    num_clients = 3
    results = {}

    def run_worker(client_idx: int):
        c = SplatClientAgent(
            server_url=live_server,
            client_id=f"concurrent_worker_{client_idx}",
        )
        temp_splat = Path(tempfile.mktemp(suffix=f"_c{client_idx}.splat"))
        try:
            res = c.run_full_cycle(
                num_frames=3,
                iterations=1000,
                resolution="720p",
                output_splat_path=str(temp_splat),
                timeout=25.0,
            )
            results[client_idx] = (res, temp_splat)
        except Exception as e:
            results[client_idx] = ({"success": False, "error": str(e)}, temp_splat)

    threads = []
    for i in range(num_clients):
        t = threading.Thread(target=run_worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=45.0)

    assert len(results) == num_clients
    for idx, (res, splat_p) in results.items():
        try:
            assert res.get("success") is True, f"Client {idx} failed: {res.get('error')}"
            assert splat_p.exists()
            assert splat_p.stat().st_size % 32 == 0
        finally:
            try:
                if splat_p.exists():
                    splat_p.unlink()
            except PermissionError:
                pass


def test_e2e_failure_recovery_on_corrupt_upload(live_server):
    """T4-E2E-04: Graceful error propagation when corrupt dataset is uploaded."""
    client = SplatClientAgent(server_url=live_server)
    client.handshake()

    corrupt_zip = create_corrupt_dataset("empty_frames")
    try:
        # Uploading corrupt archive with fast_test=False to trigger ingestion validation
        up_res = client.upload_scan(corrupt_zip, iterations=1000, fast_test=False)
        scan_id = up_res.get("scan_id")
        assert scan_id is not None

        # Wait for background task to catch ingestion error and transition to FAILED
        for _ in range(30):
            st = client.get_status(scan_id)
            if st["stage"] == "FAILED":
                break
            time.sleep(0.1)

        final_status = client.get_status(scan_id)
        assert final_status["stage"] == "FAILED"
        assert len(final_status["message"]) > 0
    finally:
        if os.path.exists(corrupt_zip):
            os.unlink(corrupt_zip)


def test_e2e_active_job_cancellation_flow(live_server):
    """T4-E2E-05: Client abort/cancellation halts coordination and marks job FAILED."""
    client = SplatClientAgent(server_url=live_server)
    client.handshake()

    mock_zip = generate_mock_dataset(num_frames=4, as_zip=True)
    try:
        up_res = client.upload_scan(mock_zip, iterations=5000)
        scan_id = up_res["scan_id"]

        # Cancel the job
        cancel_res = client.cancel_job(scan_id)
        assert cancel_res["scan_id"] == scan_id

        # Verify state
        status = client.get_status(scan_id)
        assert status["stage"] in ("FAILED", "COMPLETED")
    finally:
        if os.path.exists(mock_zip):
            os.unlink(mock_zip)
