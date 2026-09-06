"""
Milestone 5 Adversarial Verification Suite — Challenger 2.
Empirically verifies:
1. Live server interaction with client_agent.py (Python API and CLI invocation).
2. Network failure / socket drop handling during active WebSocket telemetry streaming.
3. Floater pruning on various density distributions and geometry preservation integrity.
4. Empirical stress tests (concurrency, path traversal, mid-stream cancellation, corrupt archives).
"""

import asyncio
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Dict, List, Tuple

import httpx
import numpy as np
from plyfile import PlyData, PlyElement
import pytest
import uvicorn
import websockets

# Setup search paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_DIR = PROJECT_ROOT / "Android_Gaussian_Splatting" / "server"
CLIENT_AGENT_DIR = PROJECT_ROOT / "Android_Gaussian_Splatting" / "client_agent"

for p in (str(SERVER_DIR), str(CLIENT_AGENT_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from server import app, coordinator
from client_agent import SplatClientAgent
from mock_capture import generate_mock_dataset, create_corrupt_dataset
from prune_floaters import prune_gaussians, compute_scene_radius, PruneFloatersError
from verify_splat import verify_splat_file
from pipeline_runner import PipelineConfig, PipelineRunner, PipelineStage


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server():
    """Launches live FastAPI/Uvicorn server on a dynamically assigned free port."""
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

    healthy = False
    for _ in range(50):
        try:
            r = httpx.get(f"{base_url}/health", timeout=0.5)
            if r.status_code == 200:
                healthy = True
                break
        except Exception:
            time.sleep(0.1)

    assert healthy, "Uvicorn live server failed to initialize within 5 seconds"
    yield base_url

    server.should_exit = True
    thread.join(timeout=3.0)


# =============================================================================
# 1. LIVE SERVER INTERACTION WITH CLIENT_AGENT.PY
# =============================================================================

def test_01_client_agent_full_lifecycle(live_server):
    """
    Empirically verify end-to-end client-server interaction:
    Capture handoff -> parameter negotiation -> upload -> WebSocket telemetry across stages -> .splat download.
    """
    client = SplatClientAgent(
        server_url=live_server,
        client_id="challenger2_full_cycle_agent",
        device_model="Pixel 8 Pro Testbed",
        app_version="2.0.0",
    )

    # 1. Health check
    health = client.check_health()
    assert health["status"] == "online"

    # 2. Handshake
    hs_data = client.handshake(
        capabilities={"sensor_fusion": True, "vulkan_version": "1.3", "device_ram_gb": 12}
    )
    assert client.session_id is not None
    assert "session_id" in hs_data
    assert hs_data["prune_floaters_supported"] is True

    # 3. Parameter negotiation
    neg_data = client.negotiate(
        iterations=1500,
        resolution="720p",
        prune_floaters=True,
        sh_degree=3,
    )
    assert neg_data["accepted"] is True
    assert neg_data["config"]["iterations"] == 1500

    # 4. Generate mock dataset & upload
    mock_zip = generate_mock_dataset(num_frames=4, as_zip=True)
    try:
        up_data = client.upload_scan(
            zip_path=mock_zip,
            iterations=1500,
            resolution="720p",
            prune_floaters=True,
            fast_test=True,
        )
        scan_id = up_data["scan_id"]
        assert scan_id is not None
        assert up_data["status"] == "QUEUED"

        # 5. Monitor WebSocket telemetry across stages
        received_stages = set()
        telemetry_frames = []

        def on_frame(f):
            telemetry_frames.append(f)
            stage = f.get("stage", f.get("status"))
            if stage:
                received_stages.add(stage)

        frames = client.listen_telemetry(scan_id, timeout=30.0, on_frame=on_frame)
        assert len(frames) > 0, "Failed to receive any telemetry frames over WebSocket"

        # Verify key pipeline stages were observed
        assert any(st in received_stages for st in ("QUEUED", "INGESTING", "TRAINING"))
        assert any(st in received_stages for st in ("PRUNING", "CONVERTING", "COMPLETED"))

        # 6. Verify final status
        final_status = client.get_status(scan_id)
        assert final_status["status"] in ("completed", "COMPLETED")
        assert final_status["stage"] == "COMPLETED"
        assert final_status["progress"] >= 1.0

        # 7. Download final .splat file
        temp_splat = tempfile.mktemp(suffix=".splat")
        try:
            dl_path = client.download_splat(f"{scan_id}.splat", temp_splat)
            assert os.path.exists(dl_path)
            file_size = os.path.getsize(dl_path)
            assert file_size > 0
            assert file_size % 32 == 0, f"Splat size {file_size} is not divisible by 32"

            # 8. Strict binary format verification
            v_res = verify_splat_file(dl_path, min_splat_count=50)
            assert v_res["valid"] is True
            assert v_res["num_splats"] == file_size // 32
        finally:
            if os.path.exists(temp_splat):
                try:
                    os.unlink(temp_splat)
                except PermissionError:
                    pass
    finally:
        if os.path.exists(mock_zip):
            try:
                os.unlink(mock_zip)
            except Exception:
                pass


def test_02_client_agent_cli_invocation(live_server):
    """
    Empirically verify executing client_agent.py as a standalone CLI script
    simulating Android headless execution and returning code 0.
    """
    temp_splat = tempfile.mktemp(suffix=".splat")
    cmd = [
        sys.executable,
        str(CLIENT_AGENT_DIR / "client_agent.py"),
        "--server", live_server,
        "--frames", "3",
        "--iterations", "1000",
        "--output", temp_splat,
        "--timeout", "30.0",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        assert proc.returncode == 0, f"CLI client failed (code {proc.returncode}):\n{proc.stderr}\n{proc.stdout}"

        # Parse output JSON
        json_start = proc.stdout.find("{")
        assert json_start != -1, f"No JSON found in stdout: {proc.stdout}"
        data = json.loads(proc.stdout[json_start:])
        assert data.get("success") is True
        assert data["steps"]["handshake"]["status"] == "PASS"
        assert data["steps"]["upload"]["status"] == "PASS"
        assert data["steps"]["download"]["status"] == "PASS"

        # Verify downloaded splat file
        assert os.path.exists(temp_splat)
        sz = os.path.getsize(temp_splat)
        assert sz > 0 and sz % 32 == 0
        v_check = verify_splat_file(temp_splat, min_splat_count=50)
        assert v_check["valid"] is True
    finally:
        if os.path.exists(temp_splat):
            try:
                os.unlink(temp_splat)
            except PermissionError:
                pass


# =============================================================================
# 2. NETWORK FAILURE / SOCKET DROP HANDLING DURING WEBSOCKET TELEMETRY
# =============================================================================

def test_03_websocket_abrupt_client_disconnect(live_server):
    """
    Test network failure: WebSocket client abruptly drops connection mid-stream.
    Server must NOT crash, TelemetryHub must clean up dead socket, and pipeline must complete.
    """
    client = SplatClientAgent(server_url=live_server)
    client.handshake()

    mock_zip = generate_mock_dataset(num_frames=3, as_zip=True)
    try:
        up_res = client.upload_scan(mock_zip, iterations=1500)
        scan_id = up_res["scan_id"]

        ws_port = live_server.split(":")[-1]
        ws_url = f"ws://127.0.0.1:{ws_port}/ws/telemetry/{scan_id}"

        # Connect, receive 2 frames, then abruptly break socket
        async def drop_client():
            async with websockets.connect(ws_url, open_timeout=5.0) as ws:
                # Receive first frame
                msg1 = await asyncio.wait_for(ws.recv(), timeout=2.0)
                assert msg1 is not None
                # Force close transport without sending clean close frame
                if hasattr(ws, "transport") and ws.transport:
                    ws.transport.close()
                else:
                    await ws.close()

        asyncio.run(drop_client())

        # Give server time to process drop and continue pipeline
        time.sleep(1.0)

        # Confirm coordinator and server are fully functional and subscriber count was cleaned up
        sub_count = coordinator.telemetry_hub.get_subscriber_count(scan_id)
        assert sub_count == 0, f"Expected 0 subscribers after socket drop, found {sub_count}"

        # Poll status until pipeline reaches terminal stage
        completed = False
        for _ in range(40):
            st = client.get_status(scan_id)
            if st["stage"] in ("COMPLETED", "FAILED"):
                completed = True
                assert st["stage"] == "COMPLETED", f"Pipeline failed: {st.get('message')}"
                break
            time.sleep(0.1)

        assert completed, "Pipeline did not reach COMPLETED after client socket drop"
    finally:
        if os.path.exists(mock_zip):
            try:
                os.unlink(mock_zip)
            except Exception:
                pass


def test_04_websocket_reconnection_after_drop(live_server):
    """
    Test reconnecting to active job's WebSocket stream after a network drop.
    The reconnected socket must receive live telemetry snapshots and frames.
    """
    client = SplatClientAgent(server_url=live_server)
    client.handshake()

    mock_zip = generate_mock_dataset(num_frames=3, as_zip=True)
    try:
        up_res = client.upload_scan(mock_zip, iterations=2000)
        scan_id = up_res["scan_id"]
        ws_port = live_server.split(":")[-1]
        ws_url = f"ws://127.0.0.1:{ws_port}/ws/telemetry/{scan_id}"

        # 1. Connect first client and drop after 1 frame
        async def client_1_drop():
            async with websockets.connect(ws_url, open_timeout=5.0) as ws:
                await asyncio.wait_for(ws.recv(), timeout=2.0)
                await ws.close()

        asyncio.run(client_1_drop())
        time.sleep(0.1)

        # 2. Reconnect with client 2 to the same scan_id
        reconnected_frames = []

        async def client_2_reconnect():
            async with websockets.connect(ws_url, open_timeout=5.0) as ws:
                for _ in range(25):
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        f = json.loads(raw if isinstance(raw, str) else raw.decode())
                        reconnected_frames.append(f)
                        if f.get("stage") in ("COMPLETED", "FAILED"):
                            break
                    except asyncio.TimeoutError:
                        break

        asyncio.run(client_2_reconnect())

        assert len(reconnected_frames) > 0, "Reconnected client received 0 frames"
        assert reconnected_frames[0]["scan_id"] == scan_id
    finally:
        if os.path.exists(mock_zip):
            try:
                os.unlink(mock_zip)
            except Exception:
                pass


def test_05_concurrent_multi_subscribers_with_mixed_drops(live_server):
    """
    Test 3 concurrent WebSocket subscribers where 2 disconnect mid-stream
    and 1 stays connected. The persistent client must receive all frames uninterrupted.
    """
    client = SplatClientAgent(server_url=live_server)
    client.handshake()

    mock_zip = generate_mock_dataset(num_frames=3, as_zip=True)
    try:
        up_res = client.upload_scan(mock_zip, iterations=1500)
        scan_id = up_res["scan_id"]
        ws_port = live_server.split(":")[-1]
        ws_url = f"ws://127.0.0.1:{ws_port}/ws/telemetry/{scan_id}"

        persistent_frames = []
        c1_frames = []
        c2_frames = []

        async def run_subscriber_mesh():
            # Connect all 3
            ws_persist = await websockets.connect(ws_url)
            ws_drop1 = await websockets.connect(ws_url)
            ws_drop2 = await websockets.connect(ws_url)

            # Receive 1 frame on drop1, then close
            f1 = await asyncio.wait_for(ws_drop1.recv(), timeout=2.0)
            c1_frames.append(f1)
            await ws_drop1.close()

            # Receive 2 frames on drop2, then close
            for _ in range(2):
                f2 = await asyncio.wait_for(ws_drop2.recv(), timeout=2.0)
                c2_frames.append(f2)
            await ws_drop2.close()

            # Persistent client continues listening until COMPLETED
            for _ in range(30):
                try:
                    f = await asyncio.wait_for(ws_persist.recv(), timeout=2.0)
                    frame_data = json.loads(f if isinstance(f, str) else f.decode())
                    persistent_frames.append(frame_data)
                    if frame_data.get("stage") in ("COMPLETED", "FAILED"):
                        break
                except asyncio.TimeoutError:
                    break

            await ws_persist.close()

        asyncio.run(run_subscriber_mesh())

        assert len(c1_frames) == 1
        assert len(c2_frames) == 2
        assert len(persistent_frames) >= 3, "Persistent subscriber missed expected frames"
        assert persistent_frames[-1].get("stage") == "COMPLETED"
    finally:
        if os.path.exists(mock_zip):
            try:
                os.unlink(mock_zip)
            except Exception:
                pass


# =============================================================================
# 3. FLOATER PRUNING ON VARIOUS DENSITY DISTRIBUTIONS & GEOMETRY INTEGRITY
# =============================================================================

def _build_test_ply(
    xyz: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    ply_path: Path,
) -> None:
    """Helper to write a valid 3DGS PLY file with given geometry and attributes."""
    n = len(xyz)
    normals = np.zeros((n, 3), dtype=np.float32)
    f_dc = np.full((n, 3), 0.5, dtype=np.float32)
    rots = np.zeros((n, 4), dtype=np.float32)
    rots[:, 0] = 1.0

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    elements = np.empty(n, dtype=dtype)
    elements["x"] = xyz[:, 0]
    elements["y"] = xyz[:, 1]
    elements["z"] = xyz[:, 2]
    elements["nx"] = normals[:, 0]
    elements["ny"] = normals[:, 1]
    elements["nz"] = normals[:, 2]
    elements["f_dc_0"] = f_dc[:, 0]
    elements["f_dc_1"] = f_dc[:, 1]
    elements["f_dc_2"] = f_dc[:, 2]
    elements["opacity"] = opacities
    elements["scale_0"] = scales[:, 0]
    elements["scale_1"] = scales[:, 1]
    elements["scale_2"] = scales[:, 2]
    elements["rot_0"] = rots[:, 0]
    elements["rot_1"] = rots[:, 1]
    elements["rot_2"] = rots[:, 2]
    elements["rot_3"] = rots[:, 3]

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    el = PlyElement.describe(elements, "vertex")
    PlyData([el], text=False).write(str(ply_path))


def test_06_floater_pruning_uniform_density_and_geometry_preservation():
    """
    Test floater pruning on uniform density sphere with synthetic floaters.
    Verify:
    1. Valid core Gaussians are 100% retained.
    2. Low-opacity and oversized floaters are 100% culled.
    3. Retained Gaussians retain their exact XYZ positions (zero alteration of valid geometry).
    """
    np.random.seed(101)
    num_core = 500
    num_low_op = 100
    num_oversized = 100
    total = num_core + num_low_op + num_oversized

    # Core Gaussians: clustered in sphere r=0.5, opacity raw=2.0 (sigmoid 0.88), log scale = -3.0
    core_xyz = np.random.uniform(-0.5, 0.5, size=(num_core, 3)).astype(np.float32)
    # Low opacity floaters: scattered in r=0.5, opacity raw=-5.0 (sigmoid ~0.0067 < 0.04)
    low_op_xyz = np.random.uniform(-0.5, 0.5, size=(num_low_op, 3)).astype(np.float32)
    # Oversized floaters: scattered far out with huge scale
    oversized_xyz = np.random.uniform(-1.0, 1.0, size=(num_oversized, 3)).astype(np.float32)

    xyz = np.vstack([core_xyz, low_op_xyz, oversized_xyz])

    opacities = np.zeros(total, dtype=np.float32)
    opacities[:num_core] = 2.0
    opacities[num_core : num_core + num_low_op] = -5.0
    opacities[num_core + num_low_op :] = 2.0

    scales = np.full((total, 3), -3.0, dtype=np.float32)
    scales[num_core + num_low_op :] = 2.5  # exp(2.5) ~ 12.18 >> threshold

    temp_dir = Path(tempfile.mkdtemp(prefix="test_prune_uniform_"))
    in_ply = temp_dir / "input.ply"
    out_ply = temp_dir / "pruned.ply"

    try:
        _build_test_ply(xyz, opacities, scales, in_ply)

        stats = prune_gaussians(
            input_ply=in_ply,
            output_ply=out_ply,
            min_opacity=0.04,
            max_scale=0.15,
            relative_scale=True,
        )

        assert stats["initial_gaussians"] == 700
        assert stats["kept_gaussians"] == num_core
        assert stats["culled_gaussians"] == num_low_op + num_oversized
        assert stats["culled_opacity"] == num_low_op
        assert stats["culled_scale"] == num_oversized

        # Verify Geometry Preservation
        pruned_data = PlyData.read(str(out_ply))
        v = pruned_data["vertex"].data
        pruned_xyz = np.column_stack([v["x"], v["y"], v["z"]])

        assert len(pruned_xyz) == num_core
        # Verify exact numerical match with core_xyz
        np.testing.assert_allclose(pruned_xyz, core_xyz, rtol=1e-6, atol=1e-6)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_07_floater_pruning_bimodal_cluster_distribution():
    """
    Test floater pruning on bimodal distribution (two dense clusters separated by distance).
    Verifies that scene radius calculation via 95th percentile does not cull valid distant clusters.
    """
    np.random.seed(202)
    num_c1 = 300
    num_c2 = 300
    num_floaters = 80
    total = num_c1 + num_c2 + num_floaters

    # Cluster 1 centered at (-2.0, 0, 0)
    c1_xyz = (np.random.normal(loc=0.0, scale=0.15, size=(num_c1, 3)) + np.array([-2.0, 0.0, 0.0])).astype(np.float32)
    # Cluster 2 centered at (+2.0, 0, 0)
    c2_xyz = (np.random.normal(loc=0.0, scale=0.15, size=(num_c2, 3)) + np.array([2.0, 0.0, 0.0])).astype(np.float32)
    # Sparse floaters scattered widely
    flt_xyz = np.random.uniform(-6.0, 6.0, size=(num_floaters, 3)).astype(np.float32)

    xyz = np.vstack([c1_xyz, c2_xyz, flt_xyz])

    opacities = np.full(total, 2.5, dtype=np.float32)
    # Give all diffuse floaters low opacity < 0.04
    opacities[num_c1 + num_c2 :] = -4.5

    scales = np.full((total, 3), -3.0, dtype=np.float32)

    temp_dir = Path(tempfile.mkdtemp(prefix="test_prune_bimodal_"))
    in_ply = temp_dir / "input.ply"
    out_ply = temp_dir / "pruned.ply"

    try:
        _build_test_ply(xyz, opacities, scales, in_ply)

        stats = prune_gaussians(
            input_ply=in_ply,
            output_ply=out_ply,
            min_opacity=0.04,
            max_scale=0.15,
            relative_scale=True,
        )

        assert stats["kept_gaussians"] == num_c1 + num_c2
        assert stats["culled_gaussians"] == num_floaters

        # Verify points from both clusters are preserved
        pruned_data = PlyData.read(str(out_ply))
        v = pruned_data["vertex"].data
        pruned_x = np.asarray(v["x"])

        kept_c1 = np.sum(pruned_x < 0)
        kept_c2 = np.sum(pruned_x > 0)
        assert kept_c1 == num_c1, f"Expected {num_c1} in Cluster 1, kept {kept_c1}"
        assert kept_c2 == num_c2, f"Expected {num_c2} in Cluster 2, kept {kept_c2}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_08_floater_pruning_all_floaters_fallback_safeguard():
    """
    Test degenerate edge case: dataset consisting ENTIRELY of floaters (opacity < 0.04).
    Pruner must NOT generate an empty PLY; fallback safeguard must retain top 10% highest opacity.
    """
    np.random.seed(303)
    n = 200
    xyz = np.random.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    # All opacities are negative logits between -6.0 and -4.0 (all < 0.04)
    opacities = np.linspace(-6.0, -4.0, n, dtype=np.float32)
    scales = np.full((n, 3), -3.0, dtype=np.float32)

    temp_dir = Path(tempfile.mkdtemp(prefix="test_prune_fallback_"))
    in_ply = temp_dir / "input.ply"
    out_ply = temp_dir / "pruned.ply"

    try:
        _build_test_ply(xyz, opacities, scales, in_ply)

        stats = prune_gaussians(
            input_ply=in_ply,
            output_ply=out_ply,
            min_opacity=0.04,
            max_scale=0.15,
        )

        assert stats["kept_gaussians"] == 20  # Top 10% of 200
        assert stats["kept_percentage"] == 10.0
        assert out_ply.exists()
        pruned_data = PlyData.read(str(out_ply))
        assert len(pruned_data["vertex"].data) == 20
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_09_floater_pruning_zero_floaters_identity():
    """
    Test clean dataset with zero floaters: all Gaussians valid.
    Verify 100% retention and zero modifications.
    """
    np.random.seed(404)
    n = 300
    xyz = np.random.uniform(-0.4, 0.4, size=(n, 3)).astype(np.float32)
    opacities = np.full(n, 3.0, dtype=np.float32)
    scales = np.full((n, 3), -3.5, dtype=np.float32)

    temp_dir = Path(tempfile.mkdtemp(prefix="test_prune_identity_"))
    in_ply = temp_dir / "input.ply"
    out_ply = temp_dir / "pruned.ply"

    try:
        _build_test_ply(xyz, opacities, scales, in_ply)

        stats = prune_gaussians(
            input_ply=in_ply,
            output_ply=out_ply,
            min_opacity=0.04,
            max_scale=0.15,
        )

        assert stats["initial_gaussians"] == n
        assert stats["kept_gaussians"] == n
        assert stats["culled_gaussians"] == 0
        assert stats["kept_percentage"] == 100.0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_10_floater_pruning_direct_splat_export():
    """
    Test prune_gaussians with direct export_splat output.
    Verify exported .splat passes strict binary validation (verify_splat_file).
    """
    np.random.seed(505)
    n = 250
    xyz = np.random.uniform(-0.5, 0.5, size=(n, 3)).astype(np.float32)
    opacities = np.full(n, 2.0, dtype=np.float32)
    scales = np.full((n, 3), -3.0, dtype=np.float32)

    temp_dir = Path(tempfile.mkdtemp(prefix="test_prune_splat_"))
    in_ply = temp_dir / "input.ply"
    out_splat = temp_dir / "pruned.splat"

    try:
        _build_test_ply(xyz, opacities, scales, in_ply)

        stats = prune_gaussians(
            input_ply=in_ply,
            export_splat=out_splat,
            min_opacity=0.04,
            max_scale=0.15,
        )

        assert out_splat.exists()
        file_size = out_splat.stat().st_size
        assert file_size == n * 32

        v_res = verify_splat_file(out_splat, min_splat_count=50)
        assert v_res["valid"] is True
        assert v_res["num_splats"] == n
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# 4. EMPIRICAL STRESS TESTS & BOUNDARY CONDITIONS
# =============================================================================

def test_11_concurrent_multi_client_agent_stress(live_server):
    """
    Stress test: 4 concurrent SplatClientAgent instances running full cycles simultaneously.
    Verifies session isolation, queue management, and valid splat output for all clients.
    """
    num_clients = 4
    results = {}

    def worker(idx: int):
        c = SplatClientAgent(
            server_url=live_server,
            client_id=f"stress_agent_{idx}",
        )
        temp_splat = tempfile.mktemp(suffix=f"_stress_{idx}.splat")
        try:
            report = c.run_full_cycle(
                num_frames=3,
                iterations=1000,
                output_splat_path=temp_splat,
                timeout=30.0,
            )
            results[idx] = (report, temp_splat)
        except Exception as e:
            results[idx] = ({"success": False, "error": str(e)}, temp_splat)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=45.0)

    assert len(results) == num_clients
    for idx, (res, splat_path) in results.items():
        try:
            assert res.get("success") is True, f"Client {idx} failed: {res.get('error')}"
            assert os.path.exists(splat_path)
            assert os.path.getsize(splat_path) % 32 == 0
        finally:
            if os.path.exists(splat_path):
                try:
                    os.unlink(splat_path)
                except PermissionError:
                    pass


def test_12_download_security_path_traversal_defense(live_server):
    """
    Adversarial test: path traversal attempts on GET /download/{filename}.
    Server must sanitize path traversal patterns and return 404 (never leak system files).
    """
    traversal_attacks = [
        "../../server.py",
        "..\\..\\server.py",
        "....//....//server.py",
        "%2e%2e%2fserver.py",
        "etc/passwd",
    ]

    with httpx.Client(timeout=5.0) as client:
        for attack in traversal_attacks:
            resp = client.get(f"{live_server}/download/{attack}")
            # Must return 404 Not Found (or 400 Bad Request) — NEVER 200 with server code
            assert resp.status_code in (404, 400, 422), f"Attack '{attack}' returned unexpected status {resp.status_code}"
            assert "FastAPI" not in resp.text
            assert "coordinator" not in resp.text


def test_13_cancellation_during_active_ws_streaming(live_server):
    """
    Empirically verify job cancellation while WebSocket is actively streaming.
    Coordinator must terminate process, transition to FAILED, and notify subscribers.
    """
    client = SplatClientAgent(server_url=live_server)
    client.handshake()

    mock_zip = generate_mock_dataset(num_frames=4, as_zip=True)
    try:
        up_res = client.upload_scan(mock_zip, iterations=5000)
        scan_id = up_res["scan_id"]

        ws_port = live_server.split(":")[-1]
        ws_url = f"ws://127.0.0.1:{ws_port}/ws/telemetry/{scan_id}"

        terminal_frames = []

        async def cancel_flow():
            async with websockets.connect(ws_url) as ws:
                # Wait for 1 frame
                f1 = await asyncio.wait_for(ws.recv(), timeout=2.0)
                # Now cancel job via REST
                cancel_res = client.cancel_job(scan_id)
                assert cancel_res["cancelled"] is True

                # Wait for next frame
                for _ in range(10):
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        frame = json.loads(raw if isinstance(raw, str) else raw.decode())
                        terminal_frames.append(frame)
                        if frame.get("stage") in ("FAILED", "CANCELLED", "COMPLETED"):
                            break
                    except asyncio.TimeoutError:
                        break

        asyncio.run(cancel_flow())

        # Verify final status
        st = client.get_status(scan_id)
        assert st["stage"] in ("FAILED", "CANCELLED")
    finally:
        if os.path.exists(mock_zip):
            try:
                os.unlink(mock_zip)
            except Exception:
                pass


def test_14_parameter_negotiation_boundary_clamping(live_server):
    """
    Boundary stress test: parameter negotiation requests with extreme out-of-bounds values.
    Server must gracefully clamp values to safe bounds without 500 error.
    """
    client = SplatClientAgent(server_url=live_server)
    client.handshake()

    # Extreme low iterations (< 50)
    low_res = client.negotiate(iterations=-100)
    assert low_res["accepted"] is True
    assert low_res["config"]["iterations"] == 50

    # Extreme high iterations (> 30000)
    high_res = client.negotiate(iterations=500000)
    assert high_res["accepted"] is True
    assert high_res["config"]["iterations"] == 30000

    # Invalid resolution -> server falls back to default "1080p" or "720p"
    bad_res = client.negotiate(resolution="16k_super_extreme")
    assert bad_res["accepted"] is True
    assert bad_res["config"]["resolution"] in ("720p", "1080p")
