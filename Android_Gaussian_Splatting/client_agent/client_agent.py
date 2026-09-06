"""Headless Android Capture Client Agent for 3D Gaussian Splatting.

Simulates an Android client (Mobile3DGS CaptureActivity / ApiClient) interacting with
the FastAPI synchronization server:
- Coordinates handshake (/handshake) and parameter negotiation (/negotiate).
- Uploads ARCore synthetic datasets (/upload).
- Subscribes to real-time WebSocket telemetry (/ws/telemetry/{scan_id}).
- Downloads trained .splat binary models (/download/{filename}).
- Validates model delivery and binary specifications.
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import httpx
import websockets

# Import local mock capture
try:
    from .mock_capture import generate_mock_dataset
except (ImportError, ValueError):
    from mock_capture import generate_mock_dataset


class SplatClientAgent:
    """Headless simulation client agent for Android 3DGS."""

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000",
        client_id: Optional[str] = None,
        device_model: str = "Google Pixel 8 Pro",
        app_version: str = "1.0.0",
        timeout: float = 30.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.client_id = client_id or f"android_client_{uuid.uuid4().hex[:8]}"
        self.device_model = device_model
        self.app_version = app_version
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self.negotiated_params: Dict[str, Any] = {}
        self.server_capabilities: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # REST Endpoints
    # -------------------------------------------------------------------------

    def check_health(self) -> Dict[str, Any]:
        """Queries GET /health."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.server_url}/health")
            resp.raise_for_status()
            return resp.json()

    def handshake(
        self,
        capabilities: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Executes POST /handshake to initialize a session and obtain server limits."""
        if capabilities is None:
            capabilities = {
                "sensor_fusion": True,
                "vulkan_version": "1.3",
                "camera_fov": 65.5,
                "device_ram_gb": 12,
            }

        payload = {
            "client_id": self.client_id,
            "device_model": self.device_model,
            "app_version": self.app_version,
            "capabilities": capabilities,
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.server_url}/handshake", json=payload)
            resp.raise_for_status()
            data = resp.json()
            self.session_id = data.get("session_id")
            self.server_capabilities = data
            return data

    def negotiate(
        self,
        iterations: int = 2000,
        resolution: str = "720p",
        prune_floaters: bool = True,
        sh_degree: int = 3,
        session_id: Optional[str] = None,
        fast_test: bool = True,
    ) -> Dict[str, Any]:
        """Executes POST /negotiate to set pipeline parameters."""
        sid = session_id or self.session_id
        if not sid:
            # Auto-handshake if not done yet
            self.handshake()
            sid = self.session_id

        payload = {
            "session_id": sid,
            "iterations": iterations,
            "resolution": resolution,
            "prune_floaters": prune_floaters,
            "sh_degree": sh_degree,
            "fast_test": fast_test,
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.server_url}/negotiate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            self.negotiated_params = data.get("params", data.get("config", payload))
            return data

    def upload_scan(
        self,
        zip_path: str,
        iterations: Optional[int] = None,
        resolution: Optional[str] = None,
        prune_floaters: Optional[bool] = None,
        session_id: Optional[str] = None,
        fast_test: bool = True,
    ) -> Dict[str, Any]:
        """Executes POST /upload with a multipart zip archive."""
        file_path = Path(zip_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset archive not found: {zip_path}")

        sid = session_id or self.session_id
        data_fields: Dict[str, Any] = {}
        if sid:
            data_fields["session_id"] = sid
        if iterations is not None:
            data_fields["iterations"] = str(iterations)
        elif "iterations" in self.negotiated_params:
            data_fields["iterations"] = str(self.negotiated_params["iterations"])

        if resolution is not None:
            data_fields["resolution"] = resolution
        elif "resolution" in self.negotiated_params:
            data_fields["resolution"] = str(self.negotiated_params["resolution"])

        if prune_floaters is not None:
            data_fields["prune_floaters"] = str(prune_floaters).lower()
        elif "prune_floaters" in self.negotiated_params:
            data_fields["prune_floaters"] = str(self.negotiated_params["prune_floaters"]).lower()

        data_fields["fast_test"] = "true" if fast_test else "false"

        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/zip")}
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{self.server_url}/upload",
                    data=data_fields,
                    files=files,
                )
                resp.raise_for_status()
                return resp.json()

    def get_status(self, scan_id: str) -> Dict[str, Any]:
        """Queries GET /status/{scan_id}."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.server_url}/status/{scan_id}")
            resp.raise_for_status()
            return resp.json()

    def cancel_job(self, scan_id: str) -> Dict[str, Any]:
        """Executes DELETE /jobs/{scan_id} to cancel an active job."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.delete(f"{self.server_url}/jobs/{scan_id}")
            resp.raise_for_status()
            return resp.json()

    def list_models(self) -> Dict[str, Any]:
        """Queries GET /models."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.server_url}/models")
            resp.raise_for_status()
            return resp.json()

    def download_splat(
        self,
        filename_or_url: str,
        output_path: str,
    ) -> str:
        """Downloads binary .splat file via GET /download/{filename}."""
        if "/" in filename_or_url:
            filename = Path(filename_or_url).name
        else:
            filename = filename_or_url

        dest_file = Path(output_path)
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        download_url = f"{self.server_url}/download/{filename}"
        with httpx.Client(timeout=120.0) as client:
            with client.stream("GET", download_url) as resp:
                resp.raise_for_status()
                with open(dest_file, "wb") as out:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        out.write(chunk)

        return str(dest_file.resolve())

    # -------------------------------------------------------------------------
    # WebSocket Telemetry
    # -------------------------------------------------------------------------

    async def _listen_telemetry_async(
        self,
        scan_id: str,
        timeout: float = 60.0,
        on_frame: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Asynchronously listens to ws://host:port/ws/telemetry/{scan_id}."""
        parsed = urlparse(self.server_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        ws_url = f"{ws_scheme}://{parsed.netloc}/ws/telemetry/{scan_id}"

        frames = []
        start_time = time.time()

        try:
            async with websockets.connect(ws_url, open_timeout=10.0) as ws:
                while time.time() - start_time < timeout:
                    try:
                        raw_msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        if isinstance(raw_msg, str):
                            frame = json.loads(raw_msg)
                        else:
                            frame = json.loads(raw_msg.decode("utf-8"))

                        frames.append(frame)
                        if on_frame:
                            on_frame(frame)

                        stage = str(frame.get("stage", frame.get("state", ""))).upper()
                        if stage in ("COMPLETED", "FAILED", "CANCELLED"):
                            break
                    except asyncio.TimeoutError:
                        # Check HTTP status as fallback
                        try:
                            status_data = self.get_status(scan_id)
                            status_str = str(status_data.get("status", "")).upper()
                            if status_str in ("COMPLETED", "FAILED", "CANCELLED"):
                                frames.append(status_data)
                                break
                        except Exception:
                            pass
                        continue
        except Exception as e:
            # If websocket fails to connect, fallback to polling HTTP status
            while time.time() - start_time < timeout:
                try:
                    status_data = self.get_status(scan_id)
                    frames.append(status_data)
                    if on_frame:
                        on_frame(status_data)
                    status_str = str(status_data.get("status", "")).upper()
                    if status_str in ("COMPLETED", "FAILED", "CANCELLED"):
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.5)

        return frames

    def listen_telemetry(
        self,
        scan_id: str,
        timeout: float = 60.0,
        on_frame: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for telemetry listener."""
        return asyncio.run(self._listen_telemetry_async(scan_id, timeout, on_frame))

    # -------------------------------------------------------------------------
    # Full End-to-End Execution Cycle
    # -------------------------------------------------------------------------

    def run_full_cycle(
        self,
        num_frames: int = 8,
        iterations: int = 2000,
        resolution: str = "720p",
        prune_floaters: bool = True,
        output_splat_path: Optional[str] = None,
        timeout: float = 90.0,
    ) -> Dict[str, Any]:
        """Runs the complete simulated mobile capture, handshake, upload, telemetry,

        and download cycle.
        """
        report: Dict[str, Any] = {
            "client_id": self.client_id,
            "steps": {},
            "success": False,
        }

        # 1. Generate Mock Capture Dataset
        t0 = time.time()
        mock_zip = generate_mock_dataset(num_frames=num_frames, as_zip=True)
        report["steps"]["mock_dataset"] = {
            "status": "PASS",
            "path": mock_zip,
            "duration_sec": round(time.time() - t0, 3),
        }

        # 2. Handshake
        t0 = time.time()
        hs_res = self.handshake()
        report["steps"]["handshake"] = {
            "status": "PASS",
            "session_id": self.session_id,
            "response": hs_res,
            "duration_sec": round(time.time() - t0, 3),
        }

        # 3. Negotiate
        t0 = time.time()
        neg_res = self.negotiate(
            iterations=iterations,
            resolution=resolution,
            prune_floaters=prune_floaters,
        )
        report["steps"]["negotiate"] = {
            "status": "PASS",
            "response": neg_res,
            "duration_sec": round(time.time() - t0, 3),
        }

        # 4. Upload Dataset
        t0 = time.time()
        up_res = self.upload_scan(
            zip_path=mock_zip,
            iterations=iterations,
            resolution=resolution,
            prune_floaters=prune_floaters,
        )
        scan_id = up_res.get("scan_id")
        report["steps"]["upload"] = {
            "status": "PASS",
            "scan_id": scan_id,
            "response": up_res,
            "duration_sec": round(time.time() - t0, 3),
        }

        # 5. Listen to Telemetry Stream
        t0 = time.time()
        frames = self.listen_telemetry(scan_id, timeout=timeout)
        report["steps"]["telemetry"] = {
            "status": "PASS" if frames else "WARN",
            "frame_count": len(frames),
            "final_stage": frames[-1].get("stage", frames[-1].get("status")) if frames else "UNKNOWN",
            "duration_sec": round(time.time() - t0, 3),
        }

        # 6. Check final status
        final_status = self.get_status(scan_id)
        report["steps"]["final_status"] = final_status

        # 7. Download Model Artifact if completed
        splat_url = final_status.get("splat_url")
        if not splat_url:
            filename = f"{scan_id}.splat"
        else:
            filename = Path(splat_url).name

        if output_splat_path is None:
            fd, output_splat_path = tempfile.mkstemp(suffix=".splat", prefix=f"model_{scan_id}_")
            os.close(fd)

        t0 = time.time()
        try:
            downloaded = self.download_splat(filename, output_splat_path)
            file_size = Path(downloaded).stat().st_size
            report["steps"]["download"] = {
                "status": "PASS",
                "path": downloaded,
                "file_size_bytes": file_size,
                "duration_sec": round(time.time() - t0, 3),
            }

            # 8. Binary format verification
            is_valid_size = (file_size > 0) and (file_size % 32 == 0)
            report["steps"]["splat_verification"] = {
                "status": "PASS" if is_valid_size else "FAIL",
                "splats_count": file_size // 32,
                "size_divisible_by_32": is_valid_size,
            }
            report["success"] = is_valid_size
        except Exception as e:
            report["steps"]["download"] = {
                "status": "FAIL",
                "error": str(e),
                "duration_sec": round(time.time() - t0, 3),
            }
            report["success"] = False

        # Cleanup mock zip
        if Path(mock_zip).exists():
            try:
                os.unlink(mock_zip)
            except Exception:
                pass

        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless Android 3DGS Simulation Client")
    parser.add_argument("--server", default="http://127.0.0.1:8000", help="FastAPI Server URL")
    parser.add_argument("--frames", type=int, default=8, help="Number of synthetic frames")
    parser.add_argument("--iterations", type=int, default=2000, help="Training iterations")
    parser.add_argument("--output", "-o", default=None, help="Output .splat destination")
    parser.add_argument("--timeout", type=float, default=60.0, help="Operation timeout in seconds")
    args = parser.parse_args()

    client = SplatClientAgent(server_url=args.server)
    print(f"Starting simulated capture client for server: {args.server}")
    result = client.run_full_cycle(
        num_frames=args.frames,
        iterations=args.iterations,
        output_splat_path=args.output,
        timeout=args.timeout,
    )
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["success"] else 1)
