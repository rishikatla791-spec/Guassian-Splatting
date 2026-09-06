"""
Pipeline Runner: Coordinates the end-to-end 3D Gaussian Splatting desktop pipeline.

Pipeline Sequence:
  1. Ingestion: Convert ARCore transforms.json -> COLMAP sparse/0/ (cameras.txt, images.txt, points3D.ply)
  2. Training: Headless CUDA training (gaussian-splatting/train.py --disable_viewer) with stdout telemetry
  3. Floater Pruning: Headless post-training opacity & scale culling (prune_floaters.py)
  4. Splat Conversion: Binary 32-byte .splat packing (ply_to_splat.py)
  5. Verification: Strict binary validator confirming 32-byte alignment, finite floats, valid scales & quaternions (verify_splat.py)

Includes high-speed dry-run / simulation mode for integration testing and automated CI.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from plyfile import PlyData, PlyElement

# Import local pipeline modules
from dataset_adapter import convert_transforms_to_colmap
from prune_floaters import prune_gaussians
from verify_splat import verify_splat_file
from ply_to_splat import convert_ply_to_splat


class PipelineStage(str, Enum):
    IDLE = "IDLE"
    INGESTION = "INGESTION"
    TRAINING = "TRAINING"
    PRUNING = "PRUNING"
    CONVERTING = "CONVERTING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class TelemetryEvent:
    """Structured telemetry event emitted during pipeline execution."""
    scan_id: str
    stage: PipelineStage
    iteration: int = 0
    total_iterations: int = 0
    progress: float = 0.0
    loss: Optional[float] = None
    eta_seconds: Optional[float] = None
    message: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "stage": self.stage.value,
            "iteration": self.iteration,
            "total_iterations": self.total_iterations,
            "progress": round(self.progress, 4),
            "loss": round(self.loss, 6) if self.loss is not None else None,
            "eta_seconds": round(self.eta_seconds, 1) if self.eta_seconds is not None else None,
            "message": self.message,
            "timestamp": self.timestamp
        }


@dataclass
class PipelineConfig:
    """Configuration options for pipeline execution."""
    scan_id: str
    scan_dir: Path
    output_dir: Path
    models_dir: Path
    iterations: int = 2000
    resolution: int = 2
    prune_floaters: bool = True
    min_opacity: float = 0.04
    max_scale: float = 0.15
    white_background: bool = False
    simulation_mode: bool = False
    simulation_steps: int = 20
    python_exe: Optional[str] = None
    gaussian_repo: Optional[Path] = None


@dataclass
class PipelineResult:
    """Final result of a pipeline execution."""
    scan_id: str
    success: bool
    final_stage: PipelineStage
    splat_path: Optional[Path] = None
    splat_stats: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    telemetry_history: List[Dict[str, Any]] = field(default_factory=list)


class PipelineRunner:
    """
    Executes and coordinates the 3D Gaussian Splatting desktop training pipeline.
    """

    def __init__(self):
        self._current_process: Optional[subprocess.Popen] = None
        self._cancel_requested: bool = False
        self._lock = threading.Lock()

    def cancel(self) -> None:
        """Requests cancellation of running pipeline subprocess."""
        with self._lock:
            self._cancel_requested = True
            if self._current_process and self._current_process.poll() is None:
                try:
                    self._current_process.terminate()
                    time.sleep(0.5)
                    if self._current_process.poll() is None:
                        self._current_process.kill()
                except Exception:
                    pass

    def run(
        self,
        config: PipelineConfig,
        telemetry_callback: Optional[Callable[[TelemetryEvent], None]] = None
    ) -> PipelineResult:
        """
        Runs the complete 5-stage pipeline sequence.
        """
        start_time = time.time()
        telemetry_history: List[Dict[str, Any]] = []

        def emit(stage: PipelineStage, iteration: int = 0, total: int = 0,
                 prog: float = 0.0, loss: Optional[float] = None,
                 eta: Optional[float] = None, msg: str = ""):
            evt = TelemetryEvent(
                scan_id=config.scan_id,
                stage=stage,
                iteration=iteration,
                total_iterations=total,
                progress=prog,
                loss=loss,
                eta_seconds=eta,
                message=msg
            )
            telemetry_history.append(evt.to_dict())
            if telemetry_callback:
                try:
                    telemetry_callback(evt)
                except Exception:
                    pass

        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.models_dir.mkdir(parents=True, exist_ok=True)

        try:
            # =================================================================
            # STAGE 1: DATASET INGESTION
            # =================================================================
            emit(PipelineStage.INGESTION, msg="Starting dataset ingestion and COLMAP sparse conversion...")
            sparse_dir = config.scan_dir / "sparse" / "0"

            # Check if conversion needed
            if not (sparse_dir / "cameras.txt").exists() or not (sparse_dir / "images.txt").exists():
                ingest_summary = convert_transforms_to_colmap(
                    scan_dir=config.scan_dir,
                    output_sparse_dir=sparse_dir,
                    opengl_to_colmap=True
                )
                emit(PipelineStage.INGESTION, prog=0.1,
                     msg=f"Ingestion complete: {ingest_summary['num_images']} images, {ingest_summary['num_points']} sparse points")
            else:
                emit(PipelineStage.INGESTION, prog=0.1, msg="Existing COLMAP sparse/0 model found; skipping conversion")

            if self._cancel_requested:
                emit(PipelineStage.CANCELLED, msg="Pipeline cancelled by user")
                return PipelineResult(config.scan_id, False, PipelineStage.CANCELLED, elapsed_seconds=time.time() - start_time)

            # =================================================================
            # STAGE 2: TRAINING (CUDA OR SIMULATION)
            # =================================================================
            if config.simulation_mode:
                ply_path = self._run_simulation_training(config, emit)
            else:
                ply_path = self._run_cuda_training(config, emit)

            if self._cancel_requested:
                emit(PipelineStage.CANCELLED, msg="Pipeline cancelled by user")
                return PipelineResult(config.scan_id, False, PipelineStage.CANCELLED, elapsed_seconds=time.time() - start_time)

            # =================================================================
            # STAGE 3: FLOATER PRUNING
            # =================================================================
            pruned_ply = config.output_dir / "pruned.ply"
            if config.prune_floaters:
                emit(PipelineStage.PRUNING, prog=0.85, msg="Pruning floaters and oversized Gaussians...")
                prune_stats = prune_gaussians(
                    input_ply=ply_path,
                    output_ply=pruned_ply,
                    min_opacity=config.min_opacity,
                    max_scale=config.max_scale,
                    relative_scale=True
                )
                emit(PipelineStage.PRUNING, prog=0.90,
                     msg=f"Pruning complete: kept {prune_stats['kept_gaussians']}/{prune_stats['initial_gaussians']} ({prune_stats['kept_percentage']}%)")
                source_ply_for_splat = pruned_ply
            else:
                source_ply_for_splat = ply_path

            # =================================================================
            # STAGE 4: SPLAT CONVERSION
            # =================================================================
            emit(PipelineStage.CONVERTING, prog=0.92, msg="Packing Gaussians to 32-byte binary .splat...")
            splat_path = config.models_dir / f"{config.scan_id}.splat"
            convert_ply_to_splat(str(source_ply_for_splat), str(splat_path))
            emit(PipelineStage.CONVERTING, prog=0.96, msg=f"Saved binary .splat to {splat_path.name}")

            # =================================================================
            # STAGE 5: STRICT VERIFICATION
            # =================================================================
            emit(PipelineStage.VERIFYING, prog=0.98, msg="Validating binary .splat format integrity...")
            validation_stats = verify_splat_file(splat_path, min_splat_count=100)

            total_elapsed = time.time() - start_time
            emit(
                PipelineStage.COMPLETED,
                iteration=config.iterations,
                total=config.iterations,
                prog=1.0,
                msg=f"Pipeline completed successfully in {total_elapsed:.2f}s ({validation_stats['num_splats']} splats)"
            )

            return PipelineResult(
                scan_id=config.scan_id,
                success=True,
                final_stage=PipelineStage.COMPLETED,
                splat_path=splat_path,
                splat_stats=validation_stats,
                elapsed_seconds=total_elapsed,
                telemetry_history=telemetry_history
            )

        except Exception as e:
            total_elapsed = time.time() - start_time
            err_msg = str(e)
            emit(PipelineStage.FAILED, msg=f"Error: {err_msg}")
            return PipelineResult(
                scan_id=config.scan_id,
                success=False,
                final_stage=PipelineStage.FAILED,
                error=err_msg,
                elapsed_seconds=total_elapsed,
                telemetry_history=telemetry_history
            )

    def _run_simulation_training(
        self,
        config: PipelineConfig,
        emit: Callable[..., None]
    ) -> Path:
        """
        Fast high-speed simulation mode for automated testing without requiring GPU execution.
        Synthesizes genuine 3D Gaussian Splats and writes a valid point_cloud.ply.
        """
        steps = max(5, config.simulation_steps)
        total_iters = config.iterations
        step_size = max(1, total_iters // steps)

        sim_start = time.time()
        for i in range(1, steps + 1):
            if self._cancel_requested:
                break
            current_iter = min(total_iters, i * step_size)
            prog = 0.1 + 0.75 * (current_iter / total_iters)
            # Realistic synthetic loss curve: exponential decay + residual
            loss = 0.35 * math.exp(-3.0 * (current_iter / total_iters)) + 0.025
            sim_elapsed = time.time() - sim_start
            rate = current_iter / max(sim_elapsed, 1e-4)
            eta = (total_iters - current_iter) / max(rate, 1e-4)

            emit(
                PipelineStage.TRAINING,
                iteration=current_iter,
                total=total_iters,
                prog=prog,
                loss=loss,
                eta=eta,
                msg=f"Simulating optimization: step {current_iter}/{total_iters} (loss={loss:.4f})"
            )
            time.sleep(0.01)  # Micro-sleep for telemetry observation

        # Synthesize a genuine 3D Gaussian point cloud
        out_ply_dir = config.output_dir / "point_cloud" / f"iteration_{total_iters}"
        out_ply_dir.mkdir(parents=True, exist_ok=True)
        ply_file = out_ply_dir / "point_cloud.ply"

        self._synthesize_simulation_ply(ply_file, num_gaussians=2500)
        return ply_file

    def _synthesize_simulation_ply(self, output_ply: Path, num_gaussians: int = 2500) -> None:
        """Synthesize a genuine 3D Gaussian PLY cloud with realistic geometry and attributes."""
        np.random.seed(42)
        n = num_gaussians

        # Helix / Torus geometry
        t = np.linspace(0, 8 * np.pi, n)
        r = 1.0 + 0.2 * np.sin(3.0 * t)
        x = r * np.cos(t) + np.random.normal(0, 0.02, n)
        y = (t / (8 * np.pi) - 0.5) * 2.0 + np.random.normal(0, 0.02, n)
        z = r * np.sin(t) + np.random.normal(0, 0.02, n)

        # Spherical harmonics DC component (RGB)
        C0 = 0.28209479177387814
        rgb = np.column_stack([
            (np.sin(t * 0.5) * 0.5 + 0.5),
            (np.cos(t * 0.3) * 0.5 + 0.5),
            (np.sin(t * 0.7 + 1.0) * 0.5 + 0.5)
        ])
        f_dc = (rgb - 0.5) / C0

        # Opacity: mix of solid points and ~5% low-opacity floaters to exercise pruning
        raw_opacity = np.random.normal(2.5, 0.5, n)  # sigmoid(2.5) ~ 0.92
        floater_indices = np.random.choice(n, size=int(n * 0.08), replace=False)
        raw_opacity[floater_indices] = -4.0  # sigmoid(-4.0) ~ 0.018 < 0.04

        # Scale: log-space scales
        raw_scale = np.random.normal(-3.5, 0.3, (n, 3))  # exp(-3.5) ~ 0.03
        oversize_indices = np.random.choice(n, size=int(n * 0.04), replace=False)
        raw_scale[oversize_indices] = 1.5  # exp(1.5) ~ 4.48 (oversized floater)

        # Rotation: normalized quaternions [rot_0, rot_1, rot_2, rot_3]
        rot = np.random.normal(0, 1.0, (n, 4))
        rot /= np.linalg.norm(rot, axis=1, keepdims=True)

        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
        ]
        elements = np.empty(n, dtype=dtype)
        elements['x'] = x
        elements['y'] = y
        elements['z'] = z
        elements['nx'] = 0.0; elements['ny'] = 0.0; elements['nz'] = 0.0
        elements['f_dc_0'] = f_dc[:, 0]
        elements['f_dc_1'] = f_dc[:, 1]
        elements['f_dc_2'] = f_dc[:, 2]
        elements['opacity'] = raw_opacity
        elements['scale_0'] = raw_scale[:, 0]
        elements['scale_1'] = raw_scale[:, 1]
        elements['scale_2'] = raw_scale[:, 2]
        elements['rot_0'] = rot[:, 0]
        elements['rot_1'] = rot[:, 1]
        elements['rot_2'] = rot[:, 2]
        elements['rot_3'] = rot[:, 3]

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el], text=False).write(str(output_ply))

    def _run_cuda_training(
        self,
        config: PipelineConfig,
        emit: Callable[..., None]
    ) -> Path:
        """
        Executes genuine headless CUDA training using gaussian-splatting/train.py.
        Monitors stdout in real time, parses tqdm iteration and loss, and calculates ETA.
        """
        workspace_root = Path(__file__).resolve().parent.parent.parent
        # Locate train.py
        if config.gaussian_repo and (config.gaussian_repo / "train.py").exists():
            gaussian_repo = config.gaussian_repo
        elif (workspace_root / "Laptop_Gaussian_Splatting" / "gaussian-splatting" / "train.py").exists():
            gaussian_repo = workspace_root / "Laptop_Gaussian_Splatting" / "gaussian-splatting"
        elif (workspace_root / "gaussian-splatting" / "train.py").exists():
            gaussian_repo = workspace_root / "gaussian-splatting"
        else:
            raise FileNotFoundError("Could not locate gaussian-splatting/train.py repository")

        train_script = gaussian_repo / "train.py"
        python_exe = config.python_exe or sys.executable

        # Prepare Environment with injected PYTHONPATH for compiled CUDA extensions
        env = os.environ.copy()
        submodules_dir = gaussian_repo / "submodules"
        rasterizer_dir = submodules_dir / "diff-gaussian-rasterization"
        fused_ssim_dir = submodules_dir / "fused-ssim"

        pythonpath_entries = [str(gaussian_repo)]
        if rasterizer_dir.exists():
            pythonpath_entries.append(str(rasterizer_dir))
        if fused_ssim_dir.exists():
            pythonpath_entries.append(str(fused_ssim_dir))
        if "PYTHONPATH" in env:
            pythonpath_entries.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = ";".join(pythonpath_entries)

        # Force unbuffered python output
        env["PYTHONUNBUFFERED"] = "1"

        cmd = [
            str(python_exe),
            str(train_script),
            "-s", str(config.scan_dir),
            "-m", str(config.output_dir),
            "--iterations", str(config.iterations),
            "--save_iterations", str(config.iterations),
            "--resolution", str(config.resolution),
            "--disable_viewer"
        ]
        if config.white_background:
            cmd.append("--white_background")

        emit(PipelineStage.TRAINING, msg=f"Spawning CUDA trainer: iterations={config.iterations}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(gaussian_repo),
            text=True,
            bufsize=1
        )
        with self._lock:
            self._current_process = process

        # Parse output stream with carriage return awareness
        start_train_time = time.time()
        last_emit_iter = 0

        buffer = ""
        while True:
            char = process.stdout.read(1)
            if not char:
                break
            if char in ('\r', '\n'):
                line = buffer.strip()
                buffer = ""
                if not line:
                    continue

                # Check for tqdm output e.g. "Training progress:  45%|████▍ | 1350/3000 [00:45<00:55, 29.80it/s, Loss=0.034512]"
                if "Training progress" in line:
                    try:
                        # Extract percentage
                        pct_match = re.search(r'(\d+)%', line)
                        percent = float(pct_match.group(1)) if pct_match else 0.0

                        # Extract step/total
                        step_match = re.search(r'(\d+)/(\d+)', line)
                        if step_match:
                            cur_iter = int(step_match.group(1))
                            tot_iter = int(step_match.group(2))
                        else:
                            cur_iter = int(percent * config.iterations / 100.0)
                            tot_iter = config.iterations

                        # Extract loss
                        loss_match = re.search(r'Loss[:=]\s*([0-9.]+)', line)
                        loss_val = float(loss_match.group(1)) if loss_match else None

                        # Calculate ETA
                        elapsed = time.time() - start_train_time
                        rate = cur_iter / max(elapsed, 1e-4)
                        eta_sec = (tot_iter - cur_iter) / max(rate, 1e-4)
                        prog = 0.1 + 0.75 * (cur_iter / max(tot_iter, 1))

                        if cur_iter >= last_emit_iter + 20 or cur_iter >= tot_iter:
                            last_emit_iter = cur_iter
                            emit(
                                PipelineStage.TRAINING,
                                iteration=cur_iter,
                                total=tot_iter,
                                prog=prog,
                                loss=loss_val,
                                eta=eta_sec,
                                msg=f"Optimization step {cur_iter}/{tot_iter} (loss={loss_val if loss_val is not None else 0.0:.5f})"
                            )
                    except Exception:
                        pass
            else:
                buffer += char

        process.wait()
        with self._lock:
            self._current_process = None

        if process.returncode != 0 and not self._cancel_requested:
            raise RuntimeError(f"CUDA training subprocess failed with exit code {process.returncode}")

        # Locate point_cloud.ply
        expected_ply = config.output_dir / "point_cloud" / f"iteration_{config.iterations}" / "point_cloud.ply"
        if expected_ply.exists():
            return expected_ply

        # Fallback search
        for p in config.output_dir.glob("**/point_cloud.ply"):
            return p

        raise FileNotFoundError(f"Training finished but point_cloud.ply not found in {config.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Desktop 3DGS Pipeline Runner")
    parser.add_argument("--scan-dir", "-s", required=True, type=str, help="Directory of uploaded scan")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory")
    parser.add_argument("--models-dir", "-m", type=str, default=None, help="Models output directory")
    parser.add_argument("--iterations", "-i", type=int, default=2000, help="Total training iterations")
    parser.add_argument("--resolution", "-r", type=int, default=2, help="Resolution downscale factor")
    parser.add_argument("--no-prune", action="store_true", help="Skip floater pruning")
    parser.add_argument("--simulation", action="store_true", help="Run high-speed dry-run simulation mode")

    args = parser.parse_args()

    scan_path = Path(args.scan_dir).resolve()
    scan_id = scan_path.name
    output_path = Path(args.output_dir or (scan_path.parent.parent / "outputs" / scan_id)).resolve()
    models_path = Path(args.models_dir or (scan_path.parent.parent / "models")).resolve()

    cfg = PipelineConfig(
        scan_id=scan_id,
        scan_dir=scan_path,
        output_dir=output_path,
        models_dir=models_path,
        iterations=args.iterations,
        resolution=args.resolution,
        prune_floaters=not args.no_prune,
        simulation_mode=args.simulation
    )

    runner = PipelineRunner()

    def on_telemetry(evt: TelemetryEvent):
        print(f"[{evt.stage.value}] step={evt.iteration}/{evt.total_iterations} prog={evt.progress:.2f} msg={evt.message}")

    result = runner.run(cfg, telemetry_callback=on_telemetry)
    print("\n--- Pipeline Result ---")
    print(json.dumps({
        "scan_id": result.scan_id,
        "success": result.success,
        "final_stage": result.final_stage.value,
        "splat_path": str(result.splat_path) if result.splat_path else None,
        "elapsed_seconds": round(result.elapsed_seconds, 2),
        "error": result.error,
        "splat_stats": result.splat_stats
    }, indent=2))

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
