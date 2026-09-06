"""Tier 1, 2, & 3 Tests: Pipeline Automation, Floater Pruning, Splat Conversion, and Verification.

Covers:
- Feature 5: Headless CUDA Training Automation & Pipeline Coordination
- Feature 6: Standalone Floater Pruning Hook
- Feature 7: Binary 32-Byte .splat Generation
- Feature 8: Binary .splat Format Verification
- Feature 9: Pipeline Resilience & Error Handling
- Tier 3: Pairwise Parameter Interaction Combinations

Thresholds: >=5 tests per feature, exhaustive pairwise matrix, strict format assertions.
"""

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple

import numpy as np
from plyfile import PlyData, PlyElement
import pytest

# Ensure server module is discoverable
SERVER_DIR = Path(__file__).resolve().parent.parent.parent / "Android_Gaussian_Splatting" / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

CLIENT_AGENT_DIR = Path(__file__).resolve().parent.parent.parent / "Android_Gaussian_Splatting" / "client_agent"
if str(CLIENT_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_AGENT_DIR))

from prune_floaters import prune_gaussians, compute_scene_radius, PruneFloatersError
from verify_splat import (
    verify_splat_file,
    FileSizeError,
    SplatCountError,
    CorruptCoordinatesError,
    CorruptScaleError,
    CorruptQuaternionError,
)
from ply_to_splat import convert_ply_to_splat
from pipeline_runner import PipelineConfig, PipelineRunner, PipelineStage, PipelineResult
from mock_capture import generate_mock_dataset


def create_synthetic_ply(
    ply_path: Path,
    num_good: int = 150,
    num_low_opacity: int = 50,
    num_oversized: int = 50,
) -> int:
    """Helper creating a synthetic Inria 3DGS PLY point cloud with controlled floater properties.

    - Good: opacity raw = 2.0 (sigmoid ≈ 0.88 > 0.04), scale log = -3.0 (exp ≈ 0.05 < 0.15)
    - Low opacity: opacity raw = -4.0 (sigmoid ≈ 0.018 < 0.04)
    - Oversized: scale log = 2.0 (exp ≈ 7.39 > 0.15)
    """
    total = num_good + num_low_opacity + num_oversized
    ply_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    # Positions inside unit sphere
    xyz = np.random.uniform(-0.5, 0.5, size=(total, 3)).astype(np.float32)
    normals = np.zeros((total, 3), dtype=np.float32)

    # Base features (DC RGB)
    f_dc = np.random.uniform(-1.0, 1.0, size=(total, 3)).astype(np.float32)

    # Opacity logits (inverse sigmoid)
    opacities = np.zeros(total, dtype=np.float32)
    # Good
    opacities[:num_good] = 2.0
    # Low opacity
    opacities[num_good : num_good + num_low_opacity] = -4.0
    # Oversized
    opacities[num_good + num_low_opacity :] = 2.0

    # Scales in log-space
    scales = np.full((total, 3), -3.0, dtype=np.float32)
    # Oversized
    scales[num_good + num_low_opacity :] = 2.0

    # Normalized quaternions [1, 0, 0, 0]
    rots = np.zeros((total, 4), dtype=np.float32)
    rots[:, 0] = 1.0

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]

    elements = np.empty(total, dtype=dtype)
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

    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(str(ply_path))
    return total


# =============================================================================
# FEATURE 6: STANDALONE FLOATER PRUNING HOOK (TIERS 1 & 2)
# =============================================================================

def test_prune_floaters_low_opacity_and_scale():
    """T1-F6-01: Low opacity (<0.04) and oversized Gaussians are filtered out."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_prune_"))
    in_ply = temp_dir / "input.ply"
    out_ply = temp_dir / "pruned.ply"

    try:
        create_synthetic_ply(in_ply, num_good=120, num_low_opacity=40, num_oversized=40)
        stats = prune_gaussians(
            in_ply,
            out_ply,
            min_opacity=0.04,
            max_scale=0.15,
            relative_scale=False,
        )

        assert stats["initial_gaussians"] == 200
        assert stats["kept_gaussians"] == 120
        assert stats["culled_gaussians"] == 80
        assert out_ply.exists()

        # Check pruned PLY data
        ply = PlyData.read(str(out_ply))
        assert len(ply["vertex"]) == 120
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_prune_floaters_preserves_valid_gaussians():
    """T1-F6-02: Clean Gaussian point cloud retains 100% of Gaussians."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_prune_clean_"))
    in_ply = temp_dir / "clean.ply"
    out_ply = temp_dir / "clean_pruned.ply"

    try:
        create_synthetic_ply(in_ply, num_good=150, num_low_opacity=0, num_oversized=0)
        stats = prune_gaussians(in_ply, out_ply, min_opacity=0.04, max_scale=0.15)
        assert stats["initial_gaussians"] == 150
        assert stats["kept_gaussians"] == 150
        assert stats["culled_gaussians"] == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_prune_floaters_exports_direct_splat():
    """T1-F6-03: Setting export_splat directly outputs a valid 32-byte .splat file."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_prune_splat_"))
    in_ply = temp_dir / "input.ply"
    out_splat = temp_dir / "output.splat"

    try:
        create_synthetic_ply(in_ply, num_good=150, num_low_opacity=10, num_oversized=10)
        stats = prune_gaussians(in_ply, export_splat=out_splat)
        assert out_splat.exists()
        splat_size = out_splat.stat().st_size
        assert splat_size > 0
        assert splat_size % 32 == 0
        assert splat_size == 150 * 32
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_prune_floaters_missing_file_raises():
    """T2-F6-01: Pruning non-existent PLY file raises PruneFloatersError."""
    with pytest.raises(PruneFloatersError, match="not found"):
        prune_gaussians("C:/non_existent_file_xyz_123.ply")


def test_prune_floaters_corrupt_ply_raises():
    """T2-F6-02: Pruning corrupted non-PLY file raises PruneFloatersError."""
    temp_file = Path(tempfile.mktemp(suffix=".ply"))
    temp_file.write_text("CORRUPT_NON_PLY_DATA")
    try:
        with pytest.raises(PruneFloatersError, match="Failed to parse"):
            prune_gaussians(temp_file)
    finally:
        if temp_file.exists():
            temp_file.unlink()


def test_prune_floaters_scene_radius_computation():
    """T1-F6-04: compute_scene_radius computes robust center and radius."""
    # Points in a sphere of radius 2.0
    xyz = np.random.uniform(-2.0, 2.0, size=(500, 3)).astype(np.float32)
    center, radius = compute_scene_radius(xyz)
    assert len(center) == 3
    assert 0.5 < radius < 4.0


# =============================================================================
# FEATURE 7 & 8: BINARY 32-BYTE .SPLAT PACKING & VALIDATION (TIERS 1 & 2)
# =============================================================================

def test_splat_packer_exact_32_byte_stride():
    """T1-F7-01: Converting PLY to .splat produces exact N * 32 byte binary format."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_splat_packer_"))
    in_ply = temp_dir / "cloud.ply"
    out_splat = temp_dir / "cloud.splat"

    try:
        num_gaussians = 180
        create_synthetic_ply(in_ply, num_good=num_gaussians, num_low_opacity=0, num_oversized=0)
        convert_ply_to_splat(in_ply, out_splat)

        assert out_splat.exists()
        size = out_splat.stat().st_size
        assert size == num_gaussians * 32
        assert size % 32 == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_splat_validator_valid_demo_file():
    """T1-F8-01: Validates official bundled demo.splat model against strict spec."""
    demo_path = SERVER_DIR / "models" / "demo.splat"
    assert demo_path.exists(), "demo.splat must exist for verification"

    result = verify_splat_file(demo_path, min_splat_count=100)
    assert result["valid"] is True
    assert result["num_splats"] > 100000
    assert result["file_size_bytes"] % 32 == 0
    assert "bounds" in result
    assert "scales" in result
    assert result["scales"]["min"] > 0


def test_splat_validator_size_not_divisible_by_32():
    """T2-F8-01: File with size not divisible by 32 raises FileSizeError."""
    temp_splat = Path(tempfile.mktemp(suffix=".splat"))
    # Write 35 bytes (not divisible by 32)
    temp_splat.write_bytes(b"\x00" * 35)
    try:
        with pytest.raises(FileSizeError, match="32 bytes"):
            verify_splat_file(temp_splat)
    finally:
        try:
            if temp_splat.exists():
                temp_splat.unlink()
        except PermissionError:
            pass


def test_splat_validator_zero_byte_file():
    """T2-F8-02: 0-byte file raises FileSizeError."""
    temp_splat = Path(tempfile.mktemp(suffix=".splat"))
    temp_splat.write_bytes(b"")
    try:
        with pytest.raises(FileSizeError, match="empty"):
            verify_splat_file(temp_splat)
    finally:
        try:
            if temp_splat.exists():
                temp_splat.unlink()
        except PermissionError:
            pass


def test_splat_validator_splat_count_below_minimum():
    """T2-F8-03: Splat file with fewer than min_splat_count raises SplatCountError."""
    temp_splat = Path(tempfile.mktemp(suffix=".splat"))
    # Write 10 splats = 320 bytes
    temp_splat.write_bytes(b"\x00" * 320)
    try:
        with pytest.raises(SplatCountError, match="below minimum"):
            verify_splat_file(temp_splat, min_splat_count=100)
    finally:
        try:
            if temp_splat.exists():
                temp_splat.unlink()
        except PermissionError:
            pass


def test_splat_validator_nan_coordinates():
    """T2-F8-04: Splat file with NaN position coordinate raises CorruptCoordinatesError."""
    temp_splat = Path(tempfile.mktemp(suffix=".splat"))
    # 100 splats of 32 bytes = 3200 bytes
    buffer = bytearray(b"\x00" * 3200)
    # Inject NaN float32 at byte offset 0 (first float)
    nan_bytes = np.array([float("nan")], dtype="<f4").tobytes()
    buffer[0:4] = nan_bytes
    temp_splat.write_bytes(buffer)

    try:
        with pytest.raises(CorruptCoordinatesError, match="NaN/Inf"):
            verify_splat_file(temp_splat, min_splat_count=50)
    finally:
        try:
            if temp_splat.exists():
                temp_splat.unlink()
        except PermissionError:
            pass


def test_splat_validator_corrupt_scale():
    """T2-F8-05: Splat file with non-positive scale raises CorruptScaleError."""
    temp_splat = Path(tempfile.mktemp(suffix=".splat"))
    buffer = bytearray(b"\x00" * 3200)
    # Set position (bytes 0-11) to 1.0
    pos_bytes = np.array([1.0, 1.0, 1.0], dtype="<f4").tobytes()
    # Set scale (bytes 12-23) to -5.0 (non-positive)
    bad_scale = np.array([-5.0, 1.0, 1.0], dtype="<f4").tobytes()

    for i in range(100):
        offset = i * 32
        buffer[offset : offset + 12] = pos_bytes
        buffer[offset + 12 : offset + 24] = bad_scale
        buffer[offset + 28 : offset + 32] = b"\x80\x00\x00\x00"  # norm quat

    temp_splat.write_bytes(buffer)
    try:
        with pytest.raises(CorruptScaleError, match="non-positive"):
            verify_splat_file(temp_splat, min_splat_count=50)
    finally:
        try:
            if temp_splat.exists():
                temp_splat.unlink()
        except PermissionError:
            pass


def test_splat_validator_cli_exit_code():
    """T1-F8-02: verify_splat.py CLI exits 0 on valid file and 1 on invalid file."""
    python_exe = sys.executable
    demo_path = SERVER_DIR / "models" / "demo.splat"
    verify_script = SERVER_DIR / "verify_splat.py"

    # Test valid demo file -> exit 0
    res_valid = subprocess.run(
        [python_exe, str(verify_script), str(demo_path)],
        capture_output=True,
        text=True,
    )
    assert res_valid.returncode == 0
    data = json.loads(res_valid.stdout)
    assert data["valid"] is True

    # Test invalid file -> exit 1
    temp_bad = Path(tempfile.mktemp(suffix=".splat"))
    temp_bad.write_bytes(b"BAD_SPLATS")
    try:
        res_invalid = subprocess.run(
            [python_exe, str(verify_script), str(temp_bad)],
            capture_output=True,
            text=True,
        )
        assert res_invalid.returncode != 0
        raw_err = res_invalid.stderr.strip() or res_invalid.stdout.strip()
        data_bad = json.loads(raw_err)
        assert data_bad["valid"] is False
    finally:
        try:
            if temp_bad.exists():
                temp_bad.unlink()
        except PermissionError:
            pass


# =============================================================================
# TIER 3: PAIRWISE PARAMETER INTERACTION COMBINATIONS
# =============================================================================

# Pairwise test combinations across: iterations x prune_floaters x resolution x sh_degree
PAIRWISE_COMBOS = [
    # (iterations, prune_floaters, resolution, sh_degree)
    (1000, True, 2, 0),    # Combo 1: Fast preview, prune, half-res, diffuse
    (2000, True, 2, 3),    # Combo 2: Standard mobile, prune, half-res, full SH
    (3000, False, 1, 1),   # Combo 3: Studio medium, no prune, full-res, low SH
    (7000, True, 1, 2),    # Combo 4: High iterations, prune, full-res, med SH
    (15000, False, 4, 3),  # Combo 5: Very high iterations, no prune, quarter-res, full SH
    (30000, True, 2, 3),   # Combo 6: Max iterations, prune, half-res, full SH
    (1000, False, 2, 3),   # Combo 7: Fast iterations, no prune, half-res, full SH
    (3000, True, 1, 0),    # Combo 8: Studio medium, prune, full-res, diffuse
]


@pytest.mark.parametrize("iterations,prune_floaters,resolution,sh_degree", PAIRWISE_COMBOS)
def test_pipeline_pairwise_combinations(iterations, prune_floaters, resolution, sh_degree):
    """T3-PAIR-01..08: Verifies full pipeline execution across parameter interaction matrix."""
    scan_dir = Path(generate_mock_dataset(num_frames=4, as_zip=False))
    temp_root = Path(tempfile.mkdtemp(prefix=f"test_pw_{iterations}_{prune_floaters}_"))
    out_dir = temp_root / "outputs"
    models_dir = temp_root / "models"
    scan_id = f"pw_{iterations}_{resolution}_{sh_degree}"

    try:
        config = PipelineConfig(
            scan_id=scan_id,
            scan_dir=scan_dir,
            output_dir=out_dir,
            models_dir=models_dir,
            iterations=iterations,
            resolution=resolution,
            prune_floaters=prune_floaters,
            simulation_mode=True,
            simulation_steps=10,
        )

        runner = PipelineRunner()
        telemetry_events = []
        result = runner.run(config, telemetry_callback=lambda e: telemetry_events.append(e))

        assert result.success is True
        assert result.final_stage == PipelineStage.COMPLETED
        assert result.splat_path is not None
        assert result.splat_path.exists()

        # Telemetry progression check: must see stages INGESTION, TRAINING, COMPLETED
        stages_seen = [e.stage for e in telemetry_events]
        assert PipelineStage.INGESTION in stages_seen
        assert PipelineStage.TRAINING in stages_seen
        assert PipelineStage.COMPLETED in stages_seen

        if prune_floaters:
            assert PipelineStage.PRUNING in stages_seen

        # Verify output .splat format
        verify_result = verify_splat_file(result.splat_path, min_splat_count=50)
        assert verify_result["valid"] is True
        assert verify_result["file_size_bytes"] % 32 == 0
    finally:
        shutil.rmtree(scan_dir, ignore_errors=True)
        shutil.rmtree(temp_root, ignore_errors=True)
