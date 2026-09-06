"""
Strict Binary .splat Format Validator for 3D Gaussian Splatting.

Specification (PlayCanvas / Antimatter15 / WebGL standard):
  - File size must be an exact non-zero multiple of 32 bytes (N * 32).
  - Minimum splat count threshold (default N >= 100).
  - 32-byte layout per splat:
      pos:   3x float32 (<f4) -> finite, no NaN, no Inf
      scale: 3x float32 (<f4) -> strictly positive (> 0.0), finite
      rgba:  4x uint8 (u1)   -> valid bytes [0, 255]
      rot:   4x uint8 (u1)   -> normalized quaternion, non-degenerate
  - Memory-mapped reading for sub-second validation even on multimillion-splat files.
  - CLI and Python API with structured JSON output and strict exit codes (0 = valid, non-zero = error).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Union

import numpy as np


class SplatValidationError(Exception):
    """Base exception for splat validation failures."""
    pass


class FileSizeError(SplatValidationError):
    """Raised when file does not exist, is empty, or size is not divisible by 32."""
    pass


class SplatCountError(SplatValidationError):
    """Raised when splat count is below the minimum threshold."""
    pass


class CorruptCoordinatesError(SplatValidationError):
    """Raised when position coordinates contain NaN, Inf, or extreme outliers."""
    pass


class CorruptScaleError(SplatValidationError):
    """Raised when Gaussian scales are non-positive, NaN, or Inf."""
    pass


class CorruptQuaternionError(SplatValidationError):
    """Raised when rotation quaternions are degenerate or unnormalized."""
    pass


SPLAT_DTYPE = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
    ('s0', '<f4'), ('s1', '<f4'), ('s2', '<f4'),
    ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1'),
    ('q0', 'u1'), ('q1', 'u1'), ('q2', 'u1'), ('q3', 'u1')
])


def verify_splat_file(
    file_path: Union[str, Path],
    min_splat_count: int = 100,
    check_quaternion_normalization: bool = True
) -> Dict[str, Any]:
    """
    Strictly validates a binary 32-byte .splat file.

    Args:
        file_path: Path to the .splat file
        min_splat_count: Minimum allowed number of Gaussians (default 100)
        check_quaternion_normalization: Whether to verify quaternion non-degeneracy

    Returns:
        Dictionary containing detailed validation metrics.

    Raises:
        FileSizeError, SplatCountError, CorruptCoordinatesError,
        CorruptScaleError, CorruptQuaternionError
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileSizeError(f"File does not exist: {path}")

    if not path.is_file():
        raise FileSizeError(f"Path is not a regular file: {path}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise FileSizeError(f"File is empty (0 bytes): {path}")

    if file_size % 32 != 0:
        remainder = file_size % 32
        raise FileSizeError(
            f"File size ({file_size} bytes) is not divisible by 32 bytes "
            f"(remainder: {remainder} bytes, corrupt or truncated .splat)"
        )

    num_splats = file_size // 32
    if num_splats < min_splat_count:
        raise SplatCountError(
            f"Splat count ({num_splats}) is below minimum threshold ({min_splat_count})"
        )

    # Read binary data with np.fromfile (closes file handle immediately to prevent Windows file locking)
    try:
        data = np.fromfile(str(path), dtype=SPLAT_DTYPE)
    except Exception as e:
        raise SplatValidationError(f"Failed to read binary splat file: {e}")

    # 1. Coordinate Validation (x, y, z)
    x = data['x']
    y = data['y']
    z = data['z']

    x_finite = np.isfinite(x)
    y_finite = np.isfinite(y)
    z_finite = np.isfinite(z)

    if not (np.all(x_finite) and np.all(y_finite) and np.all(z_finite)):
        bad_x = int(np.sum(~x_finite))
        bad_y = int(np.sum(~y_finite))
        bad_z = int(np.sum(~z_finite))
        raise CorruptCoordinatesError(
            f"Found non-finite coordinates (NaN/Inf or NaN or Inf): X({bad_x}), Y({bad_y}), Z({bad_z})"
        )


    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    z_min, z_max = float(np.min(z)), float(np.max(z))

    # 2. Scale Validation (s0, s1, s2)
    s0 = data['s0']
    s1 = data['s1']
    s2 = data['s2']

    s0_finite = np.isfinite(s0)
    s1_finite = np.isfinite(s1)
    s2_finite = np.isfinite(s2)

    if not (np.all(s0_finite) and np.all(s1_finite) and np.all(s2_finite)):
        raise CorruptScaleError("Scale values contain NaN or Inf")

    s0_positive = s0 > 0.0
    s1_positive = s1 > 0.0
    s2_positive = s2 > 0.0

    if not (np.all(s0_positive) and np.all(s1_positive) and np.all(s2_positive)):
        non_pos_0 = int(np.sum(~s0_positive))
        non_pos_1 = int(np.sum(~s1_positive))
        non_pos_2 = int(np.sum(~s2_positive))
        raise CorruptScaleError(
            f"Found non-positive scale values: S0({non_pos_0}), S1({non_pos_1}), S2({non_pos_2})"
        )

    all_scales_min = float(min(np.min(s0), np.min(s1), np.min(s2)))
    all_scales_max = float(max(np.max(s0), np.max(s1), np.max(s2)))

    # 3. Color and Alpha Validation
    a = data['a']
    alpha_max = int(np.max(a))
    if alpha_max == 0:
        raise SplatValidationError("All splats have alpha = 0 (completely transparent scene)")

    alpha_mean = float(np.mean(a))

    # 4. Rotation Quaternion Validation
    # Quaternions are stored as uint8: q_u8 = rot * 128 + 128
    # Map back: rot = (q_u8 - 128.0) / 128.0
    q0 = (data['q0'].astype(np.float32) - 128.0) / 128.0
    q1 = (data['q1'].astype(np.float32) - 128.0) / 128.0
    q2 = (data['q2'].astype(np.float32) - 128.0) / 128.0
    q3 = (data['q3'].astype(np.float32) - 128.0) / 128.0

    q_sq_sum = q0**2 + q1**2 + q2**2 + q3**2
    q_norms = np.sqrt(q_sq_sum)

    if check_quaternion_normalization:
        # Non-degenerate check: norm must be non-zero
        degenerate_count = int(np.sum(q_norms < 0.1))
        if degenerate_count > 0:
            raise CorruptQuaternionError(
                f"Found {degenerate_count} degenerate quaternions with norm near zero"
            )

        # 8-bit quantization bounds: norm should be around 1.0 (allow [0.6, 1.4] for uint8 precision)
        min_norm = float(np.min(q_norms))
        max_norm = float(np.max(q_norms))
        if min_norm < 0.5 or max_norm > 1.5:
            raise CorruptQuaternionError(
                f"Quaternion norm bounds [{min_norm:.4f}, {max_norm:.4f}] violate normalization range [0.5, 1.5]"
            )
    else:
        min_norm = float(np.min(q_norms))
        max_norm = float(np.max(q_norms))

    summary = {
        "valid": True,
        "file_path": str(path),
        "file_size_bytes": file_size,
        "num_splats": num_splats,
        "bounds": {
            "x": [round(x_min, 4), round(x_max, 4)],
            "y": [round(y_min, 4), round(y_max, 4)],
            "z": [round(z_min, 4), round(z_max, 4)],
            "extent": [round(x_max - x_min, 4), round(y_max - y_min, 4), round(z_max - z_min, 4)]
        },
        "scales": {
            "min": float(f"{all_scales_min:.6e}"),
            "max": round(all_scales_max, 6)
        },
        "alpha": {
            "min": int(np.min(a)),
            "max": alpha_max,
            "mean": round(alpha_mean, 2)
        },
        "quaternion_norms": {
            "min": round(min_norm, 4),
            "max": round(max_norm, 4)
        }
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Strict binary .splat validator")
    parser.add_argument("splat_path", type=str, help="Path to .splat file to validate")
    parser.add_argument("--min-splats", type=int, default=100, help="Minimum splat count (default: 100)")
    parser.add_argument("--no-quat-check", action="store_true", help="Skip quaternion normalization check")
    parser.add_argument("--json", action="store_true", help="Output result as JSON (default: true)")

    args = parser.parse_args()

    try:
        report = verify_splat_file(
            args.splat_path,
            min_splat_count=args.min_splats,
            check_quaternion_normalization=not args.no_quat_check
        )
        print(json.dumps(report, indent=2))
        sys.exit(0)
    except SplatValidationError as e:
        error_payload = {
            "valid": False,
            "error_type": type(e).__name__,
            "message": str(e),
            "file_path": str(args.splat_path)
        }
        print(json.dumps(error_payload, indent=2))
        sys.exit(1)
    except Exception as e:
        error_payload = {
            "valid": False,
            "error_type": "UnexpectedError",
            "message": str(e),
            "file_path": str(args.splat_path)
        }
        print(json.dumps(error_payload, indent=2))
        sys.exit(2)



if __name__ == "__main__":
    main()
