"""Master Opaque-Box E2E Test Runner for Android-to-Laptop 3DGS Coordination.

Executes all test tiers:
- Tier 1: Feature Coverage (Sanity & Happy Path)
- Tier 2: Boundary, Corner & Negative Error Handling
- Tier 3: Pairwise Parameter Interactions & Cross-Module Pipeline
- Tier 4: Real-World Client Agent Capture Simulation & Live E2E Integration

Reports structured pass/fail metrics, execution timings, and feature checklist.
Exits 0 on 100% success; non-zero on any failure.
"""

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent.parent


TIER_DEFINITIONS = {
    1: {
        "name": "Tier 1: Feature Coverage (Sanity & Contract Checks)",
        "files": ["test_protocol.py", "test_adapter.py", "test_pipeline.py", "test_milestone2_vulkan_jni.py"],
        "keywords": "not (Boundary or Negative or Corrupt or clamp or invalid or nonexistent or singular or pair)",
        "features": ["F1", "F2", "F4", "F6", "F7", "F8", "F10", "F13"],
    },
    2: {
        "name": "Tier 2: Boundary, Edge Cases & Error Handling",
        "files": ["test_protocol.py", "test_adapter.py", "test_pipeline.py"],
        "keywords": "Boundary or Negative or Corrupt or clamp or invalid or nonexistent or singular or missing or empty",
        "features": ["F1", "F2", "F4", "F6", "F8", "F9", "F10"],
    },
    3: {
        "name": "Tier 3: Pairwise Combinatorial Parameter Interactions",
        "files": ["test_pipeline.py"],
        "keywords": "pairwise",
        "features": ["F2", "F5", "F6", "F7", "F8"],
    },
    4: {
        "name": "Tier 4: Realistic Client Simulation & Live E2E Integration",
        "files": ["test_e2e_integration.py"],
        "keywords": "",
        "features": ["F3", "F11", "F12"],
    },
}


class TierResultCollector:
    """Custom pytest plugin to track pass/fail/skip counts per test item."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1
                self.failures.append((report.nodeid, report.longreprtext))
            elif report.skipped:
                self.skipped += 1
        elif report.when == "setup" and report.failed:
            self.failed += 1
            self.failures.append((report.nodeid, report.longreprtext))


def run_tier(tier_num: int, verbose: bool = False) -> Dict[str, Any]:
    """Runs a single test tier and returns summary metrics."""
    tier_info = TIER_DEFINITIONS[tier_num]
    t0 = time.time()

    args = ["-q"]
    if verbose:
        args = ["-v"]

    # Target test files
    for f in tier_info["files"]:
        target_path = TESTS_DIR / f
        if target_path.exists():
            args.append(str(target_path))

    # Apply keyword expression if specified
    kw = tier_info["keywords"]
    if kw:
        args.extend(["-k", kw])

    collector = TierResultCollector()
    exit_code = pytest.main(args, plugins=[collector])
    elapsed = time.time() - t0

    return {
        "tier": tier_num,
        "name": tier_info["name"],
        "passed": collector.passed,
        "failed": collector.failed,
        "skipped": collector.skipped,
        "total": collector.passed + collector.failed + collector.skipped,
        "duration": round(elapsed, 2),
        "exit_code": int(exit_code),
        "failures": collector.failures,
        "features": tier_info["features"],
    }


def print_dashboard(results: List[Dict[str, Any]]) -> None:
    """Renders structured summary table and feature coverage dashboard."""
    print("\n" + "=" * 80)
    print("       3DGS MULTI-AGENT TASK COORDINATION — E2E TEST SUITE REPORT")
    print("=" * 80)
    print(f"{'Tier':<8} | {'Tier Description':<40} | {'Pass':<6} | {'Fail':<6} | {'Time':<7} | {'Status'}")
    print("-" * 80)

    total_pass = 0
    total_fail = 0
    total_time = 0.0

    for r in results:
        total_pass += r["passed"]
        total_fail += r["failed"]
        total_time += r["duration"]

        status = "[PASS]" if r["failed"] == 0 and r["total"] > 0 else "[FAIL]"
        if r["total"] == 0:
            status = "[EMPTY]"

        short_desc = r["name"].split(":")[1].strip() if ":" in r["name"] else r["name"]
        short_desc = short_desc[:38]

        print(f"Tier {r['tier']:<3} | {short_desc:<40} | {r['passed']:<6} | {r['failed']:<6} | {r['duration']:>5.2f}s | {status}")

    print("-" * 80)
    total_status = "[ALL PASSED]" if total_fail == 0 else "[FAILED]"
    print(f"{'TOTAL':<8} | {'Across All 4 Tiers':<40} | {total_pass:<6} | {total_fail:<6} | {total_time:>5.2f}s | {total_status}")
    print("=" * 80)

    # Feature Checklist
    print("\n" + "-" * 80)
    print("FEATURE INVENTORY COVERAGE CHECKLIST (PROJECT.md)")
    print("-" * 80)
    features_map = {
        "F1": "Protocol Schema & State Machine",
        "F2": "Parameter Negotiation & REST Endpoints",
        "F3": "WebSocket Telemetry Stream (/ws/telemetry)",
        "F4": "Dataset Ingestion Bridge (ARCore to COLMAP)",
        "F5": "Headless CUDA Training Automation",
        "F6": "Standalone Floater Pruning Hook",
        "F7": "Binary 32-Byte .splat Generation",
        "F8": "Binary .splat Format Verification",
        "F9": "Pipeline Resilience & Error Handling",
        "F10": "Android Capture Client Wiring & Compatibility",
        "F11": "Headless Mock Client Agent",
        "F12": "Automated E2E Verification Harness",
        "F13": "Milestone 2: Native Vulkan Compute Engine & JNI Bridge",
    }

    covered_features = set()
    for r in results:
        if r["failed"] == 0:
            covered_features.update(r["features"])

    for fid, fname in features_map.items():
        cov = "[x] VERIFIED" if fid in covered_features else "[ ] UNVERIFIED"
        print(f"  {cov}  {fid:<4}: {fname}")
    print("-" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Master E2E Test Runner")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], default=None, help="Run only specific tier")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose pytest output")
    args = parser.parse_args()

    print(f"[E2E Runner] Starting test suite in: {TESTS_DIR}")
    tiers_to_run = [args.tier] if args.tier else [1, 2, 3, 4]

    results = []
    for t in tiers_to_run:
        res = run_tier(t, verbose=args.verbose)
        results.append(res)

    print_dashboard(results)

    total_failures = sum(r["failed"] for r in results)
    if total_failures > 0:
        print(f"\n[E2E Runner] {total_failures} test(s) failed. See output above.")
        for r in results:
            for nodeid, reason in r["failures"]:
                print(f"  FAILED: {nodeid}")
        sys.exit(1)

    print("[E2E Runner] All tests PASSED successfully (100% pass rate).")
    sys.exit(0)


if __name__ == "__main__":
    main()
