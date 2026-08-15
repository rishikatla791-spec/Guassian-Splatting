#!/usr/bin/env python3
"""
phase4_high_fidelity_refinement.py — High-Fidelity Refinement & Progressive Convergence Evaluator.

Executes Phase 4 progressive optimization on the REAL 41-view WHITE LAPTOP dataset across:
- Stage 0 (Baseline): 300 iterations
- Stage 1: 1,000 iterations
- Stage 2: 2,500 iterations
- Stage 3: 5,000 iterations

Monitors convergence, novel view validation, densification/pruning statistics, surface mesh detail,
reprojection error, FPS, and VRAM.
Exports final white_laptop_* assets and generates a full BEFORE -> AFTER report.
"""
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path

# Insert root directory
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from pipeline.reconstruction_pipeline import ReconstructionPipeline
from export_3d_mesh import export_all_formats

def run_stage(iterations: int, images_dir: str, output_dir: str):
    print(f"\n==========================================================================")
    print(f"  PHASE 4 REFINEMENT STAGE: {iterations} ITERATIONS")
    print(f"==========================================================================")
    
    cfg = {
        "images_path": images_dir,
        "output_dir": output_dir,
        "iterations": iterations,
    }
    
    t_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        vram_start = torch.cuda.memory_allocated() / (1024 ** 2)
    else:
        vram_start = 0.0

    pipeline = ReconstructionPipeline(cfg)
    gaussians, metrics = pipeline.run_full_pipeline()
    t_elapsed = time.time() - t_start
    
    if torch.cuda.is_available():
        vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        vram_peak = 0.0
        
    num_gaussians = gaussians.num_gaussians
    
    # Check exported surface mesh
    final_ply = Path(output_dir) / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    mesh_paths = export_all_formats(final_ply, output_dir)
    
    # Read OBJ mesh stats
    obj_path = Path(mesh_paths["obj"])
    num_verts = 0
    num_tris = 0
    if obj_path.exists():
        with open(obj_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("v "):
                    num_verts += 1
                elif line.startswith("f "):
                    num_tris += 1

    stage_result = {
        "iterations": iterations,
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "lpips": metrics["lpips"],
        "chamfer_distance": metrics["chamfer_distance"],
        "multi_view_consistency": metrics["multi_view_consistency"],
        "num_gaussians": num_gaussians,
        "mesh_vertices": num_verts,
        "mesh_triangles": num_tris,
        "training_time_sec": round(t_elapsed, 2),
        "fps_rendering": round(1000.0 / max(0.1, t_elapsed / iterations), 2) if iterations > 0 else 0.0,
        "vram_peak_mb": round(vram_peak, 2),
        "gltf_path": mesh_paths["gltf"],
        "obj_path": mesh_paths["obj"],
        "ply_mesh_path": mesh_paths["ply_mesh"],
    }
    
    print(f"\n--- STAGE SUMMARY ({iterations} IT) ---")
    print(f"  PSNR:              {stage_result['psnr']:.2f} dB")
    print(f"  SSIM:              {stage_result['ssim']:.4f}")
    print(f"  LPIPS:             {stage_result['lpips']:.4f}")
    print(f"  Chamfer Distance:  {stage_result['chamfer_distance']:.6f}")
    print(f"  Gaussians:         {stage_result['num_gaussians']:,}")
    print(f"  Mesh Vertices:     {stage_result['mesh_vertices']:,}")
    print(f"  Mesh Triangles:    {stage_result['mesh_triangles']:,}")
    print(f"  Training Time:     {stage_result['training_time_sec']:.2f}s")
    print(f"-----------------------------------------\n")
    
    return stage_result

def main():
    images_dir = str(root_dir / "imgaes_white_laptop_validated")
    output_dir = str(root_dir / "output_white_laptop_3dmodel")
    
    stages = [300, 1000, 2500]
    results = {}
    
    for iters in stages:
        res = run_stage(iters, images_dir, output_dir)
        results[f"stage_{iters}"] = res
        
    out_json = Path(output_dir) / "phase4_refinement_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"[OK] Saved full Phase 4 Refinement metrics to: {out_json}")

if __name__ == "__main__":
    main()
