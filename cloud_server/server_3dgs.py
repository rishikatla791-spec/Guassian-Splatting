import os
import sys
import shutil
import zipfile
import subprocess
import uuid
import struct
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU"
except Exception:
    CUDA_AVAILABLE = False
    GPU_NAME = "CPU"

app = FastAPI(title="Mobile 3DGS Cloud Training Server", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_DIR = Path("/content/jobs") if os.path.exists("/content") else Path("./jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job progress tracker
job_status = {}

def convert_ply_to_splat(ply_path: Path, splat_path: Path) -> int:
    """Converts standard 3DGS PLY to 32-byte mobile binary splat format."""
    from plyfile import PlyData
    ply = PlyData.read(str(ply_path))
    v = ply['vertex']
    num_splats = len(v)

    # Prepare 32-byte structured buffer
    # Format: pos(3f=12B) + scale(3f=12B) + rgba(4B) + rot(4B) = 32B
    buffer = bytearray(num_splats * 32)
    
    xyz = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)
    scales = np.stack([np.exp(v['scale_0']), np.exp(v['scale_1']), np.exp(v['scale_2'])], axis=-1).astype(np.float32)
    
    # Sigmoid opacity * 255
    opacities = (1.0 / (1.0 + np.exp(-v['opacity'])) * 255.0).clip(0, 255).astype(np.uint8)
    
    # SH0 to RGB [0, 255]
    SH_C0 = 0.28209479177387814
    r = ((0.5 + SH_C0 * v['f_dc_0']) * 255.0).clip(0, 255).astype(np.uint8)
    g = ((0.5 + SH_C0 * v['f_dc_1']) * 255.0).clip(0, 255).astype(np.uint8)
    b = ((0.5 + SH_C0 * v['f_dc_2']) * 255.0).clip(0, 255).astype(np.uint8)
    
    colors = np.stack([r, g, b, opacities], axis=-1)

    # Normalized quaternions encoded to uint8
    rots = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1).astype(np.float32)
    norms = np.linalg.norm(rots, axis=-1, keepdims=True) + 1e-8
    rots = rots / norms
    rots_u8 = ((rots + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    # Pack into bytearray
    offset = 0
    for i in range(num_splats):
        struct.pack_into('<fff', buffer, offset, xyz[i, 0], xyz[i, 1], xyz[i, 2])
        struct.pack_into('<fff', buffer, offset + 12, scales[i, 0], scales[i, 1], scales[i, 2])
        struct.pack_into('<BBBB', buffer, offset + 24, colors[i, 0], colors[i, 1], colors[i, 2], colors[i, 3])
        struct.pack_into('<BBBB', buffer, offset + 28, rots_u8[i, 0], rots_u8[i, 1], rots_u8[i, 2], rots_u8[i, 3])
        offset += 32

    with open(splat_path, 'wb') as f:
        f.write(buffer)
        
    return num_splats

def run_training_job(job_id: str, zip_path: Path):
    job_dir = JOBS_DIR / job_id
    dataset_dir = job_dir / "dataset"
    input_images = dataset_dir / "input"
    output_dir = job_dir / "output"

    try:
        job_status[job_id] = {"status": "extracting", "progress": 10, "message": "Extracting scan photos..."}
        input_images.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(input_images)

        job_status[job_id] = {"status": "sfm", "progress": 30, "message": "Running COLMAP camera pose estimation..."}
        cmd_convert = ["python", "convert.py", "-s", str(dataset_dir)]
        res_convert = subprocess.run(cmd_convert, capture_output=True, text=True)
        
        job_status[job_id] = {"status": "training", "progress": 55, "message": "Training 1.5M+ 3D Gaussians (SH Degree 3)..."}
        cmd_train = [
            "python", "train.py",
            "-s", str(dataset_dir),
            "-m", str(output_dir),
            "--iterations", "7000",
            "--sh_degree", "3",
            "--save_iterations", "7000"
        ]
        res_train = subprocess.run(cmd_train, capture_output=True, text=True)

        job_status[job_id] = {"status": "converting", "progress": 85, "message": "Generating mobile 3D splat..."}
        ply_file = output_dir / "point_cloud" / "iteration_7000" / "point_cloud.ply"
        splat_file = job_dir / "model.splat"

        if ply_file.exists():
            num_gaussians = convert_ply_to_splat(ply_file, splat_file)
            job_status[job_id] = {
                "status": "completed",
                "progress": 100,
                "message": f"Successfully trained {num_gaussians:,} Gaussians!",
                "splat_file": str(splat_file),
                "num_gaussians": num_gaussians,
                "model_url": f"/download/{job_id}.splat"
            }
        else:
            job_status[job_id] = {"status": "failed", "progress": 0, "message": f"Training failed: {res_train.stderr}"}

    except Exception as e:
        job_status[job_id] = {"status": "failed", "progress": 0, "message": str(e)}

@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "status": "online",
        "gpu": GPU_NAME,
        "cuda_available": CUDA_AVAILABLE,
        "active_jobs": len(job_status)
    }

@app.post("/upload")
@app.post("/api/train")
async def start_training(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = job_dir / "scan.zip"
    with open(zip_path, "wb") as f:
        content = await file.read()
        f.write(content)

    job_status[job_id] = {
        "scan_id": job_id,
        "job_id": job_id,
        "status": "queued",
        "progress": 5,
        "message": "Scan uploaded. Starting training job..."
    }
    background_tasks.add_task(run_training_job, job_id, zip_path)

    return {"scan_id": job_id, "job_id": job_id, "status": "queued", "message": "Training started in background"}

@app.get("/status/{job_id}")
@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    clean_id = job_id.replace(".splat", "")
    if clean_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_status[clean_id]

@app.get("/download/{filename}")
@app.get("/api/download/{filename}")
def download_splat(filename: str):
    clean_id = filename.replace(".splat", "")
    if clean_id not in job_status or job_status[clean_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not ready for download")
    
    splat_path = job_status[clean_id]["splat_file"]
    return FileResponse(splat_path, media_type="application/octet-stream", filename=f"{clean_id}.splat")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
