import os
import sys
import json
import uuid
import shutil
import zipfile
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent.resolve()
WORKSPACE_DIR = BASE_DIR.parent.parent
GAUSSIAN_REPO = WORKSPACE_DIR / "gaussian-splatting"
SCANS_DIR = BASE_DIR / "scans"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

SCANS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Mobile 3DGS Training Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active training jobs state
JOBS = {}

class JobStatus(BaseModel):
    id: str
    status: str # 'pending', 'training', 'converting', 'completed', 'failed'
    progress: float # 0.0 - 1.0
    iteration: int
    total_iterations: int
    message: str
    splat_url: Optional[str] = None

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "Mobile 3DGS Server",
        "gpu_available": True,
        "cuda_version": "12.1"
    }

@app.get("/models")
def list_models():
    models = []
    for f in MODELS_DIR.glob("*.splat"):
        models.append({
            "name": f.stem,
            "filename": f.name,
            "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
            "download_url": f"/download/{f.name}"
        })
    return {"models": models}

@app.post("/upload")
async def upload_scan(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    scan_id = str(uuid.uuid4())[:8]
    scan_path = SCANS_DIR / f"{scan_id}.zip"
    extract_dir = SCANS_DIR / scan_id
    
    with open(scan_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Unpack
    with zipfile.ZipFile(scan_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    JOBS[scan_id] = {
        "id": scan_id,
        "status": "pending",
        "progress": 0.0,
        "iteration": 0,
        "total_iterations": 2000,
        "message": "Scan uploaded and extracted.",
        "splat_url": None
    }
    
    # Trigger background training
    if background_tasks:
        background_tasks.add_task(run_training_pipeline, scan_id, extract_dir)
        
    return {"scan_id": scan_id, "status": "pending", "message": "Training queued"}

@app.get("/status/{scan_id}")
def get_status(scan_id: str):
    if scan_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[scan_id]

@app.get("/download/{filename}")
def download_model(filename: str):
    file_path = MODELS_DIR / filename
    if not file_path.exists():
        # Check outputs dir
        for p in OUTPUTS_DIR.glob(f"**/{filename}"):
            file_path = p
            break
            
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")
        
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream"
    )

def run_training_pipeline(scan_id: str, scan_dir: Path):
    try:
        JOBS[scan_id]["status"] = "training"
        JOBS[scan_id]["message"] = "Optimizing 3D Gaussian Splats on GPU..."
        
        output_dir = OUTPUTS_DIR / scan_id
        output_dir.mkdir(exist_ok=True)
        
        python_exe = sys.executable
        train_script = GAUSSIAN_REPO / "train.py"
        
        cmd = [
            python_exe, str(train_script),
            "-s", str(scan_dir),
            "-m", str(output_dir),
            "--eval",
            "--iterations", "2000",
            "--save_iterations", "2000",
            "--resolution", "2"
        ]
        
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        
        for line in process.stdout:
            if "Training progress:" in line:
                try:
                    parts = line.split("%")[0].split()[-1]
                    percent = float(parts)
                    JOBS[scan_id]["progress"] = percent / 100.0
                    JOBS[scan_id]["iteration"] = int(percent * 20)
                except:
                    pass
                    
        process.wait()
        
        # Locate output point_cloud.ply
        ply_file = output_dir / "point_cloud" / "iteration_2000" / "point_cloud.ply"
        if not ply_file.exists():
            for p in output_dir.glob("**/point_cloud.ply"):
                ply_file = p
                break
                
        if not ply_file.exists():
            raise RuntimeError("Training finished but point_cloud.ply was not found")
            
        # Convert to .splat
        JOBS[scan_id]["status"] = "converting"
        JOBS[scan_id]["message"] = "Compressing to mobile .splat format..."
        
        splat_file = MODELS_DIR / f"{scan_id}.splat"
        from ply_to_splat import convert_ply_to_splat
        convert_ply_to_splat(str(ply_file), str(splat_file))
        
        JOBS[scan_id]["status"] = "completed"
        JOBS[scan_id]["progress"] = 1.0
        JOBS[scan_id]["message"] = "Model trained and converted successfully!"
        JOBS[scan_id]["splat_url"] = f"/download/{scan_id}.splat"
        
    except Exception as e:
        JOBS[scan_id]["status"] = "failed"
        JOBS[scan_id]["message"] = f"Error: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    print("Starting Mobile 3DGS Training Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
