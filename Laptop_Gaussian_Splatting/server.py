#!/usr/bin/env python3
"""
Gaussian 3D Studio Enterprise Backend Server (server.py).

Provides local HTTP API endpoints for Web Studio:
- /api/experiments : Returns cross-object comparison & verification history
- /api/object_data : Returns detailed metrics, error maps, & mesh URLs per object (truck, white_laptop, box, custom_upload)
- /api/images      : Returns multi-view image thumbnail URLs per object dataset
- /api/upload      : Handles multi-view image uploads and triggers 3DGS pipeline
- Serves index.html & static assets (OBJ, GLTF, PLY, images, JSONs)
"""
import io
import os
import re
import sys
import json
import shutil
import base64
import mimetypes
import threading
import subprocess
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

# Python with CUDA support
CUDA_PYTHON_EXE = r"C:\Users\Rishi\anaconda3\envs\gaussian_cuda\python.exe"

# Ensure custom mime types for 3D assets
mimetypes.add_type("model/obj", ".obj")
mimetypes.add_type("model/gltf+json", ".gltf")
mimetypes.add_type("application/x-ply", ".ply")

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def get_image_files(directory: Path):
    if not directory.exists():
        return []
    files = []
    for p in directory.glob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            files.append(p)
    return sorted(files)

reconstruction_state = {
    "status": "idle",
    "message": "Ready for upload",
    "active_object": "truck"
}

def run_background_reconstruction(images_dir, output_dir, obj_name):
    global reconstruction_state
    try:
        reconstruction_state["status"] = "processing"
        reconstruction_state["message"] = "Starting 3DGS Pose Estimation & Feature Triangulation..."
        reconstruction_state["active_object"] = obj_name

        import torch
        has_cuda = torch.cuda.is_available()

        # If current Python does not have CUDA but Python 3.12 with CUDA is installed, delegate to it
        if not has_cuda and Path(CUDA_PYTHON_EXE).exists():
            print(f"[Server] Delegating reconstruction to CUDA environment: {CUDA_PYTHON_EXE}")
            reconstruction_state["message"] = "Delegating to NVIDIA GPU CUDA Python 3.12 Engine..."
            worker_code = f"""
import sys
from pathlib import Path
root_dir = Path(r'{root_dir}')
sys.path.insert(0, str(root_dir))

from pipeline.reconstruction_pipeline import ReconstructionPipeline
cfg = {{
    'images_path': r'{images_dir}',
    'output_dir': r'{output_dir}',
    'iterations': 300,
}}
pipeline = ReconstructionPipeline(cfg)
pipeline.run_full_pipeline()
print('SUCCESS_RECONSTRUCTION')
"""
            res = subprocess.run(
                [CUDA_PYTHON_EXE, "-c", worker_code],
                cwd=str(root_dir),
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                err_msg = res.stderr.strip() or res.stdout.strip()
                raise RuntimeError(err_msg)
        else:
            from pipeline.reconstruction_pipeline import ReconstructionPipeline
            cfg = {
                "images_path": str(images_dir),
                "output_dir": str(output_dir),
                "iterations": 300,
            }
            pipeline = ReconstructionPipeline(cfg)
            reconstruction_state["message"] = "Seeding Volumetric Visual Hull & GPU CUDA Training..."
            pipeline.run_full_pipeline()

        reconstruction_state["status"] = "complete"
        reconstruction_state["message"] = "3D Reconstruction Completed Successfully!"
    except Exception as e:
        print("[Reconstruction Background Error]", e)
        reconstruction_state["status"] = "error"
        reconstruction_state["message"] = f"Reconstruction Error: {str(e)}"


def parse_multipart_form(body: bytes, content_type: str):
    """
    Robust multipart/form-data parser that extracts files without corrupting binary data.
    """
    match = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type, re.IGNORECASE)
    if not match:
        raise ValueError("Could not find valid boundary in Content-Type header")

    boundary_str = (match.group(1) or match.group(2)).strip()
    boundary = boundary_str.encode('utf-8')
    delimiter = b'--' + boundary

    parts = body.split(delimiter)
    saved_files = []

    for part in parts:
        if not part or part.startswith(b'--'):
            continue

        if part.startswith(b'\r\n'):
            part = part[2:]
        if part.endswith(b'\r\n'):
            part = part[:-2]

        if b'\r\n\r\n' not in part:
            continue

        header_bytes, content = part.split(b'\r\n\r\n', 1)
        header_str = header_bytes.decode('utf-8', errors='ignore')

        fn_match = re.search(r'filename\*?=(?:"([^"]+)"|\'([^\']+)\'|([^\s;]+))', header_str, re.IGNORECASE)
        if fn_match:
            raw_filename = fn_match.group(1) or fn_match.group(2) or fn_match.group(3)
            from urllib.parse import unquote
            if raw_filename.lower().startswith("utf-8''"):
                raw_filename = raw_filename[7:]
            raw_filename = unquote(raw_filename)
            filename = os.path.basename(raw_filename)
            if filename:
                saved_files.append((filename, content))

    return saved_files


class StudioRequestHandler(SimpleHTTPRequestHandler):

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/favicon.ico":
            self.send_response(200)
            self.send_header("Content-Type", "image/x-icon")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if path == "/" or path == "/index.html":
            self.path = "/index.html"
            return SimpleHTTPRequestHandler.do_GET(self)

        # Endpoint 0: Live Reconstruction Status
        if path == "/api/reconstruct_status":
            return self.send_json(reconstruction_state)

        # Endpoint 1: Summary of all experiments
        if path == "/api/experiments":
            summary_path = root_dir / "phase6_cross_object_generalization.json"
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return self.send_json(data)
            else:
                return self.send_json({"error": "phase6_cross_object_generalization.json not found"}, 404)

        # Endpoint 2: Detailed dataset info per object
        if path == "/api/object_data":
            obj_id = query.get("object_id", ["truck"])[0]

            if obj_id == "truck":
                out_dir = root_dir / "output_truck_3dmodel"
                img_dir = root_dir / "imgaes" / "truck" / "images"
                history_path = None
                refinement_path = None
                error_map_dir = out_dir / "comparisons"
            elif obj_id == "custom_upload":
                out_dir = root_dir / "output_custom_3dmodel" if (root_dir / "output_custom_3dmodel").exists() else root_dir / "output_truck_3dmodel"
                img_dir = root_dir / "imgaes_custom_upload" if (root_dir / "imgaes_custom_upload").exists() else root_dir / "imgaes" / "truck" / "images"
                history_path = None
                refinement_path = None
                error_map_dir = out_dir / "comparisons"
            elif obj_id == "white_laptop":
                out_dir = root_dir / "output_white_laptop_3dmodel"
                img_dir = root_dir / "imgaes_white_laptop_validated"
                history_path = out_dir / "phase5_closed_loop_history.json"
                refinement_path = out_dir / "phase4_refinement_metrics.json"
                error_map_dir = out_dir / "phase6_error_maps" / "Object_A_White_Laptop"
            else: # box
                out_dir = root_dir / "output_box_3dmodel"
                img_dir = root_dir / "imgaes_box_validated"
                history_path = None
                refinement_path = None
                error_map_dir = out_dir / "phase6_error_maps" / "Object_B_Cardboard_Box"

            history = []
            if history_path and history_path.exists():
                with open(history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)

            refinement = {}
            if refinement_path and refinement_path.exists():
                with open(refinement_path, "r", encoding="utf-8") as f:
                    refinement = json.load(f)

            # Error map images
            error_maps = []
            if error_map_dir.exists():
                for p in sorted(error_map_dir.glob("*.png")):
                    rel_url = "/" + str(p.relative_to(root_dir)).replace("\\", "/")
                    error_maps.append({"name": p.name, "url": rel_url})

            # Check exported models (look for specific names or first available in out_dir)
            obj_path = out_dir / f"{obj_id}_3d_model.obj"
            if not obj_path.exists():
                obj_files = list(out_dir.glob("*.obj"))
                obj_path = obj_files[0] if obj_files else obj_path

            gltf_path = out_dir / f"{obj_id}_3d_model.gltf"
            if not gltf_path.exists():
                gltf_files = list(out_dir.glob("*.gltf"))
                gltf_path = gltf_files[0] if gltf_files else gltf_path

            ply_path = out_dir / f"{obj_id}_3d_mesh.ply"
            if not ply_path.exists():
                ply_files = [f for f in out_dir.glob("*.ply") if "mesh" in f.name]
                ply_path = ply_files[0] if ply_files else ply_path

            # Load 3D Point Cloud PLY points for dynamic Splatting WebGL viewer
            ply_point_file = out_dir / "point_cloud.ply"
            if not ply_point_file.exists():
                ply_point_file = out_dir / "point_cloud" / "iteration_300" / "point_cloud.ply"
            if not ply_point_file.exists():
                ply_point_file = out_dir / "point_cloud" / "iteration_60" / "point_cloud.ply"

            points_data = []
            if ply_point_file.exists():
                try:
                    from export_3d_mesh import load_gaussian_ply
                    import numpy as np
                    pts, colors, opacities, scales = load_gaussian_ply(ply_point_file)
                    center = np.mean(pts, axis=0)
                    pts_centered = pts - center
                    max_r = np.max(np.linalg.norm(pts_centered, axis=1)) or 1.0
                    pts_norm = pts_centered / max_r

                    # Sample up to 6,000 points for WebGL responsiveness
                    step = max(1, len(pts_norm) // 6000)
                    for i in range(0, len(pts_norm), step):
                        r, g, b = (colors[i] * 255.0).astype(int).tolist()
                        px, py, pz = pts_norm[i].tolist()
                        points_data.append({
                            "pos": [round(float(px), 4), round(float(py), 4), round(float(pz), 4)],
                            "color": [r, g, b]
                        })
                except Exception as e:
                    print("Error loading point cloud PLY for server:", e)

            response = {
                "object_id": obj_id,
                "output_dir": str(out_dir.relative_to(root_dir)).replace("\\", "/"),
                "obj_url": ("/" + str(obj_path.relative_to(root_dir)).replace("\\", "/")) if obj_path.exists() else None,
                "gltf_url": ("/" + str(gltf_path.relative_to(root_dir)).replace("\\", "/")) if gltf_path.exists() else None,
                "ply_url": ("/" + str(ply_path.relative_to(root_dir)).replace("\\", "/")) if ply_path.exists() else None,
                "error_maps": error_maps,
                "history": history,
                "refinement": refinement,
                "points": points_data,
            }
            return self.send_json(response)

        # Endpoint 3: Multi-view Image list
        if path == "/api/images":
            obj_id = query.get("object_id", ["truck"])[0]

            if obj_id == "truck":
                img_dir = root_dir / "imgaes" / "truck" / "images"
            elif obj_id == "custom_upload":
                img_dir = root_dir / "imgaes_custom_upload" if (root_dir / "imgaes_custom_upload").exists() else root_dir / "imgaes" / "truck" / "images"
            elif obj_id == "white_laptop":
                img_dir = root_dir / "imgaes_white_laptop_validated"
            else:
                img_dir = root_dir / "imgaes_box_validated"

            image_files = get_image_files(img_dir)
            images = []
            for p in image_files:
                rel_url = "/" + str(p.relative_to(root_dir)).replace("\\", "/")
                images.append({"filename": p.name, "url": rel_url})

            return self.send_json({"object_id": obj_id, "total": len(images), "images": images})

        # Static File Handler fallback
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/api/upload":
            try:
                content_type = self.headers.get("Content-Type", "")
                content_length = int(self.headers.get("Content-Length", 0))

                if content_length <= 0:
                    return self.send_json({"error": "No content received"}, 400)

                is_first_batch = query.get("is_first", ["true"])[0].lower() in ["true", "1"]
                is_last_batch = query.get("is_last", ["true"])[0].lower() in ["true", "1"]

                # Read body safely in 64KB buffer chunks to prevent socket timeouts
                remaining = content_length
                chunk_size = 64 * 1024
                body_chunks = []
                while remaining > 0:
                    read_size = min(remaining, chunk_size)
                    chunk = self.rfile.read(read_size)
                    if not chunk:
                        break
                    body_chunks.append(chunk)
                    remaining -= len(chunk)
                body = b"".join(body_chunks)

                upload_dir = root_dir / "imgaes_custom_upload"
                output_dir = root_dir / "output_custom_3dmodel"

                # Clear old upload folder on first batch only
                if is_first_batch:
                    if upload_dir.exists():
                        shutil.rmtree(upload_dir)
                    upload_dir.mkdir(parents=True, exist_ok=True)

                if "multipart/form-data" in content_type.lower():
                    parsed_files = parse_multipart_form(body, content_type)
                    saved_names = []

                    for filename, file_bytes in parsed_files:
                        save_path = upload_dir / filename
                        with open(save_path, "wb") as f:
                            f.write(file_bytes)
                        saved_names.append(filename)

                    if not saved_names and is_first_batch:
                        return self.send_json({"error": "No valid files were extracted from upload payload"}, 400)

                    # Trigger GPU Reconstruction Pipeline on last batch
                    if is_last_batch:
                        t = threading.Thread(
                            target=run_background_reconstruction,
                            args=(upload_dir, output_dir, "custom_upload"),
                            daemon=True
                        )
                        t.start()

                    total_saved = len(get_image_files(upload_dir))
                    return self.send_json({
                        "status": "success",
                        "count": total_saved,
                        "batch_saved": len(saved_names),
                        "reconstructing": is_last_batch,
                        "message": f"Successfully uploaded batch ({len(saved_names)} images). Total uploaded: {total_saved} images."
                    })
                else:
                    return self.send_json({"error": "Unsupported Content-Type header. Expected multipart/form-data."}, 400)
            except Exception as e:
                print("[Upload Handler Exception]", e)
                return self.send_json({"error": str(e)}, 500)

        return self.send_json({"error": "Not Found"}, 404)


def run_server(port=8000):
    os.chdir(str(root_dir))
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, StudioRequestHandler)
    print(f"\n=======================================================")
    print(f"[OK] Gaussian 3D Web Studio Server Running at http://localhost:{port}")
    print(f"=======================================================\n")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
