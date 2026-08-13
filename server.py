#!/usr/bin/env python3
"""
Gaussian 3D Studio Local Backend Server (server.py).

Provides instant API endpoint `/api/generate_3d`:
- Accepts single image upload (POST multipart/form-data or base64)
- Runs SingleImageTripoSRPipeline with PyTorch differentiable Gaussian rasterization
- Returns photorealistic 3D PLY model & normalized point cloud JSON
- Serves index.html at http://localhost:8000
"""
import io
import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir.parent))

from gaussian.pipeline.single_image_triposr import SingleImageTripoSRPipeline

pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        print("⚡ [Server] Initializing SingleImageTripoSRPipeline...")
        pipeline = SingleImageTripoSRPipeline()
    return pipeline


class StudioRequestHandler(SimpleHTTPRequestHandler):

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.path = "/index.html"
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if self.path == "/api/generate_3d":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                # Expecting JSON with image_base64 and iterations
                data = json.loads(body.decode("utf-8"))
                img_b64 = data.get("image_base64", "")
                iterations = int(data.get("iterations", 50))

                if "," in img_b64:
                    img_b64 = img_b64.split(",")[1]

                import base64
                from PIL import Image

                img_bytes = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(img_bytes))

                # Run Single-Image 3D Pipeline
                p = get_pipeline()
                out_dir = root_dir / "output_single_image"
                res = p.generate_3d_from_single_image(img, num_refine_iterations=iterations, output_dir=out_dir)

                gaussians = res["gaussians"]
                xyz = gaussians.get_xyz.detach().cpu().numpy()
                colors = gaussians.get_features[:, 0, :].detach().cpu().numpy()
                colors = np.clip((colors + 0.5), 0.0, 1.0)

                # Normalize for viewer
                center = np.mean(xyz, axis=0)
                xyz_centered = xyz - center
                max_r = np.max(np.linalg.norm(xyz_centered, axis=1))
                xyz_norm = xyz_centered / (max_r or 1.0)

                points_json = []
                for i in range(len(xyz_norm)):
                    r, g, b = (colors[i] * 255.0).astype(int).tolist()
                    px, py, pz = xyz_norm[i].tolist()
                    points_json.append({
                        "pos": [round(px, 4), round(py, 4), round(pz, 4)],
                        "color": [r, g, b]
                    })

                response_data = {
                    "status": "SUCCESS",
                    "dialogue": res.get("dialogue", "HULK SMASH SUCCESS"),
                    "num_points": len(points_json),
                    "ply_url": f"file:///{res['ply_path'].resolve()}".replace("\\", "/"),
                    "points": points_json
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))

            except Exception as e:
                print("Error in /api/generate_3d:", e)
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_server(port=8000):
    os.chdir(str(root_dir))
    server_address = ("", port)
    httpd = HTTPServer(server_address, StudioRequestHandler)
    print(f"\n=======================================================")
    print(f"🚀 Gaussian 3D Studio Server Running at http://localhost:{port}")
    print(f"=======================================================\n")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
