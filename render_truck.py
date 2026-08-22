import sys
import os
sys.path.append(r"C:\Users\Rishi\Downloads\test\gaussian-splatting")

import json
import math
import numpy as np
import torch
from plyfile import PlyData
from PIL import Image
from scene.cameras import MiniCam
from gaussian_renderer import render
from utils.graphics_utils import getProjectionMatrix, getWorld2View2

def render_truck():
    out_dir = r"C:\Users\Rishi\Downloads\test\output\truck_rendered"
    os.makedirs(out_dir, exist_ok=True)
    
    ply_path = r"C:\Users\Rishi\Downloads\test\output\truck_rendered\point_cloud_1000.ply"
    cameras_json = r"C:\Users\Rishi\Downloads\test\output\truck\cameras.json"
    
    print(f"Loading Truck PLY: {ply_path}", flush=True)
    plydata = PlyData.read(ply_path)
    v = plydata['vertex'].data
    matrix = np.column_stack([v[name] for name in v.dtype.names]).astype(np.float32)
    num_pts = len(matrix)
    
    xyz = torch.tensor(matrix[:, :3], dtype=torch.float32, device="cuda")
    features_dc = torch.tensor(matrix[:, 6:9], dtype=torch.float32, device="cuda").unsqueeze(1)
    features_rest = torch.tensor(matrix[:, 9:54], dtype=torch.float32, device="cuda").reshape(num_pts, 15, 3)
    opacity = torch.sigmoid(torch.tensor(matrix[:, 54:55], dtype=torch.float32, device="cuda"))
    scaling = torch.exp(torch.tensor(matrix[:, 55:58], dtype=torch.float32, device="cuda"))
    rotation = torch.tensor(matrix[:, 58:62], dtype=torch.float32, device="cuda")
    rotation = rotation / torch.norm(rotation, dim=-1, keepdim=True)
    
    print(f"Loaded {num_pts} Gaussians onto GPU!", flush=True)
    
    class FastGaussianModel:
        def __init__(self, xyz, f_dc, f_rest, op, sc, rot):
            self._xyz = xyz
            self._features_dc = f_dc
            self._features_rest = f_rest
            self._opacity = op
            self._scaling = sc
            self._rotation = rot
            self.active_sh_degree = 0
            self.max_sh_degree = 3
        @property
        def get_xyz(self): return self._xyz
        @property
        def get_features(self): return torch.cat([self._features_dc, self._features_rest], dim=1)
        @property
        def get_opacity(self): return self._opacity
        @property
        def get_scaling(self): return self._scaling
        @property
        def get_rotation(self): return self._rotation
        
    gaussians = FastGaussianModel(xyz, features_dc, features_rest, opacity, scaling, rotation)
    
    class PipeParams:
        def __init__(self):
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.debug = False
            self.antialiasing = False
            
    pipe = PipeParams()
    bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    
    with open(cameras_json, 'r') as f:
        cams_data = json.load(f)
        
    print(f"Loaded {len(cams_data)} camera poses from dataset", flush=True)
    
    indices = [int(i) for i in np.linspace(0, len(cams_data) - 1, 16)]
    frames = []
    
    print("Rendering novel views on RTX 3050...", flush=True)
    for idx, c_idx in enumerate(indices):
        c = cams_data[c_idx]
        w = c['width']
        h = c['height']
        fx = c['fx']
        fy = c['fy']
        fovx = 2 * math.atan(w / (2 * fx))
        fovy = 2 * math.atan(h / (2 * fy))
        
        R = np.array(c['rotation'])
        T = np.array(c['position'])
        
        world_view_transform = torch.tensor(getWorld2View2(R, T), dtype=torch.float32, device="cuda").transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
        full_proj = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        
        cam = MiniCam(
            width=w, height=h, fovy=fovy, fovx=fovx,
            znear=0.01, zfar=100.0,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj
        )
        
        with torch.no_grad():
            out = render(cam, gaussians, pipe, bg_color)["render"]
            img_np = (out.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            img.save(os.path.join(out_dir, f"truck_view_{idx:03d}.png"))
            frames.append(img.resize((w // 4, h // 4), Image.Resampling.LANCZOS))
            print(f"Rendered View {idx+1}/16 ({w}x{h})", flush=True)
            
    gif_path = os.path.join(out_dir, "truck_flythrough.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=120, loop=0)
    print(f"Saved animation to {gif_path}!", flush=True)

if __name__ == "__main__":
    render_truck()
