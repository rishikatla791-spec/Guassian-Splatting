"""
3D Gaussian Splatting Unified Studio & Editor
============================================
Single-file pipeline to load, edit, filter, render, and export 3D Gaussian Splatting scenes.

Features & Modifications:
- Camera Trajectories: 360 Orbit/Turntable, Dataset Path, Spiral
- Gaussian 3D Editing: Opacity filtering, scale pruning, ROI cropping, brightness/tint adjustment
- Render Adjustments: Background color (black/white/custom), custom resolution, FOV, SH degree
- Multi-format Export: MP4 video, GIF animation, PNG sequence, and Edited PLY file
- Integrated SIBR Viewer launcher
"""

import os
import sys
import math
import json
import time
import argparse
import subprocess
import numpy as np
import torch
from PIL import Image

# Add gaussian-splatting submodule to python path
GS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gaussian-splatting")
if GS_PATH not in sys.path:
    sys.path.append(GS_PATH)

try:
    from plyfile import PlyData, PlyElement
    from scene.cameras import MiniCam
    from gaussian_renderer import render
    from utils.graphics_utils import getProjectionMatrix, getWorld2View2, focal2fov
except ImportError as e:
    print(f"[Warning] Gaussian Splatting dependencies missing: {e}")


# ==============================================================================
# 1. Pipeline & Model Definitions
# ==============================================================================

class RenderPipelineParams:
    def __init__(self, convert_shs=False, compute_cov3d=False, debug=False, antialiasing=False):
        self.convert_SHs_python = convert_shs
        self.compute_cov3D_python = compute_cov3d
        self.debug = debug
        self.antialiasing = antialiasing


class GaussianCloud:
    """Encapsulates GPU Gaussian tensors and editing operations."""
    def __init__(self, xyz, features_dc, features_rest, opacity, scaling, rotation, active_sh=3):
        self.xyz = xyz                  # (N, 3)
        self.features_dc = features_dc  # (N, 1, 3)
        self.features_rest = features_rest  # (N, 15, 3)
        self.opacity = opacity          # (N, 1) (sigmoid activated)
        self.scaling = scaling          # (N, 3) (exp activated)
        self.rotation = rotation        # (N, 4) (quaternions normalized)
        self.active_sh_degree = active_sh
        self.max_sh_degree = 3

    @property
    def get_xyz(self): return self.xyz
    @property
    def get_features(self): return torch.cat([self.features_dc, self.features_rest], dim=1)
    @property
    def get_opacity(self): return self.opacity
    @property
    def get_scaling(self): return self.scaling
    @property
    def get_rotation(self): return self.rotation

    def apply_filters(self, min_opacity=0.0, max_scale=None, crop_radius=None, center=None):
        """Filter out floaters, oversized Gaussians, or points outside an ROI."""
        mask = torch.ones(len(self.xyz), dtype=torch.bool, device="cuda")
        
        if min_opacity > 0.0:
            mask = mask & (self.opacity.squeeze(-1) >= min_opacity)
            
        if max_scale is not None:
            max_s = torch.max(self.scaling, dim=-1)[0]
            mask = mask & (max_s <= max_scale)
            
        if crop_radius is not None:
            c = center if center is not None else torch.median(self.xyz, dim=0)[0]
            dist = torch.norm(self.xyz - c, dim=-1)
            mask = mask & (dist <= crop_radius)

        kept = mask.sum().item()
        print(f"[Filter] Kept {kept:,}/{len(self.xyz):,} Gaussians ({kept/len(self.xyz)*100:.1f}%)")

        self.xyz = self.xyz[mask]
        self.features_dc = self.features_dc[mask]
        self.features_rest = self.features_rest[mask]
        self.opacity = self.opacity[mask]
        self.scaling = self.scaling[mask]
        self.rotation = self.rotation[mask]

    def adjust_colors(self, brightness=1.0, tint=(1.0, 1.0, 1.0)):
        """Adjust overall brightness and color tint."""
        if brightness != 1.0:
            self.features_dc = self.features_dc * brightness
        if tint != (1.0, 1.0, 1.0):
            tint_t = torch.tensor(tint, device="cuda", dtype=torch.float32).view(1, 1, 3)
            self.features_dc = self.features_dc * tint_t

    def export_splat(self, splat_path):
        """Export to standard .splat format for web 3D viewers (PlayCanvas/SuperSplat/WebGL)."""
        os.makedirs(os.path.dirname(os.path.abspath(splat_path)), exist_ok=True)
        xyz_np = self.xyz.detach().cpu().numpy().astype(np.float32)
        scales_np = self.scaling.detach().cpu().numpy().astype(np.float32)
        
        rot = self.rotation.detach().cpu().numpy()
        rot_uint8 = np.clip((rot * 128 + 128), 0, 255).astype(np.uint8)
        
        # Color from DC SH component (C0 = 0.28209479177387814)
        SH_C0 = 0.28209479177387814
        rgb_t = self.features_dc.squeeze(1).detach().cpu().numpy() * SH_C0 + 0.5
        rgb_uint8 = np.clip(rgb_t * 255, 0, 255).astype(np.uint8)
        alpha_uint8 = np.clip(self.opacity.detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
        rgba_uint8 = np.concatenate([rgb_uint8, alpha_uint8], axis=-1)
        
        with open(splat_path, 'wb') as f:
            for i in range(len(xyz_np)):
                f.write(xyz_np[i].tobytes())
                f.write(scales_np[i].tobytes())
                f.write(rgba_uint8[i].tobytes())
                f.write(rot_uint8[i].tobytes())
        print(f"[Export] Saved WebGL 3D .splat file ({len(xyz_np):,} Gaussians) to {splat_path}")

    def save_ply(self, output_path):
        """Export the (potentially modified/cleaned) Gaussian model to PLY format."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        xyz_np = self.xyz.detach().cpu().numpy()
        f_dc_np = self.features_dc.detach().cpu().numpy().reshape(-1, 3)
        f_rest_np = self.features_rest.detach().cpu().numpy().reshape(-1, 45)
        # Invert activations to match original ply storage
        op_np = torch.logit(self.opacity.clamp(1e-6, 1.0 - 1e-6)).detach().cpu().numpy()
        scale_np = torch.log(self.scaling.clamp(min=1e-6)).detach().cpu().numpy()
        rot_np = self.rotation.detach().cpu().numpy()

        dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                      ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
        for i in range(3):
            dtype_list.append((f'f_dc_{i}', 'f4'))
        for i in range(45):
            dtype_list.append((f'f_rest_{i}', 'f4'))
        dtype_list.append(('opacity', 'f4'))
        for i in range(3):
            dtype_list.append((f'scale_{i}', 'f4'))
        for i in range(4):
            dtype_list.append((f'rot_{i}', 'f4'))

        elements = np.empty(len(xyz_np), dtype=dtype_list)
        elements['x'] = xyz_np[:, 0]
        elements['y'] = xyz_np[:, 1]
        elements['z'] = xyz_np[:, 2]
        elements['nx'] = 0; elements['ny'] = 0; elements['nz'] = 0
        for i in range(3):
            elements[f'f_dc_{i}'] = f_dc_np[:, i]
        for i in range(45):
            elements[f'f_rest_{i}'] = f_rest_np[:, i]
        elements['opacity'] = op_np[:, 0]
        for i in range(3):
            elements[f'scale_{i}'] = scale_np[:, i]
        for i in range(4):
            elements[f'rot_{i}'] = rot_np[:, i]

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(output_path)
        print(f"[Export] Saved modified PLY ({len(xyz_np):,} Gaussians) to {output_path}")


# ==============================================================================
# 2. Scene Loader & Camera Utilities
# ==============================================================================

def find_model_files(model_dir_or_ply):
    """Auto-detect PLY and cameras.json from directory or ply file path."""
    if os.path.isfile(model_dir_or_ply) and model_dir_or_ply.endswith(".ply"):
        ply_path = model_dir_or_ply
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(ply_path)))
        cams_path = os.path.join(base_dir, "cameras.json")
        if not os.path.exists(cams_path):
            cams_path = os.path.join(os.path.dirname(ply_path), "cameras.json")
    else:
        # Search model dir for point_cloud iterations
        model_dir = model_dir_or_ply
        cams_path = os.path.join(model_dir, "cameras.json")
        ply_path = None
        # Try latest iteration first
        for it in [30000, 7000, 5000, 3000, 2000, 1000]:
            candidate = os.path.join(model_dir, "point_cloud", f"iteration_{it}", "point_cloud.ply")
            if os.path.exists(candidate):
                ply_path = candidate
                break
        if ply_path is None:
            # Check direct ply in folder
            for f in os.listdir(model_dir):
                if f.endswith(".ply"):
                    ply_path = os.path.join(model_dir, f)
                    break

    if not ply_path or not os.path.exists(ply_path):
        raise FileNotFoundError(f"Could not locate PLY file in {model_dir_or_ply}")
    if not os.path.exists(cams_path):
        cams_path = None

    return ply_path, cams_path


def load_gaussian_cloud(ply_path, device="cuda"):
    """Fast GPU-vectorized loader for 3D Gaussian PLY."""
    print(f"[Loader] Reading PLY: {ply_path}", flush=True)
    t0 = time.time()
    plydata = PlyData.read(ply_path)
    v = plydata['vertex'].data
    matrix = np.column_stack([v[name] for name in v.dtype.names]).astype(np.float32)
    num_pts = len(matrix)

    xyz = torch.tensor(matrix[:, :3], dtype=torch.float32, device=device)
    features_dc = torch.tensor(matrix[:, 6:9], dtype=torch.float32, device=device).unsqueeze(1)
    
    # Handle SH properties
    if matrix.shape[1] >= 54:
        features_rest = torch.tensor(matrix[:, 9:54], dtype=torch.float32, device=device).reshape(num_pts, 15, 3)
        sh_degree = 3
    else:
        features_rest = torch.zeros((num_pts, 15, 3), dtype=torch.float32, device=device)
        sh_degree = 0

    opacity = torch.sigmoid(torch.tensor(matrix[:, 54:55], dtype=torch.float32, device=device))
    scaling = torch.exp(torch.tensor(matrix[:, 55:58], dtype=torch.float32, device=device))
    rotation = torch.tensor(matrix[:, 58:62], dtype=torch.float32, device=device)
    rotation = rotation / torch.norm(rotation, dim=-1, keepdim=True)

    print(f"[Loader] Loaded {num_pts:,} Gaussians to GPU in {time.time()-t0:.2f}s", flush=True)
    return GaussianCloud(xyz, features_dc, features_rest, opacity, scaling, rotation, active_sh=sh_degree)


def generate_orbit_cameras(cams_data=None, gaussians=None, radius=None, height=None,
                           num_frames=60, width=1280, height_px=720, fov=60.0):
    """Generate a smooth 360-degree orbital camera trajectory centered on the scene."""
    # 1. Determine Center, Up vector, and Radius
    if cams_data is not None and len(cams_data) > 0:
        world_centers = []
        ups = []
        for c in cams_data:
            R = np.array(c['rotation'], dtype=np.float32)
            t = np.array(c['position'], dtype=np.float32)
            w2v = getWorld2View2(R, t)
            c2w = np.linalg.inv(w2v)
            world_centers.append(c2w[:3, 3])
            ups.append(c2w[:3, :3] @ np.array([0, -1, 0], dtype=np.float32))

        world_centers = np.array(world_centers)
        scene_center = np.median(world_centers, axis=0)
        
        # Invert slightly towards model center if available
        if gaussians is not None:
            g_center = np.median(gaussians.xyz.detach().cpu().numpy(), axis=0)
            scene_center = 0.7 * g_center + 0.3 * scene_center

        up_vector = np.mean(ups, axis=0)
        up_vector = up_vector / np.linalg.norm(up_vector)
        
        auto_radius = np.mean(np.linalg.norm(world_centers - scene_center, axis=1)) * 0.95
        auto_height = float(np.mean(world_centers[:, 1]))
    else:
        scene_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if gaussians is not None:
            scene_center = np.median(gaussians.xyz.detach().cpu().numpy(), axis=0)
        up_vector = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        auto_radius = 3.6
        auto_height = -0.3

    final_radius = radius if radius is not None else auto_radius
    final_height = height if height is not None else auto_height

    print(f"[Orbit] 360 Trajectory: Center={scene_center.round(2)}, Radius={final_radius:.2f}, Frames={num_frames}")

    # Build reference coordinate frame (horizontal plane perpendicular to up_vector)
    temp_ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(np.dot(temp_ref, up_vector)) > 0.9:
        temp_ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    u_axis = np.cross(up_vector, temp_ref)
    u_axis = u_axis / np.linalg.norm(u_axis)
    v_axis = np.cross(up_vector, u_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)

    cams = []
    fovx = math.radians(fov)
    fovy = 2 * math.atan(math.tan(fovx / 2) * (height_px / width))

    for i in range(num_frames):
        theta = 2 * math.pi * (i / num_frames)
        
        # Position in orbit circle
        cam_pos = scene_center + (u_axis * math.cos(theta) + v_axis * math.sin(theta)) * final_radius + up_vector * (final_height * 0.3)
        
        # Look-at matrix
        forward = scene_center - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up_vector)
        if np.linalg.norm(right) < 1e-4:
            right = u_axis
        right = right / np.linalg.norm(right)
        
        cam_up = np.cross(right, forward)
        cam_up = cam_up / np.linalg.norm(cam_up)

        # Build Camera-to-World (COLMAP / 3DGS format: X=right, Y=-cam_up, Z=forward)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = -cam_up
        c2w[:3, 2] = forward
        c2w[:3, 3] = cam_pos

        w2v = np.linalg.inv(c2w)
        R = w2v[:3, :3].T
        T = w2v[:3, 3]

        w2v_t = torch.tensor(getWorld2View2(R, T), dtype=torch.float32, device="cuda").transpose(0, 1)
        proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
        full_proj = (w2v_t.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

        cams.append(MiniCam(
            width=width, height=height_px, fovy=fovy, fovx=fovx,
            znear=0.01, zfar=100.0,
            world_view_transform=w2v_t,
            full_proj_transform=full_proj
        ))
    return cams


def generate_dataset_cameras(cams_data, num_frames=30, scale_res=1.0):
    """Interpolate/subsample viewpoints from dataset cameras.json."""
    step = max(1, len(cams_data) // num_frames)
    selected = cams_data[::step][:num_frames]
    cams = []

    for c in selected:
        w = int(c['width'] * scale_res)
        h = int(c['height'] * scale_res)
        fx = c['fx'] * scale_res
        fy = c['fy'] * scale_res
        fovx = 2 * math.atan(w / (2 * fx))
        fovy = 2 * math.atan(h / (2 * fy))

        R = np.array(c['rotation'], dtype=np.float32)
        pos = np.array(c['position'], dtype=np.float32)
        T = pos

        w2v = torch.tensor(getWorld2View2(R, T), dtype=torch.float32, device="cuda").transpose(0, 1)
        proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
        full_proj = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

        cams.append(MiniCam(
            width=w, height=h, fovy=fovy, fovx=fovx,
            znear=0.01, zfar=100.0,
            world_view_transform=w2v,
            full_proj_transform=full_proj
        ))
    return cams


# ==============================================================================
# 3. Main Rendering & Video Export
# ==============================================================================

def render_scene(gaussians, cameras, bg_color=(0.0, 0.0, 0.0), out_dir="./output/rendered",
                 export_gif=True, export_mp4=True, export_pngs=True, fps=30):
    """Render camera views and export GIF, MP4, and PNG images."""
    os.makedirs(out_dir, exist_ok=True)
    pipe = RenderPipelineParams()
    bg_t = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f"\n[Render] Rendering {len(cameras)} frames on GPU (Resolution: {cameras[0].image_width}x{cameras[0].image_height})...", flush=True)
    frames_pil = []
    frames_np = []
    
    t_start = time.time()
    for idx, cam in enumerate(cameras):
        t0 = time.time()
        with torch.no_grad():
            out = render(cam, gaussians, pipe, bg_t)["render"]
            torch.cuda.synchronize()

        img_np = (out.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        if export_pngs:
            img_pil.save(os.path.join(out_dir, f"frame_{idx:04d}.png"))

        frames_pil.append(img_pil)
        frames_np.append(img_np)

        frame_ms = (time.time() - t0) * 1000
        print(f"  Frame {idx+1:02d}/{len(cameras)} rendered in {frame_ms:.1f}ms ({1000/max(frame_ms,1):.1f} FPS)", flush=True)

    total_time = time.time() - t_start
    print(f"[Render] All {len(cameras)} frames completed in {total_time:.2f}s (Avg: {len(cameras)/total_time:.1f} FPS)")

    # Export GIF
    if export_gif and len(frames_pil) > 0:
        gif_path = os.path.join(out_dir, "orbit_360.gif")
        gif_frames = [f.resize((min(f.width, 800), int(f.height * min(f.width, 800) / f.width)), Image.Resampling.LANCZOS) for f in frames_pil]
        gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=int(1000/fps), loop=0)
        print(f"[Output] 360 Orbit GIF saved to: {gif_path}")

    # Export MP4 Video
    if export_mp4 and len(frames_np) > 0:
        try:
            import imageio
            mp4_path = os.path.join(out_dir, "orbit_360.mp4")
            # Ensure dimensions divisible by 16 for ffmpeg
            h, w = frames_np[0].shape[:2]
            pad_h = (16 - (h % 16)) % 16
            pad_w = (16 - (w % 16)) % 16
            
            writer = imageio.get_writer(mp4_path, fps=fps, quality=9)
            for f in frames_np:
                if pad_h > 0 or pad_w > 0:
                    f = np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
                writer.append_data(f)
            writer.close()
            print(f"[Output] 360 Orbit MP4 video saved to: {mp4_path}")
        except Exception as e:
            print(f"[Notice] MP4 export skipped: {e}")

    return out_dir


def launch_viewer(model_dir):
    """Launch the SIBR Interactive 3D Viewer for manual 360 navigation."""
    viewer_bin = r"C:\Users\Rishi\Downloads\test\viewers\bin\SIBR_gaussianViewer_app.exe"
    if not os.path.exists(viewer_bin):
        print(f"[Error] SIBR Viewer not found at {viewer_bin}")
        return

    print(f"[Viewer] Launching SIBR Interactive 3D Viewer for: {model_dir}")
    print("  Controls:")
    print("    - Left Click + Drag: 360 Orbit / Rotate scene")
    print("    - Right Click + Drag: Pan camera")
    print("    - Scroll: Zoom in / Zoom out")
    print("    - WASD: Free flight navigation")
    print("    - T key: Switch to Trackball mode")

    env = os.environ.copy()
    env["PATH"] = r"C:\Users\Rishi\Downloads\test\viewers\bin;C:\Users\Rishi\anaconda3\envs\gaussian_cuda\bin;" + env.get("PATH", "")
    subprocess.Popen([viewer_bin, "-m", model_dir], cwd=os.path.dirname(viewer_bin), env=env)


# ==============================================================================
# 4. CLI Interface
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting Studio & 360 Orbit Renderer")
    parser.add_argument("--model", type=str, default=r"C:\Users\Rishi\Downloads\test\output\truck",
                        help="Path to model directory or .ply file")
    parser.add_argument("--out-dir", type=str, default=r"C:\Users\Rishi\Downloads\test\output\truck_orbit360",
                        help="Directory to save rendered 360 outputs")
    
    # Orbit & Trajectory Options
    parser.add_argument("--trajectory", type=str, choices=["orbit", "dataset"], default="orbit",
                        help="Camera trajectory: 'orbit' for 360 circular orbit around object, 'dataset' for dataset path")
    parser.add_argument("--radius", type=float, default=None, help="Orbit radius (default: auto from dataset)")
    parser.add_argument("--height", type=float, default=None, help="Orbit elevation/height offset")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames for 360 full turn (default: 60)")
    parser.add_argument("--fps", type=int, default=30, help="Video framerate (default: 30)")
    parser.add_argument("--scale-res", type=float, default=1.0, help="Resolution scale factor (e.g. 0.5 for half-res)")
    
    # Visual Adjustments & Editing
    parser.add_argument("--bg", type=str, default="black", choices=["black", "white", "custom"],
                        help="Background color")
    parser.add_argument("--bg-color", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="Custom RGB background color: e.g. 0.2 0.2 0.2")
    parser.add_argument("--brightness", type=float, default=1.0, help="Brightness multiplier (e.g. 1.2)")
    parser.add_argument("--min-opacity", type=float, default=0.0, help="Filter out low opacity splats (e.g. 0.05)")
    parser.add_argument("--max-scale", type=float, default=None, help="Filter out oversized splats")
    parser.add_argument("--crop-radius", type=float, default=None, help="Crop splats outside 3D sphere radius")
    
    # Save & View Options
    parser.add_argument("--save-ply", type=str, default=None, help="Export edited Gaussian model to a new PLY file")
    parser.add_argument("--export-splat", type=str, default=None, help="Export to standard .splat format for web 3D viewers")
    parser.add_argument("--view", action="store_true", help="Launch interactive SIBR 3D viewer for manual orbit")

    args = parser.parse_args()

    # Launch interactive viewer mode if requested
    if args.view:
        launch_viewer(args.model)
        return

    # 1. Locate files
    ply_path, cams_path = find_model_files(args.model)

    # 2. Load Gaussian cloud onto GPU
    gaussians = load_gaussian_cloud(ply_path)

    # 3. Apply 3D edits and filters
    if args.min_opacity > 0.0 or args.max_scale is not None or args.crop_radius is not None:
        gaussians.apply_filters(min_opacity=args.min_opacity, max_scale=args.max_scale, crop_radius=args.crop_radius)
    
    if args.brightness != 1.0:
        gaussians.adjust_colors(brightness=args.brightness)

    # 4. Optional: Export modified PLY / SPLAT
    if args.save_ply:
        gaussians.save_ply(args.save_ply)
    if args.export_splat:
        gaussians.export_splat(args.export_splat)

    # 5. Background color
    if args.bg == "white":
        bg_rgb = (1.0, 1.0, 1.0)
    elif args.bg == "custom":
        bg_rgb = tuple(args.bg_color)
    else:
        bg_rgb = (0.0, 0.0, 0.0)

    # 6. Build Camera Trajectory
    cams_data = None
    if cams_path and os.path.exists(cams_path):
        with open(cams_path, "r") as f:
            cams_data = json.load(f)

    if args.trajectory == "dataset" and cams_data is not None:
        cameras = generate_dataset_cameras(cams_data, num_frames=args.frames, scale_res=args.scale_res)
    else:
        # 360 Full Orbit centered on truck
        w = int(1280 * args.scale_res)
        h = int(720 * args.scale_res)
        cameras = generate_orbit_cameras(
            cams_data=cams_data,
            gaussians=gaussians,
            radius=args.radius,
            height=args.height,
            num_frames=args.frames,
            width=w,
            height_px=h,
            fov=60.0
        )

    # 7. Render and Export
    render_scene(
        gaussians=gaussians,
        cameras=cameras,
        bg_color=bg_rgb,
        out_dir=args.out_dir,
        export_gif=True,
        export_mp4=True,
        export_pngs=True,
        fps=args.fps
    )


if __name__ == "__main__":
    main()

