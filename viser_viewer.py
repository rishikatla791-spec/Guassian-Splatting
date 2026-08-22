"""
Viser 3D Gaussian Splatting Interactive Visualizer
==================================================
High-performance Python 3DGS visualizer with 360 orbit, 3D grid, camera frustums, and real-time GUI controls.
"""

import os
import sys
import time
import json
import webbrowser
import numpy as np
from plyfile import PlyData
import viser
import viser.transforms as tf

# ==============================================================================
# 1. 3DGS Math Helpers (Covariances & Color)
# ==============================================================================

def load_ply_for_viser(ply_path, max_points=1_000_000):
    """Parse PLY file into centers, 3D covariances (Nx3x3), RGBs, and opacities."""
    print(f"[Viser] Loading PLY: {ply_path}...", flush=True)
    t0 = time.time()
    plydata = PlyData.read(ply_path)
    v = plydata['vertex'].data
    num_pts = len(v)

    if num_pts > max_points:
        idx = np.random.choice(num_pts, max_points, replace=False)
        v = v[idx]
        num_pts = len(v)

    # 1. Centers (x, y, z)
    centers = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)

    # 2. Scales & Rotations -> 3D Covariances: V = R * S * S^T * R^T
    scales = np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1)).astype(np.float32)
    
    # Rotations (quaternions normalized: [w, x, y, z] or [x, y, z, w])
    rots = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1)
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    
    # Quaternion to 3x3 Rotation matrix (rot_0=w, rot_1=x, rot_2=y, rot_3=z)
    qw, qx, qy, qz = rots[:, 0], rots[:, 1], rots[:, 2], rots[:, 3]
    
    R = np.zeros((num_pts, 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (qy*qy + qz*qz)
    R[:, 0, 1] = 2 * (qx*qy - qz*qw)
    R[:, 0, 2] = 2 * (qx*qz + qy*qw)
    R[:, 1, 0] = 2 * (qx*qy + qz*qw)
    R[:, 1, 1] = 1 - 2 * (qx*qx + qz*qz)
    R[:, 1, 2] = 2 * (qy*qz - qx*qw)
    R[:, 2, 0] = 2 * (qx*qz - qy*qw)
    R[:, 2, 1] = 2 * (qy*qz + qx*qw)
    R[:, 2, 2] = 1 - 2 * (qx*qx + qy*qy)

    # M = R * diag(S)
    M = R * scales[:, np.newaxis, :]
    covariances = np.einsum('nij,nkj->nik', M, M) # V = M @ M.T

    # 3. RGB Colors from Spherical Harmonics DC component (SH_C0 = 0.28209479177387814)
    SH_C0 = 0.28209479177387814
    rgbs = np.clip(np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1) * SH_C0 + 0.5, 0.0, 1.0).astype(np.float32)

    # 4. Opacities (Nx1 required by Viser)
    opacities = (1.0 / (1.0 + np.exp(-v['opacity']))).astype(np.float32)[:, np.newaxis]

    print(f"[Viser] Loaded {num_pts:,} Gaussians in {time.time()-t0:.2f}s", flush=True)
    return centers, covariances, rgbs, opacities


# ==============================================================================
# 2. Main Viser Server & Interactive GUI
# ==============================================================================

def run_viser_viewer(port=8080):
    server = viser.ViserServer(port=port)
    print(f"\n[Viser] Interactive 3D Server running at http://localhost:{port}\n")

    # Coordinate alignment (align dataset coordinate system to Viser world frame)
    scene_rotation = tf.SO3.from_x_radians(np.pi).wxyz
    
    # Add 3D Grid Floor
    grid = server.scene.add_grid(
        name="/grid",
        width=12.0,
        height=12.0,
        position=(0.0, 0.65, 0.0),
        wxyz=scene_rotation,
        plane="xz",
        cell_color=(56, 189, 248),
        section_color=(148, 163, 184),
        section_thickness=1.5
    )

    room_ply = r"C:\Users\Rishi\Downloads\test\output\room\point_cloud\iteration_3000\point_cloud.ply"
    if not os.path.exists(room_ply):
        room_ply = r"C:\Users\Rishi\Downloads\test\output\room\point_cloud\iteration_2000\point_cloud.ply"

    models = {
        "🏠 My Custom Room (Your Capture)": room_ply,
        "🎮 Playroom Scene (3,000 steps)": r"C:\Users\Rishi\Downloads\test\output\playroom\point_cloud\iteration_3000\point_cloud.ply",
        "🚚 Truck Scene (1,000 steps)": r"C:\Users\Rishi\Downloads\test\output\truck\point_cloud\iteration_1000\point_cloud.ply",
        "🏆 Pretrained Ultra-HD (30,000 steps)": r"C:\Users\Rishi\Downloads\test\output\pretrained_train\train\point_cloud\iteration_30000\point_cloud.ply"
    }

    splat_handle = None
    cameras_folder = None
    all_data = {}

    def load_model(model_key):
        nonlocal splat_handle, all_data
        ply_path = models[model_key]
        if not os.path.exists(ply_path):
            print(f"[Warning] Path not found: {ply_path}")
            return

        centers, covs, rgbs, ops = load_ply_for_viser(ply_path)
        all_data = {
            "centers": centers,
            "covs": covs,
            "rgbs": rgbs,
            "ops": ops
        }

        if splat_handle is not None:
            splat_handle.remove()

        splat_handle = server.scene.add_gaussian_splats(
            name="/gaussians",
            centers=centers,
            covariances=covs,
            rgbs=rgbs,
            opacities=ops,
            wxyz=scene_rotation
        )

    # Initial Load - Check Custom Room first, fallback to Playroom
    default_key = "🏠 My Custom Room (Your Capture)" if os.path.exists(models["🏠 My Custom Room (Your Capture)"]) else "🎮 Playroom Scene (3,000 steps)"
    load_model(default_key)

    # ==============================================================================
    # 3. Add Dataset Camera Frustums
    # ==============================================================================
    cams_json_path = r"C:\Users\Rishi\Downloads\test\output\truck\cameras.json"
    cam_handles = []
    if os.path.exists(cams_json_path):
        with open(cams_json_path, "r") as f:
            cams_data = json.load(f)
        
        # Subsample camera frustums for clean visualization
        step = max(1, len(cams_data) // 30)
        for i, c in enumerate(cams_data[::step]):
            R_w2c = np.array(c['rotation'], dtype=np.float32)
            T_w2c = np.array(c['position'], dtype=np.float32)
            
            # c2w = inv(w2v)
            Rt = np.eye(4, dtype=np.float32)
            Rt[:3, :3] = R_w2c.T
            Rt[:3, 3] = T_w2c
            c2w = np.linalg.inv(Rt)
            
            pos = c2w[:3, 3]
            rot_mat = c2w[:3, :3]
            wxyz = tf.SO3.from_matrix(rot_mat).wxyz

            h = server.scene.add_camera_frustum(
                name=f"/cameras/cam_{i}",
                fov=np.radians(50.0),
                aspect=c['width'] / c['height'],
                scale=0.15,
                color=(245, 158, 11),
                position=pos,
                wxyz=wxyz,
                visible=False
            )
            cam_handles.append(h)

    # ==============================================================================
    # 4. Interactive Sidebar GUI Controls
    # ==============================================================================
    with server.gui.add_folder("🎨 Scene Settings"):
        model_dropdown = server.gui.add_dropdown(
            "Select 3D Model",
            options=list(models.keys()),
            initial_value=default_key
        )
        @model_dropdown.on_update
        def _(_):
            load_model(model_dropdown.value)

        grid_toggle = server.gui.add_checkbox("3D Floor Grid", initial_value=True)
        @grid_toggle.on_update
        def _(_):
            grid.visible = grid_toggle.value

        cam_toggle = server.gui.add_checkbox("Dataset Camera Frustums", initial_value=False)
        @cam_toggle.on_update
        def _(_):
            for ch in cam_handles:
                ch.visible = cam_toggle.value

    with server.gui.add_folder("⚡ Filtering & Display"):
        opacity_slider = server.gui.add_slider("Min Opacity Filter", min=0.0, max=0.3, step=0.01, initial_value=0.0)
        @opacity_slider.on_update
        def _(_):
            if not all_data or splat_handle is None: return
            thresh = opacity_slider.value
            mask = (all_data["ops"] >= thresh).squeeze(-1)
            splat_handle.centers = all_data["centers"][mask]
            splat_handle.covariances = all_data["covs"][mask]
            splat_handle.rgbs = all_data["rgbs"][mask]
            splat_handle.opacities = all_data["ops"][mask]

    with server.gui.add_folder("🔄 360° Auto Orbit"):
        auto_orbit = server.gui.add_checkbox("Auto 360° Spin", initial_value=False)
        orbit_speed = server.gui.add_slider("Orbit Speed", min=0.5, max=5.0, step=0.5, initial_value=2.0)

    # Auto-orbit loop
    webbrowser.open(f"http://localhost:{port}")

    theta = 0.0
    while True:
        if auto_orbit.value:
            theta += 0.02 * orbit_speed.value
            radius = 3.5
            cam_x = -0.38 + radius * np.cos(theta)
            cam_z = 0.32 + radius * np.sin(theta)
            for client in server.get_clients().values():
                client.camera.position = np.array([cam_x, 0.4, cam_z])
                client.camera.look_at = np.array([-0.38, -0.1, 0.32])
        time.sleep(0.03)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Viser 3D Gaussian Splatting Visualizer")
    parser.add_argument("--port", type=int, default=8090, help="Port to run Viser server on (default: 8090)")
    args = parser.parse_args()
    run_viser_viewer(port=args.port)

