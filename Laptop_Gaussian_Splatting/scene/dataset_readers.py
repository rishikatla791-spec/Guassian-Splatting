"""
Scene Dataset Readers: COLMAP & Synthetic Dataset Parsers.
Direct 1:1 match with official Inria 3DGS dataset_readers.py.
"""

import os
import sys
import math
import json
from pathlib import Path
from typing import NamedTuple, List, Dict, Optional
import numpy as np
import torch
from plyfile import PlyData, PlyElement

from scene.colmap_loader import (
    read_extrinsics_binary, read_intrinsics_binary,
    read_extrinsics_text, read_intrinsics_text,
    read_points3D_binary, read_points3D_text,
    qvec2rotmat
)
from core.camera import Camera, CameraIntrinsics, CameraExtrinsics


class BasicPointCloud(NamedTuple):
    points: np.ndarray    # (N, 3) float32
    colors: np.ndarray    # (N, 3) float32 in [0, 1]
    normals: np.ndarray   # (N, 3) float32


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: List[Camera]
    test_cameras: List[Camera]
    scene_extent: float
    ply_path: str


def focal2fov(focal: float, dim: float) -> float:
    return 2.0 * math.atan(dim / (2.0 * focal))


def fov2focal(fov: float, dim: float) -> float:
    return dim / (2.0 * math.tan(fov / 2.0))


def fetch_ply(path: str) -> BasicPointCloud:
    """Read point cloud from standard PLY file."""
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T.astype(np.float32)
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T.astype(np.float32) / 255.0
    if 'nx' in vertices:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T.astype(np.float32)
    else:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def read_colmap_scene_info(
    dataset_path: str,
    images_subfolder: str = "images",
    eval_mode: bool = True,
    llffhold: int = 8,
    target_resolution: Optional[int] = None,
) -> SceneInfo:
    """
    Parse a standard COLMAP dataset directory.

    Args:
        dataset_path:       Root path containing 'images' and 'sparse/0'
        images_subfolder:   Folder containing input images (default: 'images')
        eval_mode:          If True, split cameras into train/test using llffhold
        llffhold:           Hold out every N-th camera for evaluation (default: 8)
        target_resolution:  Optional downscaling factor (1, 2, 4, 8) or max dimension
    """
    dataset_path = str(dataset_path)
    sparse_path = os.path.join(dataset_path, "sparse", "0")
    if not os.path.exists(sparse_path):
        sparse_path = os.path.join(dataset_path, "sparse")

    # 1. Read Cameras and Extrinsics
    try:
        cameras_extrinsic_file = os.path.join(sparse_path, "images.bin")
        cameras_intrinsic_file = os.path.join(sparse_path, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except Exception:
        cameras_extrinsic_file = os.path.join(sparse_path, "images.txt")
        cameras_intrinsic_file = os.path.join(sparse_path, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    images_folder = os.path.join(dataset_path, images_subfolder)

    # 2. Build Camera list
    all_cameras = []
    # Sort images by name for deterministic order
    sorted_image_ids = sorted(cam_extrinsics.keys(), key=lambda k: cam_extrinsics[k].name)

    for idx, key in enumerate(sorted_image_ids):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        
        orig_w = intr.width
        orig_h = intr.height

        # Coordinate transformation: R is W2C rotation, T is W2C translation
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec, dtype=np.float64)

        if intr.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fx = fy = float(intr.params[0])
            cx = float(intr.params[1])
            cy = float(intr.params[2])
        elif intr.model in ("PINHOLE", "OPENCV", "RADIAL"):
            fx = float(intr.params[0])
            fy = float(intr.params[1])
            cx = float(intr.params[2])
            cy = float(intr.params[3])
        else:
            fx = fy = float(intr.params[0])
            cx = float(orig_w / 2.0)
            cy = float(orig_h / 2.0)

        image_path = os.path.join(images_folder, extr.name)

        intrinsics = CameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=orig_w, height=orig_h
        )
        extrinsics = CameraExtrinsics(R=R, T=T)
        cam = Camera(uid=idx, intrinsics=intrinsics, extrinsics=extrinsics, image_path=image_path)
        all_cameras.append(cam)

    # 3. Train / Test Split
    train_cameras = []
    test_cameras = []
    for i, cam in enumerate(all_cameras):
        if eval_mode and (i % llffhold == 0):
            test_cameras.append(cam)
        else:
            train_cameras.append(cam)

    if not test_cameras:
        test_cameras = [train_cameras[0]]

    # 4. Load Point Cloud
    ply_path = os.path.join(sparse_path, "points3D.ply")
    bin_path = os.path.join(sparse_path, "points3D.bin")
    txt_path = os.path.join(sparse_path, "points3D.txt")

    if os.path.exists(ply_path):
        point_cloud = fetch_ply(ply_path)
    elif os.path.exists(bin_path):
        xyz, rgb, _ = read_points3D_binary(bin_path)
        point_cloud = BasicPointCloud(points=xyz, colors=rgb / 255.0, normals=np.zeros_like(xyz))
    elif os.path.exists(txt_path):
        xyz, rgb, _ = read_points3D_text(txt_path)
        point_cloud = BasicPointCloud(points=xyz, colors=rgb / 255.0, normals=np.zeros_like(xyz))
    else:
        # Fallback random initialization inside camera bounding box
        centers = np.stack([cam.camera_center.cpu().numpy() for cam in all_cameras])
        centroid = centers.mean(axis=0)
        radius = np.linalg.norm(centers - centroid, axis=-1).max()
        num_pts = 10_000
        xyz = np.random.uniform(-radius, radius, (num_pts, 3)).astype(np.float32) + centroid
        rgb = np.random.uniform(0.1, 0.9, (num_pts, 3)).astype(np.float32)
        point_cloud = BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros_like(xyz))

    # 5. Scene Extent Calculation (matching official Inria formula)
    centers = [cam.camera_center.cpu().numpy() for cam in all_cameras]
    avg_cam_center = np.mean(centers, axis=0)
    dists = np.linalg.norm(centers - avg_cam_center, axis=-1)
    scene_extent = float(np.max(dists) * 1.1)

    print(f"[DatasetReader] COLMAP Scene Loaded: {len(all_cameras)} total views ({len(train_cameras)} train, {len(test_cameras)} test)")
    print(f"[DatasetReader] Sparse points: {point_cloud.points.shape[0]:,} points, Scene Extent: {scene_extent:.2f}")

    return SceneInfo(
        point_cloud=point_cloud,
        train_cameras=train_cameras,
        test_cameras=test_cameras,
        scene_extent=scene_extent,
        ply_path=ply_path if os.path.exists(ply_path) else "",
    )
