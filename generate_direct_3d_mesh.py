#!/usr/bin/env python3
import sys
from pathlib import Path

# Add root and parent directory to sys.path
file_path = Path(__file__).resolve()
parent_dir = file_path.parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(file_path.parent))

from gaussian.pipeline.pose_estimation import PoseEstimator
from gaussian.pipeline.dense_geometry import DenseGeometryReconstructor
from gaussian.core.gaussians import GaussianModel
from gaussian.export_3d_mesh import export_all_formats

def main():
    root_dir = file_path.parent
    images_dir = root_dir / "imgaes_new_input_frames"
    output_dir = root_dir / "output_new_input_3dmodel"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Extracting Multi-View Poses and 3D Point Cloud for New Input Video ===")
    estimator = PoseEstimator(backend_preference="auto")
    cameras, points3d, colors, metrics = estimator.estimate_poses(images_dir)

    print(f"Points 3D extracted: {len(points3d)}")

    reconstructor = DenseGeometryReconstructor()
    clean_pts, clean_cols, normals, knn_dist = reconstructor.process_point_cloud(points3d, colors)

    # Save initial PLY
    gaussians = GaussianModel(sh_degree=3)
    gaussians.init_from_pointcloud(clean_pts, clean_cols)
    
    ply_path = output_dir / "point_cloud" / "iteration_0" / "point_cloud.ply"
    gaussians.save_ply(ply_path)

    # Convert to real 3D polygonal mesh formats (OBJ, GLTF, PLY Mesh)
    mesh_paths = export_all_formats(ply_path, output_dir)
    print("\n[OK] Generated Real 3D Model Formats:")
    for fmt, p in mesh_paths.items():
        print(f"  - {fmt.upper()}: {p}")

if __name__ == "__main__":
    main()
