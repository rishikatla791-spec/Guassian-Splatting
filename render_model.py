import os
import json
import torch
import numpy as np
import torchvision
from PIL import Image

import sys
sys.path.append(r'c:\Users\Rishi\Downloads\test\gaussian-splatting')

from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from gaussian_renderer import render
from arguments import PipelineParams

def render_scene(model_dir, output_dir, max_frames=30, iteration=30000):
    os.makedirs(output_dir, exist_ok=True)
    ply_path = os.path.join(model_dir, f'point_cloud/iteration_{iteration}/point_cloud.ply')
    if not os.path.exists(ply_path):
        # find highest available iteration
        pc_dir = os.path.join(model_dir, 'point_cloud')
        iters = [int(d.split('_')[-1]) for d in os.listdir(pc_dir) if d.startswith('iteration_')]
        max_it = max(iters)
        ply_path = os.path.join(pc_dir, f'iteration_{max_it}/point_cloud.ply')
        print(f'Using iteration {max_it}')

    cameras_json_path = os.path.join(model_dir, 'cameras.json')
    with open(cameras_json_path, 'r') as f:
        cam_data = json.load(f)

    print(f'Loaded {len(cam_data)} camera viewpoints from cameras.json')

    # Load Gaussian model
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)
    print(f'Loaded Gaussian Model with {gaussians.get_xyz.shape[0]} Gaussians!')

    pipeline = PipelineParams(None)
    pipeline.convert_SHs_python = False
    pipeline.compute_cov3D_python = False
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')

    # Subsample or take evenly spaced frames
    step = max(1, len(cam_data) // max_frames)
    selected_cams = cam_data[::step][:max_frames]

    rendered_images = []
    print(f'Rendering {len(selected_cams)} novel views...')

    with torch.no_grad():
        for i, cam_dict in enumerate(selected_cams):
            w = cam_dict['width']
            h = cam_dict['height']
            fx = cam_dict['fx']
            fy = cam_dict['fy']
            fovx = focal2fov(fx, w)
            fovy = focal2fov(fy, h)
            
            # Position and rotation from cameras.json
            R = np.array(cam_dict['rotation'])
            pos = np.array(cam_dict['position'])
            
            # In cameras.json, R and pos are stored such that:
            # T = -R @ pos in world-to-cam coordinates
            T = -np.matmul(R.T, pos)
            
            # world_view_transform
            world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
            projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

            cam = MiniCam(width=w, height=h, fovy=fovy, fovx=fovx, znear=0.01, zfar=100.0,
                          world_view_transform=world_view_transform, full_proj_transform=full_proj_transform)
            
            res = render(cam, gaussians, pipeline, background)
            img = res['render']
            img = torch.clamp(img, 0.0, 1.0)
            
            save_path = os.path.join(output_dir, f'render_{i:04d}.png')
            torchvision.utils.save_image(img, save_path)
            print(f'Rendered frame {i+1}/{len(selected_cams)} -> {save_path}')

            # Convert for gif
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            rendered_images.append(Image.fromarray(img_np))

    if rendered_images:
        gif_path = os.path.join(output_dir, 'rendered_flythrough.gif')
        rendered_images[0].save(gif_path, save_all=True, append_images=rendered_images[1:], duration=100, loop=0)
        print(f'Saved animated flythrough to {gif_path}')

if __name__ == '__main__':
    render_scene(
        model_dir=r'C:\Users\Rishi\Downloads\test\output\pretrained_train\train',
        output_dir=r'C:\Users\Rishi\Downloads\test\output\rendered_results',
        max_frames=20
    )
