import os, sys, json, time
import numpy as np
import torch
import torchvision
from PIL import Image

sys.path.append(r'c:\Users\Rishi\Downloads\test\gaussian-splatting')
from plyfile import PlyData
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from gaussian_renderer import render
from arguments import PipelineParams

model_dir = r'C:\Users\Rishi\Downloads\test\output\pretrained_train\train'
ply_path = os.path.join(model_dir, 'point_cloud/iteration_7000/point_cloud.ply')

print('Loading PLY file:', ply_path)
t0 = time.time()
plydata = PlyData.read(ply_path)
v_data = plydata.elements[0].data
print(f'Read PLY structured array in {time.time()-t0:.2f}s ({len(v_data)} points)')

gaussians = GaussianModel(sh_degree=3)

# Fast vectorized NumPy loading
xyz = np.stack([v_data['x'], v_data['y'], v_data['z']], axis=1)
opacities = v_data['opacity'][..., np.newaxis]

features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
features_dc[:, 0, 0] = v_data['f_dc_0']
features_dc[:, 1, 0] = v_data['f_dc_1']
features_dc[:, 2, 0] = v_data['f_dc_2']

extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
features_extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
for idx, attr_name in enumerate(extra_f_names):
    features_extra[:, idx] = v_data[attr_name]
features_extra = features_extra.reshape((features_extra.shape[0], 3, (gaussians.max_sh_degree + 1) ** 2 - 1))

scale_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")], key=lambda x: int(x.split('_')[-1]))
scales = np.stack([v_data[name] for name in scale_names], axis=1)

rot_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("rot")], key=lambda x: int(x.split('_')[-1]))
rots = np.stack([v_data[name] for name in rot_names], axis=1)

gaussians._xyz = torch.nn.Parameter(torch.from_numpy(xyz).float().cuda())
gaussians._features_dc = torch.nn.Parameter(torch.from_numpy(features_dc).float().cuda().transpose(1, 2).contiguous())
gaussians._features_rest = torch.nn.Parameter(torch.from_numpy(features_extra).float().cuda().transpose(1, 2).contiguous())
gaussians._opacity = torch.nn.Parameter(torch.from_numpy(opacities).float().cuda())
gaussians._scaling = torch.nn.Parameter(torch.from_numpy(scales).float().cuda())
gaussians._rotation = torch.nn.Parameter(torch.from_numpy(rots).float().cuda())
gaussians.active_sh_degree = gaussians.max_sh_degree

print(f'Model loaded onto CUDA GPU in {time.time()-t0:.2f}s total!')

with open(os.path.join(model_dir, 'cameras.json'), 'r') as f:
    cams = json.load(f)

pipeline = PipelineParams(None)
pipeline.convert_SHs_python = False
pipeline.compute_cov3D_python = False
bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')

out_dir = r'C:\Users\Rishi\Downloads\test\output\rendered_results'
os.makedirs(out_dir, exist_ok=True)

# Select 12 evenly spaced viewpoints across the trajectory
step = len(cams) // 12
selected = cams[::step][:12]

gif_frames = []
for i, c in enumerate(selected):
    w, h = c['width'], c['height']
    fx, fy = c['fx'], c['fy']
    fovx = focal2fov(fx, w)
    fovy = focal2fov(fy, h)
    R = np.array(c['rotation'])
    pos = np.array(c['position'])
    T = -np.matmul(R.T, pos)

    w2v = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    full_proj = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

    cam = MiniCam(width=w, height=h, fovy=fovy, fovx=fovx, znear=0.01, zfar=100.0,
                  world_view_transform=w2v, full_proj_transform=full_proj)

    t_render = time.time()
    with torch.no_grad():
        res = render(cam, gaussians, pipeline, bg)
        torch.cuda.synchronize()
    
    fps = 1.0 / (time.time() - t_render)
    print(f'Rendered frame {i+1}/12 ({w}x{h}) in {(time.time()-t_render)*1000:.1f}ms ({fps:.1f} FPS)!')

    img = torch.clamp(res['render'], 0.0, 1.0)
    save_p = os.path.join(out_dir, f'frame_{i:03d}.png')
    torchvision.utils.save_image(img, save_p)
    
    # create downscaled frame for gif
    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    img_pil.thumbnail((800, 450))
    gif_frames.append(img_pil)

gif_path = os.path.join(out_dir, 'scene_flythrough.gif')
gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=150, loop=0)
print(f'Saved animated flythrough to {gif_path}!')
