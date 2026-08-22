import os, sys, json, time, traceback
import numpy as np
import torch
import torchvision
from PIL import Image

sys.path.append(r'c:\Users\Rishi\Downloads\test\gaussian-splatting')

try:
    from plyfile import PlyData
    from scene.gaussian_model import GaussianModel
    from scene.cameras import MiniCam
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
    from gaussian_renderer import render

    class SimplePipe:
        def __init__(self):
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.debug = False
            self.antialiasing = False

    model_dir = r'C:\Users\Rishi\Downloads\test\output\pretrained_train\train'
    ply_path = os.path.join(model_dir, 'point_cloud/iteration_30000/point_cloud.ply')
    if not os.path.exists(ply_path):
        ply_path = os.path.join(model_dir, 'point_cloud/iteration_7000/point_cloud.ply')

    print('Loading PLY:', ply_path)
    t0 = time.time()
    ply = PlyData.read(ply_path)
    v_data = ply['vertex'].data
    num_pts = len(v_data)
    matrix = v_data.view(np.float32).reshape(num_pts, -1)
    print(f'Zero-copy parsed {num_pts} Gaussians in {time.time()-t0:.3f}s!')

    xyz = matrix[:, 0:3]
    f_dc = matrix[:, 6:9].reshape(num_pts, 3, 1)
    f_rest = matrix[:, 9:54].reshape(num_pts, 3, 15)
    opacities = matrix[:, 54:55]
    scales = matrix[:, 55:58]
    rots = matrix[:, 58:62]

    gaussians = GaussianModel(sh_degree=3)
    gaussians._xyz = torch.nn.Parameter(torch.from_numpy(xyz.copy()).cuda())
    gaussians._features_dc = torch.nn.Parameter(torch.from_numpy(f_dc.copy()).cuda().transpose(1, 2).contiguous())
    gaussians._features_rest = torch.nn.Parameter(torch.from_numpy(f_rest.copy()).cuda().transpose(1, 2).contiguous())
    gaussians._opacity = torch.nn.Parameter(torch.from_numpy(opacities.copy()).cuda())
    gaussians._scaling = torch.nn.Parameter(torch.from_numpy(scales.copy()).cuda())
    gaussians._rotation = torch.nn.Parameter(torch.from_numpy(rots.copy()).cuda())
    gaussians.active_sh_degree = 3

    print(f'Transferred to CUDA in {time.time()-t0:.3f}s!')

    with open(os.path.join(model_dir, 'cameras.json'), 'r') as f:
        cams = json.load(f)

    pipeline = SimplePipe()
    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')

    out_dir = r'C:\Users\Rishi\Downloads\test\output\rendered_results'
    os.makedirs(out_dir, exist_ok=True)

    # Select 16 viewpoints across camera trajectory
    step = max(1, len(cams) // 16)
    selected = cams[::step][:16]

    print(f'Rendering {len(selected)} high-resolution novel views on GPU...')
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
        
        render_time_ms = (time.time() - t_render) * 1000.0
        fps = 1000.0 / render_time_ms
        print(f'Rendered View {i+1:02d}/{len(selected)} ({w}x{h}) in {render_time_ms:.1f}ms ({fps:.1f} FPS)')

        img = torch.clamp(res['render'], 0.0, 1.0)
        save_p = os.path.join(out_dir, f'render_{i:03d}.png')
        torchvision.utils.save_image(img, save_p)
        
        # Save optimized frame for GIF flythrough
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        img_pil.thumbnail((800, 450))
        gif_frames.append(img_pil)

    gif_path = os.path.join(out_dir, 'flythrough.gif')
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=120, loop=0)
    print(f'Successfully rendered all frames and saved flythrough animation to {gif_path}!')

except Exception as e:
    traceback.print_exc()
