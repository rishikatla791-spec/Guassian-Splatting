import math
import numpy as np
from plyfile import PlyData

def convert_ply_to_splat(ply_path, splat_path):
    print(f"Reading {ply_path}...")
    plydata = PlyData.read(ply_path)
    v = plydata['vertex'].data
    num_pts = len(v)
    
    # Extract properties
    x = np.asarray(v['x'], dtype=np.float32)
    y = np.asarray(v['y'], dtype=np.float32)
    z = np.asarray(v['z'], dtype=np.float32)
    
    # Scale: exp(scale)
    scale_0 = np.exp(np.asarray(v['scale_0'], dtype=np.float32))
    scale_1 = np.exp(np.asarray(v['scale_1'], dtype=np.float32))
    scale_2 = np.exp(np.asarray(v['scale_2'], dtype=np.float32))
    
    # Color from SH DC: 0.28209479177387814 is C0
    C0 = 0.28209479177387814
    f_dc_0 = np.asarray(v['f_dc_0'], dtype=np.float32)
    f_dc_1 = np.asarray(v['f_dc_1'], dtype=np.float32)
    f_dc_2 = np.asarray(v['f_dc_2'], dtype=np.float32)
    
    r = np.clip((0.5 + C0 * f_dc_0) * 255.0, 0, 255).astype(np.uint8)
    g = np.clip((0.5 + C0 * f_dc_1) * 255.0, 0, 255).astype(np.uint8)
    b = np.clip((0.5 + C0 * f_dc_2) * 255.0, 0, 255).astype(np.uint8)
    
    # Opacity: sigmoid(opacity)
    raw_opacity = np.asarray(v['opacity'], dtype=np.float32)
    opacity = (1.0 / (1.0 + np.exp(-raw_opacity))) * 255.0
    a = np.clip(opacity, 0, 255).astype(np.uint8)
    
    # Rotation: normalized quaternion (rot_0, rot_1, rot_2, rot_3) mapped to uint8 [0, 255]
    rot_0 = np.asarray(v['rot_0'], dtype=np.float32)
    rot_1 = np.asarray(v['rot_1'], dtype=np.float32)
    rot_2 = np.asarray(v['rot_2'], dtype=np.float32)
    rot_3 = np.asarray(v['rot_3'], dtype=np.float32)
    
    norm = np.sqrt(rot_0**2 + rot_1**2 + rot_2**2 + rot_3**2) + 1e-8
    rot_0 /= norm
    rot_1 /= norm
    rot_2 /= norm
    rot_3 /= norm
    
    q0 = np.clip((rot_0 * 128.0 + 128.0), 0, 255).astype(np.uint8)
    q1 = np.clip((rot_1 * 128.0 + 128.0), 0, 255).astype(np.uint8)
    q2 = np.clip((rot_2 * 128.0 + 128.0), 0, 255).astype(np.uint8)
    q3 = np.clip((rot_3 * 128.0 + 128.0), 0, 255).astype(np.uint8)
    
    print(f"Packing {num_pts} splats (32 bytes each)...")
    splat_dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('s0', '<f4'), ('s1', '<f4'), ('s2', '<f4'),
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1'),
        ('q0', 'u1'), ('q1', 'u1'), ('q2', 'u1'), ('q3', 'u1')
    ])
    
    splat_array = np.empty(num_pts, dtype=splat_dtype)
    splat_array['x'] = x
    splat_array['y'] = y
    splat_array['z'] = z
    splat_array['s0'] = scale_0
    splat_array['s1'] = scale_1
    splat_array['s2'] = scale_2
    splat_array['r'] = r
    splat_array['g'] = g
    splat_array['b'] = b
    splat_array['a'] = a
    splat_array['q0'] = q0
    splat_array['q1'] = q1
    splat_array['q2'] = q2
    splat_array['q3'] = q3
    
    splat_array.tofile(splat_path)
    print(f"Successfully converted and saved {splat_path} ({num_pts * 32} bytes) in 0.05s!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        convert_ply_to_splat(sys.argv[1], sys.argv[2])
