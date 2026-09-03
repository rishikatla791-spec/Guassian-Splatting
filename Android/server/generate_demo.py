import struct
import math
import numpy as np

def generate_demo_splat(output_path):
    num_pts = 12000
    points = []
    
    # Generate a vibrant 3D double helix and sphere structure
    for i in range(num_pts):
        t = i / num_pts * 16.0 * math.pi
        r = 1.0 + 0.3 * math.sin(t * 3.0)
        
        # Double helix
        sign = 1 if i % 2 == 0 else -1
        x = r * math.cos(t) * sign + np.random.normal(0, 0.05)
        y = (i / num_pts - 0.5) * 3.0 + np.random.normal(0, 0.05)
        z = r * math.sin(t) * sign + np.random.normal(0, 0.05)
        
        # Scale
        s = 0.04 + np.random.uniform(0, 0.02)
        sx, sy, sz = s, s, s
        
        # Color gradient: Cyan -> Purple -> Orange
        cr = int((math.sin(t * 0.5) * 0.5 + 0.5) * 255)
        cg = int((math.cos(t * 0.3) * 0.5 + 0.5) * 255)
        cb = int((math.sin(t * 0.7 + 1.0) * 0.5 + 0.5) * 255)
        ca = 220
        
        # Rotation
        q0, q1, q2, q3 = 128, 128, 128, 255
        
        points.append((x, y, z, sx, sy, sz, cr, cg, cb, ca, q0, q1, q2, q3))
        
    with open(output_path, 'wb') as f:
        for p in points:
            f.write(struct.pack('<ffffffBBBBBBBB', *p))
            
    print(f"Generated demo model: {output_path} ({num_pts} Gaussians, {num_pts*32} bytes)")

if __name__ == '__main__':
    generate_demo_splat(r'c:\Users\Rishi\Downloads\test\Android\app\src\main\assets\viewer\demo.splat')
