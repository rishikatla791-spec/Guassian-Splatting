#!/usr/bin/env python3
"""
render_splats_hd.py — High-Definition 3D Gaussian Splatting Offscreen Renderer
Generates high-precision 3DGS renders from multiple camera angles.
"""

import math
import os
import numpy as np
from PIL import Image, ImageDraw

def generate_apple_gaussians(density=3200):
    gaussians = []
    N_apple = int(density * 0.65)
    N_stem = int(density * 0.08)
    N_table = int(density * 0.27)
    
    np.random.seed(42)
    
    # 1. APPLE BODY GAUSSIANS
    for i in range(N_apple):
        theta = math.acos(1 - 2 * (i + 0.5) / N_apple)
        phi = math.pi * (1 + 5**0.5) * i
        
        rx = 0.72 + (np.random.rand() - 0.5) * 0.03
        ry = 0.78 + (np.random.rand() - 0.5) * 0.03
        rz = 0.72 + (np.random.rand() - 0.5) * 0.03
        
        x = rx * math.sin(theta) * math.cos(phi)
        y = ry * math.cos(theta)
        z = rz * math.sin(theta) * math.sin(phi)
        
        if y > 0.4:
            factor = 1.0 - 0.4 * math.exp(-((y - 0.65)/0.25)**2)
            y *= factor
            x *= 0.95
            z *= 0.95
        if y < -0.4:
            y *= 0.92
            
        nx = x / (rx * rx)
        ny = y / (ry * ry)
        nz = z / (rz * rz)
        nlen = math.sqrt(nx*nx + ny*ny + nz*nz)
        normal = [nx/nlen, ny/nlen, nz/nlen]
        
        specDot = normal[0]*0.3 + normal[1]*0.5 + normal[2]*0.8
        if specDot > 0.82 and y > 0.1 and z > 0.1:
            color = (255, 245, 210)
        elif x < -0.3 and -0.2 < y < 0.3:
            t = np.random.rand()
            color = (int(220 + t*25), int(170 + t*30), int(40 + t*20))
        elif y > 0.5:
            color = (120, 20, 35)
        elif y < -0.4:
            color = (140, 15, 30)
        else:
            varR = (np.random.rand() - 0.5) * 40
            r = int(min(255, max(160, 225 + varR)))
            g = int(min(60, max(15, 30 + (np.random.rand() - 0.5)*20)))
            b = int(min(60, max(15, 35 + (np.random.rand() - 0.5)*20)))
            color = (r, g, b)
            
        scales = (0.022, 0.022, 0.008)
        gaussians.append({'pos': (x,y,z), 'normal': normal, 'color': color, 'scales': scales, 'type': 'apple'})
        
    # 2. STEM & LEAF
    for i in range(N_stem):
        t = i / N_stem
        x = 0.02 * math.sin(t * math.pi) + (np.random.rand() - 0.5) * 0.012
        y = 0.60 + t * 0.35 + (np.random.rand() - 0.5) * 0.01
        z = -0.03 * t + (np.random.rand() - 0.5) * 0.012
        
        if t < 0.25: color = (100, 50, 25)
        elif t > 0.8: color = (22, 101, 52)
        else: color = (63, 98, 18)
        
        gaussians.append({'pos': (x,y,z), 'normal': (0,1,0), 'color': color, 'scales': (0.010, 0.022, 0.010), 'type': 'stem'})
        
    # 3. WOODEN TABLE PLANE & SHADOW
    for i in range(N_table):
        angle = np.random.rand() * math.pi * 2
        dist = math.sqrt(np.random.rand()) * 2.2
        x = dist * math.cos(angle)
        z = dist * math.sin(angle)
        y = -0.76 + (np.random.rand() - 0.5) * 0.01
        
        rSq = x*x + z*z
        if rSq < 0.65:
            sf = 1.0 - (rSq / 0.65)
            color = (int(30 * (1 - sf)), int(20 * (1 - sf)), int(25 * (1 - sf)))
        else:
            stripe = math.sin(x * 8.0 + z * 2.0)
            if stripe > 0: color = (180, 100, 40)
            else: color = (130, 65, 20)
            
        gaussians.append({'pos': (x,y,z), 'normal': (0,1,0), 'color': color, 'scales': (0.045, 0.005, 0.045), 'type': 'table'})
        
    return gaussians

def render_scene(gaussians, angle_deg, width=960, height=640, render_mode='fuzzy', label=""):
    img = Image.new("RGBA", (width, height), (9, 12, 21, 255))
    
    theta = math.radians(angle_deg)
    phi = 0.26
    cam_dist = 3.1
    focal = width * 0.95
    cx, cy = width / 2.0, height / 2.0 + 30.0
    
    cosT, sinT = math.cos(theta), math.sin(theta)
    cosP, sinP = math.cos(phi), math.sin(phi)
    
    projected = []
    for g in gaussians:
        x, y, z = g['pos']
        xc = cosT * x + sinT * z
        yc = -sinP * sinT * x + cosP * y + sinP * cosT * z
        zc = -cosP * sinT * x - sinP * y + cosP * cosT * z + cam_dist
        
        if zc <= 0.2: continue
        
        u = cx + (focal * xc) / zc
        v = cy - (focal * yc) / zc
        
        if not (-100 <= u <= width + 100 and -100 <= v <= height + 100): continue
        
        depth_scale = focal / zc
        rx = max(1.5, g['scales'][0] * depth_scale * 14.0)
        ry = max(1.5, g['scales'][1] * depth_scale * 14.0)
        
        nx, ny, nz = g['normal']
        nxc = cosT * nx + sinT * nz
        nyc = -sinP * sinT * nx + cosP * ny + sinP * cosT * nz
        rot_angle = math.atan2(nyc, nxc + 1e-5)
        
        projected.append({
            'u': u, 'v': v, 'zc': zc, 'rx': rx, 'ry': ry, 'rot': rot_angle,
            'color': g['color'], 'type': g['type']
        })
        
    projected.sort(key=lambda item: item['zc'], reverse=True)
    
    if render_mode == 'fuzzy':
        for p in projected:
            u, v = p['u'], p['v']
            rx, ry = p['rx'], p['ry']
            r, g, b = p['color']
            
            size = int(max(rx, ry) * 2.0) + 4
            size = max(4, min(80, size))
            sprite = Image.new("RGBA", (size, size), (0,0,0,0))
            draw_sp = ImageDraw.Draw(sprite)
            
            scx, scy = size / 2.0, size / 2.0
            max_r = max(1.0, size / 2.0)
            
            for ring in range(int(max_r), 0, -1):
                fr = ring / max_r
                alpha = int(190 * math.exp(-4.2 * fr * fr))
                draw_sp.ellipse([scx - ring, scy - ring, scx + ring, scy + ring], fill=(r, g, b, alpha))
                
            img.paste(sprite, (int(u - scx), int(v - scy)), sprite)
            
    elif render_mode == 'ellipse':
        draw = ImageDraw.Draw(img)
        for p in projected:
            u, v, rx, ry = p['u'], p['v'], p['rx'], p['ry']
            r, g, b = p['color']
            draw.ellipse([u - rx, v - ry, u + rx, v + ry], outline=(r, g, b, 230), width=1)

    # Draw Overlay Badge Label
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 360, 60], fill=(15, 23, 42, 220), outline=(56, 189, 248, 180), width=1)
    draw.text((32, 32), f"3DGS Render | {label}", fill=(56, 189, 248, 255))
    
    return img

def main():
    out_dir = os.path.join(os.getcwd(), "imgaes")
    os.makedirs(out_dir, exist_ok=True)
    
    gaussians = generate_apple_gaussians(density=3500)
    
    targets = [
        (0, "fuzzy", "apple_splats_0deg.png", "0° (Front View) — Fuzzy 3DGS"),
        (45, "fuzzy", "apple_splats_45deg.png", "45° (Front-Right) — Fuzzy 3DGS"),
        (90, "fuzzy", "apple_splats_90deg.png", "90° (Right View) — Fuzzy 3DGS"),
        (45, "ellipse", "apple_splats_covariance.png", "45° Covariance Ellipses Wireframe"),
    ]
    
    for angle, mode, fname, label in targets:
        path = os.path.join(out_dir, fname)
        img = render_scene(gaussians, angle, render_mode=mode, label=label)
        img.save(path)
        print(f"Saved: {path}")

if __name__ == "__main__":
    main()
