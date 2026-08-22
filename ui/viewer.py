"""
Interactive 3DGS Viewer using matplotlib.

Provides:
  - GaussianViewer: interactive orbit viewer with keyboard controls
  - render_360_video(): 360° orbit video export
"""
from __future__ import annotations
import math
import time
from typing import List, Optional

import numpy as np
import torch


class GaussianViewer:
    """
    Interactive matplotlib-based viewer for 3D Gaussian Splatting scenes.

    Controls:
      W/S  — move camera forward/backward
      A/D  — strafe left/right
      Q/E  — move camera up/down
      ↑/↓  — pitch (look up/down)
      ←/→  — yaw (look left/right)
      +/-  — zoom (change FoV)
      R    — reset camera
      P    — save screenshot (viewer_{n}.png)
      ESC  — quit
    """

    def __init__(self, gaussians, cameras: List = []):
        self.gaussians = gaussians
        self.cameras = cameras

        # Camera state
        self._pos = np.array([0.0, 0.0, -3.0])
        self._yaw = 0.0    # radians, around Y
        self._pitch = 0.0  # radians, around X
        self._fovx = math.pi / 3.0  # 60°
        self._width = 800
        self._height = 600
        self._move_speed = 0.1
        self._rot_speed = 0.05
        self._screenshot_count = 0
        self._running = False

    # -----------------------------------------------------------------------
    # Camera matrix construction
    # -----------------------------------------------------------------------

    def _build_camera(self):
        from gaussian.core.camera import Camera, CameraIntrinsics, CameraExtrinsics

        # Build rotation from yaw + pitch (Euler ZYX)
        cy, sy = math.cos(self._yaw),   math.sin(self._yaw)
        cp, sp = math.cos(self._pitch), math.sin(self._pitch)

        # Combined rotation: Ryaw @ Rpitch
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        R = Ry @ Rx  # world-to-camera rotation

        T = R @ self._pos  # world-to-camera translation

        fovy = 2.0 * math.atan(math.tan(self._fovx / 2.0) * self._height / self._width)
        fx = self._width / (2.0 * math.tan(self._fovx / 2.0))
        fy = self._height / (2.0 * math.tan(fovy / 2.0))

        K = CameraIntrinsics(
            fx=fx, fy=fy,
            cx=self._width / 2.0, cy=self._height / 2.0,
            width=self._width, height=self._height
        )
        E = CameraExtrinsics(R=R, T=T)
        return Camera(uid=9999, intrinsics=K, extrinsics=E)

    # -----------------------------------------------------------------------
    # Render one frame
    # -----------------------------------------------------------------------

    def render_frame(self) -> np.ndarray:
        """Render current viewpoint. Returns (H, W, 3) uint8 numpy array."""
        from gaussian.renderer import TileBasedRasterizer

        camera = self._build_camera()
        renderer = TileBasedRasterizer()
        bg = torch.zeros(3)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = renderer.render(self.gaussians, camera, bg_color=bg)
        dt = time.perf_counter() - t0
        fps = 1.0 / max(dt, 1e-9)

        img = out['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        img_u8 = (img * 255).astype(np.uint8)
        self._last_fps = fps
        return img_u8

    # -----------------------------------------------------------------------
    # Show single frame
    # -----------------------------------------------------------------------

    def show(self, camera=None):
        """Render and display a single frame."""
        import matplotlib.pyplot as plt

        if camera is not None:
            from gaussian.renderer import TileBasedRasterizer
            renderer = TileBasedRasterizer()
            bg = torch.zeros(3)
            with torch.no_grad():
                out = renderer.render(self.gaussians, camera, bg_color=bg)
            img = out['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        else:
            img = self.render_frame() / 255.0

        plt.figure(figsize=(10, 7))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"3DGS Viewer | N={self.gaussians.num_gaussians:,} Gaussians")
        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------------------------
    # Interactive loop
    # -----------------------------------------------------------------------

    def interactive_loop(self):
        """
        Launch interactive matplotlib viewer with keyboard controls.
        Note: requires matplotlib with interactive backend (e.g., TkAgg).
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(10, 7))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.axis('off')

        img_u8 = self.render_frame()
        im = ax.imshow(img_u8)

        N = self.gaussians.num_gaussians
        title_text = ax.set_title("", color='white', fontsize=9, pad=2,
                                   backgroundcolor='black')

        self._running = True

        def update_display():
            img_u8 = self.render_frame()
            im.set_data(img_u8)
            title_text.set_text(
                f"3DGS | N={N:,} | {self._last_fps:.1f} FPS | "
                f"pos=({self._pos[0]:.1f},{self._pos[1]:.1f},{self._pos[2]:.1f}) | "
                f"yaw={math.degrees(self._yaw):.0f}° | pitch={math.degrees(self._pitch):.0f}°"
            )
            fig.canvas.draw_idle()

        def on_key(event):
            if not self._running:
                return

            # Compute forward/right directions for movement
            cy, sy = math.cos(self._yaw), math.sin(self._yaw)
            forward = np.array([sy, 0, cy])
            right   = np.array([cy, 0, -sy])
            up      = np.array([0, 1, 0])

            key = event.key
            if key == 'w':   self._pos += forward * self._move_speed
            elif key == 's': self._pos -= forward * self._move_speed
            elif key == 'a': self._pos -= right * self._move_speed
            elif key == 'd': self._pos += right * self._move_speed
            elif key == 'q': self._pos -= up * self._move_speed
            elif key == 'e': self._pos += up * self._move_speed
            elif key == 'up':    self._pitch = max(-math.pi/2, self._pitch - self._rot_speed)
            elif key == 'down':  self._pitch = min(math.pi/2,  self._pitch + self._rot_speed)
            elif key == 'left':  self._yaw -= self._rot_speed
            elif key == 'right': self._yaw += self._rot_speed
            elif key == '+' or key == '=':
                self._fovx = max(0.2, self._fovx - 0.05)
            elif key == '-':
                self._fovx = min(2.5, self._fovx + 0.05)
            elif key == 'r':
                self._pos   = np.array([0.0, 0.0, -3.0])
                self._yaw   = 0.0
                self._pitch = 0.0
            elif key == 'p':
                fname = f"viewer_{self._screenshot_count:04d}.png"
                import PIL.Image
                PIL.Image.fromarray(self.render_frame()).save(fname)
                print(f"Screenshot saved: {fname}")
                self._screenshot_count += 1
                return  # Don't re-render
            elif key == 'escape' or key == 'q':
                self._running = False
                plt.close(fig)
                return

            update_display()

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.title("3DGS Viewer — WASD move, Arrows rotate, +/- zoom, P screenshot, ESC quit")
        update_display()
        plt.show()


# ---------------------------------------------------------------------------
# 360° video export
# ---------------------------------------------------------------------------

def render_360_video(
    gaussians,
    output_path: str = "orbit.mp4",
    n_frames: int = 120,
    radius: float = 3.0,
    height: float = 0.5,
    resolution: tuple = (800, 600),
    fps: int = 30,
):
    """
    Render a 360° orbital video around the scene center.

    Args:
        gaussians:   GaussianModel
        output_path: output MP4 path (requires imageio[ffmpeg])
        n_frames:    number of frames (= video length * fps)
        radius:      orbit radius in world units
        height:      camera height above scene center
        resolution:  (width, height) in pixels
        fps:         output video frame rate
    """
    try:
        import imageio
    except ImportError:
        print("imageio not installed. Run: pip install imageio imageio-ffmpeg")
        return

    from gaussian.core.camera import Camera, CameraIntrinsics, CameraExtrinsics
    from gaussian.renderer import TileBasedRasterizer

    W, H = resolution
    renderer = TileBasedRasterizer()
    bg = torch.zeros(3)
    frames = []

    print(f"Rendering {n_frames} frames for 360° orbit video...")

    for i in range(n_frames):
        angle = 2.0 * math.pi * i / n_frames

        # Camera position on orbit circle
        eye = np.array([radius * math.sin(angle), height, radius * math.cos(angle)])
        target = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])

        # Build look-at extrinsics
        E = CameraExtrinsics.from_look_at(eye, target, up)

        fovx = math.pi / 3.0
        fovy = 2.0 * math.atan(math.tan(fovx / 2.0) * H / W)
        fx = W / (2.0 * math.tan(fovx / 2.0))
        fy = H / (2.0 * math.tan(fovy / 2.0))
        K = CameraIntrinsics(fx=fx, fy=fy, cx=W/2, cy=H/2, width=W, height=H)
        camera = Camera(uid=i, intrinsics=K, extrinsics=E)

        with torch.no_grad():
            out = renderer.render(gaussians, camera, bg_color=bg)

        img = out['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        frames.append((img * 255).astype(np.uint8))

        if (i + 1) % 10 == 0:
            print(f"  Frame {i+1}/{n_frames}")

    print(f"Writing video to {output_path}...")
    imageio.mimwrite(output_path, frames, fps=fps, quality=8)
    print("Done!")
