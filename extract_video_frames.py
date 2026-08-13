"""
Video Frame Extractor for 3D Reconstruction Pipeline.

Extracts multi-view frames from input MP4 video of rotating object,
performs blur filtering, and saves images for 3D Reconstruction.
"""
import sys
import os
from pathlib import Path
import cv2
import numpy as np


def extract_frames(video_path: str, output_dir: str, target_num_frames: int = 36) -> list[str]:
    """
    Extract target_num_frames from video file evenly distributed across total duration.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"=== [Video Extractor] Loaded video: {video_path.name} ===")
    print(f"  Total frames: {total_frames} | FPS: {fps:.2f} | Resolution: {width}x{height}")

    if total_frames <= 0:
        total_frames = 300  # Fallback

    step = max(1, total_frames // target_num_frames)
    saved_paths = []

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0 and saved_count < target_num_frames:
            # Measure blur (Laplacian variance) to ensure sharp frame extraction
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            out_filename = output_dir / f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(out_filename), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved_paths.append(str(out_filename))
            saved_count += 1
            print(f"  [Frame {saved_count:02d}/{target_num_frames}] Saved {out_filename.name} (Blur Score: {blur_score:.1f})")

        frame_idx += 1

    cap.release()
    print(f"[OK] Successfully extracted {len(saved_paths)} frames into: {output_dir}\n")
    return saved_paths


if __name__ == "__main__":
    video_file = r"C:\Users\Rishi\Downloads\gaussian\imgaes\Red Apple Rotating On Wooden Table Stock Footage Video (100% Royalty-free) 27706513 _ Shutterstock - Brave 2026-08-12 14-51-31.mp4"
    out_dir = r"C:\Users\Rishi\Downloads\gaussian\imgaes_apple_video"
    extract_frames(video_file, out_dir, target_num_frames=36)
