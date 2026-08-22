import os
import glob
import time
import cv2
from PIL import Image
import pillow_heif

# Register HEIF opener with PIL
pillow_heif.register_heif_opener()

SRC_DIR = r"C:\Users\Rishi\Downloads\test\Room"
OUT_DIR = r"C:\Users\Rishi\Downloads\test\Room_dataset\input"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  PREPARING ROOM DATASET (HEIC PHOTOS + 4K VIDEO FRAMES)")
print("=" * 60)

# 1. Process HEIC Photos
heic_files = glob.glob(os.path.join(SRC_DIR, "*.heic"))
print(f"\n[1/2] Converting {len(heic_files)} HEIC photos to JPG...")

img_idx = 0
for i, heic_path in enumerate(heic_files):
    try:
        img = Image.open(heic_path)
        # Apply EXIF rotation if present
        try:
            from PIL import ImageOps
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        
        img = img.convert("RGB")
        out_filename = f"photo_{img_idx:04d}.jpg"
        out_path = os.path.join(OUT_DIR, out_filename)
        img.save(out_path, "JPEG", quality=95)
        img_idx += 1
        if (i + 1) % 10 == 0 or i == len(heic_files) - 1:
            print(f"  Converted {i + 1}/{len(heic_files)} photos...")
    except Exception as e:
        print(f"  Error converting {heic_path}: {e}")

print(f"-> Successfully saved {img_idx} photo JPGs to {OUT_DIR}")

# 2. Extract Frames from Video
video_path = os.path.join(SRC_DIR, "20260820_225338.mp4")
if os.path.exists(video_path):
    print(f"\n[2/2] Extracting crisp frames from video: {os.path.basename(video_path)}...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    print(f"  Video Duration: {duration:.1f}s, Total Frames: {total_frames}, FPS: {fps:.2f}")

    # Extract approx 150 evenly spaced frames (roughly every 0.8 seconds)
    target_count = 150
    step = max(1, total_frames // target_count)
    
    extracted = 0
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % step == 0:
            # Measure sharpness using Laplacian variance
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Save frame if not completely blurry
            if sharpness > 15.0:
                out_filename = f"video_frame_{extracted:04d}.jpg"
                out_path = os.path.join(OUT_DIR, out_filename)
                cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                extracted += 1
        
        frame_idx += 1
        if frame_idx % 1000 == 0:
            print(f"  Processed {frame_idx}/{total_frames} video frames (extracted {extracted})...")

    cap.release()
    print(f"-> Successfully extracted {extracted} sharp video frames.")

total_images = len(glob.glob(os.path.join(OUT_DIR, "*.jpg")))
print(f"\n" + "=" * 60)
print(f"  DATASET READY: {total_images} total high-resolution images in {OUT_DIR}")
print("=" * 60)
