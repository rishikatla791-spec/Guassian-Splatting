import cv2
from pathlib import Path

img_dir = Path(r"C:\Users\Rishi\Downloads\gaussian\imgaes_apple_video")
for p in img_dir.glob("*.jpg"):
    img = cv2.imread(str(p))
    if img is not None:
        h, w = img.shape[:2]
        if w > 960:
            new_w = 960
            new_h = int(h * (960 / w))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(p), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
print("[OK] Resized video frames to 960x540 for 4x fast rendering speed!")
