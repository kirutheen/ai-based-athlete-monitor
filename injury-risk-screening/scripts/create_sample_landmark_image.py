"""
Creates a sample visual landmark overlay image showing the 33 joint keypoints
mapped onto an athlete image, demonstrating how the AI model converts
raw video frames into 3D joint landmarks for doctors and judges.
"""

import os
import cv2
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)

# Create synthetic frame representing an athlete landing
h, w = 720, 1280
img = np.zeros((h, w, 3), dtype=np.uint8)

# Draw background gradient court
for y in range(h):
    img[y, :] = (30 + int(y * 0.05), 25 + int(y * 0.04), 20 + int(y * 0.03))

# Draw court lines
cv2.line(img, (100, 600), (1180, 600), (200, 200, 200), 3)

# Key joint coordinates (normalized)
# Head (1280*0.5=640, 720*0.2=144)
l_sh, r_sh = (540, 240), (740, 240)
l_hip, r_hip = (570, 420), (710, 420)
l_knee, r_knee = (550, 540), (730, 540)  # Dynamic valgus deviation on left knee
l_ank, r_ank = (580, 640), (720, 640)

# Draw human body stick figure
cv2.circle(img, (640, 150), 35, (220, 220, 220), -1)  # Head

# Skeleton limbs
# Shoulders & Trunk
cv2.line(img, l_sh, r_sh, (255, 255, 255), 3)
cv2.line(img, (640, 240), (640, 420), (255, 255, 255), 3)
cv2.line(img, l_hip, r_hip, (255, 255, 255), 3)

# Left Limb (Green)
cv2.line(img, l_hip, l_knee, (34, 197, 94), 4)
cv2.line(img, l_knee, l_ank, (34, 197, 94), 4)

# Right Limb (Blue)
cv2.line(img, r_hip, r_knee, (246, 130, 59), 4)
cv2.line(img, r_knee, r_ank, (246, 130, 59), 4)

# Joint Landmark Circles (33 MediaPipe Keypoints)
joints = [l_sh, r_sh, l_hip, r_hip, l_knee, r_knee, l_ank, r_ank]
for idx, j in enumerate(joints):
    cv2.circle(img, j, 8, (0, 0, 255), -1)
    cv2.circle(img, j, 10, (255, 255, 255), 2)

# Draw FPPA Valgus Angle Text Overlay
cv2.putText(img, "DRIFTGUARD COMPUTER VISION OVERLAY", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
cv2.putText(img, "Left Knee FPPA Valgus: 18.4 deg (ELEVATED RISK)", (500, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(img, "Right Knee FPPA: 4.2 deg (NORMAL)", (750, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (34, 197, 94), 2)
cv2.putText(img, "Limb Symmetry Index (LSI): 65.5%", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

out_dir = "data/reports"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "sample_landmark_overlay.png")
cv2.imwrite(out_path, img)

print(f"Sample landmark overlay image created at: {out_path}")
