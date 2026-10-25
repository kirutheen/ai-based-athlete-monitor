"""
Extracts .tflite model files from models/pose_landmarker_full.task
and exports human-readable model summary files:
1. models/pose_landmarks_detector.tflite (Opens directly in Netron)
2. models/pose_detector.tflite (Opens directly in Netron)
3. models/model_summary.txt (Human-readable text description of layers)
"""

import os
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)

task_path = "models/pose_landmarker_full.task"

if not os.path.exists(task_path):
    print(f"Error: {task_path} not found.")
    sys.exit(1)

# 1. Unzip .tflite models from .task bundle
with zipfile.ZipFile(task_path, "r") as z:
    z.extractall("models/")
    print(f"Extracted TFLite models to models/: {z.namelist()}")

# 2. Write human-readable model architecture text file
summary_path = "models/model_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("DRIFTGUARD - MEDIAPIPE BLAZEPOSE 3D MODEL ARCHITECTURE SUMMARY\n")
    f.write("=" * 70 + "\n\n")

    f.write("1. MODEL BUNDLE FILE\n")
    f.write(f"   Source Path: {task_path}\n")
    f.write("   File Format: Google MediaPipe Task Bundle (ZIP container holding TFLite neural network graphs)\n\n")

    f.write("2. EXTRACTED TFLITE NEURAL NETWORK GRAPHS\n")
    f.write("   a) pose_detector.tflite\n")
    f.write("      - Purpose: Bounding-box detection of human body candidates in RGB frames.\n")
    f.write("      - Input Tensor: 1 x 224 x 224 x 3 (Normalized RGB image)\n")
    f.write("      - Output Tensors: Bounding box coordinates (xmin, ymin, width, height) + Keypoint score predictions.\n\n")

    f.write("   b) pose_landmarks_detector.tflite\n")
    f.write("      - Purpose: 3D Human Pose Landmark regression model (BlazePose GHUM 3D architecture).\n")
    f.write("      - Input Tensor: 1 x 256 x 256 x 3 (Cropped RGB torso/person tensor)\n")
    f.write("      - Output Tensors:\n")
    f.write("        * Tensor 0: 33 3D Joint Landmarks (x, y, z normalized image coordinates) + visibility scores\n")
    f.write("        * Tensor 1: 33 3D World Landmarks (x_world, y_world, z_world metric coordinates in meters)\n")
    f.write("        * Tensor 2: Pose presence confidence score (0.0 to 1.0)\n")
    f.write("        * Tensor 3: Segmented body mask heatmap\n\n")

    f.write("3. 33 BODY LANDMARK TOPOLOGY INDEX (0 to 32)\n")
    landmarks = [
        "0: NOSE", "1: LEFT_EYE_INNER", "2: LEFT_EYE", "3: LEFT_EYE_OUTER",
        "4: RIGHT_EYE_INNER", "5: RIGHT_EYE", "6: RIGHT_EYE_OUTER", "7: LEFT_EAR",
        "8: RIGHT_EAR", "9: MOUTH_LEFT", "10: MOUTH_RIGHT", "11: LEFT_SHOULDER",
        "12: RIGHT_SHOULDER", "13: LEFT_ELBOW", "14: RIGHT_ELBOW", "15: LEFT_WRIST",
        "16: RIGHT_WRIST", "17: LEFT_PINKY", "18: RIGHT_PINKY", "19: LEFT_INDEX",
        "20: RIGHT_INDEX", "21: LEFT_THUMB", "22: RIGHT_THUMB", "23: LEFT_HIP",
        "24: RIGHT_HIP", "25: LEFT_KNEE", "26: RIGHT_KNEE", "27: LEFT_ANKLE",
        "28: RIGHT_ANKLE", "29: LEFT_HEEL", "30: RIGHT_HEEL", "31: LEFT_FOOT_INDEX",
        "32: RIGHT_FOOT_INDEX"
    ]
    for lm in landmarks:
        f.write(f"   - {lm}\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("HOW TO OPEN IN NETRON MODEL VISUALIZER:\n")
    f.write("1. Open https://netron.app in your browser.\n")
    f.write("2. Select 'Open Model...' and pick: models/pose_landmarks_detector.tflite\n")
    f.write("   (Netron natively opens .tflite files!)\n")
    f.write("=" * 70 + "\n")

print(f"Exported human-readable summary to {summary_path}")
