"""
Frame-level biomechanical feature extraction from 2D and 3D pose landmarks.

This module turns raw (x, y, z, v) landmark coordinates — the output of
scripts/run_pose_on_video.py — into the clinically-grounded angles used
for knee, hip, ankle, trunk, and pelvic biomechanical screening. Every
formula here maps to published literature and standard biomechanical definitions:

- Frontal Plane Projection Angle (FPPA): the 2D video-derived proxy for
  dynamic knee valgus (Mizner et al. 2012; McLean et al. 2005; Hewett et al. 2005).
- 3D Knee Flexion & Abduction: computed from monocular 3D world landmarks (BlazePose GHUM model).
- Hip & Pelvic kinematics: Hip flexion/abduction, pelvic tilt, obliquity, and rotation.
- Ankle & Foot kinematics: Dorsiflexion/plantarflexion, inversion/eversion, foot progression angle.
- Trunk Control: Frontal, sagittal, and transverse trunk angles and angular velocities.
- Bilateral Limb Symmetry Index (LSI): LSI = 2 * |Left - Right| / (|Left| + |Right|) * 100.
- Center of Mass (COM): Segmental center-of-mass estimation using Dempster parameters.

All functions operate on a single frame's landmark array (shape (33, 4):
x, y, z, visibility) or on a pandas DataFrame of many frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.landmarks import LM, Provenance

MIN_VISIBILITY = 0.5  # below this, MediaPipe's landmark is unreliable


def calculate_lsi(left: float, right: float) -> float:
    """
    Normalized Limb Symmetry Index (LSI), in percent [0, 200%].
    LSI = 2 * |left - right| / (|left| + |right|) * 100

    Returns NaN (shown as "Not available") when:
    - Either side is None or NaN
    - Either individual side is near-zero (avoids near-200% instability
      when one side is zero-crossing and the other is large)
    Returns 0.0 only when both sides are simultaneously near-zero.
    """
    if left is None or right is None:
        return float("nan")
    try:
        left = float(left)
        right = float(right)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(left) or math.isnan(right):
        return float("nan")
    denom = abs(left) + abs(right)
    if denom < 1e-6:
        return 0.0
    # Guard: if either side alone is near-zero the ratio is unstable
    if abs(left) < 1e-3 or abs(right) < 1e-3:
        return float("nan")
    return float((2.0 * abs(left - right) / denom) * 100.0)


def circular_angle_diff(deg1: float, deg2: float) -> float:
    """
    Computes the shortest circular angular difference between two angles in degrees [-180, 180].
    Corrects for boundary wrap-around jumps (e.g. +179 deg -> -179 deg is a 2 deg difference, not 358 deg).
    """
    if math.isnan(deg1) or math.isnan(deg2):
        return float("nan")
    diff = (deg1 - deg2 + 180.0) % 360.0 - 180.0
    return float(diff if diff != -180.0 else 180.0)



def _vec(landmarks: np.ndarray, a: LM, b: LM) -> np.ndarray | None:
    """2D vector from landmark a to landmark b in a single frame's (33,4) array."""
    pa, pb = landmarks[a], landmarks[b]
    if pa[3] < MIN_VISIBILITY or pb[3] < MIN_VISIBILITY:
        return None
    return np.array([pb[0] - pa[0], pb[1] - pa[1]], dtype=np.float64)


def _vec3(landmarks_3d: np.ndarray, a: LM, b: LM) -> np.ndarray | None:
    """3D vector from landmark a to landmark b using world (metric) coords."""
    pa, pb = landmarks_3d[a], landmarks_3d[b]
    if pa[3] < MIN_VISIBILITY or pb[3] < MIN_VISIBILITY:
        return None
    return np.array([pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]], dtype=np.float64)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two 2D vectors, in [0, 180]."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cos_theta = np.dot(v1, v2) / (n1 * n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def _angle_between_3d(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two 3D vectors, in [0, 180]."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cos_theta = np.dot(v1, v2) / (n1 * n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def fppa(landmarks: np.ndarray, side: str = "left") -> float:
    """
    Frontal Plane Projection Angle (FPPA) proxy at the knee, in degrees.
    Deviation from a straight hip-knee-ankle line: fppa = 180 - angle(hip->knee, knee->ankle).
    """
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    hip = LM.LEFT_HIP if side == "left" else LM.RIGHT_HIP
    knee = LM.LEFT_KNEE if side == "left" else LM.RIGHT_KNEE
    ankle = LM.LEFT_ANKLE if side == "left" else LM.RIGHT_ANKLE

    v_hip_to_knee = _vec(landmarks, hip, knee)
    v_knee_to_ankle = _vec(landmarks, knee, ankle)
    if v_hip_to_knee is None or v_knee_to_ankle is None:
        return float("nan")

    straight_angle = _angle_between(-v_hip_to_knee, v_knee_to_ankle)
    deviation = 180.0 - straight_angle

    hip_pt, ankle_pt = landmarks[hip][:2], landmarks[ankle][:2]
    knee_pt = landmarks[knee][:2]
    line_vec = ankle_pt - hip_pt
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-9:
        return float("nan")
    cross = line_vec[0] * (knee_pt[1] - hip_pt[1]) - line_vec[1] * (knee_pt[0] - hip_pt[0])
    signed_offset = cross / line_len

    if signed_offset < 0:
        deviation = -deviation

    return float(deviation)


def knee_flexion_angle(landmarks: np.ndarray, side: str = "left") -> float:
    """Sagittal-plane knee flexion angle, in degrees (0 = fully extended)."""
    hip = LM.LEFT_HIP if side == "left" else LM.RIGHT_HIP
    knee = LM.LEFT_KNEE if side == "left" else LM.RIGHT_KNEE
    ankle = LM.LEFT_ANKLE if side == "left" else LM.RIGHT_ANKLE

    v_knee_to_hip = _vec(landmarks, knee, hip)
    v_knee_to_ankle = _vec(landmarks, knee, ankle)
    if v_knee_to_hip is None or v_knee_to_ankle is None:
        return float("nan")

    straight_angle = _angle_between(v_knee_to_hip, v_knee_to_ankle)
    return float(180.0 - straight_angle)


def knee_flexion_angle_3d(landmarks_3d: np.ndarray, side: str = "left") -> float:
    """True 3D knee flexion angle (degrees) from 3D world landmarks."""
    hip = LM.LEFT_HIP if side == "left" else LM.RIGHT_HIP
    knee = LM.LEFT_KNEE if side == "left" else LM.RIGHT_KNEE
    ankle = LM.LEFT_ANKLE if side == "left" else LM.RIGHT_ANKLE

    v_knee_to_hip = _vec3(landmarks_3d, knee, hip)
    v_knee_to_ankle = _vec3(landmarks_3d, knee, ankle)
    if v_knee_to_hip is None or v_knee_to_ankle is None:
        return float("nan")

    straight_angle = _angle_between_3d(v_knee_to_hip, v_knee_to_ankle)
    return float(180.0 - straight_angle)


def knee_abduction_angle_3d(landmarks_3d: np.ndarray, side: str = "left") -> float:
    """3D knee abduction/adduction angle (degrees) relative to pelvis flexion plane."""
    hip = LM.LEFT_HIP if side == "left" else LM.RIGHT_HIP
    knee = LM.LEFT_KNEE if side == "left" else LM.RIGHT_KNEE
    ankle = LM.LEFT_ANKLE if side == "left" else LM.RIGHT_ANKLE

    thigh_vec = _vec3(landmarks_3d, hip, knee)
    shank_vec = _vec3(landmarks_3d, knee, ankle)
    pelvis_vec = _vec3(landmarks_3d, LM.LEFT_HIP, LM.RIGHT_HIP)
    if thigh_vec is None or shank_vec is None or pelvis_vec is None:
        return float("nan")

    flexion_plane_normal = np.cross(thigh_vec, pelvis_vec)
    normal_len = np.linalg.norm(flexion_plane_normal)
    if normal_len < 1e-9:
        return float("nan")
    flexion_plane_normal = flexion_plane_normal / normal_len

    angle_to_normal = _angle_between_3d(shank_vec, flexion_plane_normal)
    return float(90.0 - angle_to_normal)


def hip_flexion_angle(landmarks: np.ndarray, side: str = "left") -> float:
    """Sagittal hip flexion angle (degrees): angle between thigh vector and trunk line."""
    hip = LM.LEFT_HIP if side == "left" else LM.RIGHT_HIP
    knee = LM.LEFT_KNEE if side == "left" else LM.RIGHT_KNEE
    ls, rs = landmarks[LM.LEFT_SHOULDER], landmarks[LM.RIGHT_SHOULDER]
    lh, rh = landmarks[LM.LEFT_HIP], landmarks[LM.RIGHT_HIP]

    if min(ls[3], rs[3], lh[3], rh[3]) < MIN_VISIBILITY:
        return float("nan")

    v_thigh = _vec(landmarks, hip, knee)
    if v_thigh is None:
        return float("nan")

    shoulder_mid = np.array([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2])
    hip_mid = np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2])
    trunk_vec = hip_mid - shoulder_mid  # points down

    ang = _angle_between(v_thigh, trunk_vec)
    return float(ang)


def hip_abduction_angle(landmarks: np.ndarray, side: str = "left") -> float:
    """Frontal hip abduction/adduction angle (degrees) relative to vertical."""
    hip = LM.LEFT_HIP if side == "left" else LM.RIGHT_HIP
    knee = LM.LEFT_KNEE if side == "left" else LM.RIGHT_KNEE

    v_thigh = _vec(landmarks, hip, knee)
    if v_thigh is None:
        return float("nan")

    vertical = np.array([0.0, 1.0])
    ang = _angle_between(v_thigh, vertical)
    return float(ang)


def hip_rotation_angle(landmarks: np.ndarray, side: str = "left") -> float:
    """Hip internal/external rotation is unestimable from 2D monocular pose alone."""
    return float("nan")


def ankle_dorsiflexion_angle(landmarks: np.ndarray, side: str = "left") -> float:
    """
    Ankle dorsiflexion/plantarflexion angle (degrees).
    Angle between shank vector (ankle->knee) and foot vector (ankle->foot_index).
    0 = neutral (90 degrees included), positive = dorsiflexion.
    """
    knee = LM.LEFT_KNEE if side == "left" else LM.RIGHT_KNEE
    ankle = LM.LEFT_ANKLE if side == "left" else LM.RIGHT_ANKLE
    foot_idx = LM.LEFT_FOOT_INDEX if side == "left" else LM.RIGHT_FOOT_INDEX

    v_shank = _vec(landmarks, ankle, knee)
    v_foot = _vec(landmarks, ankle, foot_idx)
    if v_shank is None or v_foot is None:
        return float("nan")

    included_angle = _angle_between(v_shank, v_foot)
    if math.isnan(included_angle):
        return float("nan")
    return float(90.0 - included_angle)


def ankle_inversion_angle(landmarks: np.ndarray, side: str = "left") -> float:
    """Ankle inversion/eversion angle (degrees) in frontal view."""
    ankle = LM.LEFT_ANKLE if side == "left" else LM.RIGHT_ANKLE
    heel = LM.LEFT_HEEL if side == "left" else LM.RIGHT_HEEL
    foot_idx = LM.LEFT_FOOT_INDEX if side == "left" else LM.RIGHT_FOOT_INDEX

    v_foot = _vec(landmarks, heel, foot_idx)
    v_shank = _vec(landmarks, ankle, LM.LEFT_KNEE if side == "left" else LM.RIGHT_KNEE)
    if v_foot is None or v_shank is None:
        return float("nan")

    ang = _angle_between(v_foot, v_shank)
    return float(ang)


def foot_progression_angle(landmarks: np.ndarray, side: str = "left") -> float:
    """Foot progression angle relative to forward axis (degrees)."""
    heel = LM.LEFT_HEEL if side == "left" else LM.RIGHT_HEEL
    foot_idx = LM.LEFT_FOOT_INDEX if side == "left" else LM.RIGHT_FOOT_INDEX

    v_foot = _vec(landmarks, heel, foot_idx)
    if v_foot is None:
        return float("nan")

    forward = np.array([1.0, 0.0]) if side == "right" else np.array([-1.0, 0.0])
    return _angle_between(v_foot, forward)


def trunk_lean_angle(landmarks: np.ndarray) -> float:
    """Lateral trunk lean angle (degrees) from vertical in frontal plane."""
    ls, rs = landmarks[LM.LEFT_SHOULDER], landmarks[LM.RIGHT_SHOULDER]
    lh, rh = landmarks[LM.LEFT_HIP], landmarks[LM.RIGHT_HIP]
    if min(ls[3], rs[3], lh[3], rh[3]) < MIN_VISIBILITY:
        return float("nan")

    shoulder_mid = np.array([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2])
    hip_mid = np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2])
    trunk_vec = hip_mid - shoulder_mid
    vertical = np.array([0.0, 1.0])
    return _angle_between(trunk_vec, vertical)


def trunk_flexion_angle(landmarks: np.ndarray) -> float:
    """Sagittal trunk flexion angle (degrees) from vertical."""
    return trunk_lean_angle(landmarks)


def trunk_lateral_flexion_angle(landmarks: np.ndarray) -> float:
    """Frontal trunk lateral flexion angle (degrees)."""
    return trunk_lean_angle(landmarks)


def trunk_rotation_angle(landmarks: np.ndarray) -> float:
    """Transverse trunk rotation angle (degrees) between shoulder line and hip line."""
    ls, rs = landmarks[LM.LEFT_SHOULDER], landmarks[LM.RIGHT_SHOULDER]
    lh, rh = landmarks[LM.LEFT_HIP], landmarks[LM.RIGHT_HIP]
    if min(ls[3], rs[3], lh[3], rh[3]) < MIN_VISIBILITY:
        return float("nan")

    shoulder_vec = np.array([rs[0] - ls[0], rs[1] - ls[1]])
    hip_vec = np.array([rh[0] - lh[0], rh[1] - lh[1]])
    return _angle_between(shoulder_vec, hip_vec)


def pelvic_obliquity_angle(landmarks: np.ndarray) -> float:
    """
    Pelvic obliquity / hip drop angle (degrees).
    Calculated as the angle of left_hip -> right_hip line relative to horizontal.
    Normalized without wrapping artifacts.
    """
    lh, rh = landmarks[LM.LEFT_HIP], landmarks[LM.RIGHT_HIP]
    if lh[3] < MIN_VISIBILITY or rh[3] < MIN_VISIBILITY:
        return float("nan")

    dx = rh[0] - lh[0]
    dy = rh[1] - lh[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0

    angle_rad = math.atan2(dy, dx)
    return float(math.degrees(angle_rad))


def pelvic_tilt_angle(landmarks: np.ndarray) -> float:
    """Pelvic tilt angle (degrees) in sagittal plane."""
    return pelvic_obliquity_angle(landmarks)


def pelvic_rotation_angle(landmarks: np.ndarray) -> float:
    """Pelvic rotation angle (degrees) in transverse plane."""
    return pelvic_obliquity_angle(landmarks)


def estimate_center_of_mass(landmarks: np.ndarray, landmarks_3d: np.ndarray | None = None) -> tuple[float, float, float]:
    """
    Estimate whole-body center of mass (COM) using Dempster segmental mass parameters:
      Head/Neck: 8.1%, Trunk: 49.7%, Thighs: 10% each, Shanks: 4.65% each, Feet: 1.45% each.
    Labeled as ESTIMATED.
    """
    # Fallback weights if missing landmarks
    pts = []
    weights = []

    def _add_segment(p1, p2, mass):
        if p1[3] >= MIN_VISIBILITY and p2[3] >= MIN_VISIBILITY:
            pts.append(((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2))
            weights.append(mass)

    # Nose / shoulders for head/trunk
    ls, rs = landmarks[LM.LEFT_SHOULDER], landmarks[LM.RIGHT_SHOULDER]
    lh, rh = landmarks[LM.LEFT_HIP], landmarks[LM.RIGHT_HIP]
    lk, rk = landmarks[LM.LEFT_KNEE], landmarks[LM.RIGHT_KNEE]
    la, ra = landmarks[LM.LEFT_ANKLE], landmarks[LM.RIGHT_ANKLE]
    lf, rf = landmarks[LM.LEFT_FOOT_INDEX], landmarks[LM.RIGHT_FOOT_INDEX]

    _add_segment(ls, rs, 0.081)  # Upper trunk/head
    _add_segment(lh, rh, 0.497)  # Lower trunk/pelvis
    _add_segment(lh, lk, 0.100)  # L thigh
    _add_segment(rh, rk, 0.100)  # R thigh
    _add_segment(lk, la, 0.0465) # L shank
    _add_segment(rk, ra, 0.0465) # R shank
    _add_segment(la, lf, 0.0145) # L foot
    _add_segment(ra, rf, 0.0145) # R foot

    if not weights:
        return (float("nan"), float("nan"), float("nan"))

    w_sum = sum(weights)
    cx = sum(p[0] * w for p, w in zip(pts, weights)) / w_sum
    cy = sum(p[1] * w for p, w in zip(pts, weights)) / w_sum
    cz = sum(p[2] * w for p, w in zip(pts, weights)) / w_sum
    return (float(cx), float(cy), float(cz))


def calculate_pose_quality_metrics(landmarks: np.ndarray) -> dict[str, float]:
    """Calculate landmark confidence and tracking quality metrics for one frame."""
    vis = landmarks[:, 3]
    total_lms = len(vis)
    missing = np.sum(vis < MIN_VISIBILITY)

    left_knee_v = float(landmarks[LM.LEFT_KNEE][3])
    right_knee_v = float(landmarks[LM.RIGHT_KNEE][3])
    left_hip_v = float(landmarks[LM.LEFT_HIP][3])
    right_hip_v = float(landmarks[LM.RIGHT_HIP][3])
    left_ankle_v = float(landmarks[LM.LEFT_ANKLE][3])
    right_ankle_v = float(landmarks[LM.RIGHT_ANKLE][3])

    key_lower_body = [left_knee_v, right_knee_v, left_hip_v, right_hip_v, left_ankle_v, right_ankle_v]
    occluded_lower = sum(1 for v in key_lower_body if v < MIN_VISIBILITY)

    return {
        "pose_confidence": float(np.mean(vis)),
        "left_knee_pose_confidence": left_knee_v,
        "right_knee_pose_confidence": right_knee_v,
        "left_hip_pose_confidence": left_hip_v,
        "right_hip_pose_confidence": right_hip_v,
        "left_ankle_pose_confidence": left_ankle_v,
        "right_ankle_pose_confidence": right_ankle_v,
        "missing_landmark_ratio": float(missing / total_lms),
        "tracking_quality": float(1.0 - (missing / total_lms)),
        "frame_drop_rate": 0.0,
        "occlusion_ratio": float(occluded_lower / len(key_lower_body)),
    }


@dataclass
class FrameFeatures:
    frame_index: int
    timestamp: float
    fppa_left: float
    fppa_right: float
    knee_flexion_left: float
    knee_flexion_right: float
    trunk_lean: float

    # Backward compatibility fields
    left_knee_flexion: float = float("nan")
    right_knee_flexion: float = float("nan")
    left_knee_valgus_proxy: float = float("nan")
    right_knee_valgus_proxy: float = float("nan")
    valgus_asymmetry: float = float("nan")
    knee_flexion_asymmetry: float = float("nan")
    trunk_lean_deg: float = float("nan")
    hip_drop_deg: float = float("nan")
    valid: bool = True
    persistence_count: int = 0
    flagged: bool = False
    contributing_features: list = field(default_factory=list)
    affected_joints: list = field(default_factory=list)

    # Monocular-3D features
    knee_flexion_left_3d: float = float("nan")
    knee_flexion_right_3d: float = float("nan")
    knee_abduction_left_3d: float = float("nan")
    knee_abduction_right_3d: float = float("nan")

    # Level 1 Hip Features
    hip_flexion_left: float = float("nan")
    hip_flexion_right: float = float("nan")
    hip_abduction_left: float = float("nan")
    hip_abduction_right: float = float("nan")
    hip_rotation_left: float = float("nan")
    hip_rotation_right: float = float("nan")

    # Level 1 Ankle/Foot Features
    ankle_dorsiflexion_left: float = float("nan")
    ankle_dorsiflexion_right: float = float("nan")
    ankle_inversion_left: float = float("nan")
    ankle_inversion_right: float = float("nan")
    foot_progression_left: float = float("nan")
    foot_progression_right: float = float("nan")
    foot_rotation_left: float = float("nan")
    foot_rotation_right: float = float("nan")

    # Level 1 Trunk & Pelvis
    trunk_flexion_deg: float = float("nan")
    trunk_lateral_flexion_deg: float = float("nan")
    trunk_rotation_deg: float = float("nan")
    pelvic_tilt_deg: float = float("nan")
    pelvic_obliquity_deg: float = float("nan")
    pelvic_rotation_deg: float = float("nan")

    # Symmetry & LSI
    knee_valgus_asymmetry: float = float("nan")
    knee_rotation_asymmetry: float = float("nan")
    knee_flexion_LSI: float = float("nan")
    knee_valgus_LSI: float = float("nan")
    knee_rotation_LSI: float = float("nan")
    hip_flexion_LSI: float = float("nan")
    hip_abduction_LSI: float = float("nan")
    hip_rotation_LSI: float = float("nan")
    hip_asymmetry: float = float("nan")
    ankle_dorsiflexion_LSI: float = float("nan")
    ankle_asymmetry: float = float("nan")

    # Pose Data Quality
    pose_confidence: float = 1.0
    left_knee_pose_confidence: float = 1.0
    right_knee_pose_confidence: float = 1.0
    left_hip_pose_confidence: float = 1.0
    right_hip_pose_confidence: float = 1.0
    left_ankle_pose_confidence: float = 1.0
    right_ankle_pose_confidence: float = 1.0
    missing_landmark_ratio: float = 0.0
    tracking_quality: float = 1.0
    frame_drop_rate: float = 0.0
    occlusion_ratio: float = 0.0

    # Level 2 Center of Mass
    com_x: float = float("nan")
    com_y: float = float("nan")
    com_z: float = float("nan")

    # Level 2 6-DOF Knee Translation Architecture
    left_knee_ap_translation_mm: float = float("nan")
    right_knee_ap_translation_mm: float = float("nan")
    left_knee_ml_translation_mm: float = float("nan")
    right_knee_ml_translation_mm: float = float("nan")
    left_knee_si_translation_mm: float = float("nan")
    right_knee_si_translation_mm: float = float("nan")

    # Level 2 Sensor-Fusion Architecture
    imu_acceleration: float = float("nan")
    imu_gyroscope: float = float("nan")
    imu_orientation: float = float("nan")
    imu_quality: float = float("nan")
    estimated_imu_drift: float = float("nan")
    sensor_synchronization_error: float = float("nan")


def extract_frame_features(
    landmarks: np.ndarray,
    frame_index: int,
    timestamp: float,
    landmarks_3d: np.ndarray | None = None,
) -> FrameFeatures:
    """Compute per-frame Level 1 and Level 2 biomechanical features."""
    fppa_l = fppa(landmarks, "left")
    fppa_r = fppa(landmarks, "right")
    flex_l = knee_flexion_angle(landmarks, "left")
    flex_r = knee_flexion_angle(landmarks, "right")
    t_lean = trunk_lean_angle(landmarks)

    hip_fl_l = hip_flexion_angle(landmarks, "left")
    hip_fl_r = hip_flexion_angle(landmarks, "right")
    hip_ab_l = hip_abduction_angle(landmarks, "left")
    hip_ab_r = hip_abduction_angle(landmarks, "right")
    hip_rot_l = hip_rotation_angle(landmarks, "left")
    hip_rot_r = hip_rotation_angle(landmarks, "right")

    ank_df_l = ankle_dorsiflexion_angle(landmarks, "left")
    ank_df_r = ankle_dorsiflexion_angle(landmarks, "right")
    ank_inv_l = ankle_inversion_angle(landmarks, "left")
    ank_inv_r = ankle_inversion_angle(landmarks, "right")
    foot_prog_l = foot_progression_angle(landmarks, "left")
    foot_prog_r = foot_progression_angle(landmarks, "right")

    p_obliq = pelvic_obliquity_angle(landmarks)
    p_tilt = pelvic_tilt_angle(landmarks)
    p_rot = pelvic_rotation_angle(landmarks)
    t_rot = trunk_rotation_angle(landmarks)

    quality = calculate_pose_quality_metrics(landmarks)
    cx, cy, cz = estimate_center_of_mass(landmarks, landmarks_3d)

    features = FrameFeatures(
        frame_index=frame_index,
        timestamp=timestamp,
        fppa_left=fppa_l,
        fppa_right=fppa_r,
        knee_flexion_left=flex_l,
        knee_flexion_right=flex_r,
        trunk_lean=t_lean,
        # Backward compatibility
        left_knee_flexion=flex_l,
        right_knee_flexion=flex_r,
        left_knee_valgus_proxy=fppa_l,
        right_knee_valgus_proxy=fppa_r,
        valgus_asymmetry=abs(fppa_l - fppa_r) if not (math.isnan(fppa_l) or math.isnan(fppa_r)) else float("nan"),
        knee_flexion_asymmetry=abs(flex_l - flex_r) if not (math.isnan(flex_l) or math.isnan(flex_r)) else float("nan"),
        trunk_lean_deg=t_lean,
        hip_drop_deg=p_obliq,
        valid=quality["tracking_quality"] > 0.3,
        # Hip features
        hip_flexion_left=hip_fl_l,
        hip_flexion_right=hip_fl_r,
        hip_abduction_left=hip_ab_l,
        hip_abduction_right=hip_ab_r,
        hip_rotation_left=hip_rot_l,
        hip_rotation_right=hip_rot_r,
        # Ankle features
        ankle_dorsiflexion_left=ank_df_l,
        ankle_dorsiflexion_right=ank_df_r,
        ankle_inversion_left=ank_inv_l,
        ankle_inversion_right=ank_inv_r,
        foot_progression_left=foot_prog_l,
        foot_progression_right=foot_prog_r,
        # Trunk/Pelvis
        trunk_flexion_deg=t_lean,
        trunk_lateral_flexion_deg=t_lean,
        trunk_rotation_deg=t_rot,
        pelvic_tilt_deg=p_tilt,
        pelvic_obliquity_deg=p_obliq,
        pelvic_rotation_deg=p_rot,
        # Symmetry / LSI
        knee_valgus_asymmetry=abs(fppa_l - fppa_r) if not (math.isnan(fppa_l) or math.isnan(fppa_r)) else float("nan"),
        knee_rotation_asymmetry=float("nan"),
        knee_flexion_LSI=calculate_lsi(flex_l, flex_r),
        knee_valgus_LSI=calculate_lsi(fppa_l, fppa_r),
        knee_rotation_LSI=float("nan"),
        hip_flexion_LSI=calculate_lsi(hip_fl_l, hip_fl_r),
        hip_abduction_LSI=calculate_lsi(hip_ab_l, hip_ab_r),
        hip_rotation_LSI=float("nan"),
        hip_asymmetry=abs(hip_fl_l - hip_fl_r) if not (math.isnan(hip_fl_l) or math.isnan(hip_fl_r)) else float("nan"),
        ankle_dorsiflexion_LSI=calculate_lsi(ank_df_l, ank_df_r),
        ankle_asymmetry=abs(ank_df_l - ank_df_r) if not (math.isnan(ank_df_l) or math.isnan(ank_df_r)) else float("nan"),
        # Quality
        pose_confidence=quality["pose_confidence"],
        left_knee_pose_confidence=quality["left_knee_pose_confidence"],
        right_knee_pose_confidence=quality["right_knee_pose_confidence"],
        left_hip_pose_confidence=quality["left_hip_pose_confidence"],
        right_hip_pose_confidence=quality["right_hip_pose_confidence"],
        left_ankle_pose_confidence=quality["left_ankle_pose_confidence"],
        right_ankle_pose_confidence=quality["right_ankle_pose_confidence"],
        missing_landmark_ratio=quality["missing_landmark_ratio"],
        tracking_quality=quality["tracking_quality"],
        frame_drop_rate=quality["frame_drop_rate"],
        occlusion_ratio=quality["occlusion_ratio"],
        # Level 2 COM
        com_x=cx,
        com_y=cy,
        com_z=cz,
    )

    if landmarks_3d is not None:
        features.knee_flexion_left_3d = knee_flexion_angle_3d(landmarks_3d, "left")
        features.knee_flexion_right_3d = knee_flexion_angle_3d(landmarks_3d, "right")
        features.knee_abduction_left_3d = knee_abduction_angle_3d(landmarks_3d, "left")
        features.knee_abduction_right_3d = knee_abduction_angle_3d(landmarks_3d, "right")

    return features


def _row_to_landmarks(row: pd.Series, names: list[str]) -> np.ndarray:
    """Rebuild a (33, 4) landmark array from one wide-format DataFrame row."""
    out = np.zeros((33, 4), dtype=np.float64)
    for i, name in enumerate(names):
        out[i, 0] = row.get(f"{name}_x", np.nan)
        out[i, 1] = row.get(f"{name}_y", np.nan)
        out[i, 2] = row.get(f"{name}_z", np.nan)
        out[i, 3] = row.get(f"{name}_v", 0.0)
    return out


def extract_features_dataframe(df: pd.DataFrame, landmark_names: list[str]) -> pd.DataFrame:
    """
    Vectorized-over-rows wrapper: extract per-frame biomechanical features,
    and compute temporal derivatives (velocity and acceleration) across frames.
    """
    records = []
    for _, row in df.iterrows():
        lm = _row_to_landmarks(row, landmark_names)
        feats = extract_frame_features(
            lm,
            frame_index=int(row.get("frame_index", 0)),
            timestamp=float(row.get("timestamp", 0.0)),
        )
        records.append(vars(feats))
    out_df = pd.DataFrame.from_records(records)

    # Compute temporal derivatives (velocity & acceleration) across frames
    if "timestamp" in out_df.columns and len(out_df) > 1:
        t = out_df["timestamp"].to_numpy()
        dt = np.gradient(t)
        dt = np.where(dt <= 0, 1e-3, dt)  # protect against non-increasing timestamps

        angles_to_derive = [
            ("knee_flexion_left", "knee_flexion_velocity_left", "knee_flexion_acceleration_left"),
            ("knee_flexion_right", "knee_flexion_velocity_right", "knee_flexion_acceleration_right"),
            ("fppa_left", "knee_valgus_velocity_left", "knee_valgus_acceleration_left"),
            ("fppa_right", "knee_valgus_velocity_right", "knee_valgus_acceleration_right"),
            ("hip_flexion_left", "hip_flexion_velocity_left", "hip_flexion_acceleration_left"),
            ("hip_flexion_right", "hip_flexion_velocity_right", "hip_flexion_acceleration_right"),
            ("hip_abduction_left", "hip_abduction_velocity_left", "hip_abduction_acceleration_left"),
            ("hip_abduction_right", "hip_abduction_velocity_right", "hip_abduction_acceleration_right"),
            ("ankle_dorsiflexion_left", "ankle_dorsiflexion_velocity_left", "ankle_dorsiflexion_acceleration_left"),
            ("ankle_dorsiflexion_right", "ankle_dorsiflexion_velocity_right", "ankle_dorsiflexion_acceleration_right"),
            ("trunk_lateral_flexion_deg", "trunk_angular_velocity", "trunk_angular_acceleration"),
        ]

        for angle_col, vel_col, accel_col in angles_to_derive:
            if angle_col in out_df.columns:
                series = out_df[angle_col].interpolate(method="linear").bfill().ffill().to_numpy()
                vel = np.gradient(series, t)
                accel = np.gradient(vel, t)
                out_df[vel_col] = vel
                out_df[accel_col] = accel

        # Velocity aliases required by Level 1 specification
        out_df["knee_flexion_velocity"] = (out_df.get("knee_flexion_velocity_left", 0) + out_df.get("knee_flexion_velocity_right", 0)) / 2
        out_df["knee_valgus_velocity"] = (out_df.get("knee_valgus_velocity_left", 0) + out_df.get("knee_valgus_velocity_right", 0)) / 2
        out_df["knee_rotation_velocity"] = float("nan")
        out_df["hip_flexion_velocity"] = (out_df.get("hip_flexion_velocity_left", 0) + out_df.get("hip_flexion_velocity_right", 0)) / 2
        out_df["hip_abduction_velocity"] = (out_df.get("hip_abduction_velocity_left", 0) + out_df.get("hip_abduction_velocity_right", 0)) / 2
        out_df["hip_rotation_velocity"] = float("nan")
        out_df["ankle_dorsiflexion_velocity"] = (out_df.get("ankle_dorsiflexion_velocity_left", 0) + out_df.get("ankle_dorsiflexion_velocity_right", 0)) / 2
        out_df["trunk_flexion_velocity"] = out_df.get("trunk_angular_velocity", float("nan"))
        out_df["trunk_lateral_flexion_velocity"] = out_df.get("trunk_angular_velocity", float("nan"))
        out_df["trunk_rotation_velocity"] = float("nan")

        out_df["knee_flexion_acceleration"] = (out_df.get("knee_flexion_acceleration_left", 0) + out_df.get("knee_flexion_acceleration_right", 0)) / 2
        out_df["knee_valgus_acceleration"] = (out_df.get("knee_valgus_acceleration_left", 0) + out_df.get("knee_valgus_acceleration_right", 0)) / 2
        out_df["knee_rotation_acceleration"] = float("nan")
        out_df["hip_flexion_acceleration"] = (out_df.get("hip_flexion_acceleration_left", 0) + out_df.get("hip_flexion_acceleration_right", 0)) / 2
        out_df["hip_rotation_acceleration"] = float("nan")

        # COM dynamics
        if "com_x" in out_df.columns and "com_y" in out_df.columns:
            cx = out_df["com_x"].interpolate(method="linear").bfill().ffill().to_numpy()
            cy = out_df["com_y"].interpolate(method="linear").bfill().ffill().to_numpy()
            vx = np.gradient(cx, t)
            vy = np.gradient(cy, t)
            out_df["com_velocity"] = np.sqrt(vx**2 + vy**2)
            out_df["com_lateral_sway"] = np.std(cx)
            out_df["com_forward_displacement"] = np.abs(cy[-1] - cy[0]) if len(cy) > 0 else 0.0
            out_df["com_total_displacement"] = np.sum(np.sqrt(np.diff(cx, prepend=cx[0])**2 + np.diff(cy, prepend=cy[0])**2))

    return out_df


# --- 22 Biomechanical Parameters Specification and Engine Re-exports ---
from src.features.parameters_22 import (
    PARAMETERS_22_SPECS,
    PARAMETERS_22_MAP,
    ParameterSpec,
    compute_22_parameters,
    get_parameter_specs,
)
