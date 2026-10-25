"""
Comprehensive unit and integration test suite for Level 1 and Level 2 biomechanical feature pipeline.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from src.features.biomechanics import (
    calculate_lsi,
    calculate_pose_quality_metrics,
    estimate_center_of_mass,
    extract_features_dataframe,
    extract_frame_features,
    fppa,
    hip_abduction_angle,
    hip_flexion_angle,
    knee_flexion_angle,
    pelvic_obliquity_angle,
    pelvic_tilt_angle,
    trunk_lean_angle,
    ankle_dorsiflexion_angle,
    ankle_inversion_angle,
)
from src.features.landmarks import LM, ALL_LANDMARK_NAMES
from src.features.event_segmentation import segment_landing_events
from src.features.movement_dataset import generate_movement_features_dataframe
from src.baseline.baseline_engine import compute_baseline, robust_z_score


def _build_synthetic_frame(vis: float = 0.9) -> np.ndarray:
    """Helper: build a (33, 4) landmark array with realistic 2D coordinates."""
    lm = np.zeros((33, 4), dtype=np.float64)
    lm[:, 3] = vis

    # Shoulders
    lm[LM.LEFT_SHOULDER] = [0.4, 0.2, 0.0, vis]
    lm[LM.RIGHT_SHOULDER] = [0.6, 0.2, 0.0, vis]

    # Hips
    lm[LM.LEFT_HIP] = [0.42, 0.5, 0.0, vis]
    lm[LM.RIGHT_HIP] = [0.58, 0.5, 0.0, vis]

    # Knees
    lm[LM.LEFT_KNEE] = [0.43, 0.7, 0.0, vis]
    lm[LM.RIGHT_KNEE] = [0.57, 0.7, 0.0, vis]

    # Ankles
    lm[LM.LEFT_ANKLE] = [0.42, 0.9, 0.0, vis]
    lm[LM.RIGHT_ANKLE] = [0.58, 0.9, 0.0, vis]

    # Heels & Feet
    lm[LM.LEFT_HEEL] = [0.42, 0.92, 0.0, vis]
    lm[LM.RIGHT_HEEL] = [0.58, 0.92, 0.0, vis]
    lm[LM.LEFT_FOOT_INDEX] = [0.42, 0.96, 0.0, vis]
    lm[LM.RIGHT_FOOT_INDEX] = [0.58, 0.96, 0.0, vis]

    return lm


def test_lsi_calculation_and_edge_cases():
    # Normal symmetric
    assert calculate_lsi(50.0, 50.0) == 0.0
    # Asymmetric
    assert calculate_lsi(40.0, 60.0) == pytest.approx(40.0, abs=0.1)
    # Zero denominator safe
    assert calculate_lsi(0.0, 0.0) == 0.0
    # NaN safe
    assert math.isnan(calculate_lsi(float("nan"), 50.0))


def test_knee_and_hip_angles():
    lm = _build_synthetic_frame()
    flex_l = knee_flexion_angle(lm, "left")
    flex_r = knee_flexion_angle(lm, "right")
    assert not math.isnan(flex_l)
    assert not math.isnan(flex_r)

    hip_fl = hip_flexion_angle(lm, "left")
    hip_ab = hip_abduction_angle(lm, "left")
    assert not math.isnan(hip_fl)
    assert not math.isnan(hip_ab)


def test_ankle_and_pelvic_angles():
    lm = _build_synthetic_frame()
    ank_df = ankle_dorsiflexion_angle(lm, "left")
    ank_inv = ankle_inversion_angle(lm, "left")
    p_obliq = pelvic_obliquity_angle(lm)
    p_tilt = pelvic_tilt_angle(lm)

    assert not math.isnan(ank_df)
    assert not math.isnan(ank_inv)
    assert not math.isnan(p_obliq)
    assert not math.isnan(p_tilt)


def test_angle_normalization_no_wrapping():
    lm = _build_synthetic_frame()
    # Modify hips to tilt slightly
    lm[LM.LEFT_HIP][1] = 0.52
    p_obliq = pelvic_obliquity_angle(lm)
    # Angle should stay bounded in [-90, 90] without wrapping to +/-180
    assert -90.0 <= p_obliq <= 90.0


def test_pose_quality_metrics_and_low_confidence():
    lm_high = _build_synthetic_frame(vis=0.95)
    q_high = calculate_pose_quality_metrics(lm_high)
    assert q_high["pose_confidence"] > 0.9
    assert q_high["missing_landmark_ratio"] == 0.0

    lm_low = _build_synthetic_frame(vis=0.2)
    q_low = calculate_pose_quality_metrics(lm_low)
    assert q_low["missing_landmark_ratio"] > 0.5
    assert q_low["occlusion_ratio"] == 1.0


def test_center_of_mass_estimation():
    lm = _build_synthetic_frame()
    cx, cy, cz = estimate_center_of_mass(lm)
    assert not math.isnan(cx)
    assert not math.isnan(cy)
    # Center of mass x should lie within body width range [0.4, 0.6]
    assert 0.35 <= cx <= 0.65


def test_robust_z_score_and_mad():
    # Normal deviation
    z = robust_z_score(value=10.0, median=5.0, mad=2.0)
    assert z == 2.5

    # Fallback to IQR if MAD is zero
    z_fallback = robust_z_score(value=10.0, median=5.0, mad=0.0, iqr=2.6978)
    assert z_fallback == pytest.approx(2.5, abs=0.1)

    # Protection against scale = 0
    z_zero = robust_z_score(value=10.0, median=5.0, mad=0.0, iqr=0.0)
    assert math.isnan(z_zero)


def test_dataframe_derivatives_velocity_acceleration():
    # Create 30 frames of motion
    rows = []
    names = list(ALL_LANDMARK_NAMES)
    for f in range(30):
        t = f / 30.0
        lm = _build_synthetic_frame()
        # Flexion angle changing dynamically
        lm[LM.LEFT_KNEE][1] = 0.7 + 0.1 * math.sin(t * np.pi)
        r_dict = {"frame_index": f, "timestamp": t}
        for idx, name in enumerate(names):
            r_dict[f"{name}_x"] = lm[idx, 0]
            r_dict[f"{name}_y"] = lm[idx, 1]
            r_dict[f"{name}_z"] = lm[idx, 2]
            r_dict[f"{name}_v"] = lm[idx, 3]
        rows.append(r_dict)

    df = pd.DataFrame(rows)
    extracted = extract_features_dataframe(df, names)

    assert "knee_flexion_velocity_left" in extracted.columns
    assert "knee_flexion_acceleration_left" in extracted.columns
    assert not extracted["knee_flexion_velocity_left"].isnull().all()


def test_movement_dataset_generation_and_schema():
    rows = []
    names = list(ALL_LANDMARK_NAMES)
    for f in range(60):
        t = f / 30.0
        lm = _build_synthetic_frame()
        # Dip in hip height to simulate jump-landing
        dip = 0.05 * math.sin(t * np.pi) if 0.5 <= t <= 1.5 else 0.0
        lm[LM.LEFT_HIP][1] = 0.5 - dip
        lm[LM.RIGHT_HIP][1] = 0.5 - dip
        r_dict = {"frame_index": f, "timestamp": t}
        for idx, name in enumerate(names):
            r_dict[f"{name}_x"] = lm[idx, 0]
            r_dict[f"{name}_y"] = lm[idx, 1]
            r_dict[f"{name}_z"] = lm[idx, 2]
            r_dict[f"{name}_v"] = lm[idx, 3]
        rows.append(r_dict)

    df = pd.DataFrame(rows)
    extracted = extract_features_dataframe(df, names)
    combined = pd.concat([df, extracted.drop(columns=["frame_index", "timestamp"])], axis=1)

    events = segment_landing_events(combined, fps=30.0)
    assert len(events) >= 1

    mv_df = generate_movement_features_dataframe(
        combined_df=combined,
        events=events,
        subject_id="test_sub_01",
    )
    assert not mv_df.empty
    assert "peak_knee_valgus" in mv_df.columns
    assert "peak_knee_flexion" in mv_df.columns
    assert "knee_flexion_LSI" in mv_df.columns
    assert "knee_valgus_at_initial_contact" in mv_df.columns
    assert "com_displacement" in mv_df.columns


def test_backward_compatibility_fields():
    lm = _build_synthetic_frame()
    feats = extract_frame_features(lm, frame_index=0, timestamp=0.0)

    # Verify original mandatory fields remain intact
    assert hasattr(feats, "fppa_left")
    assert hasattr(feats, "fppa_right")
    assert hasattr(feats, "knee_flexion_left")
    assert hasattr(feats, "knee_flexion_right")
    assert hasattr(feats, "trunk_lean")
    assert hasattr(feats, "left_knee_flexion")
    assert hasattr(feats, "right_knee_flexion")
    assert hasattr(feats, "left_knee_valgus_proxy")
    assert hasattr(feats, "right_knee_valgus_proxy")
    assert hasattr(feats, "trunk_lean_deg")
    assert hasattr(feats, "hip_drop_deg")
