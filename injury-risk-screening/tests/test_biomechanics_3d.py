"""
Real, runnable tests for the monocular-3D features in
src/features/biomechanics.py (knee_flexion_angle_3d, knee_abduction_angle_3d),
using synthetic (33, 4) world-landmark arrays with KNOWN geometric angles --
exactly the same "construct a known answer, check the function recovers it"
pattern as tests/test_stereo_calibration.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.biomechanics import (
    extract_frame_features,
    knee_abduction_angle_3d,
    knee_flexion_angle_3d,
)
from src.features.landmarks import LM


def _make_landmarks_3d() -> np.ndarray:
    lm = np.zeros((33, 4), dtype=np.float64)
    lm[:, 3] = 1.0
    return lm


def _set_pelvis_and_thigh(lm: np.ndarray, side: str) -> None:
    """Pelvis along the x-axis (left_hip at -x, right_hip at +x), thigh
    straight down the -y axis from the hip -- a simple, known reference
    pose that puts the flexion plane at the x-y plane (normal = +z)."""
    lm[LM.LEFT_HIP] = [-0.1, 0.0, 0.0, 1.0]
    lm[LM.RIGHT_HIP] = [0.1, 0.0, 0.0, 1.0]
    hip = LM.LEFT_HIP if side == "left" else LM.RIGHT_HIP
    knee = LM.LEFT_KNEE if side == "left" else LM.RIGHT_KNEE
    lm[knee] = np.array(lm[hip]) + np.array([0.0, -0.5, 0.0, 0.0])


def test_flexion_3d_zero_for_straight_leg():
    lm = _make_landmarks_3d()
    _set_pelvis_and_thigh(lm, "left")
    # Ankle continues straight down from the knee -- fully extended leg.
    lm[LM.LEFT_ANKLE] = np.array(lm[LM.LEFT_KNEE]) + np.array([0.0, -0.5, 0.0, 0.0])
    angle = knee_flexion_angle_3d(lm, "left")
    assert angle == pytest.approx(0.0, abs=1e-6)


def test_flexion_3d_ninety_degrees_for_right_angle_bend():
    lm = _make_landmarks_3d()
    _set_pelvis_and_thigh(lm, "left")
    # Shank perpendicular to thigh, still within the flexion plane (z=0).
    lm[LM.LEFT_ANKLE] = np.array(lm[LM.LEFT_KNEE]) + np.array([0.5, 0.0, 0.0, 0.0])
    angle = knee_flexion_angle_3d(lm, "left")
    assert angle == pytest.approx(90.0, abs=1e-6)


def test_abduction_3d_zero_when_shank_stays_in_flexion_plane():
    lm = _make_landmarks_3d()
    _set_pelvis_and_thigh(lm, "left")
    # Flexed knee, but the shank has no out-of-plane (z) component at all.
    lm[LM.LEFT_ANKLE] = np.array(lm[LM.LEFT_KNEE]) + np.array([0.3, -0.4, 0.0, 0.0])
    angle = knee_abduction_angle_3d(lm, "left")
    assert angle == pytest.approx(0.0, abs=1e-6)


def test_abduction_3d_detects_known_out_of_plane_tilt():
    lm = _make_landmarks_3d()
    _set_pelvis_and_thigh(lm, "left")
    # Shank vector (0, -0.5, 0.3): tipped out of the x-y flexion plane
    # along z. Worked out by hand (see test file docstring / PR notes):
    # angle-to-normal = arccos(0.3 / sqrt(0.5^2 + 0.3^2)) ≈ 59.04 deg,
    # so abduction = 90 - 59.04 ≈ 30.96 deg.
    lm[LM.LEFT_ANKLE] = np.array(lm[LM.LEFT_KNEE]) + np.array([0.0, -0.5, 0.3, 0.0])
    angle = knee_abduction_angle_3d(lm, "left")
    assert angle == pytest.approx(30.96, abs=0.1)


def test_abduction_3d_magnitude_grows_with_larger_tilt():
    lm_small = _make_landmarks_3d()
    _set_pelvis_and_thigh(lm_small, "left")
    lm_small[LM.LEFT_ANKLE] = np.array(lm_small[LM.LEFT_KNEE]) + np.array([0.0, -0.5, 0.1, 0.0])

    lm_large = _make_landmarks_3d()
    _set_pelvis_and_thigh(lm_large, "left")
    lm_large[LM.LEFT_ANKLE] = np.array(lm_large[LM.LEFT_KNEE]) + np.array([0.0, -0.5, 0.4, 0.0])

    small_angle = abs(knee_abduction_angle_3d(lm_small, "left"))
    large_angle = abs(knee_abduction_angle_3d(lm_large, "left"))
    assert large_angle > small_angle


def test_flexion_3d_returns_nan_on_low_visibility():
    lm = _make_landmarks_3d()
    _set_pelvis_and_thigh(lm, "left")
    lm[LM.LEFT_ANKLE] = [0.0, 0.0, 0.0, 0.0]  # zero visibility
    assert np.isnan(knee_flexion_angle_3d(lm, "left"))


def test_abduction_3d_returns_nan_on_degenerate_pelvis_vector():
    lm = _make_landmarks_3d()
    _set_pelvis_and_thigh(lm, "left")
    lm[LM.LEFT_ANKLE] = np.array(lm[LM.LEFT_KNEE]) + np.array([0.0, -0.5, 0.3, 0.0])
    # Degenerate pelvis: both hips at the same point -> zero-length pelvis vector.
    lm[LM.RIGHT_HIP] = lm[LM.LEFT_HIP].copy()
    assert np.isnan(knee_abduction_angle_3d(lm, "left"))


def test_extract_frame_features_populates_3d_fields_when_given_landmarks_3d():
    lm_2d = np.zeros((33, 4), dtype=np.float64)
    lm_2d[:, 3] = 1.0
    lm_3d = _make_landmarks_3d()
    _set_pelvis_and_thigh(lm_3d, "left")
    _set_pelvis_and_thigh(lm_3d, "right")
    lm_3d[LM.LEFT_ANKLE] = np.array(lm_3d[LM.LEFT_KNEE]) + np.array([0.0, -0.5, 0.3, 0.0])
    lm_3d[LM.RIGHT_ANKLE] = np.array(lm_3d[LM.RIGHT_KNEE]) + np.array([0.0, -0.5, 0.0, 0.0])

    features = extract_frame_features(lm_2d, frame_index=0, timestamp=0.0, landmarks_3d=lm_3d)
    assert not np.isnan(features.knee_flexion_left_3d)
    assert not np.isnan(features.knee_abduction_left_3d)
    assert abs(features.knee_abduction_left_3d) > abs(features.knee_abduction_right_3d)


def test_extract_frame_features_3d_fields_are_nan_without_landmarks_3d():
    lm_2d = np.zeros((33, 4), dtype=np.float64)
    lm_2d[:, 3] = 1.0
    features = extract_frame_features(lm_2d, frame_index=0, timestamp=0.0)
    assert np.isnan(features.knee_flexion_left_3d)
    assert np.isnan(features.knee_abduction_left_3d)
