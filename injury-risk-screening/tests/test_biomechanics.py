"""Real, runnable tests for src/features/biomechanics.py -- no mocks needed,
since these are pure geometry functions over plain numpy arrays."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.biomechanics import fppa, knee_flexion_angle, trunk_lean_angle
from src.features.landmarks import LM


def _make_landmarks() -> np.ndarray:
    """33x4 zeroed landmark array with full visibility, for tests to overwrite selectively."""
    lm = np.zeros((33, 4), dtype=np.float64)
    lm[:, 3] = 1.0  # visibility = 1 everywhere by default
    return lm


def test_fppa_straight_leg_is_zero():
    lm = _make_landmarks()
    # Perfectly vertical, collinear hip-knee-ankle (straight leg), left side.
    lm[LM.LEFT_HIP] = [0.5, 0.3, 0, 1.0]
    lm[LM.LEFT_KNEE] = [0.5, 0.5, 0, 1.0]
    lm[LM.LEFT_ANKLE] = [0.5, 0.7, 0, 1.0]
    angle = fppa(lm, "left")
    assert angle == pytest.approx(0.0, abs=1e-6)


def test_fppa_detects_sideways_knee_deviation():
    lm = _make_landmarks()
    # Hip and ankle aligned vertically, knee pulled sideways off that
    # line -- a knee-valgus-or-varus-style collapse. The sign of a
    # sideways deviation depends on camera facing direction (see
    # fppa()'s docstring) so we only assert the clinically-relevant
    # part: a large, nonzero magnitude, not a specific sign.
    lm[LM.LEFT_HIP] = [0.5, 0.3, 0, 1.0]
    lm[LM.LEFT_KNEE] = [0.6, 0.5, 0, 1.0]  # shifted sideways off the hip-ankle line
    lm[LM.LEFT_ANKLE] = [0.5, 0.7, 0, 1.0]
    angle = fppa(lm, "left")
    assert abs(angle) > 10, "sideways knee shift should register as a large FPPA deviation"


def test_fppa_sign_is_consistent_for_a_fixed_camera():
    """Two knee deviations to the same side (same camera setup) must
    produce the same sign, even though the absolute sign is convention-
    dependent (see docstring)."""
    lm1 = _make_landmarks()
    lm1[LM.LEFT_HIP] = [0.5, 0.3, 0, 1.0]
    lm1[LM.LEFT_KNEE] = [0.6, 0.5, 0, 1.0]
    lm1[LM.LEFT_ANKLE] = [0.5, 0.7, 0, 1.0]

    lm2 = _make_landmarks()
    lm2[LM.LEFT_HIP] = [0.5, 0.3, 0, 1.0]
    lm2[LM.LEFT_KNEE] = [0.65, 0.5, 0, 1.0]  # same direction, larger shift
    lm2[LM.LEFT_ANKLE] = [0.5, 0.7, 0, 1.0]

    angle1, angle2 = fppa(lm1, "left"), fppa(lm2, "left")
    assert (angle1 > 0) == (angle2 > 0)
    assert abs(angle2) > abs(angle1), "larger sideways shift should produce larger |fppa|"


def test_fppa_low_visibility_returns_nan():
    lm = _make_landmarks()
    lm[LM.LEFT_HIP] = [0.5, 0.3, 0, 0.1]  # below MIN_VISIBILITY
    lm[LM.LEFT_KNEE] = [0.5, 0.5, 0, 1.0]
    lm[LM.LEFT_ANKLE] = [0.5, 0.7, 0, 1.0]
    angle = fppa(lm, "left")
    assert angle != angle  # NaN check


def test_fppa_invalid_side_raises():
    lm = _make_landmarks()
    with pytest.raises(ValueError):
        fppa(lm, "middle")


def test_knee_flexion_straight_leg_is_zero():
    lm = _make_landmarks()
    lm[LM.LEFT_HIP] = [0.5, 0.3, 0, 1.0]
    lm[LM.LEFT_KNEE] = [0.5, 0.5, 0, 1.0]
    lm[LM.LEFT_ANKLE] = [0.5, 0.7, 0, 1.0]
    angle = knee_flexion_angle(lm, "left")
    assert angle == pytest.approx(0.0, abs=1e-6)


def test_knee_flexion_bent_leg_is_positive():
    lm = _make_landmarks()
    # Deep knee bend: shank folds back under the thigh.
    lm[LM.LEFT_HIP] = [0.5, 0.3, 0, 1.0]
    lm[LM.LEFT_KNEE] = [0.5, 0.6, 0, 1.0]
    lm[LM.LEFT_ANKLE] = [0.5, 0.4, 0, 1.0]  # ankle pulled back up near hip height
    angle = knee_flexion_angle(lm, "left")
    assert angle > 45


def test_trunk_lean_upright_is_zero():
    lm = _make_landmarks()
    lm[LM.LEFT_SHOULDER] = [0.4, 0.2, 0, 1.0]
    lm[LM.RIGHT_SHOULDER] = [0.6, 0.2, 0, 1.0]
    lm[LM.LEFT_HIP] = [0.4, 0.5, 0, 1.0]
    lm[LM.RIGHT_HIP] = [0.6, 0.5, 0, 1.0]
    angle = trunk_lean_angle(lm)
    assert angle == pytest.approx(0.0, abs=1e-6)


def test_trunk_lean_sideways_is_nonzero():
    lm = _make_landmarks()
    lm[LM.LEFT_SHOULDER] = [0.4, 0.2, 0, 1.0]
    lm[LM.RIGHT_SHOULDER] = [0.6, 0.2, 0, 1.0]
    lm[LM.LEFT_HIP] = [0.5, 0.5, 0, 1.0]
    lm[LM.RIGHT_HIP] = [0.7, 0.5, 0, 1.0]  # hips shifted sideways relative to shoulders
    angle = trunk_lean_angle(lm)
    assert angle > 5
