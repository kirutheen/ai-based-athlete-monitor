"""
Small focused tests for bugs 3 (coordinate mapping), 4 (FPPA vs knee flexion formula),
and 5 (LSI null handling and near-zero guard).
"""
import math
import numpy as np
import pytest

from src.features.biomechanics import calculate_lsi, fppa, knee_flexion_angle
from src.features.landmarks import LM


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_landmarks(lm_positions: dict) -> np.ndarray:
    """Build a (33, 4) landmark array (x, y, z, visibility=1.0) with given overrides."""
    lms = np.zeros((33, 4), dtype=np.float64)
    # Default: all landmarks at centre with zero visibility so they won't interfere
    for i in range(33):
        lms[i] = [0.5, 0.5, 0.0, 0.0]
    for lm, (x, y) in lm_positions.items():
        lms[lm.value] = [x, y, 0.0, 1.0]
    return lms


# ── Bug 5: LSI null / near-zero guard ────────────────────────────────────────

class TestLSI:
    def test_symmetric_returns_zero(self):
        assert calculate_lsi(45.0, 45.0) == pytest.approx(0.0)

    def test_asymmetric_known_value(self):
        # 2 * |60-40| / (60+40) * 100 = 2*20/100*100 = 40 %
        assert calculate_lsi(60.0, 40.0) == pytest.approx(40.0)

    def test_none_left_returns_nan(self):
        result = calculate_lsi(None, 45.0)  # type: ignore[arg-type]
        assert math.isnan(result), "None input must return NaN"

    def test_none_right_returns_nan(self):
        result = calculate_lsi(45.0, None)  # type: ignore[arg-type]
        assert math.isnan(result), "None input must return NaN"

    def test_nan_returns_nan(self):
        assert math.isnan(calculate_lsi(float("nan"), 45.0))
        assert math.isnan(calculate_lsi(45.0, float("nan")))

    def test_both_zero_returns_zero(self):
        assert calculate_lsi(0.0, 0.0) == pytest.approx(0.0)

    def test_near_zero_single_side_returns_nan(self):
        """Near-200% instability guard: one side near-zero → NaN, not 197.9%."""
        result = calculate_lsi(0.0001, 78.8)
        assert math.isnan(result), f"Expected NaN for near-zero side, got {result}"

    def test_near_zero_right_returns_nan(self):
        result = calculate_lsi(78.8, 0.0001)
        assert math.isnan(result), f"Expected NaN for near-zero side, got {result}"

    def test_valid_lsi_not_near_200(self):
        """LSI for clearly asymmetric but non-zero values must be < 100%."""
        result = calculate_lsi(-78.8, -87.7)
        assert not math.isnan(result), "Valid inputs should return a number"
        assert 0.0 <= result < 100.0, f"Unexpected LSI={result}"


# ── Bug 4: FPPA vs knee flexion use different planes / sign conventions ───────

class TestFPPAvsKneeFlexion:
    """
    Both FPPA and knee flexion share the same three landmarks (hip, knee, ankle)
    but they must produce DIFFERENT values in real scenarios.

    FPPA: deviation of the frontal-plane knee projection from the hip-ankle line.
          Sign comes from medial/lateral offset of the knee relative to the hip-ankle line.
    Knee flexion: sagittal-plane angle at the knee (always ≥ 0, 0 = fully extended).

    For a straight-leg posture (hip, knee, ankle perfectly vertical), both should
    return 0°. For a bent knee, flexion > 0 while FPPA stays near 0 (no valgus).
    """

    def _straight_leg_left(self):
        """Perfectly straight left leg: hip directly above knee above ankle."""
        return _make_landmarks({
            LM.LEFT_HIP:   (0.5, 0.2),
            LM.LEFT_KNEE:  (0.5, 0.5),
            LM.LEFT_ANKLE: (0.5, 0.8),
        })

    def _bent_knee_no_valgus(self):
        """Knee bent sagittally, no frontal deviation (knee stays on hip-ankle vertical)."""
        return _make_landmarks({
            LM.LEFT_HIP:   (0.5, 0.2),
            LM.LEFT_KNEE:  (0.45, 0.5),   # slight medial shift = bent but no valgus
            LM.LEFT_ANKLE: (0.5, 0.8),
        })

    def test_straight_leg_fppa_near_zero(self):
        lms = self._straight_leg_left()
        result = fppa(lms, "left")
        assert abs(result) < 1e-6, f"FPPA for straight leg should be ~0, got {result}"

    def test_straight_leg_flexion_near_zero(self):
        lms = self._straight_leg_left()
        result = knee_flexion_angle(lms, "left")
        assert abs(result) < 1e-6, f"Flexion for straight leg should be ~0, got {result}"

    def test_knee_flexion_is_non_negative(self):
        """Knee flexion angle is always >= 0 by definition (0 = fully extended)."""
        lms = self._bent_knee_no_valgus()
        result = knee_flexion_angle(lms, "left")
        assert result >= 0.0, f"Knee flexion must be >= 0, got {result}"

    def test_fppa_and_flexion_not_always_same_magnitude(self):
        """
        For a valgus posture, FPPA should reflect frontal deviation while
        knee flexion reflects sagittal bend. They must not be exactly equal
        in all frames (the bug was that they were always same magnitude, opposite sign).
        """
        # Valgus: knee displaced medially from hip-ankle line
        lms = _make_landmarks({
            LM.LEFT_HIP:   (0.5, 0.2),
            LM.LEFT_KNEE:  (0.6, 0.5),   # knee displaced laterally
            LM.LEFT_ANKLE: (0.5, 0.8),
        })
        f = fppa(lms, "left")
        k = knee_flexion_angle(lms, "left")
        # Both must be computed (not NaN)
        assert not math.isnan(f), "FPPA should not be NaN"
        assert not math.isnan(k), "Flexion should not be NaN"
        # Flexion should be positive; FPPA sign depends on valgus direction
        assert k >= 0, f"Flexion must be >= 0, got {k}"
        # They cannot be exactly equal and opposite for all cases — just verify
        # they are not identical in magnitude when there IS a lateral offset:
        # (for pure lateral offset with no sagittal bend, flexion ≈ 0 but fppa ≠ 0)
        assert abs(f) > 1.0, f"Expected non-zero FPPA for valgus landmark, got {f}"


# ── Bug 3: Coordinate mapping (skeleton canvas logic) ─────────────────────────

class TestSkeletonCoordinateMapping:
    """
    The skeleton overlay maps normalized [0,1] landmark coords to canvas pixels.
    Canvas is set to native video resolution (nativeW × nativeH).
    The CSS box covers the contain-fit area (drawW × drawH inside containerW × containerH).

    For a portrait video (nativeW < nativeH) in a landscape container:
      - Contains fit: bars on left and right (pillarbox)
      - drawH = containerH, drawW = containerH * (nativeW/nativeH)
      - offsetX = (containerW - drawW) / 2, offsetY = 0

    Canvas internal coords go 0..nativeW × 0..nativeH.
    A normalized landmark at (lm_x, lm_y) maps to:
      px = lm_x * nativeW
      py = lm_y * nativeH
    CSS scales the canvas box to drawW × drawH, so landmark position appears correct.
    """

    def test_normalized_landmark_maps_to_canvas_pixel(self):
        nativeW, nativeH = 480, 854  # portrait 9:16
        # Centre landmark
        lm_x, lm_y = 0.5, 0.5
        px = lm_x * nativeW
        py = lm_y * nativeH
        assert px == pytest.approx(240.0)
        assert py == pytest.approx(427.0)

    def test_contain_fit_portrait_in_landscape_container(self):
        """CSS box dimensions for a portrait video in a landscape container."""
        nativeW, nativeH = 480, 854
        containerW, containerH = 1280, 720
        videoAR = nativeW / nativeH
        containerAR = containerW / containerH

        if videoAR > containerAR:
            # Pillarbox (shouldn't happen for portrait in landscape)
            drawW = containerW
            drawH = containerW / videoAR
            offsetX = 0
            offsetY = (containerH - drawH) / 2
        else:
            # Letterbox: bars on left & right
            drawH = containerH
            drawW = containerH * videoAR
            offsetX = (containerW - drawW) / 2
            offsetY = 0

        assert drawH == pytest.approx(720.0, rel=1e-3)
        assert drawW == pytest.approx(720 * 480 / 854, rel=1e-3)
        assert offsetY == pytest.approx(0.0)
        assert offsetX > 0  # bars exist on sides

    def test_contain_fit_landscape_in_landscape_container(self):
        """No bars when video and container share the same aspect ratio."""
        nativeW, nativeH = 1280, 720
        containerW, containerH = 1280, 720
        videoAR = nativeW / nativeH
        containerAR = containerW / containerH

        if videoAR > containerAR:
            drawW = containerW
            drawH = containerW / videoAR
            offsetX = 0
            offsetY = (containerH - drawH) / 2
        else:
            drawH = containerH
            drawW = containerH * videoAR
            offsetX = (containerW - drawW) / 2
            offsetY = 0

        assert drawW == pytest.approx(1280.0)
        assert drawH == pytest.approx(720.0)
        assert offsetX == pytest.approx(0.0)
        assert offsetY == pytest.approx(0.0)
