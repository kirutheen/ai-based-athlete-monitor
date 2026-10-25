"""
Phase 5 tests: 22 biomechanical parameters -- computation, null-safety, metrics_summary coverage.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from src.features.parameters_22 import compute_22_parameters, PARAMETERS_22_SPECS, PARAMETERS_22_MAP
from src.features.biomechanics import (
    fppa, knee_flexion_angle, hip_flexion_angle, hip_abduction_angle,
    ankle_dorsiflexion_angle, trunk_lean_angle, pelvic_obliquity_angle,
    calculate_lsi, extract_frame_features,
)
from src.features.landmarks import LM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_landmarks(
    l_hip=(0.40, 0.40), l_knee=(0.41, 0.65), l_ank=(0.41, 0.85),
    r_hip=(0.60, 0.40), r_knee=(0.59, 0.65), r_ank=(0.59, 0.85),
    vis=0.9,
) -> np.ndarray:
    """Build a (33,4) array with realistic lower-body landmarks."""
    lm = np.zeros((33, 4), dtype=np.float64)
    lm[:, 3] = vis
    lm[LM.LEFT_HIP] = [*l_hip, 0.0, vis]
    lm[LM.LEFT_KNEE] = [*l_knee, 0.0, vis]
    lm[LM.LEFT_ANKLE] = [*l_ank, 0.0, vis]
    lm[LM.RIGHT_HIP] = [*r_hip, 0.0, vis]
    lm[LM.RIGHT_KNEE] = [*r_knee, 0.0, vis]
    lm[LM.RIGHT_ANKLE] = [*r_ank, 0.0, vis]
    lm[LM.LEFT_SHOULDER] = [0.38, 0.15, 0.0, vis]
    lm[LM.RIGHT_SHOULDER] = [0.62, 0.15, 0.0, vis]
    lm[LM.LEFT_FOOT_INDEX] = [0.39, 0.92, 0.0, vis]
    lm[LM.RIGHT_FOOT_INDEX] = [0.61, 0.92, 0.0, vis]
    lm[LM.LEFT_HEEL] = [0.40, 0.90, 0.0, vis]
    lm[LM.RIGHT_HEEL] = [0.60, 0.90, 0.0, vis]
    return lm


def _build_df(n_frames=10, **kwargs) -> pd.DataFrame:
    """Build a small features DataFrame from synthetic landmarks."""
    from src.features.biomechanics import extract_features_dataframe
    lm = _synthetic_landmarks(**kwargs)
    rows = []
    from src.features.landmarks import ALL_LANDMARK_NAMES
    names = list(ALL_LANDMARK_NAMES)
    for i in range(n_frames):
        row = {"frame_index": i, "timestamp": i / 30.0}
        for j, name in enumerate(names):
            row[f"{name}_x"] = float(lm[j, 0])
            row[f"{name}_y"] = float(lm[j, 1])
            row[f"{name}_z"] = float(lm[j, 2])
            row[f"{name}_v"] = float(lm[j, 3])
        rows.append(row)
    pose_df = pd.DataFrame(rows)
    return extract_features_dataframe(pose_df, names)


# ---------------------------------------------------------------------------
# 1. Registry completeness
# ---------------------------------------------------------------------------

class TestParameterRegistry:

    def test_exactly_22_specs(self):
        assert len(PARAMETERS_22_SPECS) == 22

    def test_level1_count(self):
        level1 = [s for s in PARAMETERS_22_SPECS if s.level == 1]
        assert len(level1) == 18

    def test_level2_count(self):
        level2 = [s for s in PARAMETERS_22_SPECS if s.level == 2]
        assert len(level2) == 4

    def test_all_level2_require_depth(self):
        for s in PARAMETERS_22_SPECS:
            if s.level == 2:
                assert s.requires_depth_or_calibration is True, f"{s.name} must require depth"

    def test_no_level1_require_depth(self):
        for s in PARAMETERS_22_SPECS:
            if s.level == 1:
                assert s.requires_depth_or_calibration is False, f"{s.name} must not require depth"

    def test_each_param_has_valid_classification(self):
        valid = {"measured", "estimate", "proxy", "model-derived"}
        for s in PARAMETERS_22_SPECS:
            assert s.classification in valid

    def test_param_ids_are_1_to_22(self):
        ids = sorted(s.param_id for s in PARAMETERS_22_SPECS)
        assert ids == list(range(1, 23))

    def test_all_required_names_present(self):
        required = {
            "fppa_right", "knee_flexion_left", "knee_flexion_right",
            "hip_flexion_left", "hip_flexion_right",
            "hip_abduction_left", "hip_abduction_right",
            "ankle_dorsiflexion_left", "ankle_dorsiflexion_right",
            "trunk_lean", "trunk_flexion_deg", "pelvic_obliquity_deg",
            "knee_valgus_LSI", "knee_flexion_LSI",
            "knee_valgus_at_initial_contact", "knee_flexion_at_initial_contact",
            "valgus_duration", "com_displacement",
            "knee_ap_translation_mm", "knee_ml_translation_mm",
            "knee_si_translation_mm", "knee_int_ext_rot_deg",
        }
        actual = {s.name for s in PARAMETERS_22_SPECS}
        assert required == actual


# ---------------------------------------------------------------------------
# 2. Per-function null safety (low visibility => NaN, never fabricated)
# ---------------------------------------------------------------------------

class TestNullSafetyLowVisibility:

    def _low_vis_lm(self):
        lm = _synthetic_landmarks(vis=0.1)  # below MIN_VISIBILITY=0.5
        return lm

    def test_fppa_returns_nan_low_vis(self):
        assert math.isnan(fppa(self._low_vis_lm(), "left"))

    def test_knee_flexion_returns_nan_low_vis(self):
        assert math.isnan(knee_flexion_angle(self._low_vis_lm(), "left"))

    def test_hip_flexion_returns_nan_low_vis(self):
        assert math.isnan(hip_flexion_angle(self._low_vis_lm(), "left"))

    def test_hip_abduction_returns_nan_low_vis(self):
        assert math.isnan(hip_abduction_angle(self._low_vis_lm(), "left"))

    def test_ankle_df_returns_nan_low_vis(self):
        assert math.isnan(ankle_dorsiflexion_angle(self._low_vis_lm(), "left"))

    def test_trunk_lean_returns_nan_low_vis(self):
        assert math.isnan(trunk_lean_angle(self._low_vis_lm()))

    def test_pelvic_obliquity_returns_nan_low_vis(self):
        assert math.isnan(pelvic_obliquity_angle(self._low_vis_lm()))


# ---------------------------------------------------------------------------
# 3. Compute_22_parameters: Level 2 gating
# ---------------------------------------------------------------------------

class TestLevel2Gating:

    def test_level2_not_available_monocular(self):
        df = _build_df()
        result = compute_22_parameters(df, has_depth_input=False, has_stereo_calibration=False)
        for name in ["knee_ap_translation_mm", "knee_ml_translation_mm",
                     "knee_si_translation_mm", "knee_int_ext_rot_deg"]:
            assert result[name]["value"] == "not available", f"{name} must be not available"
            assert result[name]["status"] == "not available"

    def test_level2_available_with_depth(self):
        df = _build_df()
        df["knee_ap_translation_mm"] = 3.5
        result = compute_22_parameters(df, has_depth_input=True)
        assert result["knee_ap_translation_mm"]["status"] == "available"

    def test_fppa_left_not_in_22_output(self):
        """fppa_left is the clinical anchor — it must NOT appear in the 22-param output."""
        df = _build_df()
        result = compute_22_parameters(df)
        assert "fppa_left" not in result


# ---------------------------------------------------------------------------
# 4. Compute_22_parameters: Level 1 values are numeric when data exists
# ---------------------------------------------------------------------------

class TestLevel1Values:

    def test_fppa_right_available(self):
        df = _build_df()
        result = compute_22_parameters(df)
        assert result["fppa_right"]["status"] == "available"
        assert isinstance(result["fppa_right"]["value"], float)

    def test_knee_flexion_bilateral(self):
        df = _build_df()
        result = compute_22_parameters(df)
        for name in ["knee_flexion_left", "knee_flexion_right"]:
            assert result[name]["status"] == "available"
            v = result[name]["value"]
            assert isinstance(v, float) and 0 <= v <= 180

    def test_hip_flexion_bilateral(self):
        df = _build_df()
        result = compute_22_parameters(df)
        for name in ["hip_flexion_left", "hip_flexion_right"]:
            assert result[name]["status"] == "available"

    def test_hip_abduction_bilateral(self):
        df = _build_df()
        result = compute_22_parameters(df)
        for name in ["hip_abduction_left", "hip_abduction_right"]:
            assert result[name]["status"] == "available"

    def test_ankle_dorsiflexion_bilateral(self):
        df = _build_df()
        result = compute_22_parameters(df)
        for name in ["ankle_dorsiflexion_left", "ankle_dorsiflexion_right"]:
            r = result[name]
            # Value may be null if foot_index not visible, but must not be fabricated
            assert r["value"] == "not available" or isinstance(r["value"], float)

    def test_trunk_lean_available(self):
        df = _build_df()
        result = compute_22_parameters(df)
        assert result["trunk_lean"]["status"] == "available"

    def test_pelvic_obliquity_available(self):
        df = _build_df()
        result = compute_22_parameters(df)
        assert result["pelvic_obliquity_deg"]["status"] == "available"

    def test_valgus_lsi_available(self):
        df = _build_df()
        result = compute_22_parameters(df)
        r = result["knee_valgus_LSI"]
        assert r["status"] == "available"
        assert isinstance(r["value"], float)

    def test_flexion_lsi_available(self):
        df = _build_df()
        result = compute_22_parameters(df)
        r = result["knee_flexion_LSI"]
        assert r["status"] == "available"

    def test_valgus_duration_available(self):
        df = _build_df()
        result = compute_22_parameters(df, fps=30.0)
        r = result["valgus_duration"]
        # May be 0.0 (no high-valgus frames) — but must be a float, not fabricated
        assert r["value"] == "not available" or isinstance(r["value"], float)


# ---------------------------------------------------------------------------
# 5. LSI formula correctness
# ---------------------------------------------------------------------------

class TestLSIFormula:

    def test_perfect_symmetry(self):
        assert calculate_lsi(10.0, 10.0) == 0.0

    def test_lsi_known_value(self):
        # 2 * |10 - 20| / (10 + 20) * 100 = 2 * 10 / 30 * 100 = 66.67%
        result = calculate_lsi(10.0, 20.0)
        assert abs(result - 66.67) < 0.1

    def test_both_zero(self):
        assert calculate_lsi(0.0, 0.0) == 0.0

    def test_nan_propagates(self):
        result = calculate_lsi(float("nan"), 10.0)
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# 6. Result dict structure is complete for every parameter
# ---------------------------------------------------------------------------

class TestResultStructure:

    def test_all_22_keys_present(self):
        df = _build_df()
        result = compute_22_parameters(df)
        assert len(result) == 22

    def test_each_result_has_required_fields(self):
        required_fields = {
            "param_id", "name", "display_name", "value", "status",
            "unit", "classification", "level", "formula", "coordinate_convention",
            "description", "notes",
        }
        df = _build_df()
        result = compute_22_parameters(df)
        for name, val in result.items():
            missing = required_fields - set(val.keys())
            assert not missing, f"Parameter '{name}' missing fields: {missing}"

    def test_level_field_is_1_or_2(self):
        df = _build_df()
        result = compute_22_parameters(df)
        for name, val in result.items():
            assert val["level"] in (1, 2), f"Parameter '{name}' has invalid level: {val['level']}"
