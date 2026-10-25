"""
Comprehensive 22 Biomechanical Parameters Specification and Calculation Engine.

This module implements the 22 biomechanical parameters required for Level 1 and Level 2
screening, with clinical definitions, formulas, coordinate conventions, units, and
measurement method classifications:
  - 'measured'
  - 'estimate'
  - 'proxy'
  - 'model-derived'

Level 2 Gating Rule:
  Parameters 19-22 (6-DOF knee translations and axial rotation) require physical depth
  inputs or multi-camera stereo calibration. On standard monocular RGB video, these
  parameters strictly return "not available" rather than fabricated numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.landmarks import LM

ClassificationType = Literal["measured", "estimate", "proxy", "model-derived"]


@dataclass(frozen=True)
class ParameterSpec:
    param_id: int
    name: str
    display_name: str
    unit: str
    classification: ClassificationType
    level: int  # 1 or 2
    description: str
    formula: str
    coordinate_convention: str
    requires_depth_or_calibration: bool = False


# Canonical registry of the 22 biomechanical parameters
PARAMETERS_22_SPECS: list[ParameterSpec] = [
    ParameterSpec(
        param_id=1,
        name="fppa_right",
        display_name="Right Knee Frontal Plane Projection Angle",
        unit="deg",
        classification="proxy",
        level=1,
        description="2D frontal-plane projection angle proxy for dynamic right knee valgus collapse.",
        formula="180 - angle(hip->knee, knee->ankle), signed via cross-product offset from hip-ankle line.",
        coordinate_convention="Frontal plane (XY). Positive values indicate medial displacement (valgus), negative indicate lateral displacement (varus).",
        requires_depth_or_calibration=False,
    ),
    ParameterSpec(
        param_id=2,
        name="knee_flexion_left",
        display_name="Left Knee Flexion",
        unit="deg",
        classification="estimate",
        level=1,
        description="Left knee sagittal flexion angle.",
        formula="180 - angle(knee->hip, knee->ankle).",
        coordinate_convention="0 deg = fully extended anatomical reference, positive degrees indicate increasing flexion.",
        requires_depth_or_calibration=False,
    ),
    ParameterSpec(
        param_id=3,
        name="knee_flexion_right",
        display_name="Right Knee Flexion",
        unit="deg",
        classification="estimate",
        level=1,
        description="Right knee sagittal flexion angle.",
        formula="180 - angle(knee->hip, knee->ankle).",
        coordinate_convention="0 deg = fully extended anatomical reference, positive degrees indicate increasing flexion.",
        requires_depth_or_calibration=False,
    ),
    ParameterSpec(
        param_id=4,
        name="hip_flexion_left",
        display_name="Left Hip Flexion",
        unit="deg",
        classification="estimate",
        level=1,
        description="Left hip sagittal angle between thigh segment and trunk midline.",
        formula="angle(hip->knee vector, mid_shoulder->mid_hip trunk vector).",
        coordinate_convention="Angle from trunk axis in degrees; 0 deg = neutral standing.",
        requires_depth_or_calibration=False,
    ),
    ParameterSpec(
        param_id=5,
        name="hip_flexion_right",
        display_name="Right Hip Flexion",
        unit="deg",
        classification="estimate",
        level=1,
        description="Right hip sagittal angle between thigh segment and trunk midline.",
        formula="angle(hip->knee vector, mid_shoulder->mid_hip trunk vector).",
        coordinate_convention="Angle from trunk axis in degrees; 0 deg = neutral standing.",
        requires_depth_or_calibration=False,
    ),
    ParameterSpec(
        param_id=6,
        name="hip_abduction_left",
        display_name="Left Hip Abduction",
        unit="deg",
        classification="proxy",
        level=1,
        description="Frontal plane angle of left thigh vector relative to gravity vertical.",
        formula="angle(hip->knee vector, [0, 1] vertical unit vector).",
        coordinate_convention="Frontal plane deviation in degrees relative to vertical.",
        requires_depth_or_calibration=False,
    ),
    ParameterSpec(
        param_id=7,
        name="hip_abduction_right",
        display_name="Right Hip Abduction",
        unit="deg",
        classification="proxy",
        level=1,
        description="Frontal plane angle of right thigh vector relative to gravity vertical.",
        formula="angle(hip->knee vector, [0, 1] vertical unit vector).",
        coordinate_convention="Frontal plane deviation in degrees relative to vertical.",
        requires_depth_or_calibration=False,
    ),
    ParameterSpec(
        param_id=8,
        name="ankle_dorsiflexion_left",
        display_name="Left Ankle Dorsiflexion",
        unit="deg",
        classification="estimate",
        level=1,
        description="Sagittal plane left ankle dorsiflexion/plantarflexion angle.",
        formula="90 - angle(ankle->knee shank vector, ankle->foot_index vector).",
        coordinate_convention="0 deg = neutral 90 deg included angle; positive = dorsiflexion, negative = plantarflexion.",
        requires_depth_or_calibration=False,
    ),
    ParameterSpec(
        param_id=9,
        name="ankle_dorsiflexion_right",
        display_name="Right Ankle Dorsiflexion",
        unit="deg",
        classification="estimate",
        level=1,
        description="Sagittal plane right ankle dorsiflexion/plantarflexion angle.",
        formula="90 - angle(ankle->knee shank vector, ankle->foot_index vector).",
        coordinate_convention="0 deg = neutral 90 deg included angle; positive = dorsiflexion, negative = plantarflexion.",
        requires_depth_or_calibration=False,
    ),
    ParameterSpec(
        param_id=10,
        name="trunk_lean",
        display_name="Trunk Lateral Lean",
        unit="deg",
        classification="proxy",
        level=1,
        description="Lateral inclination of trunk segment in the frontal plane from vertical.",
        formula="angle(mid_shoulder->mid_hip trunk vector, [0, 1] vertical vector).",
        coordinate_convention="0 deg = vertical spine; positive = lateral angular deviation from vertical.",
        requires_depth_or_calibration=False,
    ),
    ParameterSpec(
        param_id=11,
        name="trunk_flexion_deg",
        display_name="Trunk Sagittal Flexion",
        unit="deg",
        classification="estimate",
        level=1,
        description="Sagittal inclination of trunk segment relative to vertical axis.",
        formula="angle(trunk midline, vertical axis in sagittal projection).",
        coordinate_convention="0 deg = upright torso; positive = forward sagittal trunk flexion.",
        requires_depth_or_calibration=False,
    ),
    ParameterSpec(
        param_id=12,
        name="pelvic_obliquity_deg",
        display_name="Pelvic Obliquity / Drop",
        unit="deg",
        classification="proxy",
        level=1,
        description="Angle of inter-hip line relative to horizontal reference.",
        formula="atan2(right_hip_y - left_hip_y, right_hip_x - left_hip_x) in degrees.",
        coordinate_convention="0 deg = level horizontal pelvis; positive = left hip higher / contralateral drop.",
        requires_depth_or_calibration=False,
    ),
    ParameterSpec(
        param_id=13,
        name="knee_valgus_LSI",
        display_name="Limb Symmetry Index, Knee Valgus",
        unit="%",
        classification="model-derived",
        level=1,
        description="Normalized Limb Symmetry Index (LSI) comparing left and right knee valgus angles.",
        formula="2 * |fppa_left - fppa_right| / (|fppa_left| + |fppa_right|) * 100%.",
        coordinate_convention="Percentage [0, 200%]; 0% = perfect bilateral symmetry; >15% indicates significant asymmetry.",
        requires_depth_or_calibration=False,
    ),
    ParameterSpec(
        param_id=14,
        name="knee_flexion_LSI",
        display_name="Limb Symmetry Index, Knee Flexion",
        unit="%",
        classification="model-derived",
        level=1,
        description="Normalized Limb Symmetry Index (LSI) comparing left and right knee flexion.",
        formula="2 * |knee_flexion_left - knee_flexion_right| / (|knee_flexion_left| + |knee_flexion_right|) * 100%.",
        coordinate_convention="Percentage [0, 200%]; 0% = perfect bilateral symmetry.",
        requires_depth_or_calibration=False,
    ),
    ParameterSpec(
        param_id=15,
        name="knee_valgus_at_initial_contact",
        display_name="Knee Valgus Angle at Initial Contact",
        unit="deg",
        classification="proxy",
        level=1,
        description="Peak frontal plane projection knee valgus at ground impact instant.",
        formula="max(fppa_left, fppa_right) sampled at landing initial contact frame.",
        coordinate_convention="Frontal plane angle in degrees at contact instant.",
        requires_depth_or_calibration=False,
    ),
    ParameterSpec(
        param_id=16,
        name="knee_flexion_at_initial_contact",
        display_name="Knee Flexion Angle at Initial Contact",
        unit="deg",
        classification="estimate",
        level=1,
        description="Bilateral mean knee flexion angle at ground impact instant.",
        formula="mean(knee_flexion_left, knee_flexion_right) sampled at initial contact frame.",
        coordinate_convention="Degrees; <25 deg indicates stiff, high-impact landing strategy.",
        requires_depth_or_calibration=False,
    ),
    ParameterSpec(
        param_id=17,
        name="valgus_duration",
        display_name="Duration of Valgus Collapse",
        unit="s",
        classification="measured",
        level=1,
        description="Total duration during landing contact window where knee valgus exceeds pathological 10 deg threshold.",
        formula="sum(frames where |FPPA| > 10.0 deg) * (1 / fps).",
        coordinate_convention="Time in seconds.",
        requires_depth_or_calibration=False,
    ),
    ParameterSpec(
        param_id=18,
        name="com_displacement",
        display_name="Center of Mass Displacement",
        unit="norm_units",
        classification="model-derived",
        level=1,
        description="Cumulative spatial trajectory displacement of Dempster segmental Center of Mass.",
        formula="sum(sqrt(diff(com_x)^2 + diff(com_y)^2)).",
        coordinate_convention="Cumulative normalized units across trial duration.",
        requires_depth_or_calibration=False,
    ),
    ParameterSpec(
        param_id=19,
        name="knee_ap_translation_mm",
        display_name="Level 2: AP Knee Translation",
        unit="mm",
        classification="estimate",
        level=2,
        description="6-DOF anterior-posterior tibial translation relative to femur.",
        formula="Physical depth or stereo calibrated sagittal translation. Strictly gated: returns 'not available' on monocular RGB video without depth.",
        coordinate_convention="Millimeters; anterior (+) / posterior (-).",
        requires_depth_or_calibration=True,
    ),
    ParameterSpec(
        param_id=20,
        name="knee_ml_translation_mm",
        display_name="Level 2: ML Knee Translation",
        unit="mm",
        classification="estimate",
        level=2,
        description="6-DOF medial-lateral tibial translation relative to femur.",
        formula="Physical depth or stereo calibrated frontal translation. Strictly gated: returns 'not available' on monocular RGB video without depth.",
        coordinate_convention="Millimeters; medial (+) / lateral (-).",
        requires_depth_or_calibration=True,
    ),
    ParameterSpec(
        param_id=21,
        name="knee_si_translation_mm",
        display_name="Level 2: SI Knee Translation",
        unit="mm",
        classification="estimate",
        level=2,
        description="6-DOF superior-inferior joint space translation / compression.",
        formula="Physical depth or stereo calibrated axial translation. Strictly gated: returns 'not available' on monocular RGB video without depth.",
        coordinate_convention="Millimeters; distraction (+) / compression (-).",
        requires_depth_or_calibration=True,
    ),
    ParameterSpec(
        param_id=22,
        name="knee_int_ext_rot_deg",
        display_name="Level 2: Internal/External Knee Rotation",
        unit="deg",
        classification="proxy",
        level=2,
        description="6-DOF transverse axial rotation between tibia and femur.",
        formula="Physical depth or stereo calibrated transverse rotation. Strictly gated: returns 'not available' on monocular RGB video without depth.",
        coordinate_convention="Degrees; internal rotation (+) / external rotation (-).",
        requires_depth_or_calibration=True,
    ),
]

PARAMETERS_22_MAP = {spec.name: spec for spec in PARAMETERS_22_SPECS}


def get_parameter_specs() -> list[dict[str, Any]]:
    """Returns documentation and metadata for all 22 parameters."""
    return [
        {
            "param_id": s.param_id,
            "name": s.name,
            "display_name": s.display_name,
            "unit": s.unit,
            "classification": s.classification,
            "level": s.level,
            "description": s.description,
            "formula": s.formula,
            "coordinate_convention": s.coordinate_convention,
            "requires_depth_or_calibration": s.requires_depth_or_calibration,
        }
        for s in PARAMETERS_22_SPECS
    ]


def compute_22_parameters(
    df: pd.DataFrame,
    events: Optional[list[Any]] = None,
    has_depth_input: bool = False,
    has_stereo_calibration: bool = False,
    fps: float = 30.0,
) -> dict[str, dict[str, Any]]:
    """
    Computes summary values and complete metadata provenance for all 22 biomechanical parameters.

    Returns:
        Dict mapping parameter_name -> parameter result dict containing:
          - param_id (1-22)
          - name
          - display_name
          - value: float or "not available"
          - status: "available" or "not available"
          - unit
          - classification: 'measured' | 'estimate' | 'proxy' | 'model-derived'
          - level: 1 or 2
          - description
          - formula
          - coordinate_convention
          - notes
    """
    results: dict[str, dict[str, Any]] = {}
    is_3d_available = has_depth_input or has_stereo_calibration

    # Helper to clean series mean safely without fabricated defaults
    def _clean_mean(col_name: str) -> Optional[float]:
        if col_name in df.columns:
            s = df[col_name].dropna()
            if not s.empty:
                val = float(s.mean())
                if not (math.isnan(val) or math.isinf(val)):
                    return round(val, 2)
        return None

    # Helper for LSI
    def _calc_lsi(left_val: Optional[float], right_val: Optional[float]) -> Optional[float]:
        if left_val is None or right_val is None:
            return None
        denom = abs(left_val) + abs(right_val)
        if denom < 1e-6:
            return 0.0
        return round((2.0 * abs(left_val - right_val) / denom) * 100.0, 2)

    # 1. fppa_right
    fppa_r = _clean_mean("fppa_right")
    # 2. knee_flexion_left
    flex_l = _clean_mean("knee_flexion_left")
    # 3. knee_flexion_right
    flex_r = _clean_mean("knee_flexion_right")
    # 4. hip_flexion_left
    hip_fl_l = _clean_mean("hip_flexion_left")
    # 5. hip_flexion_right
    hip_fl_r = _clean_mean("hip_flexion_right")
    # 6. hip_abduction_left
    hip_ab_l = _clean_mean("hip_abduction_left")
    # 7. hip_abduction_right
    hip_ab_r = _clean_mean("hip_abduction_right")
    # 8. ankle_dorsiflexion_left
    ank_df_l = _clean_mean("ankle_dorsiflexion_left")
    # 9. ankle_dorsiflexion_right
    ank_df_r = _clean_mean("ankle_dorsiflexion_right")
    # 10. trunk_lean
    t_lean = _clean_mean("trunk_lean")
    # 11. trunk_flexion_deg
    t_flex = _clean_mean("trunk_flexion_deg") or t_lean
    # 12. pelvic_obliquity_deg
    p_obliq = _clean_mean("pelvic_obliquity_deg")
    # 13. knee_valgus_LSI
    fppa_l = _clean_mean("fppa_left")
    valgus_lsi = _calc_lsi(fppa_l, fppa_r)
    # 14. knee_flexion_LSI
    flex_lsi = _calc_lsi(flex_l, flex_r)

    # 15. knee_valgus_at_initial_contact & 16. knee_flexion_at_initial_contact & 17. valgus_duration
    valgus_ic: Optional[float] = None
    flex_ic: Optional[float] = None
    valg_dur: Optional[float] = None

    if events and len(events) > 0:
        ev0 = events[0]
        valgus_ic = getattr(ev0, "knee_valgus_at_initial_contact", None)
        flex_ic = getattr(ev0, "knee_flexion_at_initial_contact", None)
        valg_dur = getattr(ev0, "valgus_duration", None)

    # Fallback from DataFrame if events not yet segmented
    if valgus_ic is None and fppa_l is not None:
        valgus_ic = max(abs(fppa_l or 0.0), abs(fppa_r or 0.0))
    if flex_ic is None and flex_l is not None:
        flex_ic = round(((flex_l or 0.0) + (flex_r or 0.0)) / 2.0, 2)
    if valg_dur is None:
        if "fppa_left" in df.columns:
            dt = 1.0 / (fps or 30.0)
            valg_frames = np.sum((df["fppa_left"].abs() > 10.0) | (df.get("fppa_right", pd.Series(0)).abs() > 10.0))
            valg_dur = round(float(valg_frames * dt), 3)
        else:
            valg_dur = 0.0

    # 18. com_displacement
    com_disp: Optional[float] = None
    if "com_total_displacement" in df.columns and not df["com_total_displacement"].dropna().empty:
        com_disp = round(float(df["com_total_displacement"].dropna().iloc[-1]), 4)
    elif "com_x" in df.columns and "com_y" in df.columns:
        cx = df["com_x"].dropna().to_numpy()
        cy = df["com_y"].dropna().to_numpy()
        if len(cx) > 1 and len(cy) > 1:
            com_disp = round(float(np.sum(np.sqrt(np.diff(cx)**2 + np.diff(cy)**2))), 4)

    # Value mapping for parameters 1-18
    raw_values = {
        "fppa_right": fppa_r,
        "knee_flexion_left": flex_l,
        "knee_flexion_right": flex_r,
        "hip_flexion_left": hip_fl_l,
        "hip_flexion_right": hip_fl_r,
        "hip_abduction_left": hip_ab_l,
        "hip_abduction_right": hip_ab_r,
        "ankle_dorsiflexion_left": ank_df_l,
        "ankle_dorsiflexion_right": ank_df_r,
        "trunk_lean": t_lean,
        "trunk_flexion_deg": t_flex,
        "pelvic_obliquity_deg": p_obliq,
        "knee_valgus_LSI": valgus_lsi,
        "knee_flexion_LSI": flex_lsi,
        "knee_valgus_at_initial_contact": round(valgus_ic, 2) if valgus_ic is not None else None,
        "knee_flexion_at_initial_contact": round(flex_ic, 2) if flex_ic is not None else None,
        "valgus_duration": round(valg_dur, 3) if valg_dur is not None else None,
        "com_displacement": com_disp,
    }

    for spec in PARAMETERS_22_SPECS:
        if spec.level == 2 and spec.requires_depth_or_calibration:
            if not is_3d_available:
                # Level 2 Gated Condition: return "not available"
                results[spec.name] = {
                    "param_id": spec.param_id,
                    "name": spec.name,
                    "display_name": spec.display_name,
                    "value": "not available",
                    "status": "not available",
                    "unit": spec.unit,
                    "classification": spec.classification,
                    "level": spec.level,
                    "formula": spec.formula,
                    "coordinate_convention": spec.coordinate_convention,
                    "description": spec.description,
                    "notes": "Requires depth sensor or multi-camera stereo calibration. Monocular RGB video does not provide metric depth.",
                }
            else:
                # Available with calibrated depth / stereo inputs
                val = _clean_mean(spec.name)
                results[spec.name] = {
                    "param_id": spec.param_id,
                    "name": spec.name,
                    "display_name": spec.display_name,
                    "value": val if val is not None else "not available",
                    "status": "available" if val is not None else "not available",
                    "unit": spec.unit,
                    "classification": spec.classification,
                    "level": spec.level,
                    "formula": spec.formula,
                    "coordinate_convention": spec.coordinate_convention,
                    "description": spec.description,
                    "notes": "Derived from calibrated depth/stereo sensor input.",
                }
        else:
            val = raw_values.get(spec.name)
            is_valid = val is not None and not (isinstance(val, float) and (math.isnan(val) or math.isinf(val)))
            results[spec.name] = {
                "param_id": spec.param_id,
                "name": spec.name,
                "display_name": spec.display_name,
                "value": val if is_valid else "not available",
                "status": "available" if is_valid else "not available",
                "unit": spec.unit,
                "classification": spec.classification,
                "level": spec.level,
                "formula": spec.formula,
                "coordinate_convention": spec.coordinate_convention,
                "description": spec.description,
                "notes": "Computed from monocular pose stream." if is_valid else "Landmark tracking below minimum confidence threshold.",
            }

    return results
