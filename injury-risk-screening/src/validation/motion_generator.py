"""
Parametric reference knee-motion generator for validation.

============================= HONEST SCOPE =================================
These are SIMULATED, parametric motion profiles -- smooth mathematical
curves shaped to stay within commonly-reported clinical ranges of motion
for each movement type (e.g. knee flexion roughly 0-60 degrees during
normal-speed gait, deeper during sit-to-stand). They are built to
exercise the validation pipeline end-to-end (motion_generator ->
opensim_knee_model -> noise_model -> trajectory_metrics), NOT measured
from a real person and NOT a substitute for real motion-capture
reference data. The abstract itself frames this stage as "development
and validation preparation rather than real-world clinical validation"
-- this module is exactly that preparation stage, nothing more claimed.
==============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.opensim_knee_model import KneePose

MOVEMENT_TYPES = ("gait", "sit_to_stand", "stair_descent", "pivot")


def generate_reference_trajectory(movement_type: str, duration_s: float = 2.0, fps: float = 100.0) -> pd.DataFrame:
    """
    Generate one cycle of a simulated reference 6-DOF knee trajectory.

    Returns a DataFrame with columns: time_s, flexion_extension_deg,
    abduction_adduction_deg, internal_external_rotation_deg,
    ap_translation_mm, ml_translation_mm, si_translation_mm.
    """
    if movement_type not in MOVEMENT_TYPES:
        raise ValueError(f"movement_type must be one of {MOVEMENT_TYPES}")

    n = int(round(duration_s * fps))
    t = np.linspace(0, duration_s, n, endpoint=False)
    phase = t / duration_s  # 0..1 across the cycle

    if movement_type == "gait":
        # One gait cycle: ~5 deg at heel strike, peak ~60 deg swing flexion.
        flex = 5 + 55 * np.clip(np.sin(phase * 2 * np.pi - 0.3), 0, None) ** 1.5
        abd = 4 * np.sin(phase * 2 * np.pi) * np.exp(-((phase - 0.15) ** 2) / 0.02)
        rot = 6 * np.sin(phase * 2 * np.pi + 0.5)
    elif movement_type == "sit_to_stand":
        # Monotonic extension from deep seated flexion to standing.
        flex = 95 * (1 - phase) ** 1.2
        abd = 3 * np.sin(phase * np.pi)
        rot = 4 * np.sin(phase * np.pi * 1.5)
    elif movement_type == "stair_descent":
        flex = 20 + 60 * np.clip(np.sin(phase * 2 * np.pi), 0, None)
        abd = 7 * np.sin(phase * 2 * np.pi) ** 2 * np.sign(np.sin(phase * 2 * np.pi))
        rot = 8 * np.sin(phase * 2 * np.pi + 1.0)
    else:  # pivot
        flex = 25 + 10 * np.sin(phase * 2 * np.pi)
        abd = 10 * np.sin(phase * 2 * np.pi)
        rot = 15 * np.sin(phase * 2 * np.pi)

    # Small, smooth translational drift (mm), well within typical
    # reported tibiofemoral translation magnitudes during functional
    # movement (a few mm), used here purely to exercise the
    # translational-RMSE path of the metrics pipeline.
    ap = 4 * np.sin(phase * 2 * np.pi + 0.2)
    ml = 2 * np.sin(phase * 2 * np.pi + 0.8)
    si = 3 * np.cos(phase * 2 * np.pi)

    return pd.DataFrame(
        {
            "time_s": t,
            "flexion_extension_deg": flex,
            "abduction_adduction_deg": abd,
            "internal_external_rotation_deg": rot,
            "ap_translation_mm": ap,
            "ml_translation_mm": ml,
            "si_translation_mm": si,
        }
    )


def trajectory_row_to_pose(row: pd.Series) -> KneePose:
    return KneePose(
        flexion_extension_deg=float(row["flexion_extension_deg"]),
        abduction_adduction_deg=float(row["abduction_adduction_deg"]),
        internal_external_rotation_deg=float(row["internal_external_rotation_deg"]),
        ap_translation_mm=float(row["ap_translation_mm"]),
        ml_translation_mm=float(row["ml_translation_mm"]),
        si_translation_mm=float(row["si_translation_mm"]),
    )
