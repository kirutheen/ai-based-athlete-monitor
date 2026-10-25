"""
Movement-level feature dataset generator (`movement_features.csv`).

Produces structured trial-level biomechanical summaries (1 row per complete movement/trial)
combining Level 1 features, selected Level 2 features, quality metrics, and baseline deviations.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from src.features.biomechanics import calculate_lsi
from src.features.event_segmentation import LandingEvent


def create_movement_feature_row(
    combined_df: pd.DataFrame,
    event: LandingEvent,
    subject_id: str = "subject_001",
    trial_id: str = "trial_001",
    movement_type: str = "drop_jump",
    baseline_medians: dict[str, float] | None = None,
    baseline_iqrs: dict[str, float] | None = None,
) -> dict:
    """Extract a single trial-level summary feature row for a LandingEvent."""
    window = combined_df[
        (combined_df["frame_index"] >= event.window_start_frame) &
        (combined_df["frame_index"] <= event.window_end_frame)
    ]
    if window.empty:
        window = combined_df

    # Knee Valgus / FPPA
    val_l = window["fppa_left"].abs() if "fppa_left" in window else pd.Series([np.nan])
    val_r = window["fppa_right"].abs() if "fppa_right" in window else pd.Series([np.nan])
    val_comb = pd.concat([val_l, val_r], axis=1).max(axis=1)

    peak_knee_valgus = float(val_comb.max()) if pd.notna(val_comb.max()) else float("nan")
    mean_knee_valgus = float(val_comb.mean()) if pd.notna(val_comb.mean()) else float("nan")
    knee_valgus_rom = float(val_comb.max() - val_comb.min()) if pd.notna(val_comb.max()) and pd.notna(val_comb.min()) else float("nan")

    # Knee Flexion
    flex_l = window["knee_flexion_left"] if "knee_flexion_left" in window else pd.Series([np.nan])
    flex_r = window["knee_flexion_right"] if "knee_flexion_right" in window else pd.Series([np.nan])
    flex_comb = pd.concat([flex_l, flex_r], axis=1).mean(axis=1)

    peak_knee_flexion = float(flex_comb.max()) if pd.notna(flex_comb.max()) else float("nan")
    minimum_knee_flexion = float(flex_comb.min()) if pd.notna(flex_comb.min()) else float("nan")
    knee_flexion_rom = float(flex_comb.max() - flex_comb.min()) if pd.notna(flex_comb.max()) and pd.notna(flex_comb.min()) else float("nan")

    # Hip Kinematics
    hip_fl = window[["hip_flexion_left", "hip_flexion_right"]].mean(axis=1) if {"hip_flexion_left", "hip_flexion_right"}.issubset(window.columns) else pd.Series([np.nan])
    hip_ab = window[["hip_abduction_left", "hip_abduction_right"]].mean(axis=1) if {"hip_abduction_left", "hip_abduction_right"}.issubset(window.columns) else pd.Series([np.nan])

    peak_hip_flexion = float(hip_fl.max()) if pd.notna(hip_fl.max()) else float("nan")
    peak_hip_abduction = float(hip_ab.max()) if pd.notna(hip_ab.max()) else float("nan")

    # Ankle Kinematics
    ank_df = window[["ankle_dorsiflexion_left", "ankle_dorsiflexion_right"]].mean(axis=1) if {"ankle_dorsiflexion_left", "ankle_dorsiflexion_right"}.issubset(window.columns) else pd.Series([np.nan])
    peak_ankle_dorsiflexion = float(ank_df.max()) if pd.notna(ank_df.max()) else float("nan")

    # Trunk & Pelvis
    trunk_fl = float(window["trunk_flexion_deg"].mean()) if "trunk_flexion_deg" in window else float("nan")
    trunk_lat = float(window["trunk_lateral_flexion_deg"].mean()) if "trunk_lateral_flexion_deg" in window else float("nan")
    trunk_rot = float(window["trunk_rotation_deg"].mean()) if "trunk_rotation_deg" in window else float("nan")

    p_tilt = float(window["pelvic_tilt_deg"].mean()) if "pelvic_tilt_deg" in window else float("nan")
    p_obliq = float(window["pelvic_obliquity_deg"].mean()) if "pelvic_obliquity_deg" in window else float("nan")
    p_rot = float(window["pelvic_rotation_deg"].mean()) if "pelvic_rotation_deg" in window else float("nan")

    # LSI
    knee_flex_lsi = calculate_lsi(float(flex_l.mean()), float(flex_r.mean())) if pd.notna(flex_l.mean()) and pd.notna(flex_r.mean()) else float("nan")
    knee_valg_lsi = calculate_lsi(float(val_l.mean()), float(val_r.mean())) if pd.notna(val_l.mean()) and pd.notna(val_r.mean()) else float("nan")

    # Baseline deviation & z-score
    base_dev = float("nan")
    base_z = float("nan")
    if baseline_medians and baseline_iqrs and "fppa_left" in baseline_medians:
        b_med = baseline_medians.get("fppa_left", float("nan"))
        b_iqr = baseline_iqrs.get("fppa_left", float("nan"))
        if pd.notna(b_med) and pd.notna(b_iqr) and b_iqr > 0:
            base_dev = peak_knee_valgus - b_med
            base_z = base_dev / b_iqr

    # COM
    com_disp = float(window["com_total_displacement"].iloc[-1]) if "com_total_displacement" in window and not window["com_total_displacement"].empty else float("nan")

    # Velocities & accelerations
    val_vel = float(window["knee_valgus_velocity"].abs().max()) if "knee_valgus_velocity" in window else float("nan")
    val_accel = float(window["knee_valgus_acceleration"].abs().max()) if "knee_valgus_acceleration" in window else float("nan")

    pose_q = float(window["pose_confidence"].mean()) if "pose_confidence" in window else 1.0
    occ_r = float(window["occlusion_ratio"].mean()) if "occlusion_ratio" in window else 0.0
    track_q = float(window["tracking_quality"].mean()) if "tracking_quality" in window else 1.0

    return {
        "subject_id": subject_id,
        "trial_id": trial_id,
        "movement_type": movement_type,
        "timestamp_start": float(window["timestamp"].iloc[0]) if "timestamp" in window and not window.empty else event.landing_timestamp,
        "timestamp_end": float(window["timestamp"].iloc[-1]) if "timestamp" in window and not window.empty else event.landing_timestamp,
        "movement_duration": event.movement_duration,
        "peak_knee_valgus": peak_knee_valgus,
        "mean_knee_valgus": mean_knee_valgus,
        "knee_valgus_rom": knee_valgus_rom,
        "knee_valgus_velocity": val_vel,
        "knee_valgus_acceleration": val_accel,
        "peak_knee_flexion": peak_knee_flexion,
        "minimum_knee_flexion": minimum_knee_flexion,
        "knee_flexion_rom": knee_flexion_rom,
        "peak_knee_rotation": float("nan"),
        "knee_rotation_rom": float("nan"),
        "peak_hip_flexion": peak_hip_flexion,
        "peak_hip_abduction": peak_hip_abduction,
        "peak_hip_rotation": float("nan"),
        "peak_ankle_dorsiflexion": peak_ankle_dorsiflexion,
        "trunk_flexion": trunk_fl,
        "trunk_lateral_flexion": trunk_lat,
        "trunk_rotation": trunk_rot,
        "pelvic_tilt": p_tilt,
        "pelvic_obliquity": p_obliq,
        "pelvic_rotation": p_rot,
        "knee_flexion_LSI": knee_flex_lsi,
        "knee_valgus_LSI": knee_valg_lsi,
        "knee_rotation_LSI": float("nan"),
        "knee_valgus_at_initial_contact": event.knee_valgus_at_initial_contact,
        "knee_flexion_at_initial_contact": event.knee_flexion_at_initial_contact,
        "hip_flexion_at_initial_contact": event.hip_flexion_at_initial_contact,
        "time_to_peak_valgus": event.time_to_peak_valgus_after_contact,
        "time_to_peak_flexion": event.time_to_peak_knee_flexion,
        "valgus_duration": event.valgus_duration,
        "com_displacement": com_disp,
        "pose_quality": pose_q,
        "occlusion_ratio": occ_r,
        "tracking_quality": track_q,
        "baseline_deviation": base_dev,
        "baseline_zscore": base_z,
        "persistence_count": 0,
        "flagged": False,
    }


def generate_movement_features_dataframe(
    combined_df: pd.DataFrame,
    events: list[LandingEvent],
    subject_id: str = "subject_001",
    trial_prefix: str = "trial",
    movement_type: str = "drop_jump",
    baseline_medians: dict[str, float] | None = None,
    baseline_iqrs: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Generate a DataFrame representing `movement_features.csv` (1 row per LandingEvent)."""
    rows = []
    for idx, e in enumerate(events):
        t_id = f"{trial_prefix}_{idx+1:03d}"
        row = create_movement_feature_row(
            combined_df=combined_df,
            event=e,
            subject_id=subject_id,
            trial_id=t_id,
            movement_type=movement_type,
            baseline_medians=baseline_medians,
            baseline_iqrs=baseline_iqrs,
        )
        rows.append(row)
    return pd.DataFrame(rows)
