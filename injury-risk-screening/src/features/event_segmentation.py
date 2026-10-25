"""
Event segmentation: turn a continuous per-frame feature stream into
discrete landing events.

Segmenting first before scoring risk makes baseline calculations robust and
allows athlete-ID-based train/test splitting.

Detection approach: a landing event is identified from the vertical (y) trajectory
of the hip midpoint. Landing instant (initial contact) is identified as the first frame
where downward velocity crosses through zero as height returns near baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _safe_nanmean(vals: list[float]) -> float:
    valid = [v for v in vals if not math.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def _safe_nanmax(vals: list[float]) -> float:
    valid = [v for v in vals if not math.isnan(v)]
    return float(np.max(valid)) if valid else float("nan")


@dataclass
class LandingEvent:
    event_id: int
    takeoff_frame: int
    peak_frame: int  # highest point of the jump (lowest y)
    landing_frame: int  # estimated ground-contact frame
    landing_timestamp: float
    window_start_frame: int  # a few frames before landing
    window_end_frame: int  # a few frames after landing

    # Level 1 Initial Contact & Landing features
    initial_contact_frame: int | None = None
    initial_contact_time: float = float("nan")
    knee_flexion_at_initial_contact: float = float("nan")
    knee_valgus_at_initial_contact: float = float("nan")
    hip_flexion_at_initial_contact: float = float("nan")
    hip_abduction_at_initial_contact: float = float("nan")
    ankle_angle_at_initial_contact: float = float("nan")
    trunk_position_at_initial_contact: float = float("nan")

    # Post-contact metrics
    peak_knee_valgus_after_contact: float = float("nan")
    peak_knee_flexion_after_contact: float = float("nan")
    peak_trunk_lateral_lean_after_contact: float = float("nan")
    peak_hip_rotation_after_contact: float = float("nan")
    time_to_peak_valgus_after_contact: float = float("nan")

    # Movement timing metrics
    movement_duration: float = float("nan")
    time_to_peak_knee_valgus: float = float("nan")
    time_to_peak_knee_flexion: float = float("nan")
    time_to_peak_knee_rotation: float = float("nan")
    time_to_peak_hip_rotation: float = float("nan")
    time_to_peak_trunk_lean: float = float("nan")

    # Abnormal movement durations (seconds)
    valgus_duration: float = 0.0
    high_valgus_duration: float = 0.0
    low_knee_flexion_duration: float = 0.0
    excessive_rotation_duration: float = 0.0


def _hip_height_series(df: pd.DataFrame) -> pd.Series:
    """Mean of left/right hip y-coordinate per frame."""
    cols = [f"{side}_hip_y" for side in ("left", "right") if f"{side}_hip_y" in df.columns]
    if not cols:
        raise ValueError("DataFrame must contain left_hip_y and/or right_hip_y columns")
    return df[cols].mean(axis=1)


def segment_landing_events(
    df: pd.DataFrame,
    fps: float,
    min_jump_height_px: float = 0.03,
    min_event_separation_s: float = 1.0,
    pre_landing_s: float = 0.15,
    post_landing_s: float = 0.15,
) -> list[LandingEvent]:
    """Detect discrete jump-landing events and compute event-level landing & timing metrics."""
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column")

    hip_y = _hip_height_series(df).to_numpy()
    n = len(hip_y)
    if n < 5:
        return []

    smoothed = pd.Series(hip_y).rolling(window=3, center=True, min_periods=1).mean().to_numpy()
    baseline_height = np.nanmedian(smoothed)
    velocity = np.gradient(smoothed)

    events: list[LandingEvent] = []
    min_sep_frames = int(round(min_event_separation_s * fps))
    i = 1
    last_landing_frame = -min_sep_frames

    while i < n - 1:
        is_local_min = smoothed[i] < smoothed[i - 1] and smoothed[i] <= smoothed[i + 1]
        dips_enough = (baseline_height - smoothed[i]) >= min_jump_height_px
        if is_local_min and dips_enough:
            peak_frame = i

            takeoff_frame = peak_frame
            for j in range(peak_frame, 0, -1):
                if smoothed[j] >= baseline_height - min_jump_height_px * 0.25:
                    takeoff_frame = j
                    break

            landing_frame = None
            for j in range(peak_frame + 1, n - 1):
                returned_to_ground = smoothed[j] >= baseline_height - min_jump_height_px * 0.25
                decel = velocity[j] >= 0 and velocity[j + 1] <= velocity[j]
                if returned_to_ground and decel:
                    landing_frame = j
                    break

            if landing_frame is not None and (landing_frame - last_landing_frame) >= min_sep_frames:
                pre_frames = int(round(pre_landing_s * fps))
                post_frames = int(round(post_landing_s * fps))

                start_idx = max(0, landing_frame - pre_frames)
                end_idx = min(n - 1, landing_frame + post_frames)

                takeoff_f_idx = int(df.iloc[takeoff_frame]["frame_index"])
                peak_f_idx = int(df.iloc[peak_frame]["frame_index"])
                landing_f_idx = int(df.iloc[landing_frame]["frame_index"])
                landing_ts = float(df.iloc[landing_frame]["timestamp"])
                start_f_idx = int(df.iloc[start_idx]["frame_index"])
                end_f_idx = int(df.iloc[end_idx]["frame_index"])

                event = LandingEvent(
                    event_id=len(events),
                    takeoff_frame=takeoff_f_idx,
                    peak_frame=peak_f_idx,
                    landing_frame=landing_f_idx,
                    landing_timestamp=landing_ts,
                    window_start_frame=start_f_idx,
                    window_end_frame=end_f_idx,
                    initial_contact_frame=landing_f_idx,
                    initial_contact_time=landing_ts,
                )

                window_df = df.iloc[start_idx:end_idx + 1]
                t_window = window_df["timestamp"].to_numpy()
                duration = float(t_window[-1] - t_window[0]) if len(t_window) > 1 else (1.0 / fps)
                event.movement_duration = duration

                ic_row = df.iloc[landing_frame]
                val_l = float(ic_row.get("fppa_left", np.nan))
                val_r = float(ic_row.get("fppa_right", np.nan))
                event.knee_valgus_at_initial_contact = _safe_nanmax([val_l, val_r])

                flex_l = float(ic_row.get("knee_flexion_left", np.nan))
                flex_r = float(ic_row.get("knee_flexion_right", np.nan))
                event.knee_flexion_at_initial_contact = _safe_nanmean([flex_l, flex_r])

                hip_l = float(ic_row.get("hip_flexion_left", np.nan))
                hip_r = float(ic_row.get("hip_flexion_right", np.nan))
                event.hip_flexion_at_initial_contact = _safe_nanmean([hip_l, hip_r])

                hip_ab_l = float(ic_row.get("hip_abduction_left", np.nan))
                hip_ab_r = float(ic_row.get("hip_abduction_right", np.nan))
                event.hip_abduction_at_initial_contact = _safe_nanmean([hip_ab_l, hip_ab_r])

                ank_l = float(ic_row.get("ankle_dorsiflexion_left", np.nan))
                ank_r = float(ic_row.get("ankle_dorsiflexion_right", np.nan))
                event.ankle_angle_at_initial_contact = _safe_nanmean([ank_l, ank_r])

                event.trunk_position_at_initial_contact = float(ic_row.get("trunk_lean", np.nan))

                post_df = df.iloc[landing_frame:end_idx + 1]
                if not post_df.empty:
                    val_post_l = float(post_df["fppa_left"].max()) if "fppa_left" in post_df else float("nan")
                    val_post_r = float(post_df["fppa_right"].max()) if "fppa_right" in post_df else float("nan")
                    event.peak_knee_valgus_after_contact = _safe_nanmax([val_post_l, val_post_r])

                    flex_post_l = float(post_df["knee_flexion_left"].max()) if "knee_flexion_left" in post_df else float("nan")
                    flex_post_r = float(post_df["knee_flexion_right"].max()) if "knee_flexion_right" in post_df else float("nan")
                    event.peak_knee_flexion_after_contact = _safe_nanmax([flex_post_l, flex_post_r])

                    event.peak_trunk_lateral_lean_after_contact = float(post_df["trunk_lean"].max()) if "trunk_lean" in post_df else float("nan")

                    if "fppa_left" in post_df.columns or "fppa_right" in post_df.columns:
                        combined_val = post_df[["fppa_left", "fppa_right"]].max(axis=1) if {"fppa_left", "fppa_right"}.issubset(post_df.columns) else post_df.get("fppa_left", post_df.get("fppa_right"))
                        max_idx = combined_val.idxmax()
                        if pd.notna(max_idx):
                            peak_ts = float(post_df.loc[max_idx, "timestamp"])
                            event.time_to_peak_valgus_after_contact = max(0.0, float(peak_ts - landing_ts))

                dt_frame = 1.0 / fps
                if "fppa_left" in window_df.columns and "fppa_right" in window_df.columns:
                    val_series = window_df[["fppa_left", "fppa_right"]].abs().max(axis=1)
                    event.valgus_duration = float(np.sum(val_series > 10.0) * dt_frame)
                    event.high_valgus_duration = float(np.sum(val_series > 15.0) * dt_frame)

                if "knee_flexion_left" in window_df.columns and "knee_flexion_right" in window_df.columns:
                    flex_series = window_df[["knee_flexion_left", "knee_flexion_right"]].mean(axis=1)
                    event.low_knee_flexion_duration = float(np.sum(flex_series < 45.0) * dt_frame)

                events.append(event)
                last_landing_frame = landing_frame
                i = landing_frame + min_sep_frames
                continue
        i += 1

    return events


def events_to_dataframe(events: list[LandingEvent]) -> pd.DataFrame:
    """Flatten a list of LandingEvent into a DataFrame for storage/inspection."""
    return pd.DataFrame([vars(e) for e in events])
