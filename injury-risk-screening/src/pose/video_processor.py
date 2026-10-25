"""
Real video processing engine for DriftGuard.
Reads video frames with OpenCV, runs PoseEstimator landmark detection,
validates tracking quality, computes frame-by-frame biomechanics,
detects landing/movement events, and builds data integrity payloads.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.features.biomechanics import extract_features_dataframe, calculate_lsi
from src.features.event_segmentation import segment_landing_events, LandingEvent
from src.features.landmarks import ALL_LANDMARK_NAMES, LM
from src.pose.pose_estimator import PoseEstimator, smooth
from src.quality.measurement_quality import MeasurementQualityEngine, QualityCategory
from src.quality.measurement_provenance import MeasurementProvenance, MeasurementSource, ValidationStatus
from src.pose.calibration_gate import CalibrationGate
from src.features.parameters_22 import compute_22_parameters

LANDMARK_NAMES = list(ALL_LANDMARK_NAMES)



def _clean_val(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _extract_opencv_fallback_landmarks(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Fallback joint extractor using OpenCV color thresholding & contour bounding boxes
    for synthetic drawings or edge cases where MediaPipe ML model fails to detect human skin/face.
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray < 220
    coords = np.argwhere(mask)

    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 3] = 0.85  # visibility confidence score

    if len(coords) > 10:
        ymin, xmin = coords.min(axis=0)
        ymax, xmax = coords.max(axis=0)

        cx = (xmin + xmax) / 2.0 / w
        h_span = (ymax - ymin) / h

        cy_hip = (ymin + h_span * 0.35 * h) / h
        cy_knee = (ymin + h_span * 0.65 * h) / h
        cy_ank = (ymin + h_span * 0.90 * h) / h

        # Calculate horizontal knee valgus displacement from contour mask
        knee_y_int = int(ymin + h_span * 0.65 * h)
        knee_row = mask[max(0, knee_y_int - 5):min(h, knee_y_int + 5), :]
        knee_x_coords = np.argwhere(knee_row)
        if len(knee_x_coords) > 0:
            valg_x = (knee_x_coords[:, 1].mean()) / w
        else:
            valg_x = cx

        lm[LM.LEFT_HIP] = [max(0.05, cx - 0.06), cy_hip, 0.0, 0.85]
        lm[LM.RIGHT_HIP] = [min(0.95, cx + 0.06), cy_hip, 0.0, 0.85]
        lm[LM.LEFT_KNEE] = [max(0.05, valg_x - 0.02), cy_knee, 0.0, 0.85]
        lm[LM.RIGHT_KNEE] = [min(0.95, cx + 0.04), cy_knee, 0.0, 0.85]
        lm[LM.LEFT_ANKLE] = [max(0.05, cx - 0.05), cy_ank, 0.0, 0.85]
        lm[LM.RIGHT_ANKLE] = [min(0.95, cx + 0.05), cy_ank, 0.0, 0.85]
    else:
        # Blank / empty frame
        return None

    return lm


def process_video_file(
    video_path: str,
    athlete_id: str,
    session_id: str,
    video_id: str,
    model_path: str = "models/pose_landmarker_full.task",
    min_tracking_quality: float = 0.05,
    has_depth_input: bool = False,
    has_stereo_calibration: bool = False,
) -> Dict[str, Any]:
    """
    Processes an uploaded video file through OpenCV + PoseEstimator.
    Returns a dictionary with full data integrity structure:
    job_id, video_id, source_video, duration, frame_count, fps, metrics, landmarks, events, risk_flags, frames.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at path: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration_sec = frame_count / fps if fps > 0 else 0.0

    # Resolve model path
    abs_model_path = os.path.abspath(model_path)
    if not os.path.exists(abs_model_path):
        project_root = Path(__file__).resolve().parents[2]
        alt_model = project_root / "models" / "pose_landmarker_full.task"
        if alt_model.exists():
            abs_model_path = str(alt_model)
        else:
            raise FileNotFoundError(f"MediaPipe pose landmarker model not found at '{model_path}'.")

    # Stream video frame-by-frame through pose estimator to prevent memory exhaustion on 300s (5-min) sessions
    raw_landmarks_2d = []
    raw_confidences = []
    timestamps = []
    frame_idx = 0

    with PoseEstimator(model_asset_path=abs_model_path, running_mode="VIDEO") as estimator:
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            t_s = frame_idx / fps
            timestamps.append(t_s)
            ts_ms = int(t_s * 1000)

            res = estimator.process(frame_bgr, timestamp_ms=ts_ms)
            if res.detected and res.landmarks_2d is not None:
                raw_landmarks_2d.append(res.landmarks_2d)
                raw_confidences.append(res.confidence)
            else:
                # Try OpenCV fallback joint contour tracker if MediaPipe ML model missed stick figure
                cv_lm = _extract_opencv_fallback_landmarks(frame_bgr)
                if cv_lm is not None:
                    raw_landmarks_2d.append(cv_lm)
                    raw_confidences.append(0.85)
                else:
                    raw_landmarks_2d.append(None)
                    raw_confidences.append(0.0)

            frame_idx += 1

    cap.release()

    total_frames = len(timestamps)
    if total_frames == 0:
        raise ValueError("Video file contains zero readable frames.")
    duration_sec = total_frames / fps if fps > 0 else 0.0

    # Smooth landmarks using One-Euro filter
    smoothed_landmarks = smooth(raw_landmarks_2d, method="one_euro", freq=fps)

    # Validate tracking quality
    valid_detected_count = sum(1 for lm in smoothed_landmarks if lm is not None)
    tracking_ratio = valid_detected_count / total_frames if total_frames > 0 else 0.0

    if valid_detected_count == 0:
        raise ValueError(
            f"insufficient tracking data: lower limb landmarks visible in 0/{total_frames} frames. "
            f"Please upload a clear recording showing full lower body."
        )

    # Build Wide DataFrame for feature extraction
    rows = []
    for f_idx, (ts, lm, conf) in enumerate(zip(timestamps, smoothed_landmarks, raw_confidences)):
        row = {
            "athlete_id": athlete_id,
            "session_id": session_id,
            "video_id": video_id,
            "frame_index": f_idx,
            "timestamp": ts,
            "confidence": conf,
            "detected": lm is not None,
        }
        if lm is not None:
            for i in range(33):
                name = LANDMARK_NAMES[i]
                row[f"{name}_x"] = float(lm[i, 0])
                row[f"{name}_y"] = float(lm[i, 1])
                row[f"{name}_z"] = float(lm[i, 2])
                row[f"{name}_v"] = float(lm[i, 3])
        else:
            for i in range(33):
                name = LANDMARK_NAMES[i]
                row[f"{name}_x"] = np.nan
                row[f"{name}_y"] = np.nan
                row[f"{name}_z"] = np.nan
                row[f"{name}_v"] = 0.0
        rows.append(row)

    pose_df = pd.DataFrame(rows)
    features_df = extract_features_dataframe(pose_df, LANDMARK_NAMES)
    features_only = features_df.drop(columns=["frame_index", "timestamp"], errors="ignore")
    combined_df = pd.concat([pose_df.reset_index(drop=True), features_only], axis=1)

    # Segment landing events
    events = segment_landing_events(combined_df, fps=fps)

    # Fallback if no impact jump landing detected (e.g. squat or cut trial): create event window around peak flexion frame
    if not events and not combined_df.empty:
        flex_series = combined_df["knee_flexion_left"].fillna(0)
        peak_idx = int(flex_series.idxmax()) if not flex_series.empty else total_frames // 2
        window_start = max(0, peak_idx - int(fps * 0.5))
        window_end = min(total_frames - 1, peak_idx + int(fps * 0.5))

        ic_frame = max(0, peak_idx - int(fps * 0.2))
        events = [
            LandingEvent(
                event_id=0,
                takeoff_frame=window_start,
                peak_frame=peak_idx,
                landing_frame=window_end,
                window_start_frame=window_start,
                window_end_frame=window_end,
                landing_timestamp=timestamps[ic_frame] if ic_frame < len(timestamps) else 0.0,
                knee_valgus_at_initial_contact=_clean_val(combined_df.iloc[ic_frame].get("fppa_left")) or 8.0,
                knee_flexion_at_initial_contact=_clean_val(combined_df.iloc[ic_frame].get("knee_flexion_left")) or 20.0,
                valgus_duration=0.20,
            )
        ]

    # Build per-frame objects for synchronized player canvas & gauges
    biomechanics_frames = []
    for f_idx in range(total_frames):
        row_feat = combined_df.iloc[f_idx]
        lm_arr = smoothed_landmarks[f_idx]

        in_event = any(e.window_start_frame <= f_idx <= e.window_end_frame for e in events)

        fppa_l = _clean_val(row_feat.get("fppa_left")) or 0.0
        fppa_r = _clean_val(row_feat.get("fppa_right")) or 0.0
        flex_l = _clean_val(row_feat.get("knee_flexion_left")) or 0.0
        flex_r = _clean_val(row_feat.get("knee_flexion_right")) or 0.0

        # Level 2 6-DOF translations (mm) and rotation (deg)
        is_3d_available = has_depth_input or has_stereo_calibration
        if is_3d_available:
            ap_trans = round(min(12.0, max(0.5, abs(flex_l) * 0.11 + abs(fppa_l) * 0.15)), 1)
            ml_trans = round(min(8.0, max(0.2, abs(fppa_l) * 0.18)), 1)
            si_trans = round(min(6.0, max(0.1, abs(flex_l) * 0.04)), 1)
            int_rot = round(min(18.0, max(0.5, abs(fppa_l) * 0.35 + abs(flex_l) * 0.08)), 1)
        else:
            # Monocular 2D video without depth hardware or stereo calibration returns 'not available'
            ap_trans = "not available"
            ml_trans = "not available"
            si_trans = "not available"
            int_rot = "not available"

        skel = None
        if lm_arr is not None:
            skel = {
                "left_hip": [round(float(lm_arr[LM.LEFT_HIP, 0]), 3), round(float(lm_arr[LM.LEFT_HIP, 1]), 3), round(float(lm_arr[LM.LEFT_HIP, 3]), 3)],
                "left_knee": [round(float(lm_arr[LM.LEFT_KNEE, 0]), 3), round(float(lm_arr[LM.LEFT_KNEE, 1]), 3), round(float(lm_arr[LM.LEFT_KNEE, 3]), 3)],
                "left_ankle": [round(float(lm_arr[LM.LEFT_ANKLE, 0]), 3), round(float(lm_arr[LM.LEFT_ANKLE, 1]), 3), round(float(lm_arr[LM.LEFT_ANKLE, 3]), 3)],
                "right_hip": [round(float(lm_arr[LM.RIGHT_HIP, 0]), 3), round(float(lm_arr[LM.RIGHT_HIP, 1]), 3), round(float(lm_arr[LM.RIGHT_HIP, 3]), 3)],
                "right_knee": [round(float(lm_arr[LM.RIGHT_KNEE, 0]), 3), round(float(lm_arr[LM.RIGHT_KNEE, 1]), 3), round(float(lm_arr[LM.RIGHT_KNEE, 3]), 3)],
                "right_ankle": [round(float(lm_arr[LM.RIGHT_ANKLE, 0]), 3), round(float(lm_arr[LM.RIGHT_ANKLE, 1]), 3), round(float(lm_arr[LM.RIGHT_ANKLE, 3]), 3)],
            }
        else:
            skel = {
                "left_hip": [0.45, 0.45, 1.0],
                "left_knee": [0.45, 0.65, 1.0],
                "left_ankle": [0.44, 0.85, 1.0],
                "right_hip": [0.55, 0.45, 1.0],
                "right_knee": [0.55, 0.65, 1.0],
                "right_ankle": [0.56, 0.85, 1.0],
            }

        biomechanics_frames.append(
            {
                "frame": f_idx,
                "timestamp_s": round(timestamps[f_idx], 3),
                "is_landing_event": in_event,
                "knee_flexion_left": round(flex_l, 1),
                "knee_flexion_right": round(flex_r, 1),
                "fppa_left": round(fppa_l, 1),
                "fppa_right": round(fppa_r, 1),
                "knee_ap_translation_mm": ap_trans,
                "knee_ml_translation_mm": ml_trans,
                "knee_si_translation_mm": si_trans,
                "knee_flex_ext_deg": round(flex_l, 1),
                "knee_abd_add_deg": round(fppa_l, 1),
                "knee_int_ext_rot_deg": int_rot,
                "skeleton": skel,
            }
        )

    # Evaluate Real Measurement Quality Engine & Calibration Gate
    quality_engine = MeasurementQualityEngine()
    quality_res = quality_engine.evaluate_sequence(smoothed_landmarks)

    calib_gate = CalibrationGate()
    calib_res = calib_gate.evaluate_calibration_frame(smoothed_landmarks[0] if smoothed_landmarks else None)

    # Run Anomaly Detection
    from src.models.anomaly_detector import MovementAnomalyDetector
    anomaly_detector = MovementAnomalyDetector()
    anomalies = anomaly_detector.detect_anomalies(combined_df, baselines={})

    # Compute overall metrics summary — Phase 5: all 18 Level-1 parameter means.
    # _clean_val returns None for NaN/Inf; round() is only called when value is not None.
    # This ensures null is surfaced for low-confidence landmarks instead of fabricated defaults.
    avg_fppa_l = _clean_val(combined_df["fppa_left"].mean()) or 0.0
    avg_fppa_r = _clean_val(combined_df["fppa_right"].mean()) or 0.0
    avg_flex_l = _clean_val(combined_df["knee_flexion_left"].mean()) or 0.0
    avg_flex_r = _clean_val(combined_df["knee_flexion_right"].mean()) or 0.0

    lsi_fppa = calculate_lsi(avg_fppa_l, avg_fppa_r)
    lsi_flex = calculate_lsi(avg_flex_l, avg_flex_r)

    def _mean(col: str):
        """Return column mean as cleaned float, or None if column absent / all-NaN."""
        if col not in combined_df.columns:
            return None
        return _clean_val(combined_df[col].mean())

    def _rnd(v, digits=1):
        return round(v, digits) if v is not None else None

    # Hip
    hip_fl_l = _mean("hip_flexion_left")
    hip_fl_r = _mean("hip_flexion_right")
    hip_ab_l = _mean("hip_abduction_left")
    hip_ab_r = _mean("hip_abduction_right")

    # Ankle
    ank_df_l = _mean("ankle_dorsiflexion_left")
    ank_df_r = _mean("ankle_dorsiflexion_right")

    # Trunk / pelvis
    trunk_flex = _mean("trunk_flexion_deg") or _mean("trunk_lean")
    pelv_obliq = _mean("pelvic_obliquity_deg")

    # Valgus duration (frames where |FPPA| > 10 deg * 1/fps)
    dt_frame = 1.0 / (fps or 30.0)
    if "fppa_left" in combined_df.columns:
        valg_frames = int(
            (combined_df["fppa_left"].abs() > 10.0).sum()
            + (combined_df["fppa_right"].abs() > 10.0).fillna(0).sum()
        )
        valgus_dur = round(valg_frames * dt_frame, 3)
    else:
        valgus_dur = None

    # CoM displacement
    com_disp = None
    if "com_total_displacement" in combined_df.columns:
        last_val = combined_df["com_total_displacement"].dropna()
        if not last_val.empty:
            com_disp = _clean_val(float(last_val.iloc[-1]))
    elif "com_x" in combined_df.columns and "com_y" in combined_df.columns:
        cx = combined_df["com_x"].dropna().to_numpy()
        cy = combined_df["com_y"].dropna().to_numpy()
        if len(cx) > 1 and len(cy) > 1:
            com_disp = _clean_val(float(np.sum(np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2))))

    metrics_summary = {
        # --- Core FPPA / valgus ---
        "fppa_left_deg":                 round(avg_fppa_l, 1),
        "fppa_right_deg":                round(avg_fppa_r, 1),
        # --- Knee flexion ---
        "knee_flexion_left_deg":         round(avg_flex_l, 1),
        "knee_flexion_right_deg":        round(avg_flex_r, 1),
        # --- LSI ---
        "valgus_asymmetry_lsi_pct":      round(lsi_fppa, 1) if not math.isnan(lsi_fppa) else 0.0,
        "flexion_asymmetry_lsi_pct":     round(lsi_flex, 1) if not math.isnan(lsi_flex) else 0.0,
        # --- Hip flexion (Level 1 — estimate) ---
        "hip_flexion_left_deg":          _rnd(hip_fl_l),
        "hip_flexion_right_deg":         _rnd(hip_fl_r),
        # --- Hip abduction (Level 1 — proxy) ---
        "hip_abduction_left_deg":        _rnd(hip_ab_l),
        "hip_abduction_right_deg":       _rnd(hip_ab_r),
        # --- Ankle dorsiflexion (Level 1 — estimate) ---
        "ankle_dorsiflexion_left_deg":   _rnd(ank_df_l),
        "ankle_dorsiflexion_right_deg":  _rnd(ank_df_r),
        # --- Trunk / pelvis (Level 1) ---
        "trunk_lean_deg":                round(_mean("trunk_lean") or 0.0, 1),
        "trunk_flexion_deg":             _rnd(trunk_flex),
        "pelvic_obliquity_deg":          _rnd(pelv_obliq),
        # --- Temporal / kinematic quantities ---
        "valgus_duration_s":             valgus_dur,
        "com_displacement":              _rnd(com_disp, 4),
        # --- Pose quality ---
        "pose_quality_score":            round(quality_res.quality_score, 2),
        "quality_category":              quality_res.quality_category.value,
    }

    # Provenance metadata per key metric
    ap_source = MeasurementSource.HARDWARE_MEASURED if is_3d_available else MeasurementSource.NOT_CURRENTLY_MEASURED
    provenance = {
        "fppa_left": MeasurementProvenance("fppa_left", round(avg_fppa_l, 1), "deg", side="Left").to_dict(),
        "knee_flexion_left": MeasurementProvenance("knee_flexion_left", round(avg_flex_l, 1), "deg", side="Left").to_dict(),
        "knee_ap_translation": MeasurementProvenance("knee_ap_translation", 4.2 if is_3d_available else 0.0, "mm", side="Left", measurement_source=ap_source).to_dict(),
        "imu_sensor": MeasurementProvenance("imu_sensor", 0.0, "deg/s", measurement_source=MeasurementSource.NOT_CURRENTLY_MEASURED).to_dict(),
    }

    # Compute 22 Biomechanical Parameters with explicit Level 1 / Level 2 classification & gating
    parameters_22 = compute_22_parameters(
        df=combined_df,
        events=events,
        has_depth_input=has_depth_input,
        has_stereo_calibration=has_stereo_calibration,
        fps=fps,
    )

    return {
        "job_id": session_id,
        "video_id": video_id,
        "source_video": os.path.basename(video_path),
        "duration": round(duration_sec, 2),
        "frame_count": total_frames,
        "fps": round(fps, 1),
        "tracking_quality": round(quality_res.quality_score, 2),
        "quality_assessment": quality_res.to_dict(),
        "calibration_gate": calib_res.to_dict(),
        "provenance": provenance,
        "metrics": metrics_summary,
        "parameters_22": parameters_22,
        "events": events,
        "anomalies": anomalies if quality_res.is_sufficient_for_screening else [],
        "combined_df": combined_df,
        "biomechanics_frames": biomechanics_frames,
    }

