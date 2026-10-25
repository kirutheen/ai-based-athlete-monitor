"""
Multi-Athlete Video Processor for DriftGuard.

Architecture:
  PoseEstimator(num_poses=N)  →  per-frame list[PoseResult] (one per person)
  MultiPersonTracker.update()  →  list[TrackedPose] (stable track_id per person)
  per-athlete PlayerState       →  independent pose/biomechanics history
  biomechanics.py functions     →  real calculated angles per athlete, per frame

This module is entirely additive: it does NOT touch video_processor.py or
the existing single-athlete pipeline. It exposes one public function:

    process_multi_athlete_video(video_path, session_id, ...) -> dict

The returned dict is served by two new API endpoints in src/api/main.py:
  GET /sessions/{id}/multi-athlete
  GET /sessions/{id}/multi-athlete/athletes

Tracking limitations are documented honestly in src/capture/multi_person_tracker.py.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.capture.multi_person_tracker import (
    MultiPersonTracker,
    TRACK_STATUS_UNCERTAIN,
    TRACK_STATUS_ACTIVE,
    TRACK_STATUS_TEMPORARILY_LOST,
)
from src.features.biomechanics import extract_features_dataframe
from src.features.landmarks import ALL_LANDMARK_NAMES, LM
from src.features.player_state import PlayerState
from src.pose.pose_estimator import LandmarkOneEuroFilter, PoseEstimator

LANDMARK_NAMES = list(ALL_LANDMARK_NAMES)

# Deterministic color palette derived from track_id — ensures the same
# athlete always gets the same color within a session.
# Colors are (B, G, R) for OpenCV, then converted to "#rrggbb" for the frontend.
_TRACK_PALETTE_BGR = [
    (255,  80,  80),   # 0 → blue
    ( 80, 200,  80),   # 1 → green
    ( 80,  80, 255),   # 2 → red
    (255, 180,   0),   # 3 → cyan
    (  0, 160, 255),   # 4 → orange
    (200,   0, 200),   # 5 → magenta
    (  0, 220, 220),   # 6 → yellow-green
    (128, 128, 255),   # 7 → pink
]

def _track_color_hex(track_id: int) -> str:
    b, g, r = _TRACK_PALETTE_BGR[track_id % len(_TRACK_PALETTE_BGR)]
    return f"#{r:02x}{g:02x}{b:02x}"


def _clean_val(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _landmarks_to_row(lm: np.ndarray) -> Dict[str, float]:
    """Convert a (33,4) landmark array into the flat column dict expected by
    extract_features_dataframe. NaN is used for missing visibility."""
    row: Dict[str, Any] = {
        "athlete_id": "multi",
        "session_id": "multi",
        "video_id": "multi",
        "frame_index": 0,
        "timestamp": 0.0,
        "confidence": float(np.nanmean(lm[:, 3])),
        "detected": True,
    }
    for i in range(33):
        name = LANDMARK_NAMES[i]
        row[f"{name}_x"] = float(lm[i, 0])
        row[f"{name}_y"] = float(lm[i, 1])
        row[f"{name}_z"] = float(lm[i, 2])
        row[f"{name}_v"] = float(lm[i, 3])
    return row


def _compute_biomechanics_for_frame(lm: np.ndarray, fps: float) -> Dict[str, Optional[float]]:
    """Run the real biomechanics pipeline on a single (33,4) landmark array.
    Returns a dict with at minimum: fppa_left, fppa_right,
    knee_flexion_left, knee_flexion_right.  None for any value that cannot
    be reliably computed (low-visibility landmarks).
    """
    row = _landmarks_to_row(lm)
    df_single = pd.DataFrame([row])
    try:
        features_df = extract_features_dataframe(df_single, LANDMARK_NAMES)
        feat_row = features_df.iloc[0]
    except Exception:
        return {"fppa_left": None, "fppa_right": None,
                "knee_flexion_left": None, "knee_flexion_right": None}

    return {
        "fppa_left":          _clean_val(feat_row.get("fppa_left")),
        "fppa_right":         _clean_val(feat_row.get("fppa_right")),
        "knee_flexion_left":  _clean_val(feat_row.get("knee_flexion_left")),
        "knee_flexion_right": _clean_val(feat_row.get("knee_flexion_right")),
        "hip_flexion_left":   _clean_val(feat_row.get("hip_flexion_left")),
        "hip_flexion_right":  _clean_val(feat_row.get("hip_flexion_right")),
        "trunk_lean":         _clean_val(feat_row.get("trunk_lean")),
    }


def _skeleton_from_landmarks(lm: np.ndarray) -> Dict[str, List[float]]:
    """Build the 6-joint skeleton dict used by the frontend canvas renderer."""
    def _pt(idx: int) -> List[float]:
        return [round(float(lm[idx, 0]), 3),
                round(float(lm[idx, 1]), 3),
                round(float(lm[idx, 3]), 3)]
    return {
        "left_hip":    _pt(LM.LEFT_HIP),
        "left_knee":   _pt(LM.LEFT_KNEE),
        "left_ankle":  _pt(LM.LEFT_ANKLE),
        "right_hip":   _pt(LM.RIGHT_HIP),
        "right_knee":  _pt(LM.RIGHT_KNEE),
        "right_ankle": _pt(LM.RIGHT_ANKLE),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_multi_athlete_video(
    video_path: str,
    session_id: str,
    model_path: str = "models/pose_landmarker_full.task",
    max_athletes: int = 5,
) -> Dict[str, Any]:
    """Process a video file and return per-athlete tracking, landmarks, and
    biomechanics data.

    Parameters
    ----------
    video_path:
        Absolute or relative path to the MP4/AVI/MOV file.
    session_id:
        Session identifier used in the returned payload (not stored to DB).
    model_path:
        Path to the MediaPipe pose landmarker .task bundle.
    max_athletes:
        Upper bound on simultaneous detections. Passed as num_poses to
        PoseEstimator. Does NOT assume this many athletes are present —
        the tracker handles variable counts.

    Returns
    -------
    dict with keys:
        session_id, source_video, total_frames, fps, duration_sec,
        athletes (per-athlete summary list),
        frames (per-frame list with all athletes' data).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Resolve model path
    abs_model_path = os.path.abspath(model_path)
    if not os.path.exists(abs_model_path):
        project_root = Path(__file__).resolve().parents[2]
        alt = project_root / "models" / "pose_landmarker_full.task"
        if alt.exists():
            abs_model_path = str(alt)
        else:
            raise FileNotFoundError(
                f"MediaPipe model not found at '{model_path}'. "
                "Download from https://developers.google.com/mediapipe/solutions/vision/pose_landmarker"
            )

    # Per-athlete state: track_id → PlayerState
    athlete_states: Dict[int, PlayerState] = {}
    # Per-track One-Euro smoother — each athlete has fully independent smoothing state
    track_smoothers: Dict[int, LandmarkOneEuroFilter] = {}
    tracker = MultiPersonTracker(max_match_distance=0.20, max_missed_frames=15)

    multi_frames: List[Dict[str, Any]] = []
    frame_idx = 0

    with PoseEstimator(
        model_asset_path=abs_model_path,
        running_mode="VIDEO",
        num_poses=max_athletes,
    ) as estimator:
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            t_s = frame_idx / fps
            ts_ms = int(t_s * 1000)

            # 1. Multi-pose detection → list[PoseResult] (one per person)
            pose_results = estimator.process_multi(frame_bgr, timestamp_ms=ts_ms)

            # 2. Build detection tuples for the tracker: (landmarks_2d, confidence)
            #    Confidence gate is enforced inside tracker.update(); pass all here.
            detections = [
                (pr.landmarks_2d, pr.confidence)
                for pr in pose_results
                if pr.detected and pr.landmarks_2d is not None
            ]

            # 3. Tracker assigns stable IDs → list[TrackedPose] with track_status
            tracked = tracker.update(
                detections,
                frame_index=frame_idx,
                session_id=session_id,
            )

            # 4. Per-athlete state update + biomechanics
            frame_athletes: List[Dict[str, Any]] = []
            for tp in tracked:
                tid = tp.track_id
                raw_lm = tp.landmarks_2d
                is_uncertain = (tp.track_status == TRACK_STATUS_UNCERTAIN)

                # Per-track One-Euro smoother (independent state per athlete)
                if tid not in track_smoothers:
                    track_smoothers[tid] = LandmarkOneEuroFilter(freq=fps)
                lm = track_smoothers[tid](raw_lm)

                # Create PlayerState on first sight
                if tid not in athlete_states:
                    ps = PlayerState(tracker_id=tid, first_seen=t_s)
                    athlete_states[tid] = ps
                else:
                    ps = athlete_states[tid]

                # Biomechanics: suppress (return None) when identity is uncertain.
                # Never fabricate angles for an uncertain track — flag explicitly.
                if is_uncertain:
                    bio: Dict[str, Optional[float]] = {
                        "fppa_left": None, "fppa_right": None,
                        "knee_flexion_left": None, "knee_flexion_right": None,
                        "hip_flexion_left": None, "hip_flexion_right": None,
                        "trunk_lean": None,
                    }
                else:
                    bio = _compute_biomechanics_for_frame(lm, fps)

                # Bbox comes from tracker (computed from visible landmarks)
                bbox: Tuple[float, float, float, float] = tp.landmarks_2d is not None and (
                    lambda lm_: (
                        float(lm_[lm_[:, 3] > 0.3, 0].min()) if (lm_[:, 3] > 0.3).any() else 0.0,
                        float(lm_[lm_[:, 3] > 0.3, 1].min()) if (lm_[:, 3] > 0.3).any() else 0.0,
                        float(lm_[lm_[:, 3] > 0.3, 0].max()) if (lm_[:, 3] > 0.3).any() else 1.0,
                        float(lm_[lm_[:, 3] > 0.3, 1].max()) if (lm_[:, 3] > 0.3).any() else 1.0,
                    )
                )(raw_lm) or (0.0, 0.0, 1.0, 1.0)

                ps.add_frame_data(
                    timestamp_s=t_s,
                    bbox=bbox,
                    landmarks=lm,
                    biomechanics=bio,
                )

                frame_athletes.append({
                    "track_id":           tid,
                    "session_id":         session_id,
                    "frame_index":        frame_idx,
                    "timestamp_s":        round(t_s, 3),
                    "color":              _track_color_hex(tid),
                    "display_name":       ps.display_name,
                    "confidence":         round(tp.confidence, 3),
                    "track_status":       tp.track_status,
                    "uncertainty_reason": tp.uncertainty_reason,
                    "bbox":               [round(v, 3) for v in bbox],
                    "skeleton":           _skeleton_from_landmarks(lm),
                    # Biomechanics: None when uncertain (never fabricated)
                    "fppa_left":          bio.get("fppa_left"),
                    "fppa_right":         bio.get("fppa_right"),
                    "knee_flexion_left":  bio.get("knee_flexion_left"),
                    "knee_flexion_right": bio.get("knee_flexion_right"),
                    "biomechanics_valid": not is_uncertain,
                })

            multi_frames.append({
                "frame":       frame_idx,
                "timestamp_s": round(t_s, 3),
                "athletes":    frame_athletes,
            })

            frame_idx += 1

    cap.release()

    total_frames = frame_idx
    duration_sec = total_frames / fps if fps > 0 else 0.0

    # -----------------------------------------------------------------------
    # Build per-athlete summaries
    # -----------------------------------------------------------------------
    athlete_summaries: List[Dict[str, Any]] = []
    for tid, ps in sorted(athlete_states.items()):
        # Aggregate biomechanics -- only from frames where identity was NOT uncertain
        fppa_vals = [b["fppa_left"] for b in ps.biomechanics_history
                     if b.get("fppa_left") is not None]
        flex_vals = [b["knee_flexion_left"] for b in ps.biomechanics_history
                     if b.get("knee_flexion_left") is not None]

        avg_fppa   = round(float(np.mean(fppa_vals)), 1) if fppa_vals else None
        avg_flex   = round(float(np.mean(flex_vals)), 1) if flex_vals else None
        peak_fppa  = round(float(np.max(np.abs(fppa_vals))), 1) if fppa_vals else None
        peak_flex  = round(float(np.max(flex_vals)), 1) if flex_vals else None

        frames_seen = len(ps.biomechanics_history)
        uncertain_count = sum(
            1 for b in ps.biomechanics_history
            if b.get("fppa_left") is None and b.get("knee_flexion_left") is None
        )
        coverage_pct = round(100.0 * frames_seen / max(1, total_frames), 1)
        uncertainty_pct = round(100.0 * uncertain_count / max(1, frames_seen), 1)
        bio_valid_pct = round(100.0 - uncertainty_pct, 1)

        athlete_summaries.append({
            "track_id":               tid,
            "session_id":             session_id,
            "display_name":           ps.display_name,
            "color":                  _track_color_hex(tid),
            "first_seen_s":           round(ps.first_seen, 3),
            "last_seen_s":            round(ps.last_seen, 3),
            "frames_tracked":         frames_seen,
            "coverage_pct":           coverage_pct,
            "uncertainty_pct":        uncertainty_pct,
            "biomechanics_valid_pct": bio_valid_pct,
            "avg_fppa_left":          avg_fppa,
            "avg_knee_flex_left":     avg_flex,
            "peak_fppa_left":         peak_fppa,
            "peak_knee_flex_left":    peak_flex,
        })

    return {
        "session_id":    session_id,
        "source_video":  os.path.basename(video_path),
        "total_frames":  total_frames,
        "fps":           round(fps, 1),
        "duration_sec":  round(duration_sec, 2),
        "max_athletes":  max_athletes,
        "athletes":      athlete_summaries,
        "frames":        multi_frames,
    }
