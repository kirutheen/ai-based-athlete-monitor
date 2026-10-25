"""
Real Multi-Player Video Processing Engine for DriftGuard.

Reads video frames or live camera feed, tracks multiple players with persistent tracker_ids,
resolves jersey OCR, calculates independent per-player biomechanics, detects pairwise collisions,
and generates multi-player telemetry results.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from src.capture.frame_packet import FramePacket, FrameSourceType
from src.features.collision_detector import CollisionDetector
from src.features.player_state import PlayerState
from src.pose.jersey_ocr import JerseyOCREngine
from src.pose.multi_player_overlay import MultiPlayerOverlayRenderer
from src.pose.multi_player_tracker import MultiPlayerTracker
from src.reports.csv_generator import generate_session_csv_package


def process_multiplayer_video(
    video_path: str,
    session_id: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process an uploaded MP4 video or live camera stream with the multi-player engine.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    tracker = MultiPlayerTracker()
    ocr_engine = JerseyOCREngine()
    collision_detector = CollisionDetector()
    overlay_renderer = MultiPlayerOverlayRenderer()

    all_collisions = []
    multiplayer_frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break

        timestamp_s = float(frame_idx / max(1.0, fps))

        # 1. Persistent Tracking Update with motion prediction & jersey re-identification
        active_players = tracker.update_frame(frame_bgr, timestamp_s)

        # 2. Jersey OCR Resolution
        for player in active_players:
            if player.bbox_history and player.jersey_number is None:
                jersey_cand = ocr_engine.process_player_ocr(player.tracker_id, frame_bgr, player.bbox_history[-1])
                if jersey_cand is not None:
                    player.update_identity(jersey_cand)

        # 3. Collision Detection
        collisions = collision_detector.evaluate_frame_collisions(active_players, timestamp_s, frame_idx)
        for col in collisions:
            all_collisions.append(col.to_dict())

        # 4. Record frame player data for multi-player skeleton visualization
        frame_player_data = []
        for p in active_players:
            bbox = p.bbox_history[-1] if p.bbox_history else (0.0, 0.0, 1.0, 1.0)
            lm = p.pose_history[-1] if p.pose_history else None
            skel = None
            if lm is not None:
                skel = {
                    "left_hip": [round(float(lm[23, 0]), 3), round(float(lm[23, 1]), 3), round(float(lm[23, 3]), 3)],
                    "left_knee": [round(float(lm[25, 0]), 3), round(float(lm[25, 1]), 3), round(float(lm[25, 3]), 3)],
                    "left_ankle": [round(float(lm[27, 0]), 3), round(float(lm[27, 1]), 3), round(float(lm[27, 3]), 3)],
                    "right_hip": [round(float(lm[24, 0]), 3), round(float(lm[24, 1]), 3), round(float(lm[24, 3]), 3)],
                    "right_knee": [round(float(lm[26, 0]), 3), round(float(lm[26, 1]), 3), round(float(lm[26, 3]), 3)],
                    "right_ankle": [round(float(lm[28, 0]), 3), round(float(lm[28, 1]), 3), round(float(lm[28, 3]), 3)],
                }
            latest_bio = p.biomechanics_history[-1] if p.biomechanics_history else {}
            frame_player_data.append({
                "tracker_id": p.tracker_id,
                "display_name": p.display_name,
                "jersey_number": p.jersey_number,
                "bbox": [round(float(v), 3) for v in bbox],
                "skeleton": skel,
                "fppa_left": float(latest_bio["fppa_left"]) if latest_bio.get("fppa_left") is not None else None,
                "knee_flexion_left": float(latest_bio["knee_flexion_left"]) if latest_bio.get("knee_flexion_left") is not None else None,
            })
        multiplayer_frames.append({
            "frame": frame_idx,
            "timestamp_s": round(timestamp_s, 3),
            "players": frame_player_data,
        })

        frame_idx += 1

    cap.release()

    all_players = list(tracker.players.values())
    player_summaries = [p.to_summary_dict() for p in all_players]

    # Export 8 CSV Files
    out_dir = output_dir or os.path.join(os.path.dirname(video_path), f"csv_{session_id}")
    csv_package = generate_session_csv_package(session_id, all_players, all_collisions, out_dir)

    return {
        "session_id": session_id,
        "source_video": os.path.basename(video_path),
        "total_frames": frame_idx,
        "fps": round(fps, 1),
        "duration_sec": round(frame_idx / max(1.0, fps), 2),
        "total_players_tracked": len(all_players),
        "active_players_count": len([p for p in all_players if p.active]),
        "total_collisions_detected": len(all_collisions),
        "players": player_summaries,
        "collisions": all_collisions,
        "csv_package": csv_package,
        "multiplayer_frames": multiplayer_frames,
    }
