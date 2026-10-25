#!/usr/bin/env python
"""
Real-time webcam monitoring demo.

This is the live counterpart to scripts/run_pose_on_video.py +
scripts/process_session.py: instead of recording a video and processing
it afterward, this script opens your webcam, runs pose estimation on
each frame AS IT ARRIVES, draws the skeleton live on screen, detects
landing events on the fly (via src/features/online_event_detector.py,
which never buffers more than a short rolling window of frames -- it is
safe to run for an unlimited time, unlike feeding a webcam into
run_pose_on_video.py), and -- once an athlete has enough history -- shows
a live risk flag overlay the moment a landing is confirmed.

Requires: a webcam, mediapipe + opencv installed, and the MediaPipe
pose model downloaded (scripts/download_pose_model.py).

Usage
-----
    python scripts/live_webcam_demo.py --camera 0 --athlete-id ath_abc123
    python scripts/live_webcam_demo.py --camera 0 --athlete-name "New Athlete"

Press 'q' in the video window to quit.

What this does NOT do (same honesty as the rest of this project):
  - It does not prove any flag is a correct medical finding -- see
    docs/PRODUCT_BUILD_SPEC.md and the explanation this script prints
    for every flag.
  - It is built and tested for ONE athlete in frame at a time. Pointing
    it at a crowded multi-person scene will track whichever pose
    MediaPipe locks onto and can jump between people -- use
    src/capture/multi_person_tracker.py (and its own honest caveats) if
    you need more than one person, and even then, expect ID switches in
    a crowd. This script is for the structured, one-athlete-at-a-time
    drill setup the project has always been scoped to.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.biomechanics import extract_frame_features
from src.features.online_event_detector import OnlineEventDetector
from src.pose.pose_estimator import PoseEstimator

MIN_EVENTS_FOR_BASELINE = 10  # mirrors src/baseline/baseline_engine.WARMUP_EVENTS


def ensure_athlete(api_url: str, athlete_id: str | None, athlete_name: str | None) -> str:
    if athlete_id:
        resp = requests.get(f"{api_url}/athletes/{athlete_id}", timeout=5)
        if resp.status_code == 200:
            return athlete_id
        print(f"Warning: athlete_id {athlete_id} not found, creating a new athlete instead.")
    resp = requests.post(f"{api_url}/athletes", json={"display_name": athlete_name or "Live Demo Athlete"}, timeout=5)
    resp.raise_for_status()
    new_id = resp.json()["athlete_id"]
    print(f"Created athlete: {new_id}")
    return new_id


def create_session(api_url: str, athlete_id: str, fps: float) -> str:
    from datetime import datetime, UTC

    resp = requests.post(
        api_url + "/sessions",
        json={
            "athlete_id": athlete_id,
            "recorded_at": datetime.now(UTC).isoformat(),
            "camera_setup": "webcam-live",
            "fps": fps,
        },
        timeout=5,
    )
    resp.raise_for_status()
    return resp.json()["session_id"]


def push_event(api_url: str, session_id: str, event_index: int, event) -> None:
    """Upload one confirmed live landing event to the backend. Flags (if
    any) are read back separately via GET /athletes/{id}/flags -- this
    function only does the write."""
    payload = [{
        "event_index": event_index,
        "takeoff_frame": event.takeoff_frame,
        "peak_frame": event.peak_frame,
        "landing_frame": event.landing_frame,
        "landing_timestamp_s": event.landing_timestamp,
        "fppa_left": event.averaged_features.get("fppa_left"),
        "fppa_right": event.averaged_features.get("fppa_right"),
        "knee_flexion_left": event.averaged_features.get("knee_flexion_left"),
        "knee_flexion_right": event.averaged_features.get("knee_flexion_right"),
        "trunk_lean": event.averaged_features.get("trunk_lean"),
    }]
    resp = requests.post(f"{api_url}/sessions/{session_id}/events", json=payload, timeout=10)
    resp.raise_for_status()


def draw_overlay(frame: np.ndarray, text_lines: list[str], color=(0, 255, 0)) -> np.ndarray:
    y = 30
    for line in text_lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 28
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default 0)")
    parser.add_argument("--model", default="models/pose_landmarker_full.task")
    parser.add_argument("--athlete-id", default=None)
    parser.add_argument("--athlete-name", default=None)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--no-api", action="store_true", help="Run without the backend: show live skeleton + events only, no persistence/flags")
    args = parser.parse_args()

    if not Path(args.model).is_file():
        print(f"ERROR: model file not found at '{args.model}'. Run scripts/download_pose_model.py first.")
        sys.exit(1)

    athlete_id = None
    session_id = None
    if not args.no_api:
        try:
            requests.get(f"{args.api_url}/health", timeout=3).raise_for_status()
        except requests.exceptions.RequestException as exc:
            print(f"ERROR: cannot reach API at {args.api_url}: {exc}")
            print("Start it first with: uvicorn src.api.main:app --reload --port 8000")
            print("...or pass --no-api to run this demo without persistence/flags.")
            sys.exit(1)
        athlete_id = ensure_athlete(args.api_url, args.athlete_id, args.athlete_name)
        session_id = create_session(args.api_url, athlete_id, args.target_fps)
        print(f"Athlete {athlete_id}, session {session_id}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {args.camera}.")
        sys.exit(1)

    estimator = PoseEstimator(model_asset_path=args.model, running_mode="VIDEO", num_poses=1)
    detector = OnlineEventDetector(fps=args.target_fps)

    event_count = 0
    n_events_seen_this_run = 0
    start_time = time.time()
    frame_index = 0
    last_flag_text: list[str] = []
    last_flag_until = 0.0

    print("Starting live capture. Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed -- stopping.")
                break

            timestamp = time.time() - start_time
            result = estimator.process(frame, timestamp_ms=int(timestamp * 1000))

            status_lines = [f"Frame {frame_index}  |  events detected: {event_count}"]

            if result.detected and result.landmarks_2d is not None:
                feats = extract_frame_features(
                    result.landmarks_2d,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    landmarks_3d=result.landmarks_3d,
                )
                left_hip = result.landmarks_2d[23]
                right_hip = result.landmarks_2d[24]
                hip_y = float((left_hip[1] + right_hip[1]) / 2.0)

                feature_dict = {
                    "fppa_left": feats.fppa_left,
                    "fppa_right": feats.fppa_right,
                    "knee_flexion_left": feats.knee_flexion_left,
                    "knee_flexion_right": feats.knee_flexion_right,
                    "trunk_lean": feats.trunk_lean,
                    # Monocular-3D features (from MediaPipe's pose_world_landmarks,
                    # see src/features/biomechanics.py module note) -- tracked and
                    # averaged alongside the 2D features above, but NOT YET sent to
                    # the API in push_event() below: the DB schema/LandingEventIn
                    # only has the original 2D columns. Shown live on-screen so the
                    # 3D signal is visibly working end-to-end; persisting it needs a
                    # schema migration, not just a client-side change.
                    "knee_flexion_left_3d": feats.knee_flexion_left_3d,
                    "knee_flexion_right_3d": feats.knee_flexion_right_3d,
                    "knee_abduction_left_3d": feats.knee_abduction_left_3d,
                    "knee_abduction_right_3d": feats.knee_abduction_right_3d,
                }
                live_event = detector.push(frame_index=frame_index, timestamp=timestamp, hip_y=hip_y, features=feature_dict)

                display_frame = result.annotated_frame
                status_lines.append(f"FPPA L/R (2D proxy): {feats.fppa_left:.1f} / {feats.fppa_right:.1f} deg")
                status_lines.append(f"Knee flexion L/R (2D): {feats.knee_flexion_left:.1f} / {feats.knee_flexion_right:.1f} deg")
                if not np.isnan(feats.knee_flexion_left_3d):
                    status_lines.append(
                        f"Knee flexion L/R (3D, monocular): {feats.knee_flexion_left_3d:.1f} / {feats.knee_flexion_right_3d:.1f} deg"
                    )
                    status_lines.append(
                        f"Knee abduction L/R (3D, monocular): {feats.knee_abduction_left_3d:.1f} / {feats.knee_abduction_right_3d:.1f} deg"
                    )

                if live_event is not None:
                    event_count += 1
                    print(f"Landing event #{event_count} confirmed at frame {live_event.landing_frame} "
                          f"(t={live_event.landing_timestamp:.2f}s)")
                    if session_id is not None:
                        try:
                            push_event(args.api_url, session_id, event_count - 1, live_event)
                            flags = requests.get(f"{args.api_url}/athletes/{athlete_id}/flags", timeout=5).json()
                            if flags:
                                latest = flags[0]
                                last_flag_text = [f"FLAG ({latest['band']}): {latest['headline']}"]
                                last_flag_until = time.time() + 6.0
                            else:
                                n_remaining = max(0, MIN_EVENTS_FOR_BASELINE - event_count)
                                if n_remaining > 0:
                                    last_flag_text = [f"Building baseline... {n_remaining} more event(s) needed"]
                                    last_flag_until = time.time() + 4.0
                        except requests.exceptions.RequestException as exc:
                            print(f"Warning: could not reach API to log event: {exc}")
            else:
                display_frame = frame
                status_lines.append("No pose detected")

            display_frame = draw_overlay(display_frame, status_lines, color=(0, 255, 0))
            if time.time() < last_flag_until and last_flag_text:
                display_frame = draw_overlay(display_frame, last_flag_text, color=(0, 140, 255))

            cv2.imshow("DriftGuard -- Live Webcam Monitoring (press 'q' to quit)", display_frame)
            frame_index += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        estimator.close()
        print(f"Session ended. {event_count} landing event(s) detected and {'logged' if session_id else 'NOT logged (--no-api)'}.")


if __name__ == "__main__":
    main()
