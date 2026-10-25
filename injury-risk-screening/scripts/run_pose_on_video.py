#!/usr/bin/env python
"""CLI: run VideoSource + PoseEstimator + smoothing over a video, and write:

  1. a landmarks table (CSV or Parquet), one row per frame, with all 33
     landmarks' x/y/z/visibility as columns plus athlete/session/video IDs
  2. an annotated video with the skeleton overlay drawn on the (smoothed)
     landmarks, for visual sanity-checking

Example
-------
    python scripts/run_pose_on_video.py ^
        data/my_jump_video.mp4 ^
        --model models/pose_landmarker_full.task ^
        --athlete-id athlete_001 ^
        --session-id session_01 ^
        --video-id jump_test_01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `src` imports when run as a script

from src.capture.video_source import VideoSource
from src.pose.pose_estimator import NUM_LANDMARKS, PoseEstimator, smooth
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmark

LANDMARK_NAMES = [lm.name.lower() for lm in PoseLandmark]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="Path to the input video file, or an integer camera index")
    p.add_argument("--model", default="models/pose_landmarker_full.task", help="Path to a MediaPipe Pose Landmarker .task file")
    p.add_argument("--athlete-id", required=True)
    p.add_argument("--session-id", required=True)
    p.add_argument("--video-id", required=True)
    p.add_argument("--target-fps", type=float, default=None, help="Resample input to this FPS (default: native FPS, no resampling)")
    p.add_argument("--num-poses", type=int, default=1)
    p.add_argument("--smoothing", choices=["one_euro", "moving_average", "none"], default="one_euro")
    p.add_argument("--smoothing-window", type=int, default=5, help="Window size for moving_average smoothing")
    p.add_argument("--output-landmarks", default=None, help="Output path for the landmarks table (.csv or .parquet). Default: data/processed/<video_id>_landmarks.parquet")
    p.add_argument("--output-video", default=None, help="Output path for the annotated debug video (.mp4). Default: data/processed/<video_id>_annotated.mp4")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    video_source_arg = int(args.video) if str(args.video).isdigit() else args.video

    if not Path(args.model).is_file():
        print(f"ERROR: model file not found at '{args.model}'.")
        print("Download it first -- see the comment at the top of src/pose/pose_estimator.py")
        sys.exit(1)

    out_landmarks = Path(args.output_landmarks) if args.output_landmarks else Path("data/processed") / f"{args.video_id}_landmarks.parquet"
    out_video = Path(args.output_video) if args.output_video else Path("data/processed") / f"{args.video_id}_annotated.mp4"
    out_landmarks.parent.mkdir(parents=True, exist_ok=True)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    with VideoSource(video_source_arg, target_fps=args.target_fps) as source:
        with PoseEstimator(model_asset_path=args.model, running_mode="VIDEO", num_poses=args.num_poses) as estimator:
            writer = cv2.VideoWriter(
                str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), source.output_fps, (source.width, source.height)
            )

            raw_landmarks_2d = []
            raw_confidences = []
            frame_meta = []
            raw_frames_cache = []  # kept only long enough to redraw with smoothed coords

            print(f"Reading '{args.video}' at {source.output_fps:.1f} fps ({source.width}x{source.height})...")

            for frame in source.frames():
                result = estimator.process(frame.image, timestamp_ms=int(frame.timestamp * 1000))
                raw_landmarks_2d.append(result.landmarks_2d)
                raw_confidences.append(result.confidence)
                frame_meta.append((frame.frame_index, frame.timestamp))
                raw_frames_cache.append(frame.image)

            n_frames = len(frame_meta)
            n_missing = sum(1 for lm in raw_landmarks_2d if lm is None)

            if args.smoothing == "none":
                smoothed = raw_landmarks_2d
            elif args.smoothing == "moving_average":
                smoothed = smooth(raw_landmarks_2d, method="moving_average", window=args.smoothing_window)
            else:
                smoothed = smooth(raw_landmarks_2d, method="one_euro")

            rows = []
            for (frame_index, ts), raw_frame, landmarks, confidence in zip(
                frame_meta, raw_frames_cache, smoothed, raw_confidences
            ):
                annotated = raw_frame.copy()
                row = {
                    "athlete_id": args.athlete_id,
                    "session_id": args.session_id,
                    "video_id": args.video_id,
                    "frame_index": frame_index,
                    "timestamp": ts,
                    "confidence": confidence,
                    "detected": landmarks is not None,
                }
                if landmarks is not None:
                    PoseEstimator._draw_skeleton(annotated, landmarks, source.width, source.height)
                    for i in range(NUM_LANDMARKS):
                        name = LANDMARK_NAMES[i]
                        row[f"{name}_x"] = float(landmarks[i, 0])
                        row[f"{name}_y"] = float(landmarks[i, 1])
                        row[f"{name}_z"] = float(landmarks[i, 2])
                        row[f"{name}_v"] = float(landmarks[i, 3])
                else:
                    for i in range(NUM_LANDMARKS):
                        name = LANDMARK_NAMES[i]
                        row[f"{name}_x"] = None
                        row[f"{name}_y"] = None
                        row[f"{name}_z"] = None
                        row[f"{name}_v"] = None
                rows.append(row)
                writer.write(annotated)

            writer.release()

    df = pd.DataFrame(rows)
    if out_landmarks.suffix == ".csv":
        df.to_csv(out_landmarks, index=False)
    else:
        df.to_parquet(out_landmarks, index=False)

    print(f"\nProcessed {n_frames} frames ({args.video_id}).")
    if n_frames:
        print(f"Frames with NO pose detected: {n_missing}/{n_frames} ({n_missing / n_frames:.1%})")
    else:
        print("No frames were read from the input -- check the video path.")
    print(f"Landmarks written to: {out_landmarks}")
    print(f"Annotated video written to: {out_video}")


if __name__ == "__main__":
    main()
