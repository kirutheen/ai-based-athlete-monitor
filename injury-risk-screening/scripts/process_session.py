"""
End-to-end session processor: pose CSV/Parquet -> biomechanical features
-> landing-event segmentation -> push to the DriftGuard API.

This is the script that turns the output of scripts/run_pose_on_video.py
(a raw per-frame landmark table) into exactly what src/api/main.py's
/sessions/{id}/events endpoint expects, then calls it. It's the glue
between "we ran MediaPipe on a video" and "a clinician can see a flag in
the dashboard" -- the whole point of the demo script for the doctor
pitch.

Usage:
    python scripts/process_session.py \\
        --pose-data data/athlete1_session1_pose.parquet \\
        --athlete-id ath_abc123 \\
        --fps 30 \\
        --api-url http://localhost:8000

If --athlete-id is omitted, a new athlete is created from --athlete-name
first (useful for a first-time demo athlete).
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.biomechanics import extract_features_dataframe
from src.features.event_segmentation import segment_landing_events
from src.features.landmarks import ALL_LANDMARK_NAMES

# Deliberately imported from src.features.landmarks (no mediapipe
# dependency) rather than from src.pose.pose_estimator's PoseLandmark
# enum, so this script can run on a machine that only has the feature/
# API/dashboard dependencies installed, without needing mediapipe+cv2
# just to know column names. The two orderings are kept in sync -- see
# the comment on ALL_LANDMARK_NAMES.
LANDMARK_NAMES = list(ALL_LANDMARK_NAMES)


def load_pose_data(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported pose data file type: {path.suffix} (expected .csv or .parquet)")


def ensure_athlete(api_url: str, athlete_id: str | None, athlete_name: str | None) -> str:
    if athlete_id:
        resp = requests.get(f"{api_url}/athletes/{athlete_id}", timeout=10)
        if resp.status_code == 200:
            return athlete_id
        print(f"Warning: athlete_id {athlete_id} not found, creating a new athlete instead.")

    name = athlete_name or "Demo Athlete"
    resp = requests.post(f"{api_url}/athletes", json={"display_name": name}, timeout=10)
    resp.raise_for_status()
    new_id = resp.json()["athlete_id"]
    print(f"Created new athlete: {name} ({new_id})")
    return new_id


def _clean_val(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pose-data", required=True, type=Path, help="Path to the pose CSV/Parquet from run_pose_on_video.py")
    parser.add_argument("--athlete-id", default=None, help="Existing athlete ID (skips creation)")
    parser.add_argument("--athlete-name", default=None, help="Display name if creating a new athlete")
    parser.add_argument("--fps", type=float, required=True, help="Output frame rate of the pose run")
    parser.add_argument("--camera-setup", default="frontal", help="e.g. 'frontal', 'sagittal', 'frontal+sagittal'")
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()

    if not args.pose_data.exists():
        print(f"Error: pose data file not found: {args.pose_data}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading pose data from {args.pose_data} ...")
    pose_df = load_pose_data(args.pose_data)
    if "timestamp" not in pose_df.columns or "frame_index" not in pose_df.columns:
        print("Error: pose data must contain 'timestamp' and 'frame_index' columns.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(pose_df)} frames loaded.")

    print("Extracting per-frame biomechanical features (FPPA, flexion, trunk lean)...")
    features_df = extract_features_dataframe(pose_df, LANDMARK_NAMES)
    # extract_features_dataframe's output repeats frame_index/timestamp
    # (it needs them internally to build each FrameFeatures record) --
    # drop the duplicates before concatenating so combined has exactly
    # one of each column; otherwise df["frame_index"] returns a 2-column
    # DataFrame/Series-of-Series instead of a scalar-per-row column.
    features_only = features_df.drop(columns=["frame_index", "timestamp"])
    combined = pd.concat([pose_df.reset_index(drop=True), features_only], axis=1)

    print("Segmenting landing events...")
    events = segment_landing_events(combined, fps=args.fps)
    print(f"  {len(events)} landing event(s) detected.")
    if not events:
        print("No events to upload -- nothing more to do. If this is unexpected, check that the "
              "camera captured a real jump/landing and that --fps matches the pose run's output fps.")
        return

    # Export movement_features.csv dataset (one row per trial)
    try:
        from src.features.movement_dataset import generate_movement_features_dataframe
        mv_df = generate_movement_features_dataframe(
            combined_df=combined,
            events=events,
            subject_id=args.athlete_id or "demo_athlete",
            trial_prefix="trial",
        )
        out_csv_path = args.pose_data.parent / "movement_features.csv"
        mv_df.to_csv(out_csv_path, index=False)
        print(f"  Generated movement features dataset at {out_csv_path} ({len(mv_df)} trials).")
    except Exception as exc:
        print(f"  Warning: could not write movement_features.csv: {exc}")

    print(f"Checking API at {args.api_url} ...")
    try:
        requests.get(f"{args.api_url}/health", timeout=5).raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"Error: cannot reach API at {args.api_url}: {exc}", file=sys.stderr)
        sys.exit(1)

    athlete_id = ensure_athlete(args.api_url, args.athlete_id, args.athlete_name)

    session_resp = requests.post(
        f"{args.api_url}/sessions",
        json={
            "athlete_id": athlete_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "camera_setup": args.camera_setup,
            "fps": args.fps,
        },
        timeout=10,
    )
    session_resp.raise_for_status()
    session_id = session_resp.json()["session_id"]
    print(f"Created session {session_id}")

    event_payload = []
    for e in events:
        window = combined[(combined["frame_index"] >= e.window_start_frame) & (combined["frame_index"] <= e.window_end_frame)]
        event_payload.append(
            {
                "event_index": e.event_id,
                "takeoff_frame": e.takeoff_frame,
                "peak_frame": e.peak_frame,
                "landing_frame": e.landing_frame,
                "landing_timestamp_s": e.landing_timestamp,
                "fppa_left": _clean_val(window["fppa_left"].mean()) if "fppa_left" in window else None,
                "fppa_right": _clean_val(window["fppa_right"].mean()) if "fppa_right" in window else None,
                "knee_flexion_left": _clean_val(window["knee_flexion_left"].mean()) if "knee_flexion_left" in window else None,
                "knee_flexion_right": _clean_val(window["knee_flexion_right"].mean()) if "knee_flexion_right" in window else None,
                "trunk_lean": _clean_val(window["trunk_lean"].mean()) if "trunk_lean" in window else None,
                "hip_flexion_left": _clean_val(window["hip_flexion_left"].mean()) if "hip_flexion_left" in window else None,
                "hip_flexion_right": _clean_val(window["hip_flexion_right"].mean()) if "hip_flexion_right" in window else None,
                "hip_abduction_left": _clean_val(window["hip_abduction_left"].mean()) if "hip_abduction_left" in window else None,
                "hip_abduction_right": _clean_val(window["hip_abduction_right"].mean()) if "hip_abduction_right" in window else None,
                "ankle_dorsiflexion_left": _clean_val(window["ankle_dorsiflexion_left"].mean()) if "ankle_dorsiflexion_left" in window else None,
                "ankle_dorsiflexion_right": _clean_val(window["ankle_dorsiflexion_right"].mean()) if "ankle_dorsiflexion_right" in window else None,
                "trunk_flexion_deg": _clean_val(window["trunk_flexion_deg"].mean()) if "trunk_flexion_deg" in window else None,
                "pelvic_obliquity_deg": _clean_val(window["pelvic_obliquity_deg"].mean()) if "pelvic_obliquity_deg" in window else None,
                "knee_flexion_LSI": _clean_val(window["knee_flexion_LSI"].mean()) if "knee_flexion_LSI" in window else None,
                "knee_valgus_LSI": _clean_val(window["knee_valgus_LSI"].mean()) if "knee_valgus_LSI" in window else None,
                "knee_valgus_at_initial_contact": _clean_val(e.knee_valgus_at_initial_contact),
                "knee_flexion_at_initial_contact": _clean_val(e.knee_flexion_at_initial_contact),
                "valgus_duration": _clean_val(e.valgus_duration),
                "com_displacement": _clean_val(window["com_total_displacement"].iloc[-1]) if "com_total_displacement" in window and not window["com_total_displacement"].empty else None,
                "pose_quality": _clean_val(window["pose_confidence"].mean()) if "pose_confidence" in window else None,
            }
        )

    ingest_resp = requests.post(f"{args.api_url}/sessions/{session_id}/events", json=event_payload, timeout=30)
    ingest_resp.raise_for_status()
    print(f"Uploaded {len(event_payload)} event(s) for athlete {athlete_id}.")
    print("Open the clinician dashboard (streamlit run src/dashboard/doctor_view.py) to review.")


if __name__ == "__main__":
    main()
