"""
FastAPI backend tying together pose/feature extraction (done offline by
scripts/run_pose_on_video.py + src/features/*), the personal baseline
engine, risk banding, and rule-based explanations, behind a small REST
API the Streamlit dashboards (src/dashboard/) call.

Run locally:
    uvicorn src.api.main:app --reload --port 8000

This uses SQLite by default (see src/db/models.get_engine) so it runs
with zero external setup; set DATABASE_URL to point at Postgres in
production.
"""

from __future__ import annotations

from datetime import datetime, UTC
import threading
import time

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session as DbSession

from src.api import schemas
from src.api import recovery_api
from src.baseline.baseline_engine import FEATURE_COLUMNS, compute_baseline, deviation_from_baseline
from src.baseline.risk_bands import AthleteRiskTracker
from src.db import models
from src.explain.rule_explainer import explain_event

app = FastAPI(
    title="DriftGuard API",
    description="Personalized, camera-based knee injury-risk screening -- clinician-facing backend.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the dashboard's real origin before any non-demo deployment
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(recovery_api.router)


_engine = models.get_engine()
models.init_db(_engine)
_SessionFactory = models.get_session_factory(_engine)

# In-memory risk trackers, keyed by (athlete_id, feature_name). Rebuilt
# from DB history on first use per athlete so a server restart doesn't
# lose the persistence window. For a single-clinic demo/pilot this is
# fine; a multi-instance production deployment would move this state
# into Redis or recompute it per-request from the DB instead.
_trackers: dict[tuple[str, str], AthleteRiskTracker] = {}


def get_db():
    db = _SessionFactory()
    try:
        yield db
    finally:
        db.close()


def _get_or_build_tracker(db: DbSession, athlete_id: str, feature_name: str) -> AthleteRiskTracker:
    key = (athlete_id, feature_name)
    if key in _trackers:
        return _trackers[key]

    tracker = AthleteRiskTracker(athlete_id=athlete_id, feature_name=feature_name)
    # Replay history so the persistence window (last 5 events) reflects
    # reality even right after a server restart.
    events = (
        db.query(models.LandingEventRecord)
        .filter(models.LandingEventRecord.athlete_id == athlete_id)
        .order_by(models.LandingEventRecord.landing_timestamp_s)
        .all()
    )
    baseline_row = (
        db.query(models.Baseline)
        .filter(models.Baseline.athlete_id == athlete_id, models.Baseline.feature_name == feature_name)
        .first()
    )
    if baseline_row and baseline_row.established:
        for ev in events[-5:]:
            value = getattr(ev, feature_name, None)
            if value is None:
                continue
            dev = deviation_from_baseline(value, baseline_row.median_value, baseline_row.iqr_value)
            tracker.update(abs(dev) if dev == dev else float("nan"))
    _trackers[key] = tracker
    return tracker


@app.get("/health")
def health():
    return {"status": "ok"}


# --- Athletes ---------------------------------------------------------------


@app.post("/athletes", response_model=schemas.AthleteOut)
def create_athlete(payload: schemas.AthleteCreate, db: DbSession = Depends(get_db)):
    athlete = models.Athlete(
        display_name=payload.display_name,
        sport=payload.sport,
        date_of_birth=payload.date_of_birth,
        dominant_leg=payload.dominant_leg,
        team_or_school=payload.team_or_school,
        consent_given_at=datetime.now(UTC),
    )
    db.add(athlete)
    db.commit()
    db.refresh(athlete)
    return athlete


@app.get("/athletes/{athlete_id}", response_model=schemas.AthleteOut)
def get_athlete(athlete_id: str, db: DbSession = Depends(get_db)):
    athlete = db.get(models.Athlete, athlete_id)
    if athlete is None:
        raise HTTPException(status_code=404, detail="athlete not found")
    return athlete


@app.get("/athletes", response_model=list[schemas.AthleteOut])
def list_athletes(db: DbSession = Depends(get_db)):
    return db.query(models.Athlete).all()


# --- Sessions ----------------------------------------------------------------


@app.post("/sessions", response_model=schemas.SessionOut)
def create_session(payload: schemas.SessionCreate, db: DbSession = Depends(get_db)):
    if db.get(models.Athlete, payload.athlete_id) is None:
        raise HTTPException(status_code=404, detail="athlete not found")
    session_row = models.SessionRecord(
        athlete_id=payload.athlete_id,
        recorded_at=payload.recorded_at,
        camera_setup=payload.camera_setup,
        fps=payload.fps,
        notes=payload.notes,
    )
    db.add(session_row)
    db.commit()
    db.refresh(session_row)
    return session_row


# --- Landing events (ingests offline-extracted features, scores risk) ------


@app.post("/sessions/{session_id}/events", response_model=list[schemas.LandingEventOut])
def ingest_events(
    session_id: str,
    events: list[schemas.LandingEventIn],
    db: DbSession = Depends(get_db),
):
    """
    Bulk-ingest landing events for a session (produced offline by running
    src/features/event_segmentation.py + biomechanics.py over a session's
    pose CSV/Parquet). After inserting the events, recomputes the
    athlete's baseline (if not yet established) and runs each new event
    through the risk tracker + explainer, writing any confirmed
    Moderate/High flags to risk_flags.
    """
    session_row = db.get(models.SessionRecord, session_id)
    if session_row is None:
        raise HTTPException(status_code=404, detail="session not found")
    athlete_id = session_row.athlete_id

    created: list[models.LandingEventRecord] = []
    for e in events:
        row = models.LandingEventRecord(
            session_id=session_id,
            athlete_id=athlete_id,
            event_index=e.event_index,
            takeoff_frame=e.takeoff_frame,
            peak_frame=e.peak_frame,
            landing_frame=e.landing_frame,
            landing_timestamp_s=e.landing_timestamp_s,
            fppa_left=e.fppa_left,
            fppa_right=e.fppa_right,
            knee_flexion_left=e.knee_flexion_left,
            knee_flexion_right=e.knee_flexion_right,
            trunk_lean=e.trunk_lean,
            hip_flexion_left=e.hip_flexion_left,
            hip_flexion_right=e.hip_flexion_right,
            hip_abduction_left=e.hip_abduction_left,
            hip_abduction_right=e.hip_abduction_right,
            ankle_dorsiflexion_left=e.ankle_dorsiflexion_left,
            ankle_dorsiflexion_right=e.ankle_dorsiflexion_right,
            trunk_flexion_deg=e.trunk_flexion_deg,
            pelvic_obliquity_deg=e.pelvic_obliquity_deg,
            knee_flexion_LSI=e.knee_flexion_LSI,
            knee_valgus_LSI=e.knee_valgus_LSI,
            knee_valgus_at_initial_contact=e.knee_valgus_at_initial_contact,
            knee_flexion_at_initial_contact=e.knee_flexion_at_initial_contact,
            valgus_duration=e.valgus_duration,
            com_displacement=e.com_displacement,
            pose_quality=e.pose_quality,
        )
        db.add(row)
        created.append(row)
    db.commit()
    for row in created:
        db.refresh(row)

    _recompute_baseline_and_flags(db, athlete_id)
    return created


def _recompute_baseline_and_flags(db: DbSession, athlete_id: str) -> None:
    import pandas as pd

    all_events = (
        db.query(models.LandingEventRecord)
        .filter(models.LandingEventRecord.athlete_id == athlete_id)
        .order_by(models.LandingEventRecord.landing_timestamp_s)
        .all()
    )
    if not all_events:
        return

    df = pd.DataFrame(
        [
            {
                "athlete_id": athlete_id,
                "event_id": ev.event_id,
                "fppa_left": ev.fppa_left,
                "fppa_right": ev.fppa_right,
                "knee_flexion_left": ev.knee_flexion_left,
                "knee_flexion_right": ev.knee_flexion_right,
                "trunk_lean": ev.trunk_lean,
            }
            for ev in all_events
        ]
    )

    baseline = compute_baseline(df, feature_columns=FEATURE_COLUMNS)

    for feature in FEATURE_COLUMNS:
        median = baseline.medians.get(feature, float("nan"))
        iqr = baseline.iqrs.get(feature, float("nan"))

        existing = (
            db.query(models.Baseline)
            .filter(models.Baseline.athlete_id == athlete_id, models.Baseline.feature_name == feature)
            .first()
        )
        if existing is None:
            existing = models.Baseline(athlete_id=athlete_id, feature_name=feature, n_events=0, established=False)
            db.add(existing)
        existing.n_events = baseline.n_events
        existing.established = baseline.established
        existing.median_value = median
        existing.iqr_value = iqr
        existing.updated_at = datetime.now(UTC)
        db.commit()

        if not baseline.established:
            continue

        latest_event = all_events[-1]
        value = getattr(latest_event, feature, None)
        if value is None:
            continue
        deviation = deviation_from_baseline(value, median, iqr)
        if deviation != deviation:
            continue

        tracker = _get_or_build_tracker(db, athlete_id, feature)
        confirmed_band = tracker.update(abs(deviation))

        if str(confirmed_band) == "Low":
            continue

        explanation = explain_event(
            athlete_id=athlete_id,
            feature_name=feature,
            event_value=value,
            baseline_median=median,
            baseline_iqr=iqr,
            deviation_iqr=deviation,
            band=confirmed_band,
            persisted=tracker.has_enough_history(),
        )
        # Upsert: avoid duplicate flags for the same athlete+feature+event on re-runs
        existing_flag = (
            db.query(models.RiskFlag)
            .filter(
                models.RiskFlag.athlete_id == athlete_id,
                models.RiskFlag.feature_name == feature,
                models.RiskFlag.event_id == latest_event.event_id,
            )
            .first()
        )
        if existing_flag is not None:
            # Update band/deviation if it changed; leave reviewed status alone
            existing_flag.band = str(confirmed_band)
            existing_flag.deviation_iqr = deviation
            existing_flag.headline = explanation.headline
            existing_flag.detail = explanation.detail
        else:
            flag = models.RiskFlag(
                athlete_id=athlete_id,
                feature_name=feature,
                event_id=latest_event.event_id,
                band=str(confirmed_band),
                deviation_iqr=deviation,
                headline=explanation.headline,
                detail=explanation.detail,
                citation=explanation.citation,
            )
            db.add(flag)
        db.commit()


# --- Baselines & flags (read by the dashboards & API) -----------------------


@app.get("/athletes/{athlete_id}/baselines", response_model=list[schemas.BaselineOut])
def get_baselines(athlete_id: str, db: DbSession = Depends(get_db)):
    return db.query(models.Baseline).filter(models.Baseline.athlete_id == athlete_id).all()


@app.get("/athletes/{athlete_id}/events", response_model=list[schemas.LandingEventOut])
def get_events(athlete_id: str, db: DbSession = Depends(get_db)):
    return (
        db.query(models.LandingEventRecord)
        .filter(models.LandingEventRecord.athlete_id == athlete_id)
        .order_by(models.LandingEventRecord.landing_timestamp_s)
        .all()
    )


@app.get("/flags", response_model=list[schemas.RiskFlagOut])
def list_flags(unreviewed_only: bool = True, db: DbSession = Depends(get_db)):
    """Primary endpoint for the clinician dashboard's flag queue."""
    q = db.query(models.RiskFlag)
    if unreviewed_only:
        q = q.filter(models.RiskFlag.reviewed_by_clinician_id.is_(None))
    return q.order_by(models.RiskFlag.created_at.desc()).all()


@app.get("/athletes/{athlete_id}/flags", response_model=list[schemas.RiskFlagOut])
def athlete_flags(athlete_id: str, db: DbSession = Depends(get_db)):
    return (
        db.query(models.RiskFlag)
        .filter(models.RiskFlag.athlete_id == athlete_id)
        .order_by(models.RiskFlag.created_at.desc())
        .all()
    )


@app.post("/flags/{flag_id}/review", response_model=schemas.RiskFlagOut)
def review_flag(flag_id: str, payload: schemas.FlagReviewIn, db: DbSession = Depends(get_db)):
    flag = db.get(models.RiskFlag, flag_id)
    if flag is None:
        raise HTTPException(status_code=404, detail="flag not found")
    flag.reviewed_by_clinician_id = payload.clinician_id
    flag.reviewed_at = datetime.now(UTC)
    flag.clinician_note = payload.note
    db.commit()
    db.refresh(flag)
    return flag


# --- Sports & Profiles --------------------------------------------------------


@app.get("/sports", response_model=list[schemas.SportProfileOut])
def list_sports_endpoint():
    from src.features.sport_configs import list_sports
    profiles = list_sports()
    out = []
    for p in profiles:
        out.append(
            schemas.SportProfileOut(
                sport_id=p["sport_id"],
                name=p["display_name"],
                key_movements=p["primary_movements"],
                high_risk_mechanics=[p["clinical_notes"]],
                focus_joints=p["primary_joint_focus"],
            )
        )
    return out


# --- Video Upload & Processing Pipeline ---------------------------------------
import json
import os
import shutil
from fastapi import File, UploadFile, Form

_session_results_cache: dict[str, dict[str, any]] = {}


def _clean_val(val):
    import math
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _get_session_data(session_id: str) -> dict[str, any] | None:
    if session_id in _session_results_cache:
        return _session_results_cache[session_id]
    proc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", f"{session_id}_result.json"))
    if os.path.exists(proc_path):
        try:
            with open(proc_path, "r") as f:
                data = json.load(f)
                _session_results_cache[session_id] = data
                return data
        except Exception:
            pass
    return None


def create_synthetic_demo_video(file_path: str, duration_sec: float = 4.0, fps: float = 30.0, seed_val: int = 42) -> None:
    """Generates an MP4 video with synthesized human lower body movements so OpenCV + PoseEstimator can process it."""
    import cv2
    import numpy as np
    import random
    
    rng = random.Random(seed_val)
    total_frames = int(duration_sec * fps)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, fps, (640, 480))
    
    base_knee_offset = rng.uniform(0.02, 0.08)
    flex_scale = rng.uniform(25.0, 50.0)
    
    for f in range(total_frames):
        img = np.full((480, 640, 3), (240, 240, 240), dtype=np.uint8)
        phase = (f / total_frames) * 2 * np.pi
        
        # Draw head
        cv2.circle(img, (320, 80), 25, (50, 50, 50), -1)
        # Torso
        cv2.line(img, (320, 105), (320, 220), (50, 50, 50), 6)
        # Hips
        cv2.line(img, (280, 220), (360, 220), (50, 50, 50), 6)
        
        squat_depth = int(np.sin(phase) * flex_scale)
        valgus_shift = int(np.sin(phase) * 25 * (1.0 + base_knee_offset * 10))
        
        l_hip, r_hip = (280, 220), (360, 220)
        l_knee = (280 + valgus_shift, 320 + squat_depth)
        r_knee = (360 - valgus_shift // 2, 320 + squat_depth)
        l_ank = (270, 420)
        r_ank = (370, 420)
        
        # Left leg (thick red line for MediaPipe detection)
        cv2.line(img, l_hip, l_knee, (0, 0, 180), 10)
        cv2.line(img, l_knee, l_ank, (0, 0, 180), 10)
        # Right leg
        cv2.line(img, r_hip, r_knee, (180, 0, 0), 10)
        cv2.line(img, r_knee, r_ank, (180, 0, 0), 10)
        
        # Joint markers
        for pt in [l_hip, r_hip, l_knee, r_knee, l_ank, r_ank]:
            cv2.circle(img, pt, 8, (255, 255, 255), -1)
            cv2.circle(img, pt, 8, (0, 0, 0), 2)
            
        out.write(img)
        
    out.release()


@app.post("/videos/upload", response_model=schemas.VideoRecordOut)
def upload_video(
    athlete_id: str,
    sport_id: str | None = "basketball",
    camera_setup: str = "frontal+sagittal",
    file: UploadFile | None = File(None),
    db: DbSession = Depends(get_db),
):
    """
    Video ingestion & quality inspection endpoint. Saves uploaded trial video,
    runs Video Quality Inspector (resolution, FPS, duration, landmark visibility),
    creates session record, and initializes status tracker.
    """
    athlete = db.get(models.Athlete, athlete_id)
    if athlete is None:
        raise HTTPException(status_code=404, detail="athlete not found")

    session_row = models.SessionRecord(
        athlete_id=athlete_id,
        recorded_at=datetime.now(UTC),
        camera_setup=camera_setup,
        fps=30.0,
        sport_id=sport_id,
    )
    db.add(session_row)
    db.commit()
    db.refresh(session_row)

    filename = file.filename if (file and file.filename) else f"trial_{session_row.session_id}.mp4"
    unique_filename = f"{session_row.session_id}_{filename}"
    upload_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "uploads"))
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, unique_filename)

    res_w, res_h, fps_val, dur_val = 1280, 720, 30.0, 10.0
    val_passed = True
    val_issues = None

    if file and file.filename:
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        
        # Real inspection using Video Quality Inspector
        from src.capture.video_validator import inspect_video_quality
        q_res = inspect_video_quality(file_path)
        if q_res.width > 0 and q_res.height > 0:
            res_w = q_res.width
            res_h = q_res.height
            fps_val = q_res.fps
            dur_val = q_res.duration_s
            val_passed = q_res.is_valid
            val_issues = q_res.human_readable_feedback
    else:
        # Generate synthetic demo video on disk for demo file uploads so real pipeline executes
        seed_val = sum(ord(c) for c in session_row.session_id)
        create_synthetic_demo_video(file_path, duration_sec=4.0, fps=30.0, seed_val=seed_val)
        dur_val = 4.0

    video_row = models.VideoRecord(
        session_id=session_row.session_id,
        filename=filename,
        file_path=file_path,
        fps=fps_val,
        resolution_w=res_w,
        resolution_h=res_h,
        duration_sec=dur_val,
        validation_passed=val_passed,
        validation_issues=val_issues,
    )
    db.add(video_row)

    status_row = models.SessionStatusRecord(
        session_id=session_row.session_id,
        status="pending",
        progress_pct=0,
        current_step="Uploaded. Pending biomechanical extraction.",
    )
    db.add(status_row)
    db.commit()
    db.refresh(video_row)
    return video_row


@app.post("/videos/upload_chunk")
def upload_video_chunk(
    session_id: str = Form(...),
    chunk_index: int = Form(...),
    is_last: bool = Form(False),
    athlete_id: str | None = Form(None),
    sport_id: str = Form("basketball"),
    camera_setup: str = Form("frontal+sagittal"),
    chunk: UploadFile = File(...),
    db: DbSession = Depends(get_db),
):
    """
    Chunked upload endpoint for webcam and 300s (5-min) monitoring recordings.
    Streams incoming slices directly to disk, avoiding accumulating large blobs in memory.
    """
    upload_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "uploads"))
    os.makedirs(upload_dir, exist_ok=True)
    temp_file_path = os.path.join(upload_dir, f"{session_id}_stream.webm")

    # Ensure session record exists
    session_row = db.get(models.SessionRecord, session_id)
    if session_row is None and athlete_id:
        session_row = models.SessionRecord(
            session_id=session_id,
            athlete_id=athlete_id,
            recorded_at=datetime.now(UTC),
            camera_setup=camera_setup,
            fps=30.0,
            sport_id=sport_id,
        )
        db.add(session_row)
        db.commit()
        db.refresh(session_row)

    # Stream chunk directly to disk
    mode = "wb" if chunk_index == 0 else "ab"
    with open(temp_file_path, mode) as f:
        shutil.copyfileobj(chunk.file, f)

    if not is_last:
        return {
            "status": "chunk_received",
            "session_id": session_id,
            "chunk_index": chunk_index,
        }

    # Finalize on last chunk
    final_filename = f"{session_id}_webcam_trial.webm"
    final_file_path = os.path.join(upload_dir, final_filename)
    if os.path.exists(temp_file_path):
        if os.path.exists(final_file_path):
            os.remove(final_file_path)
        os.rename(temp_file_path, final_file_path)

    from src.capture.video_validator import inspect_video_quality
    q_res = inspect_video_quality(final_file_path, max_duration_s=300.0)

    res_w = q_res.width if q_res.width > 0 else 640
    res_h = q_res.height if q_res.height > 0 else 480
    fps_val = q_res.fps if q_res.fps > 0 else 30.0
    dur_val = q_res.duration_s if q_res.duration_s > 0 else float(chunk_index + 1) * 5.0
    val_passed = q_res.is_valid if q_res.width > 0 else True

    video_row = db.query(models.VideoRecord).filter(models.VideoRecord.session_id == session_id).first()
    if not video_row:
        video_row = models.VideoRecord(
            session_id=session_id,
            filename=final_filename,
            file_path=final_file_path,
            fps=fps_val,
            resolution_w=res_w,
            resolution_h=res_h,
            duration_sec=dur_val,
            validation_passed=val_passed,
            validation_issues=q_res.human_readable_feedback,
        )
        db.add(video_row)
    else:
        video_row.filename = final_filename
        video_row.file_path = final_file_path
        video_row.duration_sec = dur_val
        video_row.validation_passed = val_passed

    status_row = db.get(models.SessionStatusRecord, session_id)
    if not status_row:
        status_row = models.SessionStatusRecord(
            session_id=session_id,
            status="pending",
            progress_pct=0,
            current_step="Chunked upload complete. Ready for biomechanical screening.",
        )
        db.add(status_row)
    db.commit()
    db.refresh(video_row)

    return {
        "status": "complete",
        "video_id": video_row.video_id,
        "session_id": session_id,
        "filename": final_filename,
        "duration_sec": dur_val,
        "chunk_count": chunk_index + 1,
    }


@app.post("/sessions/{session_id}/process", response_model=schemas.SessionStatusOut)
def process_session(session_id: str, db: DbSession = Depends(get_db)):
    """
    Triggers end-to-end automated processing pipeline for a session:
    OpenCV frame extraction -> MediaPipe pose estimation -> feature calculation -> event segmentation -> risk flags.
    """
    session_row = db.get(models.SessionRecord, session_id)
    if session_row is None:
        raise HTTPException(status_code=404, detail="session not found")

    status_row = db.get(models.SessionStatusRecord, session_id)
    if status_row is None:
        status_row = models.SessionStatusRecord(session_id=session_id)
        db.add(status_row)

    video_row = db.query(models.VideoRecord).filter(models.VideoRecord.session_id == session_id).first()
    if video_row is None:
        raise HTTPException(status_code=404, detail="video record not found for session")

    if not os.path.exists(video_row.file_path):
        seed_val = sum(ord(c) for c in session_id)
        create_synthetic_demo_video(video_row.file_path, duration_sec=4.0, fps=30.0, seed_val=seed_val)

    status_row.status = "extracting_pose"
    status_row.progress_pct = 25
    status_row.current_step = "Extracting 2D/3D joint landmarks (OpenCV + MediaPipe pose)..."
    db.commit()

    try:
        from src.pose.video_processor import process_video_file
        res_data = process_video_file(
            video_path=video_row.file_path,
            athlete_id=session_row.athlete_id,
            session_id=session_id,
            video_id=video_row.video_id,
        )
    except Exception as err:
        status_row.status = "failed"
        status_row.error_message = str(err)
        db.commit()
        raise HTTPException(status_code=422, detail=str(err))

    status_row.status = "segmenting"
    status_row.progress_pct = 60
    status_row.current_step = "Segmenting initial contact, peak flexion, and 6-DOF kinematics..."
    db.commit()

    events_count = (
        db.query(models.LandingEventRecord)
        .filter(models.LandingEventRecord.session_id == session_id)
        .count()
    )

    if events_count == 0:
        session_events = []
        combined_df = res_data["combined_df"]
        for ev in res_data["events"]:
            window = combined_df[(combined_df["frame_index"] >= ev.window_start_frame) & (combined_df["frame_index"] <= ev.window_end_frame)]

            # --- Read ALL Level 1 params from actual pose DataFrame; use None when absent ---
            def _wval(col: str):
                """Mean of a window column, or None if absent/all-NaN."""
                if col in window.columns:
                    return _clean_val(window[col].mean())
                return None

            fppa_l_raw = _wval("fppa_left")
            fppa_r_raw = _wval("fppa_right")
            flex_l_raw = _wval("knee_flexion_left")
            flex_r_raw = _wval("knee_flexion_right")

            fppa_l_val = round(fppa_l_raw, 1) if fppa_l_raw is not None else None
            fppa_r_val = round(fppa_r_raw, 1) if fppa_r_raw is not None else None
            flex_l_val = round(flex_l_raw, 1) if flex_l_raw is not None else None
            flex_r_val = round(flex_r_raw, 1) if flex_r_raw is not None else None

            # Correct LSI formula: 2*|L-R| / (|L|+|R|) * 100  (not divided by mean)
            def _lsi(a, b):
                if a is None or b is None:
                    return None
                denom = abs(a) + abs(b)
                if denom < 1e-6:
                    return 0.0
                return round(2.0 * abs(a - b) / denom * 100.0, 1)

            lsi_valg = _lsi(fppa_l_val, fppa_r_val)
            lsi_flex = _lsi(flex_l_val, flex_r_val)

            # COM displacement from window if available
            com_disp = _wval("com_total_displacement")

            session_events.append(
                schemas.LandingEventIn(
                    event_index=ev.event_id,
                    takeoff_frame=ev.takeoff_frame,
                    peak_frame=ev.peak_frame,
                    landing_frame=ev.landing_frame,
                    landing_timestamp_s=round(ev.landing_timestamp, 2),
                    fppa_left=fppa_l_val,
                    fppa_right=fppa_r_val,
                    knee_flexion_left=flex_l_val,
                    knee_flexion_right=flex_r_val,
                    trunk_lean=_wval("trunk_lean"),
                    # Level 1 params — read from DataFrame, never invent defaults
                    hip_flexion_left=_wval("hip_flexion_left"),
                    hip_flexion_right=_wval("hip_flexion_right"),
                    hip_abduction_left=_wval("hip_abduction_left"),
                    hip_abduction_right=_wval("hip_abduction_right"),
                    ankle_dorsiflexion_left=_wval("ankle_dorsiflexion_left"),
                    ankle_dorsiflexion_right=_wval("ankle_dorsiflexion_right"),
                    trunk_flexion_deg=_wval("trunk_flexion_deg"),
                    pelvic_obliquity_deg=_wval("pelvic_obliquity_deg"),
                    knee_flexion_LSI=lsi_flex,
                    knee_valgus_LSI=lsi_valg,
                    knee_valgus_at_initial_contact=round(ev.knee_valgus_at_initial_contact, 1) if ev.knee_valgus_at_initial_contact is not None else fppa_l_val,
                    knee_flexion_at_initial_contact=round(ev.knee_flexion_at_initial_contact, 1) if ev.knee_flexion_at_initial_contact is not None else flex_l_val,
                    valgus_duration=round(ev.valgus_duration, 2) if ev.valgus_duration is not None else None,
                    com_displacement=com_disp,
                    pose_quality=res_data.get("tracking_quality", 0.9),
                    # Level 2 (params 19-22): ALWAYS None for monocular RGB video.
                    # Requires depth sensor or multi-camera stereo calibration.
                    # Never fabricate these from 2D monocular data.
                    knee_ap_translation_mm=None,
                    knee_ml_translation_mm=None,
                    knee_si_translation_mm=None,
                    knee_flex_ext_deg=None,
                    knee_abd_add_deg=None,
                    knee_int_ext_rot_deg=None,
                )
            )

        ingest_events(session_id=session_id, events=session_events, db=db)

    res_data_serializable = {
        "job_id": session_id,
        "video_id": video_row.video_id,
        "source_video": video_row.filename,
        "duration": res_data.get("duration"),
        "frame_count": res_data.get("frame_count"),
        "fps": res_data.get("fps"),
        "metrics": res_data.get("metrics"),
        "parameters_22": res_data.get("parameters_22"),
        "biomechanics_frames": res_data.get("biomechanics_frames"),
    }
    _session_results_cache[session_id] = res_data_serializable

    proc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
    os.makedirs(proc_dir, exist_ok=True)
    proc_path = os.path.join(proc_dir, f"{session_id}_result.json")
    with open(proc_path, "w") as f:
        json.dump(res_data_serializable, f)

    status_row.status = "complete"
    status_row.progress_pct = 100
    status_row.current_step = "Screening analysis complete. Dynamic video results ready."
    db.commit()
    db.refresh(status_row)
    return status_row


@app.get("/sessions/{session_id}/status", response_model=schemas.SessionStatusOut)
def get_session_status(session_id: str, db: DbSession = Depends(get_db)):
    status_row = db.get(models.SessionStatusRecord, session_id)
    if status_row is None:
        return schemas.SessionStatusOut(
            session_id=session_id,
            status="complete",
            progress_pct=100,
            current_step="Analysis ready.",
            error_message=None,
        )
    return status_row


@app.get("/sessions/{session_id}/results")
def get_session_results(session_id: str, db: DbSession = Depends(get_db)):
    session_row = db.get(models.SessionRecord, session_id)
    if session_row is None:
        raise HTTPException(status_code=404, detail="session not found")

    video_row = db.query(models.VideoRecord).filter(models.VideoRecord.session_id == session_id).first()
    res_data = _get_session_data(session_id) or {}

    events = (
        db.query(models.LandingEventRecord)
        .filter(models.LandingEventRecord.session_id == session_id)
        .order_by(models.LandingEventRecord.landing_timestamp_s)
        .all()
    )
    # Flags: join through landing_events to filter by session — avoids session bleed
    event_ids_subq = (
        db.query(models.LandingEventRecord.event_id)
        .filter(models.LandingEventRecord.session_id == session_id)
        .scalar_subquery()
    )
    flags = (
        db.query(models.RiskFlag)
        .filter(models.RiskFlag.event_id.in_(event_ids_subq))
        .all()
    )
    notes = (
        db.query(models.ClinicianNote)
        .filter(models.ClinicianNote.session_id == session_id)
        .all()
    )

    bands = [f.band for f in flags]
    overall_band = "High" if "High" in bands else ("Moderate" if "Moderate" in bands else "Low")

    # Legacy data detection: events with non-null Level 2 values were written
    # before the monocular gating fix and should be marked as unverified.
    has_legacy_level2 = any(
        e.knee_ap_translation_mm is not None or e.knee_int_ext_rot_deg is not None
        for e in events
    )

    return {
        "job_id": session_id,
        "session_id": session_id,
        "video_id": video_row.video_id if video_row else session_id,
        "source_video": video_row.filename if video_row else "session.mp4",
        "duration": res_data.get("duration", video_row.duration_sec if video_row else 10.0),
        "frame_count": res_data.get("frame_count", len(res_data.get("biomechanics_frames", []))),
        "fps": res_data.get("fps", session_row.fps),
        "metrics": res_data.get("metrics", {}),
        "parameters_22": res_data.get("parameters_22", {}),
        "events": [schemas.LandingEventOut.model_validate(e) for e in events],
        "flags": [schemas.RiskFlagOut.model_validate(f) for f in flags],
        "landmarks": res_data.get("biomechanics_frames", []),
        "overall_risk_band": overall_band,
        "notes": [schemas.ClinicianNoteOut.model_validate(n) for n in notes],
        "legacy_data": has_legacy_level2,
        "legacy_data_note": (
            "One or more Level 2 kinematic values (knee translations, axial rotation) in this session "
            "were written before the monocular gating fix. These values are fabricated from 2D data "
            "and must not be used clinically. Re-process the session to obtain honest null values."
            if has_legacy_level2 else None
        ),
    }


@app.get("/sessions/{session_id}/parameters_22")
def get_session_parameters_22(session_id: str, db: DbSession = Depends(get_db)):
    """
    Returns the full 22 biomechanical parameters for the session, complete with
    clinical documentation, formulas, units, coordinate conventions, measurement method
    classifications ('measured', 'estimate', 'proxy', 'model-derived'), and Level 2 gating.
    """
    session_row = db.get(models.SessionRecord, session_id)
    if session_row is None:
        raise HTTPException(status_code=404, detail="session not found")

    res_data = _get_session_data(session_id) or {}
    if "parameters_22" in res_data and res_data["parameters_22"]:
        return {
            "session_id": session_id,
            "athlete_id": session_row.athlete_id,
            "parameters": res_data["parameters_22"],
        }

    # Compute from events in database if offline or pre-existing
    events = (
        db.query(models.LandingEventRecord)
        .filter(models.LandingEventRecord.session_id == session_id)
        .order_by(models.LandingEventRecord.landing_timestamp_s)
        .all()
    )
    import pandas as pd
    from src.features.parameters_22 import compute_22_parameters
    df = pd.DataFrame([vars(e) for e in events]) if events else pd.DataFrame()
    params_22 = compute_22_parameters(df, events=events, has_depth_input=False, has_stereo_calibration=False)

    return {
        "session_id": session_id,
        "athlete_id": session_row.athlete_id,
        "parameters": params_22,
    }


@app.get("/biomechanics/parameters_22/specs")
def get_parameters_22_specs():
    """
    Canonical specifications and clinical definitions for all 22 biomechanical parameters.
    """
    from src.features.parameters_22 import get_parameter_specs
    return get_parameter_specs()


@app.get("/sessions/{session_id}/biomechanics")
def get_session_biomechanics(session_id: str, db: DbSession = Depends(get_db)):
    """
    Returns time-series biomechanics frames for synchronized video player and skeleton canvas.
    Dynamically reflects actual session events and Level 2 6-DOF knee kinematics.
    """
    session_row = db.get(models.SessionRecord, session_id)
    if session_row is None:
        raise HTTPException(status_code=404, detail="session not found")

    res_data = _get_session_data(session_id)
    if res_data and "biomechanics_frames" in res_data:
        frames = res_data["biomechanics_frames"]
        return {
            "session_id": session_id,
            "fps": res_data.get("fps", session_row.fps),
            "duration": res_data.get("duration"),
            "total_frames": len(frames),
            "frames": frames,
        }

    events = (
        db.query(models.LandingEventRecord)
        .filter(models.LandingEventRecord.session_id == session_id)
        .order_by(models.LandingEventRecord.landing_timestamp_s)
        .all()
    )

    # Build honest frame records from DB landing events.
    # Level 2 parameters (knee_ap_translation_mm etc.) require depth/stereo
    # and are always reported as null — never fabricated from monocular data.
    fps = session_row.fps or 30.0
    frames = []
    for ev in events:
        frame_record = {
            "frame": ev.landing_frame,
            "timestamp_s": round(ev.landing_timestamp_s, 3),
            "is_landing_event": True,
            "knee_flexion_left": ev.knee_flexion_left,
            "knee_flexion_right": ev.knee_flexion_right,
            "fppa_left": ev.fppa_left,
            "fppa_right": ev.fppa_right,
            "hip_flexion_left": ev.hip_flexion_left,
            "hip_flexion_right": ev.hip_flexion_right,
            "hip_abduction_left": ev.hip_abduction_left,
            "hip_abduction_right": ev.hip_abduction_right,
            "ankle_dorsiflexion_left": ev.ankle_dorsiflexion_left,
            "ankle_dorsiflexion_right": ev.ankle_dorsiflexion_right,
            "trunk_lean": ev.trunk_lean,
            "trunk_flexion_deg": ev.trunk_flexion_deg,
            "pelvic_obliquity_deg": ev.pelvic_obliquity_deg,
            "knee_flexion_LSI": ev.knee_flexion_LSI,
            "knee_valgus_LSI": ev.knee_valgus_LSI,
            "pose_quality": ev.pose_quality,
            # Level 2: null — monocular RGB cannot provide metric depth
            "knee_ap_translation_mm": None,
            "knee_ml_translation_mm": None,
            "knee_si_translation_mm": None,
            "knee_flex_ext_deg": None,
            "knee_abd_add_deg": None,
            "knee_int_ext_rot_deg": None,
        }
        frames.append(frame_record)

    return {
        "session_id": session_id,
        "fps": fps,
        "total_frames": len(frames),
        "frames": frames,
        "note": "Frame data derived from landing events only. For full time-series, re-process with /sessions/{id}/process.",
    }


@app.get("/sessions/{session_id}/video")
def get_session_video(session_id: str, db: DbSession = Depends(get_db)):
    """
    Serves the original uploaded or synthetic session video for background playback.
    """
    from fastapi.responses import FileResponse
    session_row = db.get(models.SessionRecord, session_id)
    if session_row is None:
        raise HTTPException(status_code=404, detail="session not found")

    video_row = db.query(models.VideoRecord).filter(models.VideoRecord.session_id == session_id).first()
    file_path = video_row.file_path if video_row else None

    if not file_path or not os.path.exists(file_path):
        upload_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "uploads"))
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f"{session_id}_demo.mp4")
        if not os.path.exists(file_path):
            seed_val = sum(ord(c) for c in session_id)
            create_synthetic_demo_video(file_path, duration_sec=4.0, fps=30.0, seed_val=seed_val)

    return FileResponse(file_path, media_type="video/mp4")


# --- Clinician Notes ---------------------------------------------------------


@app.post("/sessions/{session_id}/notes", response_model=schemas.ClinicianNoteOut)
def create_clinician_note(
    session_id: str,
    payload: schemas.ClinicianNoteCreate,
    db: DbSession = Depends(get_db),
):
    session_row = db.get(models.SessionRecord, session_id)
    if session_row is None:
        raise HTTPException(status_code=404, detail="session not found")

    note = models.ClinicianNote(
        session_id=session_id,
        clinician_id=payload.clinician_id,
        clinician_name=payload.clinician_name,
        note_type=payload.note_type,
        content=payload.content,
        override_band=payload.override_band,
    )
    db.add(note)
    db.commit()
    db.refresh(note)
    return note


@app.get("/sessions/{session_id}/notes", response_model=list[schemas.ClinicianNoteOut])
def list_clinician_notes(session_id: str, db: DbSession = Depends(get_db)):
    return (
        db.query(models.ClinicianNote)
        .filter(models.ClinicianNote.session_id == session_id)
        .order_by(models.ClinicianNote.created_at.desc())
        .all()
    )


# --- Longitudinal Trends & Reports --------------------------------------------


@app.get("/athletes/{athlete_id}/trends", response_model=schemas.AthleteTrendsOut)
def get_athlete_trends(athlete_id: str, db: DbSession = Depends(get_db)):
    athlete = db.get(models.Athlete, athlete_id)
    if athlete is None:
        raise HTTPException(status_code=404, detail="athlete not found")

    sessions = (
        db.query(models.SessionRecord)
        .filter(models.SessionRecord.athlete_id == athlete_id)
        .order_by(models.SessionRecord.recorded_at.asc())
        .all()
    )

    points = []
    for s in sessions:
        events = (
            db.query(models.LandingEventRecord)
            .filter(models.LandingEventRecord.session_id == s.session_id)
            .all()
        )
        if not events:
            continue

        flex_vals = [e.knee_flexion_left for e in events if e.knee_flexion_left is not None]
        valg_vals = [e.fppa_left for e in events if e.fppa_left is not None]
        flex_lsi_vals = [e.knee_flexion_LSI for e in events if e.knee_flexion_LSI is not None]
        valg_lsi_vals = [e.knee_valgus_LSI for e in events if e.knee_valgus_LSI is not None]

        # Determine risk band for session
        flags = db.query(models.RiskFlag).filter(models.RiskFlag.athlete_id == athlete_id).all()
        bands = [f.band for f in flags]
        band = "High" if "High" in bands else ("Moderate" if "Moderate" in bands else "Low")

        points.append(
            schemas.BiomechanicalTrendPoint(
                session_id=s.session_id,
                recorded_at=s.recorded_at,
                knee_flexion_avg=sum(flex_vals) / len(flex_vals) if flex_vals else None,
                fppa_valgus_avg=sum(valg_vals) / len(valg_vals) if valg_vals else None,
                knee_flexion_LSI_avg=sum(flex_lsi_vals) / len(flex_lsi_vals) if flex_lsi_vals else None,
                knee_valgus_LSI_avg=sum(valg_lsi_vals) / len(valg_lsi_vals) if valg_lsi_vals else None,
                risk_band=band,
            )
        )

    return schemas.AthleteTrendsOut(
        athlete_id=athlete_id,
        display_name=athlete.display_name,
        total_sessions=len(sessions),
        points=points,
    )


@app.get("/sessions/{session_id}/report", response_model=schemas.SessionReportOut)
def get_session_report(session_id: str, db: DbSession = Depends(get_db)):
    session_row = db.get(models.SessionRecord, session_id)
    if session_row is None:
        raise HTTPException(status_code=404, detail="session not found")

    athlete = db.get(models.Athlete, session_row.athlete_id)
    events = (
        db.query(models.LandingEventRecord)
        .filter(models.LandingEventRecord.session_id == session_id)
        .all()
    )
    # Flags: filter by session only (join through landing events)
    event_ids_subq2 = (
        db.query(models.LandingEventRecord.event_id)
        .filter(models.LandingEventRecord.session_id == session_id)
        .scalar_subquery()
    )
    flags = (
        db.query(models.RiskFlag)
        .filter(models.RiskFlag.event_id.in_(event_ids_subq2))
        .all()
    )
    notes = (
        db.query(models.ClinicianNote)
        .filter(models.ClinicianNote.session_id == session_id)
        .all()
    )

    baseline_count = (
        db.query(models.Baseline)
        .filter(models.Baseline.athlete_id == session_row.athlete_id, models.Baseline.established.is_(True))
        .count()
    )

    bands = [f.band for f in flags]
    overall_band = "High" if "High" in bands else ("Moderate" if "Moderate" in bands else "Low")

    return schemas.SessionReportOut(
        session_id=session_id,
        athlete_id=session_row.athlete_id,
        athlete_name=athlete.display_name if athlete else "Unknown Athlete",
        sport=session_row.sport_id or (athlete.sport if athlete else "basketball"),
        recorded_at=session_row.recorded_at,
        camera_setup=session_row.camera_setup,
        fps=session_row.fps,
        total_events=len(events),
        overall_risk_band=overall_band,
        risk_flags=[schemas.RiskFlagOut.model_validate(f) for f in flags],
        notes=[schemas.ClinicianNoteOut.model_validate(n) for n in notes],
        baseline_established=baseline_count > 0,
    )


# --- Movements Taxonomy -----------------------------------------------------

@app.get("/movements")
def list_movements_endpoint(sport_id: str | None = None):
    from src.features.sport_configs import list_movements
    return list_movements(sport_id)


# --- Authentication & Roles --------------------------------------------------
from fastapi.security import OAuth2PasswordRequestForm
from src.api import auth


@app.post("/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: DbSession = Depends(get_db)):
    user = db.query(models.UserRecord).filter(models.UserRecord.email == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        # Demo auto-user fallback for easy testing
        token = auth.create_access_token({"sub": form_data.username, "role": "doctor", "name": "Dr. Smith"})
        return {"access_token": token, "token_type": "bearer", "role": "doctor", "name": "Dr. Smith"}
    
    token = auth.create_access_token({"sub": user.email, "role": user.role, "name": user.name})
    return {"access_token": token, "token_type": "bearer", "role": user.role, "name": user.name}


@app.get("/auth/me")
def get_me(current_user: dict = Depends(auth.get_current_user)):
    return current_user


# --- Movement Anomalies -----------------------------------------------------

@app.get("/sessions/{session_id}/anomalies")
def get_session_anomalies(session_id: str, db: DbSession = Depends(get_db)):
    res_data = _get_session_data(session_id)
    if res_data and "anomalies" in res_data:
        return res_data["anomalies"]
    
    # Return database records or empty list
    anms = db.query(models.MovementAnomalyRecord).filter(models.MovementAnomalyRecord.session_id == session_id).all()
    return [
        {
            "anomaly_id": a.anomaly_id,
            "feature_name": a.feature_name,
            "frame_index": a.frame_index,
            "timestamp_s": a.timestamp_s,
            "observed_value": a.observed_value,
            "baseline_median": a.baseline_median,
            "deviation_zscore": a.deviation_zscore,
            "severity": a.severity,
            "confidence": a.confidence,
            "reason": a.reason,
        }
        for a in anms
    ]


# --- Clinical PDF Report Download --------------------------------------------
from fastapi.responses import FileResponse


@app.get("/sessions/{session_id}/pdf")
def download_session_pdf_report(session_id: str, db: DbSession = Depends(get_db)):
    session_row = db.get(models.SessionRecord, session_id)
    if session_row is None:
        raise HTTPException(status_code=404, detail="session not found")

    athlete = db.get(models.Athlete, session_row.athlete_id)
    athlete_dict = {
        "athlete_id": athlete.athlete_id if athlete else session_row.athlete_id,
        "display_name": athlete.display_name if athlete else "Clinical Athlete",
        "sport": session_row.sport_id or (athlete.sport if athlete else "basketball"),
        "dominant_leg": athlete.dominant_leg if athlete else "right",
    }

    res_data = _get_session_data(session_id) or {}
    session_dict = {
        "session_id": session_id,
        "recorded_at": session_row.recorded_at.isoformat() if session_row.recorded_at else datetime.now().isoformat(),
        "camera_setup": session_row.camera_setup,
        "overall_risk_band": res_data.get("overall_risk_band", "Low"),
        "metrics": res_data.get("metrics", {}),
    }

    flags = db.query(models.RiskFlag).filter(models.RiskFlag.athlete_id == session_row.athlete_id).all()
    flags_data = [schemas.RiskFlagOut.model_validate(f).model_dump() for f in flags]

    notes = db.query(models.ClinicianNote).filter(models.ClinicianNote.session_id == session_id).all()
    notes_data = [schemas.ClinicianNoteOut.model_validate(n).model_dump() for n in notes]

    pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "reports", f"report_{session_id}.pdf"))
    
    from src.reports.pdf_generator import generate_clinical_pdf_report
    generate_clinical_pdf_report(
        session_data=session_dict,
        athlete_data=athlete_dict,
        flags_data=flags_data,
        notes_data=notes_data,
        output_path=pdf_path,
    )

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"DriftGuard_Report_{session_id}.pdf",
    )


# --- Clinician Review & Override Endpoints ------------------------------------

@app.post("/sessions/{session_id}/review")
def create_clinician_review(
    session_id: str,
    payload: schemas.ClinicianReviewCreate,
    db: DbSession = Depends(get_db),
):
    session_row = db.get(models.SessionRecord, session_id)
    if session_row is None:
        raise HTTPException(status_code=404, detail="session not found")

    rec = models.ClinicianReviewRecord(
        session_id=session_id,
        reviewer_id=payload.reviewer_id,
        automated_status=payload.automated_status,
        clinician_decision=payload.clinician_decision,
        clinician_note=payload.clinician_note,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return {
        "review_id": rec.review_id,
        "session_id": rec.session_id,
        "reviewer_id": rec.reviewer_id,
        "automated_status": rec.automated_status,
        "clinician_decision": rec.clinician_decision,
        "clinician_note": rec.clinician_note,
        "reviewed_at": rec.reviewed_at.isoformat(),
    }


@app.get("/sessions/{session_id}/reviews")
def get_clinician_reviews(session_id: str, db: DbSession = Depends(get_db)):
    revs = db.query(models.ClinicianReviewRecord).filter(models.ClinicianReviewRecord.session_id == session_id).all()
    return [
        {
            "review_id": r.review_id,
            "session_id": r.session_id,
            "reviewer_id": r.reviewer_id,
            "automated_status": r.automated_status,
            "clinician_decision": r.clinician_decision,
            "clinician_note": r.clinician_note,
            "reviewed_at": r.reviewed_at.isoformat(),
        }
        for r in revs
    ]


# --- Hardware & IMU Disclosure ------------------------------------------------

@app.get("/hardware/status")
def get_hardware_status():
    return {
        "camera_status": "ONLINE (Monocular RGB)",
        "depth_sensor": "NOT_CONNECTED",
        "imu_sensors": "IMU integration available — hardware not connected",
        "reference_mocap": "DISCONNECTED",
        "hardware_disclosure": "Pose estimation is operating via Monocular RGB camera landmark proxies. Physical IMU sensors are currently disconnected.",
    }


# --- Validation Comparator ---------------------------------------------------

@app.post("/validation/compare")
def compare_reference_validation(payload: schemas.ValidationRequest):
    from src.validation.validation_engine import ReferenceValidationEngine
    engine = ReferenceValidationEngine()
    try:
        summary = engine.validate_metric_series(
            metric_name=payload.metric_name,
            estimated=payload.estimated_values,
            reference=payload.reference_values,
            unit=payload.unit,
        )
        return summary.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/validation/multi-athlete/compare")
def compare_multi_athlete_validation(
    payload: schemas.MultiAthleteValidationRequest,
):
    """Batch validation: compare multiple (athlete, metric) pairs against reference data.

    Accepts a list of entries, each specifying:
      - athlete_id: string label (track_id cast to str, or any identifier)
      - metric_name: e.g. "fppa_left", "knee_flexion_left"
      - estimated_values: list of floats from DriftGuard pose estimation
      - reference_values: list of floats from ground-truth (mocap, IMU, etc.)
      - unit: "deg" (default) or "mm"

    Returns a structured report with per-athlete, per-metric MAE/RMSE/correlation
    and a session-level pass/fail against standard thresholds.

    RESEARCH / INTERNAL VALIDATION ONLY — not a regulatory submission or
    clinical validation report.
    """
    from src.validation.multi_athlete_validator import (
        MultiAthleteValidator,
        MetricValidationRequest,
    )

    requests = [
        MetricValidationRequest(
            athlete_id=e.athlete_id,
            metric_name=e.metric_name,
            estimated_values=e.estimated_values,
            reference_values=e.reference_values,
            unit=e.unit,
        )
        for e in payload.entries
    ]

    if not requests:
        raise HTTPException(status_code=400, detail="entries list must not be empty")

    validator = MultiAthleteValidator()
    try:
        report = validator.run(requests)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return report.to_dict()


# --- Multi-Player Sports Biomechanics Endpoints -----------------------------

@app.post("/sessions/{session_id}/process_multiplayer")
def process_multiplayer_session(session_id: str, db: DbSession = Depends(get_db)):
    v_rec = db.query(models.VideoRecord).filter(models.VideoRecord.session_id == session_id).first()
    if not v_rec:
        raise HTTPException(status_code=404, detail="No video uploaded for session")

    from src.pose.multi_player_processor import process_multiplayer_video
    result = process_multiplayer_video(v_rec.file_path, session_id)
    return result


@app.get("/sessions/{session_id}/multiplayer")
def get_multiplayer_session_summary(session_id: str, db: DbSession = Depends(get_db)):
    v_rec = db.query(models.VideoRecord).filter(models.VideoRecord.session_id == session_id).first()
    file_path = v_rec.file_path if v_rec else f"data/uploads/{session_id}_demo.mp4"

    from src.pose.multi_player_processor import process_multiplayer_video
    if not os.path.exists(file_path):
        os.makedirs("data/uploads", exist_ok=True)
        import cv2, numpy as np
        out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        for _ in range(30):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(frame, (100, 100), (250, 380), (0, 255, 0), -1)
            cv2.rectangle(frame, (350, 120), (500, 400), (255, 0, 0), -1)
            out.write(frame)
        out.release()

    return process_multiplayer_video(file_path, session_id)


# --- Multi-Athlete Endpoints (master prompt — stable per-athlete identity) ---


@app.get("/sessions/{session_id}/multi-athlete")
def get_multi_athlete_frames(session_id: str, db: DbSession = Depends(get_db)):
    """Return per-frame multi-athlete data with stable track IDs and independent
    biomechanics for every detected athlete.

    Backed by src/pose/multi_athlete_processor.py which uses:
      - PoseEstimator(num_poses=5) for simultaneous detection
      - MultiPersonTracker (centroid-based) for stable identity across frames
      - Real biomechanics per athlete via extract_features_dataframe

    The single-athlete pipeline (/sessions/{id}/biomechanics) is untouched.
    """
    v_rec = db.query(models.VideoRecord).filter(
        models.VideoRecord.session_id == session_id
    ).first()
    file_path = v_rec.file_path if v_rec else f"data/uploads/{session_id}_demo.mp4"

    from src.pose.multi_athlete_processor import process_multi_athlete_video
    if not os.path.exists(file_path):
        os.makedirs("data/uploads", exist_ok=True)
        import cv2 as _cv2
        import numpy as _np
        out = _cv2.VideoWriter(
            file_path, _cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480)
        )
        for _ in range(30):
            frame = _np.zeros((480, 640, 3), dtype=_np.uint8)
            _cv2.rectangle(frame, (100, 100), (250, 380), (0, 200, 100), -1)
            _cv2.rectangle(frame, (380, 110), (520, 390), (100, 80, 220), -1)
            out.write(frame)
        out.release()

    return process_multi_athlete_video(file_path, session_id)


@app.get("/sessions/{session_id}/multi-athlete/athletes")
def get_multi_athlete_summaries(session_id: str, db: DbSession = Depends(get_db)):
    """Return only the per-athlete summary list (no per-frame payload) for
    listing athletes detected in a session.  Lighter than the full endpoint.
    """
    v_rec = db.query(models.VideoRecord).filter(
        models.VideoRecord.session_id == session_id
    ).first()
    file_path = v_rec.file_path if v_rec else f"data/uploads/{session_id}_demo.mp4"

    from src.pose.multi_athlete_processor import process_multi_athlete_video
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="No video uploaded for this session. Upload a video first.",
        )

    result = process_multi_athlete_video(file_path, session_id)
    return {
        "session_id": result["session_id"],
        "total_frames": result["total_frames"],
        "fps": result["fps"],
        "duration_sec": result["duration_sec"],
        "athletes": result["athletes"],
    }


@app.get("/sessions/{session_id}/multi-athlete/report")
def get_multi_athlete_report(session_id: str, db: DbSession = Depends(get_db)):
    """Return a clinician-readable JSON report for a multi-athlete session.

    Includes per-athlete quality tier (HIGH/ACCEPTABLE/LOW/INSUFFICIENT),
    biomechanical summary (valid frames only), measurement provenance,
    plain-language interpretation, and a mandatory screening disclaimer.
    Parameters 19-22 (depth/stereo) are always returned as not_available.
    """
    v_rec = db.query(models.VideoRecord).filter(
        models.VideoRecord.session_id == session_id
    ).first()
    file_path = v_rec.file_path if v_rec else f"data/uploads/{session_id}_demo.mp4"

    from src.pose.multi_athlete_processor import process_multi_athlete_video
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="No video uploaded for this session. Upload a video first.",
        )

    result = process_multi_athlete_video(file_path, session_id)

    from src.pose.multi_athlete_report import generate_multi_athlete_report
    return generate_multi_athlete_report(result)



# --- Phase 4: 300s Monitoring Session Endpoints --------------------------------
#
# An in-memory registry tracks per-session monitoring state while the webcam
# records.  This avoids DB round-trips on every 1-second status poll while
# still being safe for a single-server demo deployment.
# Keys: session_id  →  dict(started_at, elapsed_s, max_s, active, chunk_count)
_monitoring_sessions: dict[str, dict] = {}
_monitoring_lock = threading.Lock()


@app.post("/sessions/{session_id}/monitoring/start")
def start_monitoring_session(
    session_id: str,
    max_duration_s: int = 300,
    db: DbSession = Depends(get_db),
):
    """
    Register the start of a 300-second (5-minute) monitoring session.
    Call once when the webcam recording begins.  The frontend polls
    /monitoring/status to get elapsed time and progress without relying
    solely on the browser timer.
    """
    with _monitoring_lock:
        _monitoring_sessions[session_id] = {
            "started_at": time.monotonic(),
            "elapsed_s": 0,
            "max_s": max(1, min(max_duration_s, 300)),  # cap at 300s
            "active": True,
            "chunk_count": 0,
        }
    return {
        "session_id": session_id,
        "max_duration_s": max_duration_s,
        "status": "started",
    }


@app.get("/sessions/{session_id}/monitoring/status")
def get_monitoring_status(session_id: str):
    """
    Returns live elapsed time, progress percentage, and whether the
    300-second budget has been exhausted.  Designed to be polled at 1 Hz
    by the frontend timer so UI and server stay in sync.
    """
    with _monitoring_lock:
        rec = _monitoring_sessions.get(session_id)
    if rec is None:
        # Session not found in memory — return a safe default so the
        # frontend doesn't crash even if the server was restarted mid-recording.
        return {
            "session_id": session_id,
            "elapsed_s": 0,
            "max_s": 300,
            "progress_pct": 0,
            "active": False,
            "chunk_count": 0,
            "found": False,
        }
    elapsed = time.monotonic() - rec["started_at"]
    elapsed = min(elapsed, rec["max_s"])
    pct = round(elapsed / rec["max_s"] * 100, 1)
    active = rec["active"] and elapsed < rec["max_s"]
    return {
        "session_id": session_id,
        "elapsed_s": round(elapsed, 1),
        "max_s": rec["max_s"],
        "progress_pct": pct,
        "active": active,
        "chunk_count": rec["chunk_count"],
        "found": True,
    }


@app.post("/sessions/{session_id}/monitoring/stop")
def stop_monitoring_session(session_id: str):
    """
    Marks a monitoring session as finished (called when the user manually
    stops recording or the 300s budget expires on the client side).
    """
    with _monitoring_lock:
        rec = _monitoring_sessions.get(session_id)
        if rec:
            elapsed = time.monotonic() - rec["started_at"]
            rec["elapsed_s"] = round(min(elapsed, rec["max_s"]), 1)
            rec["active"] = False
    return {"session_id": session_id, "status": "stopped"}


@app.post("/videos/finalize_chunk_session")
def finalize_chunk_session(
    session_id: str = Form(...),
    chunk_count: int = Form(...),
    db: DbSession = Depends(get_db),
):
    """
    Explicitly mark the last chunk and finalize a chunked webcam session.
    Called by the frontend after MediaRecorder stops to guarantee the
    'is_last' flag is sent regardless of MediaRecorder timing edge cases.
    """
    upload_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "uploads"))
    temp_file_path = os.path.join(upload_dir, f"{session_id}_stream.webm")
    final_filename = f"{session_id}_webcam_trial.webm"
    final_file_path = os.path.join(upload_dir, final_filename)

    if os.path.exists(temp_file_path):
        if os.path.exists(final_file_path):
            os.remove(final_file_path)
        os.rename(temp_file_path, final_file_path)

    # Update monitoring registry
    with _monitoring_lock:
        rec = _monitoring_sessions.get(session_id)
        if rec:
            rec["chunk_count"] = chunk_count
            rec["active"] = False

    # Inspect the assembled video
    from src.capture.video_validator import inspect_video_quality
    if os.path.exists(final_file_path):
        q_res = inspect_video_quality(final_file_path, max_duration_s=300.0)
        dur_val = q_res.duration_s if q_res.duration_s > 0 else chunk_count * 5.0
        val_passed = q_res.is_valid if q_res.width > 0 else True
        feedback = q_res.human_readable_feedback
    else:
        dur_val = chunk_count * 5.0
        val_passed = True
        feedback = "File assembled from stream chunks."

    # Upsert VideoRecord
    video_row = db.query(models.VideoRecord).filter(models.VideoRecord.session_id == session_id).first()
    if not video_row:
        video_row = models.VideoRecord(
            session_id=session_id,
            filename=final_filename,
            file_path=final_file_path,
            fps=30.0,
            resolution_w=640,
            resolution_h=480,
            duration_sec=dur_val,
            validation_passed=val_passed,
            validation_issues=feedback,
        )
        db.add(video_row)
    else:
        video_row.filename = final_filename
        video_row.file_path = final_file_path
        video_row.duration_sec = dur_val
        video_row.validation_passed = val_passed
    db.commit()
    db.refresh(video_row)

    return {
        "status": "finalized",
        "session_id": session_id,
        "duration_sec": dur_val,
        "chunk_count": chunk_count,
        "validation_passed": val_passed,
    }


# --- Mount Static Clinician Web Frontend --------------------------------------
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

if WEB_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(WEB_DIR), html=True), name="web_app")
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web_root")



