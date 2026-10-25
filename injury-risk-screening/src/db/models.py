"""
SQLAlchemy ORM models, mirroring src/db/schema.sql exactly.

Uses SQLAlchemy 2.0's typed declarative style. `get_engine()` defaults to
a local SQLite file for development/demo use (no PostgreSQL server
required to run the FastAPI app or tests) but accepts any SQLAlchemy URL
via the DATABASE_URL environment variable, so swapping to Postgres in
production is a one-line config change, not a code change.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, date, UTC

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Athlete(Base):
    __tablename__ = "athletes"

    athlete_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("ath"))
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    sport: Mapped[str | None] = mapped_column(String)
    date_of_birth: Mapped[date | None] = mapped_column(Date)
    dominant_leg: Mapped[str | None] = mapped_column(String)
    team_or_school: Mapped[str | None] = mapped_column(String)
    consent_given_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    __table_args__ = (
        CheckConstraint("dominant_leg in ('left','right','unknown')", name="ck_dominant_leg"),
    )

    sessions: Mapped[list["SessionRecord"]] = relationship(back_populates="athlete")


class SessionRecord(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("ses"))
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.athlete_id"), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    camera_setup: Mapped[str] = mapped_column(String, nullable=False)
    fps: Mapped[float] = mapped_column(Float, nullable=False)
    sport_id: Mapped[str | None] = mapped_column(String)
    raw_video_deleted_at: Mapped[datetime | None] = mapped_column(DateTime)
    notes: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    athlete: Mapped[Athlete] = relationship(back_populates="sessions")
    events: Mapped[list["LandingEventRecord"]] = relationship(back_populates="session")


class VideoRecord(Base):
    __tablename__ = "videos"

    video_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("vid"))
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    fps: Mapped[float] = mapped_column(Float, nullable=False)
    resolution_w: Mapped[int] = mapped_column(Integer, nullable=False)
    resolution_h: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_sec: Mapped[float] = mapped_column(Float, nullable=False)
    validation_passed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    validation_issues: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class SessionStatusRecord(Base):
    __tablename__ = "session_statuses"

    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), primary_key=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    progress_pct: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    current_step: Mapped[str | None] = mapped_column(String)
    error_message: Mapped[str | None] = mapped_column(String)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class ClinicianNote(Base):
    __tablename__ = "clinician_notes"

    note_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("nte"))
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), nullable=False)
    clinician_id: Mapped[str] = mapped_column(String, nullable=False, default="cln_default")
    clinician_name: Mapped[str] = mapped_column(String, nullable=False, default="Dr. Standard")
    note_type: Mapped[str] = mapped_column(String, nullable=False, default="general")
    content: Mapped[str] = mapped_column(String, nullable=False)
    override_band: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class LandingEventRecord(Base):
    __tablename__ = "landing_events"

    event_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("evt"))
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), nullable=False)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.athlete_id"), nullable=False)
    event_index: Mapped[int] = mapped_column(Integer, nullable=False)
    takeoff_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    peak_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    landing_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    landing_timestamp_s: Mapped[float] = mapped_column(Float, nullable=False)
    fppa_left: Mapped[float | None] = mapped_column(Float)
    fppa_right: Mapped[float | None] = mapped_column(Float)
    knee_flexion_left: Mapped[float | None] = mapped_column(Float)
    knee_flexion_right: Mapped[float | None] = mapped_column(Float)
    trunk_lean: Mapped[float | None] = mapped_column(Float)
    hip_flexion_left: Mapped[float | None] = mapped_column(Float)
    hip_flexion_right: Mapped[float | None] = mapped_column(Float)
    hip_abduction_left: Mapped[float | None] = mapped_column(Float)
    hip_abduction_right: Mapped[float | None] = mapped_column(Float)
    ankle_dorsiflexion_left: Mapped[float | None] = mapped_column(Float)
    ankle_dorsiflexion_right: Mapped[float | None] = mapped_column(Float)
    trunk_flexion_deg: Mapped[float | None] = mapped_column(Float)
    pelvic_obliquity_deg: Mapped[float | None] = mapped_column(Float)
    knee_flexion_LSI: Mapped[float | None] = mapped_column(Float)
    knee_valgus_LSI: Mapped[float | None] = mapped_column(Float)
    knee_valgus_at_initial_contact: Mapped[float | None] = mapped_column(Float)
    knee_flexion_at_initial_contact: Mapped[float | None] = mapped_column(Float)
    valgus_duration: Mapped[float | None] = mapped_column(Float)
    com_displacement: Mapped[float | None] = mapped_column(Float)
    pose_quality: Mapped[float | None] = mapped_column(Float)
    # Level 2 6-DOF Knee Kinematics
    knee_ap_translation_mm: Mapped[float | None] = mapped_column(Float)
    knee_ml_translation_mm: Mapped[float | None] = mapped_column(Float)
    knee_si_translation_mm: Mapped[float | None] = mapped_column(Float)
    knee_flex_ext_deg: Mapped[float | None] = mapped_column(Float)
    knee_abd_add_deg: Mapped[float | None] = mapped_column(Float)
    knee_int_ext_rot_deg: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    session: Mapped[SessionRecord] = relationship(back_populates="events")


class Baseline(Base):
    __tablename__ = "baselines"

    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.athlete_id"), primary_key=True)
    feature_name: Mapped[str] = mapped_column(String, primary_key=True)
    n_events: Mapped[int] = mapped_column(Integer, nullable=False)
    established: Mapped[bool] = mapped_column(Boolean, nullable=False)
    median_value: Mapped[float | None] = mapped_column(Float)
    iqr_value: Mapped[float | None] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class RiskFlag(Base):
    __tablename__ = "risk_flags"

    flag_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("flg"))
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.athlete_id"), nullable=False)
    feature_name: Mapped[str] = mapped_column(String, nullable=False)
    event_id: Mapped[str] = mapped_column(ForeignKey("landing_events.event_id"), nullable=False)
    band: Mapped[str] = mapped_column(String, nullable=False)
    deviation_iqr: Mapped[float | None] = mapped_column(Float)
    headline: Mapped[str] = mapped_column(String, nullable=False)
    detail: Mapped[str] = mapped_column(String, nullable=False)
    citation: Mapped[str | None] = mapped_column(String)
    reviewed_by_clinician_id: Mapped[str | None] = mapped_column(String)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime)
    clinician_note: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    __table_args__ = (
        CheckConstraint("band in ('Low','Moderate','High')", name="ck_risk_band"),
    )


class Clinician(Base):
    __tablename__ = "clinicians"

    clinician_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("cln"))
    name: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str | None] = mapped_column(String)
    hospital: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class ThermalReading(Base):
    """Hardware-dependent: populated only when a thermal camera add-on is
    in use. See src/sensors/thermal_capture.py."""

    __tablename__ = "thermal_readings"

    reading_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("thr"))
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), nullable=False)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.athlete_id"), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    left_knee_temp_c: Mapped[float | None] = mapped_column(Float)
    right_knee_temp_c: Mapped[float | None] = mapped_column(Float)
    bilateral_asymmetry_c: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class UserRecord(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("usr"))
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False, default="doctor")
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    __table_args__ = (
        CheckConstraint("role in ('doctor','physiotherapist','coach','admin')", name="ck_user_role"),
    )


class MovementRecord(Base):
    __tablename__ = "movements"

    movement_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("mvt"))
    sport_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class BiomechanicalFeatureFrameRecord(Base):
    """Structured, frame-level temporal storage for semantic joint measurements."""

    __tablename__ = "biomechanical_feature_frames"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("bff"))
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), index=True, nullable=False)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.athlete_id"), index=True, nullable=False)
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_s: Mapped[float] = mapped_column(Float, nullable=False)
    body_region: Mapped[str] = mapped_column(String, index=True, nullable=False)
    feature_name: Mapped[str] = mapped_column(String, index=True, nullable=False)
    feature_category: Mapped[str] = mapped_column(String, nullable=False)
    side: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    visibility: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    source: Mapped[str] = mapped_column(String, nullable=False, default="mediapipe_pose")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class MovementAnomalyRecord(Base):
    """Unsupervised & statistical movement anomaly records."""

    __tablename__ = "movement_anomalies"

    anomaly_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("anm"))
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), index=True, nullable=False)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.athlete_id"), index=True, nullable=False)
    feature_name: Mapped[str] = mapped_column(String, nullable=False)
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_s: Mapped[float] = mapped_column(Float, nullable=False)
    observed_value: Mapped[float] = mapped_column(Float, nullable=False)
    baseline_median: Mapped[float] = mapped_column(Float, nullable=False)
    deviation_zscore: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[str] = mapped_column(String, nullable=False, default="MODERATE")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.9)
    reason: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class ClinicianReviewRecord(Base):
    """Stores clinician review decisions and notes separately from automated outputs."""

    __tablename__ = "clinician_reviews"

    review_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("rev"))
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), index=True, nullable=False)
    reviewer_id: Mapped[str] = mapped_column(String, nullable=False, default="clinician_admin")
    automated_status: Mapped[str] = mapped_column(String, nullable=False)
    clinician_decision: Mapped[str] = mapped_column(String, nullable=False)  # "APPROVED", "OVERRIDDEN", "NEEDS_RETEST"
    clinician_note: Mapped[str | None] = mapped_column(String)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class MeasurementQualityRecord(Base):
    """Stores overall quality engine audit logs for sessions."""

    __tablename__ = "measurement_qualities"

    quality_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("mql"))
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), index=True, nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False)
    quality_category: Mapped[str] = mapped_column(String, nullable=False)
    is_sufficient: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    failed_checks: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class ReportRecord(Base):
    """Stores raw medical reports uploaded or pasted by athletes."""

    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("rep"))
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.athlete_id"), index=True, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    raw_input: Mapped[str] = mapped_column(String, nullable=False)
    extracted_json: Mapped[str] = mapped_column(String, nullable=False)


class RecoveryPlanRecord(Base):
    """Stores full recovery plans generated from confirmed medical reports."""

    __tablename__ = "recovery_plans"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("plan"))
    report_id: Mapped[str] = mapped_column(ForeignKey("reports.id"), index=True, nullable=False)
    athlete_id: Mapped[str] = mapped_column(ForeignKey("athletes.athlete_id"), index=True, nullable=False)
    diagnosis: Mapped[str] = mapped_column(String, nullable=False)
    body_part: Mapped[str] = mapped_column(String, nullable=False)
    phase: Mapped[str] = mapped_column(String, nullable=False, default="Acute Recovery")
    recovery_duration_days: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    recovery_duration_source: Mapped[str] = mapped_column(String, nullable=False, default="estimated")
    hydration_target_glasses: Mapped[int] = mapped_column(Integer, nullable=False, default=8)
    medications_json: Mapped[str] = mapped_column(String, nullable=False)
    nutrition_items_json: Mapped[str] = mapped_column(String, nullable=False)
    exercises_json: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class DayLogRecord(Base):
    """Stores daily action log progress for a specific recovery plan day."""

    __tablename__ = "day_logs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("dlog"))
    plan_id: Mapped[str] = mapped_column(ForeignKey("recovery_plans.id"), index=True, nullable=False)
    day_number: Mapped[int] = mapped_column(Integer, nullable=False)
    hydration_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    meds_taken_json: Mapped[str] = mapped_column(String, nullable=False, default="[]")
    nutrition_logged_json: Mapped[str] = mapped_column(String, nullable=False, default="[]")
    exercise_reps_json: Mapped[str] = mapped_column(String, nullable=False, default="[]")
    exercise_self_logged_json: Mapped[str] = mapped_column(String, nullable=False, default="[]")
    status: Mapped[str] = mapped_column(String, nullable=False, default="inProgress")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))




def get_engine(database_url: str | None = None):
    """
    database_url defaults to the DATABASE_URL env var, then falls back to
    a local SQLite file (./driftguard.db) so the app/tests run with zero
    external setup. Pass a postgresql+psycopg2://... URL in production.
    """
    url = database_url or os.environ.get("DATABASE_URL", "sqlite:///driftguard.db")
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


def init_db(engine) -> None:
    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        from sqlalchemy import text
        for col_def in [
            "ALTER TABLE sessions ADD COLUMN sport_id VARCHAR",
            "ALTER TABLE landing_events ADD COLUMN knee_ap_translation_mm FLOAT",
            "ALTER TABLE landing_events ADD COLUMN knee_ml_translation_mm FLOAT",
            "ALTER TABLE landing_events ADD COLUMN knee_si_translation_mm FLOAT",
            "ALTER TABLE landing_events ADD COLUMN knee_flex_ext_deg FLOAT",
            "ALTER TABLE landing_events ADD COLUMN knee_abd_add_deg FLOAT",
            "ALTER TABLE landing_events ADD COLUMN knee_int_ext_rot_deg FLOAT",
        ]:
            try:
                conn.execute(text(col_def))
                conn.commit()
            except Exception:
                pass



def get_session_factory(engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False)

