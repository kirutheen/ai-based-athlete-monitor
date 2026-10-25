"""Pydantic request/response models for the FastAPI service."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class AthleteCreate(BaseModel):
    display_name: str
    sport: str | None = None
    date_of_birth: date | None = None
    dominant_leg: str = Field(default="unknown", pattern="^(left|right|unknown)$")
    team_or_school: str | None = None


class AthleteOut(BaseModel):
    athlete_id: str
    display_name: str
    sport: str | None
    dominant_leg: str | None
    team_or_school: str | None

    model_config = {"from_attributes": True}


class ClinicianReviewCreate(BaseModel):
    reviewer_id: str = "clinician_admin"
    automated_status: str
    clinician_decision: str  # "APPROVED", "OVERRIDDEN", "NEEDS_RETEST"
    clinician_note: str | None = None


class ValidationRequest(BaseModel):
    metric_name: str
    estimated_values: list[float]
    reference_values: list[float]
    unit: str = "deg"



class SessionCreate(BaseModel):
    athlete_id: str
    recorded_at: datetime
    camera_setup: str = "frontal+sagittal"
    fps: float = 30.0
    notes: str | None = None


class SessionOut(BaseModel):
    session_id: str
    athlete_id: str
    recorded_at: datetime
    camera_setup: str
    fps: float

    model_config = {"from_attributes": True}


class LandingEventIn(BaseModel):
    event_index: int
    takeoff_frame: int
    peak_frame: int
    landing_frame: int
    landing_timestamp_s: float
    fppa_left: float | None = None
    fppa_right: float | None = None
    knee_flexion_left: float | None = None
    knee_flexion_right: float | None = None
    trunk_lean: float | None = None
    hip_flexion_left: float | None = None
    hip_flexion_right: float | None = None
    hip_abduction_left: float | None = None
    hip_abduction_right: float | None = None
    ankle_dorsiflexion_left: float | None = None
    ankle_dorsiflexion_right: float | None = None
    trunk_flexion_deg: float | None = None
    pelvic_obliquity_deg: float | None = None
    knee_flexion_LSI: float | None = None
    knee_valgus_LSI: float | None = None
    knee_valgus_at_initial_contact: float | None = None
    knee_flexion_at_initial_contact: float | None = None
    valgus_duration: float | None = None
    com_displacement: float | None = None
    pose_quality: float | None = None
    # Level 2 6-DOF Knee Kinematics
    knee_ap_translation_mm: float | None = None
    knee_ml_translation_mm: float | None = None
    knee_si_translation_mm: float | None = None
    knee_flex_ext_deg: float | None = None
    knee_abd_add_deg: float | None = None
    knee_int_ext_rot_deg: float | None = None



class LandingEventOut(LandingEventIn):
    event_id: str
    session_id: str
    athlete_id: str

    model_config = {"from_attributes": True}


class RiskFlagOut(BaseModel):
    flag_id: str
    athlete_id: str
    feature_name: str
    event_id: str
    band: str
    deviation_iqr: float | None
    headline: str
    detail: str
    citation: str | None
    reviewed_by_clinician_id: str | None
    clinician_note: str | None

    model_config = {"from_attributes": True}


class FlagReviewIn(BaseModel):
    clinician_id: str
    note: str | None = None


class BaselineOut(BaseModel):
    athlete_id: str
    feature_name: str
    n_events: int
    established: bool
    median_value: float | None
    iqr_value: float | None

    model_config = {"from_attributes": True}


class SportProfileOut(BaseModel):
    sport_id: str
    name: str
    key_movements: list[str]
    high_risk_mechanics: list[str]
    focus_joints: list[str]


class VideoValidationOut(BaseModel):
    passed: bool
    fps: float
    resolution_w: int
    resolution_h: int
    duration_sec: float
    issues: list[str]
    recommendations: list[str]


class VideoRecordOut(BaseModel):
    video_id: str
    session_id: str
    filename: str
    fps: float
    resolution_w: int
    resolution_h: int
    duration_sec: float
    validation_passed: bool
    validation_issues: str | None

    model_config = {"from_attributes": True}


class SessionStatusOut(BaseModel):
    session_id: str
    status: str
    progress_pct: int
    current_step: str | None
    error_message: str | None

    model_config = {"from_attributes": True}


class ClinicianNoteCreate(BaseModel):
    clinician_id: str = "cln_default"
    clinician_name: str = "Dr. Standard"
    note_type: str = "general"
    content: str
    override_band: str | None = None


class ClinicianNoteOut(BaseModel):
    note_id: str
    session_id: str
    clinician_id: str
    clinician_name: str
    note_type: str
    content: str
    override_band: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class BiomechanicalTrendPoint(BaseModel):
    session_id: str
    recorded_at: datetime
    knee_flexion_avg: float | None
    fppa_valgus_avg: float | None
    knee_flexion_LSI_avg: float | None
    knee_valgus_LSI_avg: float | None
    risk_band: str


class AthleteTrendsOut(BaseModel):
    athlete_id: str
    display_name: str
    total_sessions: int
    points: list[BiomechanicalTrendPoint]


class SessionReportOut(BaseModel):
    session_id: str
    athlete_id: str
    athlete_name: str
    sport: str | None
    recorded_at: datetime
    camera_setup: str
    fps: float
    total_events: int
    overall_risk_band: str
    risk_flags: list[RiskFlagOut]
    notes: list[ClinicianNoteOut]
    baseline_established: bool
    disclaimer: str = (
        "NON-DIAGNOSTIC SCREENING AID ONLY. This report contains AI-assisted screening assessment "
        "data intended strictly for clinical evaluation by a qualified medical or physiotherapy professional. "
        "It does not constitute a diagnostic medical opinion or guaranteed injury prediction."
    )



# --- Phase 4: Multi-Athlete Validation Schema ---------------------------------

class MetricValidationEntry(BaseModel):
    """One (athlete, metric) pair for batch validation."""
    athlete_id: str
    metric_name: str
    estimated_values: list[float]
    reference_values: list[float]
    unit: str = "deg"


class MultiAthleteValidationRequest(BaseModel):
    """Batch validation request: multiple athletes x multiple metrics."""
    entries: list[MetricValidationEntry]
