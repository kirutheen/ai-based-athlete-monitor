"""
FastAPI Router for Athlete Recovery Module.
Provides endpoints for report analysis, plan creation with pre-persisted days,
day log updates, and exercise media proxying.
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from datetime import datetime, UTC

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.db.models import (
    ReportRecord,
    RecoveryPlanRecord,
    DayLogRecord,
    get_engine,
    get_session_factory
)
from src.recovery.report_analyzer import analyze_medical_report
from src.recovery.wger_client import fetch_exercise_media


router = APIRouter(prefix="/api", tags=["recovery"])

# Dependency for DB Session
def get_db():
    engine = get_engine()
    factory = get_session_factory(engine)
    db = factory()
    try:
        yield db
    finally:
        db.close()


# Pydantic Request Models
class AnalyzeReportRequest(BaseModel):
    athlete_id: str
    report_text: str


class CreatePlanRequest(BaseModel):
    athlete_id: str
    report_id: str
    diagnosis: str
    body_part: str
    phase: str
    recovery_duration_days: int
    recovery_duration_source: str = "estimated"
    hydration_target_glasses: int = 8
    medications: List[Dict[str, Any]]
    nutrition_items: List[Dict[str, Any]]
    exercises: List[Dict[str, Any]]


class UpdateDayLogRequest(BaseModel):
    hydration_count: Optional[int] = None
    meds_taken: Optional[List[str]] = None
    nutrition_logged: Optional[List[str]] = None
    exercise_reps: Optional[List[Dict[str, Any]]] = None
    exercise_self_logged: Optional[List[str]] = None


# Helper to compute category and overall completion
def compute_day_status(
    plan: RecoveryPlanRecord,
    day_log: DayLogRecord,
    day_number: int
) -> Dict[str, Any]:
    meds_all = json.loads(plan.medications_json)
    nutr_all = json.loads(plan.nutrition_items_json)
    ex_all = json.loads(plan.exercises_json)

    # Filter active items for this day_number
    active_meds = [m for m in meds_all if m.get("duration_days") is None or day_number <= m.get("duration_days", 999)]
    active_nutr = [n for n in nutr_all if n.get("duration_days") is None or day_number <= n.get("duration_days", 999)]
    active_ex = [e for e in ex_all if e.get("duration_days") is None or day_number <= e.get("duration_days", 999)]

    # Current logs
    taken_meds = json.loads(day_log.meds_taken_json or "[]")
    logged_nutr = json.loads(day_log.nutrition_logged_json or "[]")
    logged_reps = json.loads(day_log.exercise_reps_json or "[]")
    self_logged_ex = json.loads(day_log.exercise_self_logged_json or "[]")

    # Hydration
    hydration_complete = day_log.hydration_count >= plan.hydration_target_glasses

    # Medication complete if all active meds are taken
    med_complete = True
    if active_meds:
        active_med_names = {m["name"] for m in active_meds}
        med_complete = active_med_names.issubset(set(taken_meds))

    # Nutrition complete if all active nutrition items logged
    nutr_complete = True
    if active_nutr:
        active_nutr_titles = {n["title"] for n in active_nutr}
        nutr_complete = active_nutr_titles.issubset(set(logged_nutr))

    # Exercise complete if all active exercises logged or self-logged
    ex_complete = True
    if active_ex:
        logged_ex_names = {r["name"] for r in logged_reps if isinstance(r, dict) and r.get("name")}.union(set(self_logged_ex))
        active_ex_names = {e["name"] for e in active_ex}
        ex_complete = active_ex_names.issubset(logged_ex_names)

    overall_complete = hydration_complete and med_complete and nutr_complete and ex_complete
    status_str = "complete" if overall_complete else "inProgress"

    return {
        "status": status_str,
        "active_medications": active_meds,
        "active_nutrition_items": active_nutr,
        "active_exercises": active_ex,
        "category_completion": {
            "hydration": hydration_complete,
            "medication": med_complete,
            "nutrition": nutr_complete,
            "exercise": ex_complete
        }
    }


# Endpoints
@router.post("/reports/analyze")
def analyze_report(req: AnalyzeReportRequest, db: Session = Depends(get_db)):
    extracted = analyze_medical_report(req.report_text)
    
    report = ReportRecord(
        athlete_id=req.athlete_id,
        raw_input=req.report_text,
        extracted_json=json.dumps(extracted)
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    return {
        "report_id": report.id,
        "extracted": extracted
    }


@router.post("/plans")
def create_recovery_plan(req: CreatePlanRequest, db: Session = Depends(get_db)):
    plan = RecoveryPlanRecord(
        report_id=req.report_id,
        athlete_id=req.athlete_id,
        diagnosis=req.diagnosis,
        body_part=req.body_part,
        phase=req.phase,
        recovery_duration_days=req.recovery_duration_days,
        recovery_duration_source=req.recovery_duration_source,
        hydration_target_glasses=req.hydration_target_glasses,
        medications_json=json.dumps(req.medications),
        nutrition_items_json=json.dumps(req.nutrition_items),
        exercises_json=json.dumps(req.exercises)
    )
    db.add(plan)
    db.commit()
    db.refresh(plan)

    # Pre-persist all N day logs
    day_logs = []
    for day in range(1, req.recovery_duration_days + 1):
        day_logs.append(
            DayLogRecord(
                plan_id=plan.id,
                day_number=day,
                hydration_count=0,
                meds_taken_json="[]",
                nutrition_logged_json="[]",
                exercise_reps_json="[]",
                exercise_self_logged_json="[]",
                status="inProgress"
            )
        )
    db.add_all(day_logs)
    db.commit()

    return {
        "plan_id": plan.id,
        "message": f"Plan created successfully with {req.recovery_duration_days} days pre-persisted."
    }


@router.get("/plans/{plan_id}")
def get_recovery_plan(plan_id: str, db: Session = Depends(get_db)):
    plan = db.query(RecoveryPlanRecord).filter(RecoveryPlanRecord.id == plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Recovery plan not found")

    day_logs = db.query(DayLogRecord).filter(DayLogRecord.plan_id == plan_id).order_by(DayLogRecord.day_number.asc()).all()

    logs_summary = [
        {
            "id": log.id,
            "day_number": log.day_number,
            "hydration_count": log.hydration_count,
            "status": log.status,
            "updated_at": log.updated_at.isoformat() if log.updated_at else None
        }
        for log in day_logs
    ]

    return {
        "id": plan.id,
        "athlete_id": plan.athlete_id,
        "diagnosis": plan.diagnosis,
        "body_part": plan.body_part,
        "phase": plan.phase,
        "recovery_duration_days": plan.recovery_duration_days,
        "recovery_duration_source": plan.recovery_duration_source,
        "hydration_target_glasses": plan.hydration_target_glasses,
        "medications": json.loads(plan.medications_json),
        "nutrition_items": json.loads(plan.nutrition_items_json),
        "exercises": json.loads(plan.exercises_json),
        "created_at": plan.created_at.isoformat() if plan.created_at else None,
        "days": logs_summary
    }


@router.get("/plans/{plan_id}/days/{day_number}")
def get_plan_day_log(plan_id: str, day_number: int, db: Session = Depends(get_db)):
    plan = db.query(RecoveryPlanRecord).filter(RecoveryPlanRecord.id == plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Recovery plan not found")

    log = db.query(DayLogRecord).filter(
        DayLogRecord.plan_id == plan_id,
        DayLogRecord.day_number == day_number
    ).first()

    if not log:
        raise HTTPException(status_code=404, detail=f"Day {day_number} log not found for plan {plan_id}")

    computed = compute_day_status(plan, log, day_number)

    return {
        "id": log.id,
        "plan_id": plan_id,
        "day_number": day_number,
        "hydration_count": log.hydration_count,
        "hydration_target": plan.hydration_target_glasses,
        "meds_taken": json.loads(log.meds_taken_json or "[]"),
        "nutrition_logged": json.loads(log.nutrition_logged_json or "[]"),
        "exercise_reps": json.loads(log.exercise_reps_json or "[]"),
        "exercise_self_logged": json.loads(log.exercise_self_logged_json or "[]"),
        "status": computed["status"],
        "active_medications": computed["active_medications"],
        "active_nutrition_items": computed["active_nutrition_items"],
        "active_exercises": computed["active_exercises"],
        "category_completion": computed["category_completion"]
    }


@router.post("/days/{log_id}")
def update_day_log(log_id: str, req: UpdateDayLogRequest, db: Session = Depends(get_db)):
    log = db.query(DayLogRecord).filter(DayLogRecord.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Day log record not found")

    plan = db.query(RecoveryPlanRecord).filter(RecoveryPlanRecord.id == log.plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Associated plan not found")

    if req.hydration_count is not None:
        log.hydration_count = req.hydration_count
    if req.meds_taken is not None:
        log.meds_taken_json = json.dumps(req.meds_taken)
    if req.nutrition_logged is not None:
        log.nutrition_logged_json = json.dumps(req.nutrition_logged)
    if req.exercise_reps is not None:
        log.exercise_reps_json = json.dumps(req.exercise_reps)
    if req.exercise_self_logged is not None:
        log.exercise_self_logged_json = json.dumps(req.exercise_self_logged)

    computed = compute_day_status(plan, log, log.day_number)
    log.status = computed["status"]
    log.updated_at = datetime.now(UTC)

    db.commit()
    db.refresh(log)

    return {
        "id": log.id,
        "plan_id": log.plan_id,
        "day_number": log.day_number,
        "hydration_count": log.hydration_count,
        "hydration_target": plan.hydration_target_glasses,
        "meds_taken": json.loads(log.meds_taken_json or "[]"),
        "nutrition_logged": json.loads(log.nutrition_logged_json or "[]"),
        "exercise_reps": json.loads(log.exercise_reps_json or "[]"),
        "exercise_self_logged": json.loads(log.exercise_self_logged_json or "[]"),
        "status": log.status,
        "category_completion": computed["category_completion"]
    }


@router.get("/exercises/media")
def get_exercise_media(name: str):
    if not name or len(name.strip()) == 0:
        raise HTTPException(status_code=400, detail="Exercise name query parameter required")
    return fetch_exercise_media(name)


@router.get("/athletes/{athlete_id}/active-plan")
def get_active_athlete_plan(athlete_id: str, db: Session = Depends(get_db)):
    plan = db.query(RecoveryPlanRecord).filter(
        RecoveryPlanRecord.athlete_id == athlete_id
    ).order_by(RecoveryPlanRecord.created_at.desc()).first()

    if not plan:
        return {"active_plan": None}

    return {"active_plan_id": plan.id}
