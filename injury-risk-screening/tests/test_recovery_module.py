"""
Comprehensive Unit Tests for Athlete Recovery Module (v3).
Tests clinical extraction, pre-persisted plan generation, per-item duration filtering,
day log updates, and exercise media proxying.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.db.models import get_engine, init_db, get_session_factory, Athlete
from src.recovery.report_analyzer import analyze_medical_report
from src.recovery.wger_client import fetch_exercise_media

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    engine = get_engine()
    init_db(engine)
    factory = get_session_factory(engine)
    db = factory()

    # Ensure test athlete exists
    ath = db.get(Athlete, "ath_test_rec")
    if not ath:
        ath = Athlete(athlete_id="ath_test_rec", display_name="Test Recovery Athlete", sport="Basketball")
        db.add(ath)
        db.commit()
    db.close()


def test_report_analyzer():
    sample_text = """
    Patient: John Doe
    Diagnosis: Left Knee ACL Grade II Sprain with mild patellar tendinopathy.
    Phase: Acute Protection & Restoration Phase
    Duration: 30 days.

    Medications: Ibuprofen 400mg twice daily for 7 days.
    Nutrition: High protein synthesis target daily for 30 days.
    Exercises: Quad Setting (3x10, 5s hold) twice daily for 14 days. Straight Leg Raise (3x12) for 30 days.
    """
    res = analyze_medical_report(sample_text)

    assert res["diagnosis"] is not None
    assert "Knee" in res["body_part"] or "ACL" in res["diagnosis"]
    assert res["recovery_duration_days"] == 30
    assert len(res["medications"]) >= 1
    assert len(res["nutrition_items"]) >= 1
    assert "example_foods" in res["nutrition_items"][0]
    assert len(res["nutrition_items"][0]["example_foods"]) >= 3
    assert len(res["exercises"]) >= 1
    assert "teach_copy" in res["exercises"][0]


def test_wger_client_fallback():
    media = fetch_exercise_media("Quad Setting")
    assert media["exercise_name"] == "Quad Setting"
    assert "image_url" in media
    assert "fallback_svg" in media
    assert "<svg" in media["fallback_svg"]


def test_create_recovery_plan_api_and_pre_persist():
    # 1. Analyze report
    report_res = client.post(
        "/api/reports/analyze",
        json={
            "athlete_id": "ath_test_rec",
            "report_text": "Diagnosis: Right Knee Sprain. Duration: 14 days."
        }
    )
    assert report_res.status_code == 200
    r_data = report_res.json()
    report_id = r_data["report_id"]
    ext = r_data["extracted"]

    # 2. Create Plan with 14 days
    plan_payload = {
        "athlete_id": "ath_test_rec",
        "report_id": report_id,
        "diagnosis": ext["diagnosis"],
        "body_part": ext["body_part"],
        "phase": ext["phase"],
        "recovery_duration_days": 14,
        "recovery_duration_source": "confirmed",
        "hydration_target_glasses": 8,
        "medications": [
            {"name": "Ibuprofen", "dosage": "400mg", "frequency": "2x/day", "duration_days": 5}
        ],
        "nutrition_items": [
            {"title": "Protein", "target": "1.6g/kg", "duration_days": 14, "example_foods": ["Eggs", "Tofu", "Chicken"]}
        ],
        "exercises": [
            {"name": "Quad Setting", "target_reps": "3x10", "frequency": "2x/day", "duration_days": 5, "teach_copy": "Squeeze quad.", "caution": "No pain", "form_cue": "Lock knee"}
        ]
    }

    create_res = client.post("/api/plans", json=plan_payload)
    assert create_res.status_code == 200
    plan_id = create_res.json()["plan_id"]

    # 3. Get Plan Details
    get_plan_res = client.get(f"/api/plans/{plan_id}")
    assert get_plan_res.status_code == 200
    plan_details = get_plan_res.json()
    assert len(plan_details["days"]) == 14  # Pre-persisted all 14 days!


def test_per_item_duration_filtering_and_day_status():
    # Create plan with 10 days duration
    plan_payload = {
        "athlete_id": "ath_test_rec",
        "report_id": "rep_test_filter",
        "diagnosis": "Patellar Tendinopathy",
        "body_part": "Knee",
        "phase": "Acute",
        "recovery_duration_days": 10,
        "recovery_duration_source": "estimated",
        "hydration_target_glasses": 8,
        "medications": [
            {"name": "Short-term NSAID", "dosage": "400mg", "frequency": "2x/day", "duration_days": 3}
        ],
        "nutrition_items": [
            {"title": "Protein", "target": "1.6g/kg", "duration_days": 10, "example_foods": ["Eggs"]}
        ],
        "exercises": [
            {"name": "Quad Setting", "target_reps": "3x10", "frequency": "2x/day", "duration_days": 3, "teach_copy": "Hold", "caution": "", "form_cue": ""}
        ]
    }

    create_res = client.post("/api/plans", json=plan_payload)
    plan_id = create_res.json()["plan_id"]

    # Day 2 should include NSAID and Quad Setting (duration_days = 3)
    day2_res = client.get(f"/api/plans/{plan_id}/days/2")
    assert day2_res.status_code == 200
    d2 = day2_res.json()
    assert len(d2["active_medications"]) == 1
    assert len(d2["active_exercises"]) == 1

    # Day 5 should QUIETLY FILTER OUT NSAID and Quad Setting (duration_days = 3 < 5)
    day5_res = client.get(f"/api/plans/{plan_id}/days/5")
    assert day5_res.status_code == 200
    d5 = day5_res.json()
    assert len(d5["active_medications"]) == 0
    assert len(d5["active_exercises"]) == 0
    # Because active meds and active exercises are 0, medication & exercise categories are complete by default for Day 5!
    assert d5["category_completion"]["medication"] is True
    assert d5["category_completion"]["exercise"] is True


def test_update_day_log_api():
    plan_payload = {
        "athlete_id": "ath_test_rec",
        "report_id": "rep_test_update",
        "diagnosis": "Hamstring Strain",
        "body_part": "Hamstring",
        "phase": "Rehab",
        "recovery_duration_days": 7,
        "hydration_target_glasses": 6,
        "medications": [{"name": "Pain Relief", "dosage": "500mg", "frequency": "1x/day", "duration_days": 7}],
        "nutrition_items": [{"title": "Protein", "target": "1.5g/kg", "duration_days": 7, "example_foods": ["Fish"]}],
        "exercises": [{"name": "Isometric Heel Dig", "target_reps": "3x10", "frequency": "1x/day", "duration_days": 7, "teach_copy": "Press heel", "caution": "", "form_cue": ""}]
    }

    create_res = client.post("/api/plans", json=plan_payload)
    plan_id = create_res.json()["plan_id"]

    # Fetch Day 1 log ID
    day1_get = client.get(f"/api/plans/{plan_id}/days/1")
    log_id = day1_get.json()["id"]

    # Update Day 1 log
    update_res = client.post(
        f"/api/days/{log_id}",
        json={
            "hydration_count": 6,
            "meds_taken": ["Pain Relief"],
            "nutrition_logged": ["Protein"],
            "exercise_self_logged": ["Isometric Heel Dig"]
        }
    )
    assert update_res.status_code == 200
    up_data = update_res.json()
    assert up_data["hydration_count"] == 6
    assert up_data["status"] == "complete"
    assert up_data["category_completion"]["hydration"] is True
    assert up_data["category_completion"]["medication"] is True
    assert up_data["category_completion"]["nutrition"] is True
    assert up_data["category_completion"]["exercise"] is True
