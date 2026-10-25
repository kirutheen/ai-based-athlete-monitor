"""
Integration tests for Clinician Review & Override API endpoints.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from src.api.main import app, get_db
from src.db import models


def test_clinician_review_flow(tmp_path):
    db_file = tmp_path / "test_reviews.db"
    engine = models.get_engine(f"sqlite:///{db_file}")
    models.init_db(engine)
    session_factory = models.get_session_factory(engine)

    def _get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _get_db
    client = TestClient(app)

    # 1. Create athlete & session
    ath_res = client.post("/athletes", json={"display_name": "Review Test Athlete", "sport": "basketball"})
    assert ath_res.status_code == 200
    athlete_id = ath_res.json()["athlete_id"]

    ses_res = client.post("/sessions", json={"athlete_id": athlete_id, "recorded_at": "2026-10-06T10:00:00Z"})
    assert ses_res.status_code == 200
    session_id = ses_res.json()["session_id"]

    # 2. Post Clinician Review / Override
    rev_payload = {
        "reviewer_id": "dr_smith",
        "automated_status": "Moderate Risk",
        "clinician_decision": "APPROVED",
        "clinician_note": "Observed mild dynamic knee valgus during landing phase. Recommended hip abductor strengthening exercises.",
    }
    post_rev = client.post(f"/sessions/{session_id}/review", json=rev_payload)
    assert post_rev.status_code == 200
    rev_data = post_rev.json()
    assert rev_data["session_id"] == session_id
    assert rev_data["clinician_decision"] == "APPROVED"

    # 3. Retrieve Clinician Reviews
    get_revs = client.get(f"/sessions/{session_id}/reviews")
    assert get_revs.status_code == 200
    revs = get_revs.json()
    assert len(revs) == 1
    assert revs[0]["reviewer_id"] == "dr_smith"
    assert revs[0]["clinician_note"] == rev_payload["clinician_note"]

    # 4. Check Hardware Status API
    hw_res = client.get("/hardware/status")
    assert hw_res.status_code == 200
    assert "hardware not connected" in hw_res.json()["imu_sensors"]

    app.dependency_overrides.clear()
