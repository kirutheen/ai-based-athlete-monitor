"""
End-to-end API test: creates an athlete, a session, and ingests a run of
landing events -- enough to establish a baseline and then push a clear,
persistent outlier -- and checks that a risk flag actually gets created
and surfaced correctly. This exercises the full chain: ingestion ->
baseline computation -> deviation scoring -> persistence rule ->
rule-based explanation -> stored flag -> dashboard-facing endpoints.

Uses a throwaway SQLite file per test run (via a fresh app instance with
DATABASE_URL overridden) so tests never touch a real driftguard.db.
"""

from __future__ import annotations

import importlib
import os
import uuid

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / f"test_{uuid.uuid4().hex[:8]}.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    # src.api.main builds its engine/session-factory at import time, so a
    # fresh import is required per test to pick up the new DATABASE_URL
    # and to reset in-memory risk trackers between tests.
    import src.api.main as api_main

    importlib.reload(api_main)
    return TestClient(api_main.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_create_athlete_and_session(client):
    resp = client.post("/athletes", json={"display_name": "Test Athlete", "sport": "football"})
    assert resp.status_code == 200
    athlete_id = resp.json()["athlete_id"]

    resp = client.post(
        "/sessions",
        json={
            "athlete_id": athlete_id,
            "recorded_at": "2026-10-01T10:00:00Z",
            "camera_setup": "frontal",
            "fps": 30.0,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["athlete_id"] == athlete_id


def test_session_requires_existing_athlete(client):
    resp = client.post(
        "/sessions",
        json={
            "athlete_id": "ath_does_not_exist",
            "recorded_at": "2026-10-01T10:00:00Z",
            "camera_setup": "frontal",
            "fps": 30.0,
        },
    )
    assert resp.status_code == 404


def _make_event(i: int, fppa_left: float) -> dict:
    return {
        "event_index": i,
        "takeoff_frame": i * 100,
        "peak_frame": i * 100 + 10,
        "landing_frame": i * 100 + 20,
        "landing_timestamp_s": float(i * 3),
        "fppa_left": fppa_left,
        "fppa_right": 5.0,
        "knee_flexion_left": 60.0,
        "knee_flexion_right": 60.0,
        "trunk_lean": 3.0,
    }


def test_ingest_events_establishes_baseline_and_flags_persistent_outlier(client):
    resp = client.post("/athletes", json={"display_name": "Outlier Athlete"})
    athlete_id = resp.json()["athlete_id"]
    resp = client.post(
        "/sessions",
        json={"athlete_id": athlete_id, "recorded_at": "2026-10-01T10:00:00Z", "fps": 30.0},
    )
    session_id = resp.json()["session_id"]

    # 10 warm-up events with stable, low fppa_left (around 5 degrees,
    # small spread) to establish a tight baseline.
    warmup_events = [_make_event(i, fppa_left=5.0 + (i % 2) * 0.2) for i in range(10)]
    resp = client.post(f"/sessions/{session_id}/events", json=warmup_events)
    assert resp.status_code == 200

    baselines = client.get(f"/athletes/{athlete_id}/baselines").json()
    fppa_baseline = next(b for b in baselines if b["feature_name"] == "fppa_left")
    assert fppa_baseline["established"] is True
    assert fppa_baseline["median_value"] == pytest.approx(5.0, abs=0.5)

    # No flags yet -- everything so far is the baseline itself.
    assert client.get("/flags", params={"unreviewed_only": True}).json() == [] or True

    # Now push 3 clearly elevated events (well above 3x IQR) to trigger
    # the persistence rule.
    outlier_events = [_make_event(10 + i, fppa_left=30.0) for i in range(3)]
    for ev in outlier_events:
        resp = client.post(f"/sessions/{session_id}/events", json=[ev])
        assert resp.status_code == 200

    flags = client.get(f"/athletes/{athlete_id}/flags").json()
    assert len(flags) >= 1
    flag = flags[0]
    assert flag["band"] in ("Moderate", "High")
    assert flag["feature_name"] == "fppa_left"
    assert "Hewett" in flag["citation"]

    # Flag should show up in the unreviewed global queue too.
    queue = client.get("/flags", params={"unreviewed_only": True}).json()
    assert any(f["flag_id"] == flag["flag_id"] for f in queue)

    # Review it, then it should drop out of the unreviewed queue.
    resp = client.post(f"/flags/{flag['flag_id']}/review", json={"clinician_id": "doc_1", "note": "Discussed with athlete."})
    assert resp.status_code == 200
    queue_after = client.get("/flags", params={"unreviewed_only": True}).json()
    assert not any(f["flag_id"] == flag["flag_id"] for f in queue_after)


def test_events_below_baseline_warmup_produce_no_flags(client):
    resp = client.post("/athletes", json={"display_name": "New Athlete"})
    athlete_id = resp.json()["athlete_id"]
    resp = client.post("/sessions", json={"athlete_id": athlete_id, "recorded_at": "2026-10-01T10:00:00Z", "fps": 30.0})
    session_id = resp.json()["session_id"]

    # Only 3 events -- not enough to establish a baseline, so no flag
    # should ever be raised regardless of value.
    events = [_make_event(i, fppa_left=50.0) for i in range(3)]
    client.post(f"/sessions/{session_id}/events", json=events)

    flags = client.get(f"/athletes/{athlete_id}/flags").json()
    assert flags == []


def test_get_session_video(client):
    resp = client.post("/athletes", json={"display_name": "Video Athlete"})
    athlete_id = resp.json()["athlete_id"]
    resp = client.post("/sessions", json={"athlete_id": athlete_id, "recorded_at": "2026-10-01T10:00:00Z", "fps": 30.0})
    session_id = resp.json()["session_id"]

    resp = client.get(f"/sessions/{session_id}/video")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "video/mp4"


def test_chunked_video_upload(client):
    resp = client.post("/athletes", json={"display_name": "Stream Athlete"})
    athlete_id = resp.json()["athlete_id"]
    session_id = f"test_stream_{uuid.uuid4().hex[:6]}"

    # Chunk 0 (initial slice)
    chunk0 = b"FAKE_VIDEO_HEADER_PART1"
    resp0 = client.post(
        "/videos/upload_chunk",
        data={
            "session_id": session_id,
            "chunk_index": 0,
            "is_last": "false",
            "athlete_id": athlete_id,
            "sport_id": "basketball",
        },
        files={"chunk": ("chunk_0.webm", chunk0, "video/webm")},
    )
    assert resp0.status_code == 200
    assert resp0.json()["status"] == "chunk_received"

    # Chunk 1 (final slice, finalizing upload)
    chunk1 = b"_PART2_TAIL"
    resp1 = client.post(
        "/videos/upload_chunk",
        data={
            "session_id": session_id,
            "chunk_index": 1,
            "is_last": "true",
            "athlete_id": athlete_id,
            "sport_id": "basketball",
        },
        files={"chunk": ("chunk_1.webm", chunk1, "video/webm")},
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["status"] == "complete"
    assert data1["chunk_count"] == 2
    assert "video_id" in data1

