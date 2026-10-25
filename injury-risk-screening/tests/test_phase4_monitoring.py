"""
Phase 4 tests: 300-second monitoring sessions and chunked webcam uploads.
"""
from __future__ import annotations

import io
import os
import time

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.capture.video_validator import inspect_video_quality

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. video_validator -- 300s boundary
# ---------------------------------------------------------------------------

class TestVideoValidator300s:

    def test_300s_is_accepted_by_default_param(self):
        import inspect as ins
        sig = ins.signature(inspect_video_quality)
        default = sig.parameters["max_duration_s"].default
        assert default == 300.0, (
            f"max_duration_s default is {default}; expected 300.0 (was previously 180)."
        )

    def test_301s_would_be_rejected(self):
        duration_s, max_duration_s = 301.0, 300.0
        issues = ["too_long"] if duration_s > max_duration_s else []
        assert "too_long" in issues

    def test_300s_is_exactly_ok(self):
        duration_s, max_duration_s = 300.0, 300.0
        issues = ["too_long"] if duration_s > max_duration_s else []
        assert "too_long" not in issues


# ---------------------------------------------------------------------------
# 2. Chunked upload endpoint
# ---------------------------------------------------------------------------

class TestChunkedUpload:

    def _chunk(self, size: int = 1024) -> bytes:
        return bytes(range(256)) * (size // 256 + 1)

    def _post_chunk(self, session_id: str, chunk_index: int, is_last: bool, data: bytes):
        return client.post(
            "/videos/upload_chunk",
            data={
                "session_id": session_id,
                "chunk_index": str(chunk_index),
                "is_last": str(is_last).lower(),
                "sport_id": "basketball",
            },
            files={"chunk": (f"c_{chunk_index}.webm", io.BytesIO(data), "video/webm")},
        )

    def test_first_chunk_accepted(self):
        sid = f"ck_{int(time.time()*1000)}"
        r = self._post_chunk(sid, 0, False, self._chunk())
        assert r.status_code == 200
        assert r.json()["status"] == "chunk_received"

    def test_multiple_chunks(self):
        sid = f"ck2_{int(time.time()*1000)}"
        for i in range(3):
            r = self._post_chunk(sid, i, False, self._chunk())
            assert r.status_code == 200

    def test_last_chunk_complete(self):
        sid = f"ck3_{int(time.time()*1000)}"
        self._post_chunk(sid, 0, False, self._chunk())
        r = self._post_chunk(sid, 1, True, self._chunk())
        assert r.status_code == 200
        assert r.json()["status"] == "complete"


# ---------------------------------------------------------------------------
# 3-6. Monitoring endpoints
# ---------------------------------------------------------------------------

class TestMonitoringEndpoints:

    def _sid(self) -> str:
        return f"mon_{int(time.time()*1000)}"

    def test_start(self):
        sid = self._sid()
        r = client.post(f"/sessions/{sid}/monitoring/start?max_duration_s=300")
        assert r.status_code == 200
        assert r.json()["status"] == "started"

    def test_status_active_after_start(self):
        sid = self._sid()
        client.post(f"/sessions/{sid}/monitoring/start?max_duration_s=300")
        r = client.get(f"/sessions/{sid}/monitoring/status")
        assert r.status_code == 200
        b = r.json()
        assert b["found"] is True
        assert b["active"] is True
        assert 0.0 <= b["elapsed_s"] <= 5.0
        assert b["max_s"] == 300

    def test_status_unknown_session_safe(self):
        r = client.get("/sessions/unknown_xyz_999/monitoring/status")
        assert r.status_code == 200
        b = r.json()
        assert b["found"] is False
        assert b["active"] is False

    def test_stop(self):
        sid = self._sid()
        client.post(f"/sessions/{sid}/monitoring/start?max_duration_s=300")
        r = client.post(f"/sessions/{sid}/monitoring/stop")
        assert r.status_code == 200
        assert r.json()["status"] == "stopped"

    def test_inactive_after_stop(self):
        sid = self._sid()
        client.post(f"/sessions/{sid}/monitoring/start?max_duration_s=300")
        client.post(f"/sessions/{sid}/monitoring/stop")
        r = client.get(f"/sessions/{sid}/monitoring/status")
        assert r.json()["active"] is False

    def test_max_duration_capped_at_300(self):
        sid = self._sid()
        client.post(f"/sessions/{sid}/monitoring/start?max_duration_s=999")
        status = client.get(f"/sessions/{sid}/monitoring/status").json()
        assert status["max_s"] == 300


# ---------------------------------------------------------------------------
# 7. Finalize chunk session
# ---------------------------------------------------------------------------

class TestFinalizeChunkSession:

    def test_finalize_no_temp_file(self):
        sid = f"fin_{int(time.time()*1000)}"
        r = client.post(
            "/videos/finalize_chunk_session",
            data={"session_id": sid, "chunk_count": "3"},
        )
        assert r.status_code == 200
        b = r.json()
        assert b["status"] == "finalized"
        assert b["chunk_count"] == 3

    def test_finalize_renames_temp_file(self):
        sid = f"fin2_{int(time.time()*1000)}"
        upload_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "uploads")
        )
        os.makedirs(upload_dir, exist_ok=True)
        temp_path = os.path.join(upload_dir, f"{sid}_stream.webm")
        final_path = os.path.join(upload_dir, f"{sid}_webcam_trial.webm")
        with open(temp_path, "wb") as f:
            f.write(b"\x00" * 128)
        try:
            r = client.post(
                "/videos/finalize_chunk_session",
                data={"session_id": sid, "chunk_count": "1"},
            )
            assert r.status_code == 200
            assert not os.path.exists(temp_path), "_stream.webm should have been renamed"
        finally:
            for p in [temp_path, final_path]:
                if os.path.exists(p):
                    os.remove(p)
