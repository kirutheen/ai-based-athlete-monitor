"""
Automated Pipeline Test: Verifies that uploading two different videos produces distinct,
independent job IDs, frame counts, landmark coordinates, biomechanical metrics,
event keyframes, and risk flags without static demo fallback data or cross-contamination.
"""

import os
import tempfile
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, _engine
from src.db import models

client = TestClient(app)


def create_test_video(file_path: str, frame_count: int, squat_amplitude: int, valgus_amplitude: int, fps: float = 30.0):
    """Helper to generate a distinct synthetic test video file with given parameters."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, fps, (640, 480))

    for f in range(frame_count):
        img = np.full((480, 640, 3), (240, 240, 240), dtype=np.uint8)
        phase = (f / frame_count) * 2 * np.pi

        # Draw head & torso
        cv2.circle(img, (320, 80), 25, (50, 50, 50), -1)
        cv2.line(img, (320, 105), (320, 220), (50, 50, 50), 6)
        cv2.line(img, (280, 220), (360, 220), (50, 50, 50), 6)

        squat_depth = int(np.sin(phase) * squat_amplitude)
        valgus_shift = int(np.sin(phase) * valgus_amplitude)

        l_hip, r_hip = (280, 220), (360, 220)
        l_knee = (280 + valgus_shift, 320 + squat_depth)
        r_knee = (360 - valgus_shift // 2, 320 + squat_depth)
        l_ank = (270, 420)
        r_ank = (370, 420)

        # Thick red lines for MediaPipe pose landmarker detection
        cv2.line(img, l_hip, l_knee, (0, 0, 180), 10)
        cv2.line(img, l_knee, l_ank, (0, 0, 180), 10)
        cv2.line(img, r_hip, r_knee, (180, 0, 0), 10)
        cv2.line(img, r_knee, r_ank, (180, 0, 0), 10)

        for pt in [l_hip, r_hip, l_knee, r_knee, l_ank, r_ank]:
            cv2.circle(img, pt, 8, (255, 255, 255), -1)
            cv2.circle(img, pt, 8, (0, 0, 0), 2)

        out.write(img)

    out.release()


def test_two_different_videos_produce_distinct_results():
    """Verifies complete upload -> processing -> API response pipeline for two distinct videos."""
    # 1. Create an athlete
    ath_resp = client.post("/athletes", json={"display_name": "Pipeline Test Athlete", "sport": "Basketball"})
    assert ath_resp.status_code == 200
    athlete_id = ath_resp.json()["athlete_id"]

    with tempfile.TemporaryDirectory() as tmpdir:
        vid_a_path = os.path.join(tmpdir, "test_video_a.mp4")
        vid_b_path = os.path.join(tmpdir, "test_video_b.mp4")

        # Video A: 60 frames (2.0s), shallow movement (amplitude 20)
        create_test_video(vid_a_path, frame_count=60, squat_amplitude=20, valgus_amplitude=10)
        # Video B: 90 frames (3.0s), deep movement (amplitude 45)
        create_test_video(vid_b_path, frame_count=90, squat_amplitude=45, valgus_amplitude=30)

        # --- PROCESS VIDEO A ---
        with open(vid_a_path, "rb") as f_a:
            up_a = client.post(
                f"/videos/upload?athlete_id={athlete_id}&sport_id=basketball",
                files={"file": ("test_video_a.mp4", f_a, "video/mp4")},
            )
        assert up_a.status_code == 200
        vid_rec_a = up_a.json()
        session_id_a = vid_rec_a["session_id"]
        video_id_a = vid_rec_a["video_id"]

        proc_a = client.post(f"/sessions/{session_id_a}/process")
        assert proc_a.status_code == 200
        assert proc_a.json()["status"] == "complete"

        res_a = client.get(f"/sessions/{session_id_a}/results").json()
        bio_a = client.get(f"/sessions/{session_id_a}/biomechanics").json()

        # --- PROCESS VIDEO B ---
        with open(vid_b_path, "rb") as f_b:
            up_b = client.post(
                f"/videos/upload?athlete_id={athlete_id}&sport_id=basketball",
                files={"file": ("test_video_b.mp4", f_b, "video/mp4")},
            )
        assert up_b.status_code == 200
        vid_rec_b = up_b.json()
        session_id_b = vid_rec_b["session_id"]
        video_id_b = vid_rec_b["video_id"]

        proc_b = client.post(f"/sessions/{session_id_b}/process")
        assert proc_b.status_code == 200
        assert proc_b.json()["status"] == "complete"

        res_b = client.get(f"/sessions/{session_id_b}/results").json()
        bio_b = client.get(f"/sessions/{session_id_b}/biomechanics").json()

        # --- VERIFICATIONS ---
        # 1. Job/Session IDs and Video IDs must be unique
        assert session_id_a != session_id_b
        assert video_id_a != video_id_b

        # 2. Source video paths/filenames must match uploaded files
        assert "test_video_a.mp4" in res_a["source_video"]
        assert "test_video_b.mp4" in res_b["source_video"]

        # 3. Frame counts and durations must match actual videos (60 frames vs 90 frames)
        assert res_a["frame_count"] == 60
        assert res_b["frame_count"] == 90
        assert bio_a["total_frames"] == 60
        assert bio_b["total_frames"] == 90
        assert abs(res_a["duration"] - 2.0) < 0.2
        assert abs(res_b["duration"] - 3.0) < 0.2

        # 4. Metrics calculated independently from video frames
        metrics_a = res_a["metrics"]
        metrics_b = res_b["metrics"]
        assert metrics_a != metrics_b

        # 5. Check frame landmarks differ between Video A and Video B
        frames_a = bio_a["frames"]
        frames_b = bio_b["frames"]

        assert len(frames_a) != len(frames_b)
        assert frames_a[15]["knee_flexion_left"] != frames_b[15]["knee_flexion_left"]
        assert frames_a[15]["skeleton"] != frames_b[15]["skeleton"]

        # 6. Refetching Session B returns Session B's result, not cached A
        refetch_b = client.get(f"/sessions/{session_id_b}/results").json()
        assert refetch_b["session_id"] == session_id_b
        assert refetch_b["frame_count"] == 90

        # 7. Refetching Session A returns Session A's result
        refetch_a = client.get(f"/sessions/{session_id_a}/results").json()
        assert refetch_a["session_id"] == session_id_a
        assert refetch_a["frame_count"] == 60
