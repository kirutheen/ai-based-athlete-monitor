"""
Regression tests for the Multi-Athlete Processor
(src/pose/multi_athlete_processor.py).

These tests use synthetic video frames created in-memory with OpenCV so
no real video file is required. MediaPipe is unlikely to detect real human
pose in solid-color rectangles, so tests verify structural correctness of
the returned payload rather than landmark quality.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict

import cv2
import numpy as np
import pytest

from src.pose.multi_athlete_processor import (
    _track_color_hex,
    process_multi_athlete_video,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_video(path: str, frames: int = 10, fps: float = 30.0) -> None:
    """Write a short synthetic MP4 with two colored rectangles (stand-ins for
    athletes) into *path*."""
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
    for _ in range(frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Left "athlete": green rectangle
        cv2.rectangle(frame, (80, 80), (220, 360), (0, 200, 80), -1)
        # Right "athlete": blue rectangle
        cv2.rectangle(frame, (380, 100), (520, 380), (100, 60, 220), -1)
        out.write(frame)
    out.release()


def _require_model() -> str:
    """Return the model path or skip the test if the .task file is absent."""
    path = os.path.abspath("models/pose_landmarker_full.task")
    if not os.path.exists(path):
        pytest.skip("MediaPipe model file not found — skipping integration test")
    return path


# ---------------------------------------------------------------------------
# Unit tests (no model / no video needed)
# ---------------------------------------------------------------------------

class TestTrackColorHex:
    def test_returns_hex_string(self):
        color = _track_color_hex(0)
        assert color.startswith("#"), f"Expected '#...', got {color!r}"
        assert len(color) == 7

    def test_different_ids_give_different_colors(self):
        colors = [_track_color_hex(i) for i in range(8)]
        # All 8 palette entries should be distinct
        assert len(set(colors)) == 8

    def test_wraps_at_palette_size(self):
        # track_id 0 and track_id 8 (palette size) should give the same color
        assert _track_color_hex(0) == _track_color_hex(8)


# ---------------------------------------------------------------------------
# Integration tests (require model + video)
# ---------------------------------------------------------------------------

class TestProcessMultiAthleteVideo:
    def test_returns_expected_top_level_keys(self):
        """Processor must return dict with all mandatory keys."""
        model_path = _require_model()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = tmp.name
        try:
            _make_synthetic_video(video_path, frames=10)
            result = process_multi_athlete_video(
                video_path, session_id="test-ma-01", model_path=model_path
            )
        finally:
            os.unlink(video_path)

        required_keys = {"session_id", "source_video", "total_frames", "fps",
                         "duration_sec", "max_athletes", "athletes", "frames"}
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_frame_count_matches_video(self):
        """total_frames must equal number of entries in the frames list."""
        model_path = _require_model()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = tmp.name
        try:
            _make_synthetic_video(video_path, frames=10)
            result = process_multi_athlete_video(
                video_path, session_id="test-ma-02", model_path=model_path
            )
        finally:
            os.unlink(video_path)

        assert result["total_frames"] == len(result["frames"]), (
            f"total_frames={result['total_frames']} but frames list has "
            f"{len(result['frames'])} entries"
        )

    def test_frame_structure(self):
        """Every frame entry must have 'frame', 'timestamp_s', 'athletes' keys."""
        model_path = _require_model()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = tmp.name
        try:
            _make_synthetic_video(video_path, frames=5)
            result = process_multi_athlete_video(
                video_path, session_id="test-ma-03", model_path=model_path
            )
        finally:
            os.unlink(video_path)

        for frame_entry in result["frames"]:
            assert "frame" in frame_entry
            assert "timestamp_s" in frame_entry
            assert "athletes" in frame_entry
            assert isinstance(frame_entry["athletes"], list)

    def test_athlete_entry_structure_when_detected(self):
        """If any athlete is detected, their entry must have required keys."""
        model_path = _require_model()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = tmp.name
        try:
            _make_synthetic_video(video_path, frames=10)
            result = process_multi_athlete_video(
                video_path, session_id="test-ma-04", model_path=model_path
            )
        finally:
            os.unlink(video_path)

        required_athlete_keys = {
            "track_id", "color", "display_name", "confidence",
            "bbox", "skeleton",
            "fppa_left", "fppa_right", "knee_flexion_left", "knee_flexion_right",
        }
        for frame_entry in result["frames"]:
            for a in frame_entry["athletes"]:
                missing = required_athlete_keys - a.keys()
                assert not missing, f"Athlete entry missing keys: {missing}"

    def test_no_cross_contamination_between_athletes(self):
        """Each track_id in a frame must be unique (no duplicate identities)."""
        model_path = _require_model()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = tmp.name
        try:
            _make_synthetic_video(video_path, frames=10)
            result = process_multi_athlete_video(
                video_path, session_id="test-ma-05", model_path=model_path
            )
        finally:
            os.unlink(video_path)

        for frame_entry in result["frames"]:
            track_ids = [a["track_id"] for a in frame_entry["athletes"]]
            assert len(track_ids) == len(set(track_ids)), (
                f"Frame {frame_entry['frame']} has duplicate track IDs: {track_ids}"
            )

    def test_athlete_summaries_contain_expected_keys(self):
        """Athletes summary list entries must contain all mandatory summary keys."""
        model_path = _require_model()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = tmp.name
        try:
            _make_synthetic_video(video_path, frames=10)
            result = process_multi_athlete_video(
                video_path, session_id="test-ma-06", model_path=model_path
            )
        finally:
            os.unlink(video_path)

        required_summary_keys = {
            "track_id", "display_name", "color", "first_seen_s", "last_seen_s",
            "frames_tracked", "coverage_pct",
            "avg_fppa_left", "avg_knee_flex_left",
            "peak_fppa_left", "peak_knee_flex_left",
        }
        for a in result["athletes"]:
            missing = required_summary_keys - a.keys()
            assert not missing, f"Athlete summary missing keys: {missing}"

    def test_color_is_valid_hex(self):
        """Every athlete's color field must be a valid 7-char hex string."""
        model_path = _require_model()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = tmp.name
        try:
            _make_synthetic_video(video_path, frames=10)
            result = process_multi_athlete_video(
                video_path, session_id="test-ma-07", model_path=model_path
            )
        finally:
            os.unlink(video_path)

        for a in result["athletes"]:
            c = a["color"]
            assert c.startswith("#") and len(c) == 7, f"Invalid color: {c!r}"

    def test_missing_video_raises(self):
        """Passing a nonexistent video path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            process_multi_athlete_video(
                "/nonexistent/path/no_video.mp4",
                session_id="test-ma-err",
                model_path=_require_model(),
            )

    def test_max_athletes_configurable(self):
        """max_athletes is reflected in the returned payload."""
        model_path = _require_model()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = tmp.name
        try:
            _make_synthetic_video(video_path, frames=5)
            result = process_multi_athlete_video(
                video_path, session_id="test-ma-08", model_path=model_path,
                max_athletes=3,
            )
        finally:
            os.unlink(video_path)

        assert result["max_athletes"] == 3

    def test_athlete_count_does_not_exceed_max(self):
        """Number of unique track IDs must never exceed max_athletes."""
        model_path = _require_model()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = tmp.name
        try:
            _make_synthetic_video(video_path, frames=10)
            max_a = 2
            result = process_multi_athlete_video(
                video_path, session_id="test-ma-09", model_path=model_path,
                max_athletes=max_a,
            )
        finally:
            os.unlink(video_path)

        all_track_ids: set = set()
        for frame_entry in result["frames"]:
            for a in frame_entry["athletes"]:
                all_track_ids.add(a["track_id"])

        # MediaPipe won't reliably find real pose in synthetic rectangles, so
        # the count could be 0. What must NOT happen is exceeding max_athletes.
        assert len(all_track_ids) <= max_a, (
            f"Detected {len(all_track_ids)} unique athletes but max_athletes={max_a}"
        )
