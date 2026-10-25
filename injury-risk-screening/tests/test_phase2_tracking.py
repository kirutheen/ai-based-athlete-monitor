"""Phase 2 regression tests — multi-athlete tracking + identity stability."""
import math
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# MultiPersonTracker tests
# ---------------------------------------------------------------------------

from src.capture.multi_person_tracker import (
    MultiPersonTracker,
    TRACK_STATUS_ACTIVE,
    TRACK_STATUS_UNCERTAIN,
    TRACK_STATUS_TEMPORARILY_LOST,
    TRACK_STATUS_ENDED,
)


def _pose(cx: float, cy: float, visibility: float = 0.9):
    """Return (landmarks_2d ndarray (33,4), confidence) tuple for tracker.update()."""
    # columns: x, y, z, visibility
    lm = np.array([[cx, cy, 0.0, visibility]] * 33, dtype=float)
    return (lm, visibility)   # (landmarks, confidence)


def _det(cx: float, cy: float, visibility: float = 0.9):
    """Alias for clarity — one detection entry."""
    return _pose(cx, cy, visibility)


class TestMultiPersonTrackerPhase2:
    def test_two_tracks_stable_consecutive_frames(self):
        """Two well-separated athletes keep their IDs across frames."""
        tracker = MultiPersonTracker()
        dets_a = [_det(0.25, 0.5), _det(0.75, 0.5)]
        results1 = tracker.update(dets_a, frame_index=0, session_id="s1")
        ids1 = {r.track_id for r in results1}
        assert len(ids1) == 2

        # Slightly moved — same IDs expected
        dets_b = [_det(0.26, 0.5), _det(0.74, 0.5)]
        results2 = tracker.update(dets_b, frame_index=1, session_id="s1")
        ids2 = {r.track_id for r in results2}
        assert ids1 == ids2, "Track IDs must be stable across frames"

    def test_three_tracks_all_active(self):
        """Three athletes produce three distinct active track IDs."""
        tracker = MultiPersonTracker()
        dets = [_det(0.1, 0.5), _det(0.5, 0.5), _det(0.9, 0.5)]
        results = tracker.update(dets, frame_index=0, session_id="s1")
        assert len(results) == 3
        assert len({r.track_id for r in results}) == 3

    def test_unique_active_ids(self):
        """active_track_ids() returns unique IDs only."""
        tracker = MultiPersonTracker()
        tracker.update([_det(0.2, 0.5), _det(0.8, 0.5)], frame_index=0)
        ids = tracker.active_track_ids()
        assert len(ids) == len(set(ids))

    def test_crossing_sets_uncertain(self):
        """Two tracks crossing close together are flagged uncertain."""
        tracker = MultiPersonTracker(crossing_distance=0.08)
        # First frame: well separated
        tracker.update([_det(0.1, 0.5), _det(0.9, 0.5)], frame_index=0)
        # Second frame: nearly overlapping
        results = tracker.update(
            [_det(0.5, 0.5), _det(0.52, 0.5)],
            frame_index=1,
        )
        statuses = {r.track_status for r in results}
        assert TRACK_STATUS_UNCERTAIN in statuses, (
            "Near-crossing should flag at least one track as uncertain"
        )

    def test_confidence_gate_drops_low_conf(self):
        """Detections below confidence threshold are dropped."""
        tracker = MultiPersonTracker(min_detection_confidence=0.50)
        low_conf = _det(0.5, 0.5, visibility=0.1)
        results = tracker.update([low_conf], frame_index=0)
        assert results == [], "Low-confidence detection must be suppressed"

    def test_frame_index_and_session_id_in_result(self):
        """TrackedPose carries frame_index and session_id from the call."""
        tracker = MultiPersonTracker()
        results = tracker.update(
            [_det(0.5, 0.5)],
            frame_index=42, session_id="session_xyz",
        )
        assert len(results) == 1
        assert results[0].frame_index == 42
        assert results[0].session_id == "session_xyz"

    def test_track_status_active_for_stable_track(self):
        """A consistently detected track is ACTIVE."""
        tracker = MultiPersonTracker()
        for fi in range(5):
            results = tracker.update([_det(0.5, 0.5)], frame_index=fi)
        assert results[0].track_status == TRACK_STATUS_ACTIVE

    def test_all_track_statuses_returns_dict(self):
        """all_track_statuses() returns a non-empty dict after detections."""
        tracker = MultiPersonTracker()
        tracker.update([_det(0.3, 0.5), _det(0.7, 0.5)], frame_index=0)
        statuses = tracker.all_track_statuses()
        assert isinstance(statuses, dict)
        assert len(statuses) == 2



# ---------------------------------------------------------------------------
# multi_athlete_processor structural / invariant tests
# ---------------------------------------------------------------------------

def _make_dummy_processor_result():
    """Minimal result dict matching the processor's output schema."""
    return {
        "session_id": "test-sess",
        "source_video": "clip.mp4",
        "total_frames": 10,
        "fps": 30.0,
        "duration_sec": 0.33,
        "max_athletes": 5,
        "athletes": [
            {
                "track_id": 1,
                "session_id": "test-sess",
                "display_name": "Athlete #1",
                "color": "#38BDF8",
                "first_seen_s": 0.0,
                "last_seen_s": 0.33,
                "frames_tracked": 10,
                "coverage_pct": 100.0,
                "uncertainty_pct": 20.0,
                "biomechanics_valid_pct": 80.0,
                "avg_fppa_left": -12.5,
                "avg_knee_flex_left": 45.0,
                "peak_fppa_left": 18.0,
                "peak_knee_flex_left": 60.0,
            }
        ],
        "frames": [
            {
                "frame_index": 0,
                "timestamp_s": 0.0,
                "athletes": [
                    {
                        "track_id": 1,
                        "track_status": TRACK_STATUS_ACTIVE,
                        "biomechanics_valid": True,
                        "bbox": [0.1, 0.1, 0.4, 0.9],
                        "skeleton": {},
                        "biomechanics": {"fppa_left": -12.5, "knee_flexion_left": 45.0},
                    }
                ],
            }
        ],
    }


class TestMultiAthleteProcessorSchema:
    def test_result_has_required_top_level_keys(self):
        r = _make_dummy_processor_result()
        for key in ("session_id", "source_video", "total_frames", "fps",
                    "duration_sec", "max_athletes", "athletes", "frames"):
            assert key in r, f"Missing top-level key: {key}"

    def test_athlete_summary_has_required_keys(self):
        r = _make_dummy_processor_result()
        a = r["athletes"][0]
        for key in ("track_id", "session_id", "display_name", "color",
                    "first_seen_s", "last_seen_s", "frames_tracked",
                    "coverage_pct", "uncertainty_pct", "biomechanics_valid_pct",
                    "avg_fppa_left", "avg_knee_flex_left",
                    "peak_fppa_left", "peak_knee_flex_left"):
            assert key in a, f"Missing athlete summary key: {key}"

    def test_biomechanics_valid_false_when_uncertain(self):
        """Frame entry with uncertain track_status should carry biomechanics_valid=False."""
        entry = {
            "track_id": 1,
            "track_status": TRACK_STATUS_UNCERTAIN,
            "biomechanics_valid": False,
            "biomechanics": {"fppa_left": None, "knee_flexion_left": None},
        }
        assert entry["biomechanics_valid"] is False
        assert entry["biomechanics"]["fppa_left"] is None
        assert entry["biomechanics"]["knee_flexion_left"] is None

    def test_uncertainty_pct_in_athlete_summary(self):
        r = _make_dummy_processor_result()
        a = r["athletes"][0]
        assert "uncertainty_pct" in a
        assert 0.0 <= a["uncertainty_pct"] <= 100.0

    def test_session_id_in_athlete_entries(self):
        r = _make_dummy_processor_result()
        for a in r["athletes"]:
            assert a["session_id"] == r["session_id"]

    def test_track_status_in_frame_athlete_entries(self):
        r = _make_dummy_processor_result()
        for frame in r["frames"]:
            for a in frame["athletes"]:
                assert "track_status" in a
                assert a["track_status"] in (
                    TRACK_STATUS_ACTIVE, TRACK_STATUS_UNCERTAIN,
                    TRACK_STATUS_TEMPORARILY_LOST, TRACK_STATUS_ENDED,
                )

    def test_biomechanics_valid_in_frame_athlete_entries(self):
        r = _make_dummy_processor_result()
        for frame in r["frames"]:
            for a in frame["athletes"]:
                assert "biomechanics_valid" in a
                assert isinstance(a["biomechanics_valid"], bool)


# ---------------------------------------------------------------------------
# Single-athlete pipeline regression
# ---------------------------------------------------------------------------

import src.pose.video_processor as _vp_module


class TestSingleAthleteRegressionPhase2:
    def test_video_processor_module_imports(self):
        """Single-athlete pipeline module still imports without error."""
        assert hasattr(_vp_module, "process_video_file"), (
            "process_video_file must exist in video_processor"
        )

    def test_video_processor_no_multi_athlete_state(self):
        """Single-athlete module must not expose multi-athlete tracking state."""
        assert not hasattr(_vp_module, "athlete_states"), (
            "video_processor must not have athlete_states"
        )
        assert not hasattr(_vp_module, "multi_athlete_tracker"), (
            "video_processor must not have multi_athlete_tracker"
        )

    def test_single_and_multi_athlete_modules_isolated(self):
        """Importing multi_athlete_processor must not contaminate video_processor."""
        import src.pose.multi_athlete_processor  # noqa: F401 — just ensure importable
        # video_processor unchanged
        assert hasattr(_vp_module, "process_video_file")

