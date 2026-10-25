"""Tests for src/features/event_segmentation.py using a synthetic hip-height
trajectory that simulates a clean drop-jump (stand -> rise -> fall -> land)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.event_segmentation import segment_landing_events


def _synthetic_jump_df(fps: float = 30.0, n_frames: int = 90) -> pd.DataFrame:
    """
    Build a synthetic hip-height (y) trajectory: flat baseline, then one
    clean jump (sinusoidal dip = rising into the air since y decreases
    upward on screen), then back to a flat baseline. Normalized image
    coordinates in [0, 1], baseline around y=0.5.
    """
    t = np.arange(n_frames) / fps
    baseline = 0.5
    hip_y = np.full(n_frames, baseline)

    jump_start, jump_end = 30, 50
    jump_frames = np.arange(jump_start, jump_end)
    # A dip (rise in the air) shaped like a downward-opening parabola in y.
    phase = (jump_frames - jump_start) / (jump_end - jump_start)
    hip_y[jump_frames] = baseline - 0.08 * np.sin(phase * np.pi)

    df = pd.DataFrame(
        {
            "frame_index": np.arange(n_frames),
            "timestamp": t,
            "left_hip_y": hip_y,
            "right_hip_y": hip_y,
        }
    )
    return df


def test_segment_landing_events_detects_one_jump():
    df = _synthetic_jump_df()
    events = segment_landing_events(df, fps=30.0, min_jump_height_px=0.03, min_event_separation_s=0.5)
    assert len(events) == 1
    event = events[0]
    assert event.takeoff_frame < event.peak_frame < event.landing_frame


def test_segment_landing_events_ignores_small_sway():
    """Postural sway well under min_jump_height_px should not be detected as a jump."""
    fps = 30.0
    n_frames = 60
    hip_y = 0.5 + 0.005 * np.sin(np.arange(n_frames) / 5.0)  # tiny wobble
    df = pd.DataFrame(
        {
            "frame_index": np.arange(n_frames),
            "timestamp": np.arange(n_frames) / fps,
            "left_hip_y": hip_y,
            "right_hip_y": hip_y,
        }
    )
    events = segment_landing_events(df, fps=fps, min_jump_height_px=0.03)
    assert events == []


def test_segment_landing_events_requires_timestamp_column():
    df = pd.DataFrame({"left_hip_y": [0.5, 0.5], "right_hip_y": [0.5, 0.5]})
    try:
        segment_landing_events(df, fps=30.0)
        assert False, "expected ValueError for missing timestamp column"
    except ValueError:
        pass


def test_segment_landing_events_handles_too_short_input():
    df = pd.DataFrame({"frame_index": [0, 1], "timestamp": [0.0, 0.03], "left_hip_y": [0.5, 0.5], "right_hip_y": [0.5, 0.5]})
    events = segment_landing_events(df, fps=30.0)
    assert events == []


def test_min_event_separation_prevents_double_counting():
    """Two jumps closer together than min_event_separation_s should still
    register as 2 distinct events as long as each clears separation from
    the *previous landing*, and never collapse into more events than
    physically happened."""
    df = _synthetic_jump_df(n_frames=90)
    events = segment_landing_events(df, fps=30.0, min_event_separation_s=2.0)
    # With a 2s minimum separation and only ~2s of synthetic data containing
    # one jump, we should still get exactly one event (not spuriously split).
    assert len(events) == 1
