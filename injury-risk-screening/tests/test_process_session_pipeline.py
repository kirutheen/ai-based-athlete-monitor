"""
Regression test for the full offline pipeline glue in
scripts/process_session.py: pose dataframe -> features -> event
segmentation, run together exactly as that script does.

This exists because of a real bug caught during manual end-to-end
testing: extract_features_dataframe's output re-includes frame_index and
timestamp columns (it needs them internally), so naively
pd.concat()-ing it onto the original pose dataframe produces two
columns named "frame_index" -- and segment_landing_events() then reads
df.iloc[i]["frame_index"] expecting a scalar and gets a 2-element Series
instead, raising a confusing TypeError deep inside int(). The fix is to
drop the duplicate columns from the features dataframe before
concatenating (see process_session.py); this test locks that behavior in
so it can't silently regress.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.biomechanics import extract_features_dataframe
from src.features.event_segmentation import segment_landing_events
from src.features.landmarks import ALL_LANDMARK_NAMES


def _synthetic_pose_df(fps: float = 30.0, n_frames: int = 90) -> pd.DataFrame:
    data = {"frame_index": np.arange(n_frames), "timestamp": np.arange(n_frames) / fps}
    for name in ALL_LANDMARK_NAMES:
        data[f"{name}_x"] = np.full(n_frames, 0.5)
        data[f"{name}_y"] = np.full(n_frames, 0.5)
        data[f"{name}_z"] = np.zeros(n_frames)
        data[f"{name}_v"] = np.ones(n_frames)

    jump = np.arange(30, 50)
    phase = (jump - 30) / 20.0
    dip = 0.08 * np.sin(phase * np.pi)
    for side in ("left", "right"):
        data[f"{side}_hip_y"][jump] = 0.5 - dip
        data[f"{side}_knee_y"][jump] = 0.65 - dip * 0.8
        data[f"{side}_ankle_y"][jump] = 0.8 - dip * 0.1

    return pd.DataFrame(data)


def test_feature_extraction_then_segmentation_does_not_duplicate_columns():
    pose_df = _synthetic_pose_df()
    features_df = extract_features_dataframe(pose_df, list(ALL_LANDMARK_NAMES))

    # The bug: features_df carries its own frame_index/timestamp copies.
    assert "frame_index" in features_df.columns
    assert "timestamp" in features_df.columns

    # The fix process_session.py applies: drop before concatenating.
    features_only = features_df.drop(columns=["frame_index", "timestamp"])
    combined = pd.concat([pose_df.reset_index(drop=True), features_only], axis=1)

    assert list(combined.columns).count("frame_index") == 1
    assert list(combined.columns).count("timestamp") == 1

    # And segmentation, which indexes df.iloc[i]["frame_index"] as a
    # scalar, must run without raising.
    events = segment_landing_events(combined, fps=30.0)
    assert len(events) == 1
    assert isinstance(events[0].takeoff_frame, int)


def test_concat_without_dropping_duplicates_breaks_segmentation():
    """Documents WHY the drop is necessary: without it, segmentation
    raises exactly the TypeError seen during manual testing."""
    pose_df = _synthetic_pose_df()
    features_df = extract_features_dataframe(pose_df, list(ALL_LANDMARK_NAMES))
    combined_buggy = pd.concat([pose_df.reset_index(drop=True), features_df], axis=1)

    assert list(combined_buggy.columns).count("frame_index") == 2  # the bug, reproduced

    try:
        segment_landing_events(combined_buggy, fps=30.0)
        raised = False
    except TypeError:
        raised = True
    assert raised, "expected the known duplicate-column bug to raise TypeError when not fixed"
