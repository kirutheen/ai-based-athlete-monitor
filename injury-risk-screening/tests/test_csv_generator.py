"""
Unit tests for Real Telemetry CSV Package Generator.
"""

from __future__ import annotations

import os
import pandas as pd
import pytest

from src.features.player_state import PlayerState
from src.reports.csv_generator import generate_session_csv_package


def test_csv_generator_package(tmp_path):
    out_dir = str(tmp_path / "csv_out")

    p1 = PlayerState(tracker_id=1, jersey_number=10)
    p2 = PlayerState(tracker_id=2, jersey_number=20)

    p1.add_frame_data(1.0, (0.1, 0.1, 0.3, 0.9), None, {"fppa_left": 12.0, "knee_flexion_left": 40.0})
    p2.add_frame_data(1.0, (0.5, 0.1, 0.7, 0.9), None, {"fppa_left": 6.0, "knee_flexion_left": 30.0})

    collisions = [{
        "event_id": "col_1",
        "player_ids": [1, 2],
        "player_names": ["Player #10", "Player #20"],
        "frame_number": 30,
        "timestamp_s": 1.0,
        "severity": "HIGH",
        "iou_overlap": 0.25,
        "relative_velocity": 0.8,
        "summary": "Collision detected",
    }]

    pkg = generate_session_csv_package("test_session_123", [p1, p2], collisions, out_dir)

    assert len(pkg) == 8
    for file_key, file_path in pkg.items():
        assert os.path.exists(file_path)

    # Validate trial_features.csv content
    df_trial = pd.read_csv(pkg["trial_features"])
    assert len(df_trial) == 2
    assert set(df_trial["tracker_id"]) == {1, 2}

    # Validate collision_events.csv content
    df_col = pd.read_csv(pkg["collision_events"])
    assert len(df_col) == 1
    assert df_col.iloc[0]["event_id"] == "col_1"
