"""
Critical Data Isolation Unit Tests for Multi-Player Engine.

Verifies that Player A's movements, abnormal events, and risk scores
NEVER alter Player B's baseline, biomechanics, or risk score.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.player_state import PlayerState


def test_player_data_isolation():
    p1 = PlayerState(tracker_id=1, jersey_number=10)
    p2 = PlayerState(tracker_id=2, jersey_number=20)

    # Player 1 experiences severe dynamic valgus spike (High Risk)
    p1.add_frame_data(
        timestamp_s=1.0,
        bbox=(0.1, 0.1, 0.3, 0.9),
        landmarks=np.ones((33, 4)),
        biomechanics={"fppa_left": 25.0, "knee_flexion_left": 40.0, "valgus_asymmetry_lsi_pct": 50.0},
    )
    p1.add_abnormal_event({
        "event_type": "Knee Valgus Spike",
        "severity": "HIGH",
        "detail": "FPPA spike 25.0 degrees",
    })

    # Player 2 experiences normal motion (Low Risk)
    p2.add_frame_data(
        timestamp_s=1.0,
        bbox=(0.5, 0.1, 0.7, 0.9),
        landmarks=np.ones((33, 4)),
        biomechanics={"fppa_left": 6.0, "knee_flexion_left": 35.0, "valgus_asymmetry_lsi_pct": 5.0},
    )

    # Assert Player 1 is HIGH risk
    r1 = p1.get_latest_risk_score()
    assert r1["risk_band"] == "HIGH"
    assert len(p1.abnormal_events) == 1

    # Assert Player 2 remains LOW risk and has ZERO abnormal events
    r2 = p2.get_latest_risk_score()
    assert r2["risk_band"] == "LOW"
    assert len(p2.abnormal_events) == 0
    assert r2["score"] > r1["score"]
