"""
Unit tests for PlayerState lifecycle management.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.player_state import PlayerState


def test_player_state_initialization():
    p = PlayerState(tracker_id=7, jersey_number=23)
    assert p.tracker_id == 7
    assert p.jersey_number == 23
    assert p.display_name == "Player #23"
    assert p.active is True


def test_player_state_identity_update():
    p = PlayerState(tracker_id=12)
    assert p.display_name == "Player 012"

    p.update_identity(24)
    assert p.jersey_number == 24
    assert p.display_name == "Player #24"


def test_player_state_timeout_deactivation():
    p = PlayerState(tracker_id=1)
    assert p.active is True

    # Mark missing 30 frames
    for _ in range(30):
        p.mark_missing(max_timeout_frames=30)

    assert p.active is False
