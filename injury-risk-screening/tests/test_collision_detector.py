"""
Unit tests for Pairwise Collision Detector.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.collision_detector import CollisionDetector
from src.features.player_state import PlayerState


def test_collision_detector_overlapping_players():
    detector = CollisionDetector(iou_collision_threshold=0.10)

    p1 = PlayerState(tracker_id=1, jersey_number=4)
    p2 = PlayerState(tracker_id=2, jersey_number=9)
    p3 = PlayerState(tracker_id=3, jersey_number=15)  # Uninvolved player

    # Player 1 and Player 2 bounding boxes overlap heavily
    p1.bbox_history.append((0.20, 0.10, 0.50, 0.90))
    p2.bbox_history.append((0.25, 0.10, 0.55, 0.90))
    p3.bbox_history.append((0.80, 0.10, 0.95, 0.90))

    events = detector.evaluate_frame_collisions([p1, p2, p3], timestamp_s=1.0, frame_number=30)

    assert len(events) == 1
    evt = events[0]
    assert set(evt.player_ids) == {1, 2}
    assert evt.iou_overlap > 0.10

    # Verify event is attached to Player 1 and Player 2, but NOT Player 3
    assert len(p1.collision_events) == 1
    assert len(p2.collision_events) == 1
    assert len(p3.collision_events) == 0
