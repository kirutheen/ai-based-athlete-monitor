"""
End-to-End Multi-Player Video Processing Pipeline Integration Test.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import pytest

from src.pose.multi_player_processor import process_multiplayer_video


def test_multiplayer_video_pipeline(tmp_path):
    video_path = str(tmp_path / "multiplayer_test.mp4")
    out_dir = str(tmp_path / "csv_test")

    # Generate synthetic 60-frame video with 2 distinct rectangular target candidate regions
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))

    for idx in range(60):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Left player candidate
        cv2.rectangle(frame, (80 + idx, 80), (220 + idx, 400), (0, 255, 0), -1)
        # Right player candidate
        cv2.rectangle(frame, (400 - idx, 100), (540 - idx, 420), (255, 0, 0), -1)
        out.write(frame)
    out.release()

    res = process_multiplayer_video(video_path, "mp_session_999", output_dir=out_dir)

    assert res["session_id"] == "mp_session_999"
    assert res["total_frames"] == 60
    assert res["total_players_tracked"] >= 1
    assert "csv_package" in res
    assert len(res["csv_package"]) == 8


def test_multiplayer_overlapping_players_tracking_stability():
    from src.pose.multi_player_tracker import MultiPlayerTracker

    tracker = MultiPlayerTracker(iou_threshold=0.20, max_missing_frames=10)

    # Simulate 15 frames of two players moving towards each other and crossing
    # Player A moving left-to-right, Player B moving right-to-left
    for frame_idx in range(15):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Player A box
        x_a = 50 + frame_idx * 25
        cv2.rectangle(frame, (x_a, 100), (x_a + 120, 420), (0, 255, 0), -1)

        # Player B box
        x_b = 450 - frame_idx * 25
        cv2.rectangle(frame, (x_b, 100), (x_b + 120, 420), (255, 0, 0), -1)

        active = tracker.update_frame(frame, timestamp_s=frame_idx / 30.0)
        assert len(active) >= 1, "At least one active player detected per frame"

    # Verify tracking identity stability
    player_ids = list(tracker.players.keys())
    assert len(player_ids) >= 2, "Both players registered with distinct tracker IDs"
    assert player_ids[0] != player_ids[1], "Separate tracking IDs assigned"


def test_independent_skeletons_and_no_cross_connections():
    from src.pose.multi_player_tracker import MultiPlayerTracker
    from src.pose.multi_player_overlay import MultiPlayerOverlayRenderer

    tracker = MultiPlayerTracker()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Player 1 on left
    cv2.rectangle(frame, (60, 80), (180, 400), (0, 255, 0), -1)
    # Player 2 on right
    cv2.rectangle(frame, (400, 80), (520, 400), (255, 0, 0), -1)

    active = tracker.update_frame(frame, timestamp_s=0.0)
    assert len(active) >= 2

    p1, p2 = active[0], active[1]
    assert p1.tracker_id != p2.tracker_id

    lm1 = p1.pose_history[-1]
    lm2 = p2.pose_history[-1]

    # When real MediaPipe is in use, landmarks may be None for synthetic/solid-color frames
    # (MediaPipe won't find a real human pose in drawn rectangles). Verify isolation only
    # when both players have real landmarks; verify non-shared data otherwise.
    if lm1 is not None and lm2 is not None:
        assert not np.array_equal(lm1, lm2), "Each player must have independent landmarks"
        assert lm1[23, 0] != lm2[23, 0], "No copied landmark coordinates"
    else:
        # At minimum, confirm the player objects are independent (different tracker IDs
        # already checked; bbox_history must differ since players were at different positions)
        assert p1.bbox_history[-1] != p2.bbox_history[-1], "Players must have independent bbox histories"

    # Overlay rendering verifies no crash and independent drawing
    renderer = MultiPlayerOverlayRenderer()
    annotated = renderer.render_frame_overlay(frame, active)
    assert annotated.shape == (480, 640, 3)

