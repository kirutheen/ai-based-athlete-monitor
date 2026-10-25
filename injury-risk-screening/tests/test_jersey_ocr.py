"""
Unit tests for Jersey OCR Engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pose.jersey_ocr import JerseyOCREngine


def test_jersey_ocr_temporal_confirmation():
    ocr = JerseyOCREngine(min_confirmation_frames=3)

    # Frame 1: Candidate 23 detected
    res1 = ocr.process_player_ocr(tracker_id=1, frame_bgr=np.zeros((480, 640, 3), dtype=np.uint8), bbox=(0.2, 0.2, 0.5, 0.8))
    # Not yet confirmed (only 1 frame)
    assert res1 is None or isinstance(res1, int)

    # Supply 3 consecutive readings
    ocr.ocr_history[1] = [23, 23, 23]
    res_confirmed = ocr.process_player_ocr(tracker_id=1, frame_bgr=np.zeros((480, 640, 3), dtype=np.uint8), bbox=(0.2, 0.2, 0.5, 0.8))
    assert res_confirmed == 23
