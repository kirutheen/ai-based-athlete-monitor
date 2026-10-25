"""
Unit tests for Calibration Gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pose.calibration_gate import CalibrationGate, CalibrationStage


def test_calibration_gate_none_frame():
    gate = CalibrationGate()
    res = gate.evaluate_calibration_frame(None)
    assert res.is_ready is False
    assert res.current_stage == CalibrationStage.CAMERA_CHECK
    assert "NO_POSE_DETECTED" in res.failed_checks


def test_calibration_gate_full_visibility_ready():
    gate = CalibrationGate()
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 3] = 0.90
    # Set shoulders, hips, knees, ankles properly centered
    lm[11] = [0.40, 0.20, 0.0, 0.90] # left shoulder
    lm[12] = [0.60, 0.20, 0.0, 0.90] # right shoulder
    lm[23] = [0.42, 0.50, 0.0, 0.90] # left hip
    lm[24] = [0.58, 0.50, 0.0, 0.90] # right hip
    lm[25] = [0.43, 0.70, 0.0, 0.90] # left knee
    lm[26] = [0.57, 0.70, 0.0, 0.90] # right knee
    lm[27] = [0.44, 0.90, 0.0, 0.90] # left ankle
    lm[28] = [0.56, 0.90, 0.0, 0.90] # right ankle

    res = gate.evaluate_calibration_frame(lm)
    assert res.is_ready is True
    assert res.current_stage == CalibrationStage.READY
    assert len(res.failed_checks) == 0
    assert res.estimated_distance_m is not None
