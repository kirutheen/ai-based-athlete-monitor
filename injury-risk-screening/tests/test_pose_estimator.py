"""Unit tests for the pose estimation + smoothing module.

These tests avoid depending on the actual MediaPipe Pose Landmarker model
weights (a `.task` file, which MediaPipe does not ship inside its pip
package and which must be downloaded separately). Instead:

- The smoothing utilities are tested directly -- they have no model
  dependency.
- The PoseEstimator wrapper is tested against a mocked
  `vision.PoseLandmarker`, verifying that landmark shapes, confidence
  scoring, and the "no detection" path all behave correctly -- i.e. that
  OUR code is correct, independent of whether the model file is present.
- If a real model file is available (set POSE_MODEL_PATH, or place one at
  models/pose_landmarker_full.task), an additional end-to-end smoke test
  runs against it on a synthetic frame; otherwise that one test is skipped
  with a clear reason.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pose.pose_estimator import (
    NUM_LANDMARKS,
    OneEuroFilter,
    PoseEstimator,
    PoseResult,
    smooth,
    smooth_moving_average,
)


# --------------------------------------------------------------------------
# Smoothing utilities -- no model dependency
# --------------------------------------------------------------------------

def _make_landmarks(x_offset: float) -> np.ndarray:
    """A (33, 4) landmark array where every landmark's x = x_offset."""
    arr = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
    arr[:, 0] = x_offset
    arr[:, 3] = 1.0  # visibility
    return arr


def test_smooth_moving_average_shape_and_averaging():
    seq = [_make_landmarks(x) for x in [0.0, 1.0, 2.0, 3.0]]
    out = smooth_moving_average(seq, window=2)
    assert len(out) == 4
    assert out[0].shape == (NUM_LANDMARKS, 4)
    # window=2 -> last element averages frames at x=2 and x=3
    assert np.isclose(out[-1][0, 0], 2.5)


def test_smooth_moving_average_passes_through_missing_frames():
    seq = [_make_landmarks(0.0), None, _make_landmarks(2.0)]
    out = smooth_moving_average(seq, window=2)
    assert out[1] is None
    assert out[0] is not None and out[2] is not None


def test_one_euro_filter_reduces_jitter():
    filt = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.0)
    noisy = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    filtered = [filt(v) for v in noisy]
    assert np.std(filtered) < np.std(noisy)


def test_smooth_dispatch_one_euro_and_moving_average():
    seq = [_make_landmarks(x) for x in range(5)]
    out_euro = smooth(seq, method="one_euro")
    out_ma = smooth(seq, method="moving_average", window=3)
    assert len(out_euro) == len(seq)
    assert len(out_ma) == len(seq)
    with pytest.raises(ValueError):
        smooth(seq, method="not_a_method")


# --------------------------------------------------------------------------
# PoseEstimator -- mocked MediaPipe backend (no model file required)
# --------------------------------------------------------------------------

class _FakeLandmark:
    def __init__(self, x, y, z, visibility):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


class _FakeResult:
    def __init__(self, pose_landmarks, pose_world_landmarks=None):
        self.pose_landmarks = pose_landmarks
        self.pose_world_landmarks = pose_world_landmarks or []


def _fake_pose_landmarks():
    return [[_FakeLandmark(0.5, 0.5, 0.0, 0.9) for _ in range(NUM_LANDMARKS)]]


@patch("src.pose.pose_estimator.vision.PoseLandmarker.create_from_options")
def test_pose_estimator_process_returns_expected_shape(mock_create):
    mock_landmarker = MagicMock()
    mock_landmarker.detect_for_video.return_value = _FakeResult(_fake_pose_landmarks())
    mock_create.return_value = mock_landmarker

    estimator = PoseEstimator(model_asset_path="unused.task", running_mode="VIDEO")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = estimator.process(frame, timestamp_ms=0)

    assert isinstance(result, PoseResult)
    assert result.detected is True
    assert result.landmarks_2d.shape == (NUM_LANDMARKS, 4)
    assert 0.0 <= result.confidence <= 1.0
    assert result.annotated_frame.shape == frame.shape


@patch("src.pose.pose_estimator.vision.PoseLandmarker.create_from_options")
def test_pose_estimator_handles_no_detection(mock_create):
    mock_landmarker = MagicMock()
    mock_landmarker.detect_for_video.return_value = _FakeResult([])
    mock_create.return_value = mock_landmarker

    estimator = PoseEstimator(model_asset_path="unused.task", running_mode="VIDEO")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = estimator.process(frame, timestamp_ms=0)

    assert result.detected is False
    assert result.landmarks_2d is None
    assert result.confidence == 0.0
    assert result.annotated_frame.shape == frame.shape


@patch("src.pose.pose_estimator.vision.PoseLandmarker.create_from_options")
def test_pose_estimator_timestamps_must_increase(mock_create):
    """VIDEO mode requires strictly increasing timestamps; the wrapper should
    self-correct rather than crash if the caller passes a non-increasing one.
    """
    mock_landmarker = MagicMock()
    mock_landmarker.detect_for_video.return_value = _FakeResult(_fake_pose_landmarks())
    mock_create.return_value = mock_landmarker

    estimator = PoseEstimator(model_asset_path="unused.task", running_mode="VIDEO")
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    estimator.process(frame, timestamp_ms=100)
    estimator.process(frame, timestamp_ms=50)  # would violate monotonicity if passed through as-is

    calls = mock_landmarker.detect_for_video.call_args_list
    ts_used = [c.args[1] for c in calls]
    assert ts_used[1] > ts_used[0]


# --------------------------------------------------------------------------
# Optional real end-to-end smoke test, only if a model file is available
# --------------------------------------------------------------------------

def _find_model_path():
    candidate = os.environ.get("POSE_MODEL_PATH") or "models/pose_landmarker_full.task"
    return candidate if Path(candidate).is_file() else None


@pytest.mark.skipif(
    _find_model_path() is None,
    reason="No MediaPipe Pose Landmarker .task model file available "
    "(set POSE_MODEL_PATH or place one at models/pose_landmarker_full.task)",
)
def test_pose_estimator_real_model_smoke_test():
    model_path = _find_model_path()
    estimator = PoseEstimator(model_asset_path=model_path, running_mode="VIDEO")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = estimator.process(frame, timestamp_ms=0)
    assert isinstance(result, PoseResult)
    estimator.close()
