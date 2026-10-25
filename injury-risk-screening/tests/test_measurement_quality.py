"""
Unit tests for Measurement Quality Engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.quality.measurement_quality import MeasurementQualityEngine, QualityCategory


def test_quality_engine_insufficient_frames():
    engine = MeasurementQualityEngine()
    res = engine.evaluate_sequence([])
    assert res.quality_category == QualityCategory.INSUFFICIENT
    assert res.is_sufficient_for_screening is False
    assert "MINIMUM_FRAME_COUNT" in res.failed_checks
    assert "INSUFFICIENT MEASUREMENT QUALITY" in res.recommendation


def test_quality_engine_high_quality_sequence():
    engine = MeasurementQualityEngine()

    # Generate 30 high-quality frames with full landmark visibility
    frames = []
    for _ in range(30):
        lm = np.ones((33, 4), dtype=np.float32)
        lm[:, 3] = 0.95  # high confidence visibility
        lm[25, 0] = 0.50 # steady left knee x coordinate
        lm[25, 1] = 0.60
        frames.append(lm)

    res = engine.evaluate_sequence(frames)
    assert res.quality_category == QualityCategory.HIGH
    assert res.is_sufficient_for_screening is True
    assert len(res.failed_checks) == 0
    assert res.quality_score >= 0.80


def test_quality_engine_low_visibility_failure():
    engine = MeasurementQualityEngine()

    # Generate 30 frames with low keypoint visibility (< 0.5)
    frames = []
    for _ in range(30):
        lm = np.ones((33, 4), dtype=np.float32)
        lm[:, 3] = 0.20  # low confidence visibility
        frames.append(lm)

    res = engine.evaluate_sequence(frames)
    assert res.is_sufficient_for_screening is False
    assert res.quality_category == QualityCategory.INSUFFICIENT
    assert "EXCESSIVE_MISSING_LANDMARKS" in res.failed_checks
