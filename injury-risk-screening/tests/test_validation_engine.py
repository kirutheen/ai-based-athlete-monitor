"""
Unit tests for Reference Validation Engine.
"""

from __future__ import annotations

import pytest

from src.validation.validation_engine import ReferenceValidationEngine


def test_validation_engine_high_agreement():
    engine = ReferenceValidationEngine()
    est = [10.0, 20.0, 30.0, 40.0, 50.0]
    ref = [10.2, 19.8, 30.1, 40.2, 49.9]

    res = engine.validate_metric_series("knee_flexion", est, ref, unit="deg")
    assert res.validation_status == "VALIDATED"
    assert res.rmse < 1.0
    assert res.correlation_r > 0.99
    assert res.target_threshold_met is True


def test_validation_engine_poor_agreement():
    engine = ReferenceValidationEngine()
    est = [10.0, 20.0, 30.0, 40.0, 50.0]
    ref = [25.0, 10.0, 60.0, 15.0, 80.0]

    res = engine.validate_metric_series("knee_flexion", est, ref, unit="deg")
    assert res.validation_status == "NOT_VALIDATED"
    assert res.rmse > 10.0
    assert res.target_threshold_met is False
