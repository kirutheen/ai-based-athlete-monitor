"""
Unit tests for Multi-Trial Baseline & Consistency Engine.
"""

from __future__ import annotations

import pytest

from src.features.multi_trial_baseline import (
    BaselineDeviationCategory,
    MultiTrialBaselineEngine,
)


def test_compute_trial_consistency():
    engine = MultiTrialBaselineEngine()
    trials = [18.2, 18.5, 18.1, 18.4, 18.3]
    res = engine.compute_trial_consistency("fppa_left", trials)

    assert res.valid_trials == 5
    assert pytest.approx(res.median_val, 0.1) == 18.3
    assert res.is_consistent is True
    assert res.cv_percent < 5.0


def test_compare_against_baseline_within():
    engine = MultiTrialBaselineEngine()
    history = [15.0, 15.2, 14.8, 15.1, 14.9, 15.0]
    res = engine.compare_against_baseline("fppa_left", 15.1, history)

    assert res.deviation_category == BaselineDeviationCategory.WITHIN_PERSONAL_BASELINE
    assert abs(res.robust_z_score) <= 1.5


def test_compare_against_baseline_deviation():
    engine = MultiTrialBaselineEngine()
    history = [15.0, 15.2, 14.8, 15.1, 14.9, 15.0]
    # Current value is 25.0 degrees (substantial spike from 15.0 baseline)
    res = engine.compare_against_baseline("fppa_left", 25.0, history)

    assert res.deviation_category in (
        BaselineDeviationCategory.PERSISTENT_DEVIATION,
        BaselineDeviationCategory.SUBSTANTIAL_DEVIATION,
    )
    assert res.robust_z_score > 3.0
