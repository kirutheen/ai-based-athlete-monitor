"""Tests for src/baseline/baseline_engine.py and src/baseline/risk_bands.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.baseline.baseline_engine import compute_baseline, deviation_from_baseline
from src.baseline.risk_bands import AthleteRiskTracker, RiskBand, band_for_deviation


def _athlete_df(fppa_values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "athlete_id": ["ath_1"] * len(fppa_values),
            "fppa_left": fppa_values,
        }
    )


def test_compute_baseline_uses_only_warmup_window():
    # 15 events: first 10 are the warmup window, last 5 should be ignored
    # by compute_baseline even though they're wildly different.
    warmup = [5.0] * 10
    later = [50.0] * 5
    df = _athlete_df(warmup + later)
    baseline = compute_baseline(df, feature_columns=("fppa_left",), warmup_events=10)
    assert baseline.established is True
    assert baseline.medians["fppa_left"] == pytest.approx(5.0)
    assert baseline.iqrs["fppa_left"] == pytest.approx(0.0)


def test_compute_baseline_not_established_before_warmup():
    df = _athlete_df([5.0] * 4)
    baseline = compute_baseline(df, feature_columns=("fppa_left",), warmup_events=10)
    assert baseline.established is False
    assert baseline.n_events == 4


def test_deviation_from_baseline_zero_iqr_returns_nan():
    dev = deviation_from_baseline(10.0, median=5.0, iqr=0.0)
    assert dev != dev  # NaN


def test_deviation_from_baseline_basic_math():
    dev = deviation_from_baseline(value=10.0, median=5.0, iqr=2.0)
    assert dev == pytest.approx(2.5)


def test_band_for_deviation_thresholds():
    assert band_for_deviation(0.5) == RiskBand.LOW
    assert band_for_deviation(1.5) == RiskBand.MODERATE
    assert band_for_deviation(2.9) == RiskBand.MODERATE
    assert band_for_deviation(3.0) == RiskBand.HIGH
    assert band_for_deviation(float("nan")) == RiskBand.LOW


def test_persistence_rule_requires_3_of_5():
    tracker = AthleteRiskTracker(athlete_id="ath_1", feature_name="fppa_left")
    # Two High-band events: not enough to confirm yet.
    assert tracker.update(4.0) == RiskBand.LOW
    assert tracker.update(4.0) == RiskBand.LOW
    # Third High-band event in the window of 5: now confirmed.
    confirmed = tracker.update(4.0)
    assert confirmed == RiskBand.HIGH


def test_persistence_rule_single_outlier_does_not_confirm():
    tracker = AthleteRiskTracker(athlete_id="ath_1", feature_name="fppa_left")
    for _ in range(4):
        tracker.update(0.2)  # solidly Low
    confirmed = tracker.update(4.0)  # one High outlier among 4 Lows
    assert confirmed == RiskBand.LOW


def test_persistence_rule_can_clear_back_to_low():
    tracker = AthleteRiskTracker(athlete_id="ath_1", feature_name="fppa_left")
    for _ in range(5):
        tracker.update(4.0)
    assert tracker.confirmed_band == RiskBand.HIGH
    for _ in range(5):
        tracker.update(0.2)
    assert tracker.confirmed_band == RiskBand.LOW


def test_has_enough_history():
    tracker = AthleteRiskTracker(athlete_id="ath_1", feature_name="fppa_left")
    assert tracker.has_enough_history() is False
    tracker.update(0.1)
    tracker.update(0.1)
    assert tracker.has_enough_history() is False
    tracker.update(0.1)
    assert tracker.has_enough_history() is True
