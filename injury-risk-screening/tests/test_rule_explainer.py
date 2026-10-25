"""Tests for src/explain/rule_explainer.py."""

from __future__ import annotations

from src.baseline.risk_bands import RiskBand
from src.explain.rule_explainer import explain_event


def test_low_band_explanation_is_reassuring():
    exp = explain_event(
        athlete_id="ath_1",
        feature_name="fppa_left",
        event_value=5.2,
        baseline_median=5.0,
        baseline_iqr=2.0,
        deviation_iqr=0.1,
        band=RiskBand.LOW,
        persisted=True,
    )
    assert "normal range" in exp.headline.lower()
    assert exp.citation  # citation is still populated for LOW band


def test_unpersisted_band_notes_insufficient_history():
    exp = explain_event(
        athlete_id="ath_1",
        feature_name="fppa_left",
        event_value=20.0,
        baseline_median=5.0,
        baseline_iqr=2.0,
        deviation_iqr=7.5,
        band=RiskBand.HIGH,
        persisted=False,
    )
    assert "not enough events" in exp.detail.lower()
    assert "normal range" in exp.headline.lower()


def test_confirmed_high_band_explanation_cites_literature():
    exp = explain_event(
        athlete_id="ath_1",
        feature_name="fppa_left",
        event_value=20.0,
        baseline_median=5.0,
        baseline_iqr=2.0,
        deviation_iqr=7.5,
        band=RiskBand.HIGH,
        persisted=True,
    )
    assert "substantially elevated" in exp.headline.lower()
    assert "Hewett" in exp.citation
    assert "3 of their last 5" in exp.detail or "3 of" in exp.detail


def test_confirmed_moderate_band_wording_differs_from_high():
    exp_mod = explain_event(
        athlete_id="ath_1",
        feature_name="trunk_lean",
        event_value=12.0,
        baseline_median=5.0,
        baseline_iqr=3.0,
        deviation_iqr=2.3,
        band=RiskBand.MODERATE,
        persisted=True,
    )
    assert "moderately elevated" in exp_mod.headline.lower()
    assert "Zazulak" in exp_mod.citation
