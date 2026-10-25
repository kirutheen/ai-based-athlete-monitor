"""
Unit tests for MovementAnomalyDetector module.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.anomaly_detector import MovementAnomalyDetector, MovementAnomaly


def test_anomaly_detector_initialization():
    detector = MovementAnomalyDetector(contamination=0.08)
    assert detector.contamination == 0.08


def test_anomaly_detector_detects_persistent_zscore_drift():
    detector = MovementAnomalyDetector()

    # Generate 30 frames with baseline values around FPPA = 4.0
    frames = []
    for f in range(30):
        val = 4.0 if f < 15 or f > 22 else 18.5  # Persistent spike between frame 15 and 22
        frames.append({
            "frame_index": f,
            "timestamp": f / 30.0,
            "fppa_left": val,
            "knee_flexion_left": 45.0,
            "trunk_lean": 3.0,
        })
    df = pd.DataFrame(frames)

    baselines = {"fppa_left": {"median_value": 4.0, "iqr_value": 2.0}}
    anomalies = detector.detect_anomalies(df, baselines=baselines, min_persistence_frames=3)

    assert len(anomalies) > 0
    fppa_anom = next(a for a in anomalies if a.feature_name == "fppa_left")
    assert fppa_anom.severity in ("MODERATE", "HIGH")
    assert fppa_anom.observed_value == 18.5
    assert fppa_anom.baseline_median == 4.0
    assert "Persistent fppa_left deviation" in fppa_anom.reason


def test_anomaly_detector_empty_dataframe_returns_empty_list():
    detector = MovementAnomalyDetector()
    assert detector.detect_anomalies(pd.DataFrame(), baselines={}) == []
