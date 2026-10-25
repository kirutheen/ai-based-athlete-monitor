"""
Unsupervised Anomaly & Movement Drift Detection Engine for DriftGuard.

Combines:
1. Robust Z-score / IQR statistical baseline deviation
2. Unsupervised Isolation Forest multi-dimensional feature anomaly detection
3. Temporal Persistence Filtering (requiring >= 3 consecutive frames) to prevent false positives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class MovementAnomaly:
    feature_name: str
    frame_index: int
    timestamp_s: float
    observed_value: float
    baseline_median: float
    deviation_zscore: float
    severity: str  # LOW, MODERATE, HIGH
    confidence: float
    reason: str


class MovementAnomalyDetector:
    """Detects multi-dimensional movement anomalies & baseline drift across trial frames."""

    def __init__(self, contamination: float = 0.10, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state

    def detect_anomalies(
        self,
        combined_df: pd.DataFrame,
        baselines: Dict[str, Dict[str, float]],
        min_persistence_frames: int = 3,
    ) -> List[MovementAnomaly]:
        """
        Runs statistical Z-score + Isolation Forest anomaly detection on trial frames.
        Returns persistent MovementAnomaly objects.
        """
        if combined_df is None or combined_df.empty or len(combined_df) < 5:
            return []

        anomalies: List[MovementAnomaly] = []

        # Target features for anomaly scanning
        feature_cols = [
            "fppa_left", "fppa_right", "knee_flexion_left", "knee_flexion_right",
            "trunk_lean", "knee_valgus_LSI", "knee_ap_translation_mm",
            "knee_ml_translation_mm", "knee_int_ext_rot_deg"
        ]
        valid_cols = [c for c in feature_cols if c in combined_df.columns]

        if not valid_cols:
            return []

        # 1. Statistical Baseline Drift Scan (Z-score / IQR)
        for col in valid_cols:
            base_info = baselines.get(col, {})
            median = base_info.get("median_value", float("nan"))
            iqr = base_info.get("iqr_value", float("nan"))

            if math.isnan(median) or math.isnan(iqr) or iqr < 1e-4:
                # Default clinical reference fallback if athlete baseline in progress
                ref_defaults = {
                    "fppa_left": (4.5, 3.0),
                    "fppa_right": (4.0, 2.8),
                    "knee_flexion_left": (48.0, 6.0),
                    "knee_flexion_right": (50.0, 5.5),
                    "trunk_lean": (4.0, 2.0),
                    "knee_valgus_LSI": (8.0, 5.0),
                    "knee_ap_translation_mm": (2.5, 1.2),
                    "knee_ml_translation_mm": (1.2, 0.8),
                    "knee_int_ext_rot_deg": (3.5, 1.5),
                }
                median, iqr = ref_defaults.get(col, (5.0, 2.5))

            series = combined_df[col].fillna(median)
            z_scores = (series - median) / (iqr * 0.7413)  # Robust Z-score

            # Identify candidate anomalous frames (Z > 2.0)
            candidate_indices = np.where(np.abs(z_scores) >= 2.0)[0]

            # Apply Temporal Persistence Filter (>= min_persistence_frames consecutive)
            if len(candidate_indices) > 0:
                groups = np.split(candidate_indices, np.where(np.diff(candidate_indices) != 1)[0] + 1)
                for grp in groups:
                    if len(grp) >= min_persistence_frames:
                        peak_frame_in_grp = grp[np.argmax(np.abs(z_scores.iloc[grp]))]
                        val = float(series.iloc[peak_frame_in_grp])
                        z = float(z_scores.iloc[peak_frame_in_grp])
                        ts = float(combined_df["timestamp"].iloc[peak_frame_in_grp]) if "timestamp" in combined_df.columns else peak_frame_in_grp / 30.0

                        sev = "HIGH" if abs(z) >= 3.5 else "MODERATE"
                        anomalies.append(
                            MovementAnomaly(
                                feature_name=col,
                                frame_index=int(peak_frame_in_grp),
                                timestamp_s=round(ts, 2),
                                observed_value=round(val, 1),
                                baseline_median=round(median, 1),
                                deviation_zscore=round(z, 2),
                                severity=sev,
                                confidence=min(0.98, round(0.70 + abs(z) * 0.08, 2)),
                                reason=f"Persistent {col} deviation ({val:.1f} vs baseline {median:.1f}, Z={z:+.1f}) across {len(grp)} frames.",
                            )
                        )

        # 2. Isolation Forest Multi-Dimensional Anomaly Detection
        X = combined_df[valid_cols].fillna(combined_df[valid_cols].median()).fillna(0.0)
        if len(X) >= 10:
            iso = IsolationForest(contamination=self.contamination, random_state=self.random_state)
            preds = iso.fit_predict(X)
            scores = iso.decision_function(X)

            iso_anomaly_frames = np.where(preds == -1)[0]
            if len(iso_anomaly_frames) > 0:
                groups = np.split(iso_anomaly_frames, np.where(np.diff(iso_anomaly_frames) != 1)[0] + 1)
                for grp in groups:
                    if len(grp) >= min_persistence_frames:
                        worst_idx = grp[np.argmin(scores[grp])]
                        ts = float(combined_df["timestamp"].iloc[worst_idx]) if "timestamp" in combined_df.columns else worst_idx / 30.0

                        anomalies.append(
                            MovementAnomaly(
                                feature_name="multi_joint_coordination",
                                frame_index=int(worst_idx),
                                timestamp_s=round(ts, 2),
                                observed_value=round(float(scores[worst_idx]), 3),
                                baseline_median=0.0,
                                deviation_zscore=round(float(scores[worst_idx]) * -10.0, 2),
                                severity="MODERATE",
                                confidence=0.88,
                                reason=f"Multi-joint kinematic coordination anomaly detected at t={ts:.2f}s (Isolation Forest score {scores[worst_idx]:.2f}).",
                            )
                        )

        return anomalies
