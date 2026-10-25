"""
Multi-Trial Consistency & Personalized Baseline Engine for DriftGuard.

Calculates robust statistical metrics across multiple recorded trials:
- Mean, Median, Standard Deviation (SD)
- Interquartile Range (IQR), Median Absolute Deviation (MAD)
- Coefficient of Variation (CV)

Distinguishes:
1. Personal Baseline Status (Within Baseline, Mild Deviation, Persistent Deviation, Substantial Deviation)
2. Personal Baseline vs. Population Reference Norms.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class BaselineDeviationCategory(str, Enum):
    WITHIN_PERSONAL_BASELINE = "WITHIN_PERSONAL_BASELINE"
    MILD_DEVIATION = "MILD_DEVIATION"
    PERSISTENT_DEVIATION = "PERSISTENT_DEVIATION"
    SUBSTANTIAL_DEVIATION = "SUBSTANTIAL_DEVIATION"
    INSUFFICIENT_BASELINE = "INSUFFICIENT_BASELINE"


@dataclass
class TrialConsistencySummary:
    feature_name: str
    mean_val: float
    median_val: float
    sd_val: float
    iqr_val: float
    mad_val: float
    cv_percent: float
    total_trials: int
    valid_trials: int
    is_consistent: bool
    summary_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "mean_val": round(self.mean_val, 2),
            "median_val": round(self.median_val, 2),
            "sd_val": round(self.sd_val, 2),
            "iqr_val": round(self.iqr_val, 2),
            "mad_val": round(self.mad_val, 2),
            "cv_percent": round(self.cv_percent, 1),
            "total_trials": self.total_trials,
            "valid_trials": self.valid_trials,
            "is_consistent": self.is_consistent,
            "summary_text": self.summary_text,
        }


@dataclass
class BaselineComparisonResult:
    feature_name: str
    current_value: float
    baseline_median: float
    baseline_iqr: float
    baseline_mad: float
    robust_z_score: float
    deviation_category: BaselineDeviationCategory
    population_reference_median: Optional[float]
    comparison_note: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "current_value": round(self.current_value, 2),
            "baseline_median": round(self.baseline_median, 2),
            "baseline_iqr": round(self.baseline_iqr, 2),
            "baseline_mad": round(self.baseline_mad, 2),
            "robust_z_score": round(self.robust_z_score, 2),
            "deviation_category": self.deviation_category.value,
            "population_reference_median": round(self.population_reference_median, 2) if self.population_reference_median is not None else None,
            "comparison_note": self.comparison_note,
        }


class MultiTrialBaselineEngine:
    """
    Computes statistical consistency across trials and compares against subject longitudinal baseline.
    """

    MIN_TRIALS_FOR_BASELINE = 3

    def compute_trial_consistency(
        self,
        feature_name: str,
        values: List[float],
    ) -> TrialConsistencySummary:
        """
        Compute mean, median, SD, IQR, MAD, and CV across a list of trial values.
        """
        valid_vals = [float(v) for v in values if v is not None and not np.isnan(v)]
        total_trials = len(values)
        valid_trials = len(valid_vals)

        if valid_trials == 0:
            return TrialConsistencySummary(
                feature_name=feature_name,
                mean_val=0.0,
                median_val=0.0,
                sd_val=0.0,
                iqr_val=0.0,
                mad_val=0.0,
                cv_percent=0.0,
                total_trials=total_trials,
                valid_trials=0,
                is_consistent=False,
                summary_text="No valid trial measurements available.",
            )

        arr = np.array(valid_vals, dtype=np.float64)
        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        sd_val = float(np.std(arr, ddof=1)) if valid_trials > 1 else 0.0

        q75, q25 = np.percentile(arr, [75, 25])
        iqr_val = float(q75 - q25)

        mad_val = float(np.median(np.abs(arr - median_val)))
        cv_percent = float((sd_val / max(1e-5, abs(mean_val))) * 100.0) if mean_val != 0 else 0.0

        # Consistency rule: CV < 20% or SD < 3.0 degrees/mm
        is_consistent = (cv_percent <= 20.0 or sd_val <= 3.0)

        if is_consistent:
            summary_text = f"Consistent across {valid_trials} trials (CV: {cv_percent:.1f}%)."
        else:
            summary_text = f"High trial variability across {valid_trials} trials (CV: {cv_percent:.1f}%)."

        return TrialConsistencySummary(
            feature_name=feature_name,
            mean_val=mean_val,
            median_val=median_val,
            sd_val=sd_val,
            iqr_val=iqr_val,
            mad_val=mad_val,
            cv_percent=cv_percent,
            total_trials=total_trials,
            valid_trials=valid_trials,
            is_consistent=is_consistent,
            summary_text=summary_text,
        )

    def compare_against_baseline(
        self,
        feature_name: str,
        current_value: float,
        baseline_history: List[float],
        population_reference_median: Optional[float] = None,
    ) -> BaselineComparisonResult:
        """
        Compare a current trial value against subject's longitudinal baseline history.
        Uses Robust Z-Score: z = (x - median) / (1.4826 * MAD)
        """
        valid_history = [float(v) for v in baseline_history if v is not None and not np.isnan(v)]

        if len(valid_history) < self.MIN_TRIALS_FOR_BASELINE:
            return BaselineComparisonResult(
                feature_name=feature_name,
                current_value=current_value,
                baseline_median=current_value,
                baseline_iqr=0.0,
                baseline_mad=0.0,
                robust_z_score=0.0,
                deviation_category=BaselineDeviationCategory.INSUFFICIENT_BASELINE,
                population_reference_median=population_reference_median,
                comparison_note=f"Insufficient baseline records ({len(valid_history)} trials). Established min: {self.MIN_TRIALS_FOR_BASELINE}.",
            )

        arr = np.array(valid_history, dtype=np.float64)
        median_val = float(np.median(arr))
        mad_val = float(np.median(np.abs(arr - median_val)))
        q75, q25 = np.percentile(arr, [75, 25])
        iqr_val = float(q75 - q25)

        # Robust Z-score calculation
        scale = 1.4826 * mad_val
        if scale > 1e-4:
            robust_z = float((current_value - median_val) / scale)
        else:
            diff = current_value - median_val
            robust_z = float(np.sign(diff) * (abs(diff) / max(1e-3, std_dev := np.std(arr))))

        abs_z = abs(robust_z)
        if abs_z <= 1.5:
            cat = BaselineDeviationCategory.WITHIN_PERSONAL_BASELINE
            note = "Value is within person's normal longitudinal baseline range."
        elif abs_z <= 2.5:
            cat = BaselineDeviationCategory.MILD_DEVIATION
            note = f"Mild deviation from personal baseline (Robust Z: {robust_z:+.2f})."
        elif abs_z <= 3.5:
            cat = BaselineDeviationCategory.PERSISTENT_DEVIATION
            note = f"Persistent deviation from personal baseline (Robust Z: {robust_z:+.2f})."
        else:
            cat = BaselineDeviationCategory.SUBSTANTIAL_DEVIATION
            note = f"Substantial deviation from personal baseline (Robust Z: {robust_z:+.2f})."

        return BaselineComparisonResult(
            feature_name=feature_name,
            current_value=current_value,
            baseline_median=median_val,
            baseline_iqr=iqr_val,
            baseline_mad=mad_val,
            robust_z_score=robust_z,
            deviation_category=cat,
            population_reference_median=population_reference_median,
            comparison_note=note,
        )
