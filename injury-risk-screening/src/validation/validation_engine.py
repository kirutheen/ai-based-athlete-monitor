"""
Reference Validation Engine for DriftGuard.

Compares pose-derived estimations against independent reference ground-truth
measurements (e.g., OpenSim, marker-based mocap, or IMU telemetry).
Calculates MAE, RMSE, maximum error, mean bias, and Pearson correlation coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MetricValidationSummary:
    metric_name: str
    sample_count: int
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Squared Error
    max_error: float
    mean_bias: float
    correlation_r: float
    validation_status: str
    unit: str
    target_threshold_met: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "sample_count": self.sample_count,
            "mae": round(self.mae, 3),
            "rmse": round(self.rmse, 3),
            "max_error": round(self.max_error, 3),
            "mean_bias": round(self.mean_bias, 3),
            "correlation_r": round(self.correlation_r, 3),
            "validation_status": self.validation_status,
            "unit": self.unit,
            "target_threshold_met": self.target_threshold_met,
        }


class ReferenceValidationEngine:
    """
    Evaluates agreement between DriftGuard camera estimation vectors and reference ground truth.
    """

    # Target clinical screening threshold targets
    TARGET_ANGLE_RMSE_DEG = 5.0  # Max acceptable RMSE for angles (degrees)
    TARGET_TRANS_RMSE_MM = 5.0   # Max acceptable RMSE for translations (mm)

    def validate_metric_series(
        self,
        metric_name: str,
        estimated: List[float],
        reference: List[float],
        unit: str = "deg",
    ) -> MetricValidationSummary:
        """
        Compare estimated vs reference floating point time-series arrays.
        """
        if not estimated or not reference or len(estimated) != len(reference):
            raise ValueError("Estimated and reference series must be non-empty and equal length.")

        est_arr = np.array(estimated, dtype=np.float64)
        ref_arr = np.array(reference, dtype=np.float64)

        # Mask out NaNs
        valid_mask = ~np.isnan(est_arr) & ~np.isnan(ref_arr)
        if np.sum(valid_mask) < 3:
            return MetricValidationSummary(
                metric_name=metric_name,
                sample_count=int(np.sum(valid_mask)),
                mae=float("nan"),
                rmse=float("nan"),
                max_error=float("nan"),
                mean_bias=float("nan"),
                correlation_r=0.0,
                validation_status="NOT_VALIDATED",
                unit=unit,
                target_threshold_met=False,
            )

        e = est_arr[valid_mask]
        r = ref_arr[valid_mask]
        n = len(e)

        diff = e - r
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        max_error = float(np.max(np.abs(diff)))
        mean_bias = float(np.mean(diff))

        if np.std(e) > 1e-6 and np.std(r) > 1e-6:
            r_matrix = np.corrcoef(e, r)
            correlation_r = float(r_matrix[0, 1])
        else:
            correlation_r = 1.0 if np.allclose(e, r) else 0.0

        target_rmse = self.TARGET_TRANS_RMSE_MM if unit == "mm" else self.TARGET_ANGLE_RMSE_DEG
        threshold_met = (rmse <= target_rmse and correlation_r >= 0.85)

        if threshold_met:
            status = "VALIDATED"
        elif rmse <= target_rmse * 1.5 or correlation_r >= 0.70:
            status = "PARTIALLY_VALIDATED"
        else:
            status = "NOT_VALIDATED"

        return MetricValidationSummary(
            metric_name=metric_name,
            sample_count=n,
            mae=mae,
            rmse=rmse,
            max_error=max_error,
            mean_bias=mean_bias,
            correlation_r=correlation_r,
            validation_status=status,
            unit=unit,
            target_threshold_met=threshold_met,
        )
