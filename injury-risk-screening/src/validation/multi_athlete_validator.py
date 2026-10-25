"""
Multi-Athlete Validation Protocol
===================================
Provides a structured, per-athlete, per-metric validation comparison against
independently collected reference measurements (e.g., marker-based mocap,
OpenSim, IMU telemetry, or clinician-labelled angles).

Design intent
-------------
- Wraps the existing ReferenceValidationEngine for multi-athlete and
  multi-metric use without modifying it.
- Accepts a batch request: list of (athlete_id, metric_name, estimated_values,
  reference_values) entries.
- Validates each pair independently; never mixes athletes' data.
- Returns a structured pass/fail report with:
    * Per-metric MAE, RMSE, correlation, bias, threshold pass/fail
    * Per-athlete roll-up: how many metrics met the target threshold
    * Session-level roll-up: overall pass/fail
    * Provenance block (engine version, thresholds used, timestamp)
    * Mandatory disclaimer (not clinical validation / regulatory clearance)

Invariants
----------
- Never modifies existing ValidationEngine, DB schema, or single-athlete paths.
- Never imputes NaN values; series with <3 valid pairs return NOT_VALIDATED.
- Never claims regulatory clearance or clinical validity.
- Safe to import without MediaPipe or OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.validation.validation_engine import (
    ReferenceValidationEngine,
    MetricValidationSummary,
)

# Validation disclaimer
VALIDATION_DISCLAIMER = (
    "RESEARCH / INTERNAL VALIDATION ONLY. "
    "This output compares DriftGuard pose-derived estimates against independently "
    "collected reference measurements using standard error metrics. "
    "Agreement with a reference system does NOT constitute regulatory clearance, "
    "clinical validation, or proof of diagnostic accuracy. "
    "Results must be interpreted by qualified engineers or researchers and must "
    "not be used to make clinical decisions."
)

_ENGINE_VERSION = "1.0"
_ANGLE_THRESHOLD_DEG = ReferenceValidationEngine.TARGET_ANGLE_RMSE_DEG  # 5.0°
_TRANS_THRESHOLD_MM  = ReferenceValidationEngine.TARGET_TRANS_RMSE_MM   # 5.0 mm


# ---------------------------------------------------------------------------
# Input / output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MetricValidationRequest:
    """One estimated-vs-reference comparison for a single athlete + metric."""
    athlete_id: str          # track_id or any string label
    metric_name: str
    estimated_values: List[float]
    reference_values: List[float]
    unit: str = "deg"


@dataclass
class AthleteValidationRollup:
    """Per-athlete summary across all metrics validated for that athlete."""
    athlete_id: str
    metrics_validated: int
    metrics_passed: int
    metrics_failed: int
    metrics_insufficient: int   # < 3 valid pairs
    all_passed: bool
    metric_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "athlete_id":            self.athlete_id,
            "metrics_validated":     self.metrics_validated,
            "metrics_passed":        self.metrics_passed,
            "metrics_failed":        self.metrics_failed,
            "metrics_insufficient":  self.metrics_insufficient,
            "all_passed":            self.all_passed,
            "metric_results":        self.metric_results,
        }


@dataclass
class ValidationSessionReport:
    """Full validation report across all athletes and metrics."""
    total_requests: int
    athletes_validated: int
    athletes_all_passed: int
    session_passed: bool
    athlete_rollups: List[AthleteValidationRollup]
    provenance: Dict[str, Any]
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disclaimer":           self.disclaimer,
            "provenance":           self.provenance,
            "session_passed":       self.session_passed,
            "total_requests":       self.total_requests,
            "athletes_validated":   self.athletes_validated,
            "athletes_all_passed":  self.athletes_all_passed,
            "athletes":             [r.to_dict() for r in self.athlete_rollups],
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class MultiAthleteValidator:
    """
    Validate multiple (athlete, metric) pairs in one call.

    Usage
    -----
    validator = MultiAthleteValidator()
    report = validator.run(requests)
    """

    def __init__(self) -> None:
        self._engine = ReferenceValidationEngine()

    def _validate_one(
        self, req: MetricValidationRequest
    ) -> Tuple[MetricValidationSummary, bool]:
        """
        Run one metric comparison.  Returns (summary, is_sufficient).
        is_sufficient = False when sample_count < 3 (engine returns NOT_VALIDATED).
        """
        summary = self._engine.validate_metric_series(
            metric_name=req.metric_name,
            estimated=req.estimated_values,
            reference=req.reference_values,
            unit=req.unit,
        )
        is_sufficient = summary.validation_status != "NOT_VALIDATED"
        return summary, is_sufficient

    def run(
        self,
        requests: List[MetricValidationRequest],
    ) -> ValidationSessionReport:
        """
        Validate all requests and return a structured report.

        Parameters
        ----------
        requests : list[MetricValidationRequest]
            One entry per (athlete, metric) pair to compare.
            Athletes are identified by athlete_id string (not required to be
            a DB primary key — can be track_id cast to str or a label).

        Returns
        -------
        ValidationSessionReport
        """
        # Group by athlete_id
        by_athlete: Dict[str, List[MetricValidationRequest]] = {}
        for req in requests:
            by_athlete.setdefault(req.athlete_id, []).append(req)

        athlete_rollups: List[AthleteValidationRollup] = []

        for athlete_id, athlete_reqs in sorted(by_athlete.items()):
            metric_results: List[Dict[str, Any]] = []
            passed = failed = insufficient = 0

            for req in athlete_reqs:
                try:
                    summary, is_sufficient = self._validate_one(req)
                except ValueError as exc:
                    # Engine raised — e.g., mismatched lengths
                    metric_results.append({
                        "metric_name":       req.metric_name,
                        "validation_status": "ERROR",
                        "error":             str(exc),
                    })
                    failed += 1
                    continue

                result_dict = summary.to_dict()
                if not is_sufficient:
                    insufficient += 1
                elif summary.target_threshold_met:
                    passed += 1
                else:
                    failed += 1

                metric_results.append(result_dict)

            total_metrics = len(athlete_reqs)
            athlete_rollups.append(AthleteValidationRollup(
                athlete_id=athlete_id,
                metrics_validated=total_metrics,
                metrics_passed=passed,
                metrics_failed=failed,
                metrics_insufficient=insufficient,
                all_passed=(failed == 0 and insufficient == 0 and passed == total_metrics),
                metric_results=metric_results,
            ))

        athletes_all_passed = sum(1 for r in athlete_rollups if r.all_passed)
        session_passed = (
            len(athlete_rollups) > 0
            and athletes_all_passed == len(athlete_rollups)
        )

        provenance = {
            "engine_version":         _ENGINE_VERSION,
            "angle_rmse_threshold_deg": _ANGLE_THRESHOLD_DEG,
            "trans_rmse_threshold_mm":  _TRANS_THRESHOLD_MM,
            "generated_at_utc":        datetime.now(timezone.utc).isoformat(),
            "total_requests":          len(requests),
        }

        return ValidationSessionReport(
            total_requests=len(requests),
            athletes_validated=len(athlete_rollups),
            athletes_all_passed=athletes_all_passed,
            session_passed=session_passed,
            athlete_rollups=athlete_rollups,
            provenance=provenance,
            disclaimer=VALIDATION_DISCLAIMER,
        )
