"""
Validation metrics comparing an estimated 6-DOF knee trajectory against
a reference trajectory -- rotational RMSE, translational RMSE, MAE,
maximum error, and cycle-level consistency, exactly the metric set the
abstract specifies, evaluated against the target of <=5 degrees
rotational RMSE and <=5 mm translational RMSE.

This module is pure statistics over two aligned time series and has no
dependency on where the reference or estimate came from -- it works
identically whether fed OpenSim-simulated trajectories (as wired up in
run_validation.py) or, later, real optical-mocap-vs-DriftGuard data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.validation.noise_model import ROTATIONAL_COLUMNS, TRANSLATIONAL_COLUMNS

DEFAULT_ROTATIONAL_RMSE_TARGET_DEG = 5.0
DEFAULT_TRANSLATIONAL_RMSE_TARGET_MM = 5.0


@dataclass
class ChannelMetrics:
    channel: str
    rmse: float
    mae: float
    max_error: float


@dataclass
class ValidationReport:
    channel_metrics: list[ChannelMetrics]
    overall_rotational_rmse_deg: float
    overall_translational_rmse_mm: float
    cycle_consistency_std_deg: float  # std of per-cycle mean rotational error, if >1 cycle given
    passes_rotational_target: bool
    passes_translational_target: bool

    @property
    def passes_overall(self) -> bool:
        return self.passes_rotational_target and self.passes_translational_target


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _max_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def compute_channel_metrics(reference_df: pd.DataFrame, estimate_df: pd.DataFrame) -> list[ChannelMetrics]:
    results = []
    for col in (*ROTATIONAL_COLUMNS, *TRANSLATIONAL_COLUMNS):
        ref = reference_df[col].to_numpy()
        est = estimate_df[col].to_numpy()
        results.append(ChannelMetrics(channel=col, rmse=_rmse(ref, est), mae=_mae(ref, est), max_error=_max_error(ref, est)))
    return results


def compute_cycle_consistency(
    reference_df: pd.DataFrame, estimate_df: pd.DataFrame, n_cycles: int = 1
) -> float:
    """
    Std deviation, across repeated cycles, of each cycle's mean
    rotational error -- a measure of whether the estimate's error is
    consistent cycle-to-cycle (low std) or varies unpredictably (high
    std). Only meaningful when the input spans multiple concatenated
    cycles (n_cycles > 1); with a single cycle this returns 0.0 (nothing
    to compare across).
    """
    if n_cycles <= 1:
        return 0.0

    combined_rot_error = np.zeros(len(reference_df))
    for col in ROTATIONAL_COLUMNS:
        combined_rot_error += (reference_df[col].to_numpy() - estimate_df[col].to_numpy()) ** 2
    combined_rot_error = np.sqrt(combined_rot_error / len(ROTATIONAL_COLUMNS))

    cycle_len = len(combined_rot_error) // n_cycles
    per_cycle_means = [
        float(np.mean(combined_rot_error[i * cycle_len: (i + 1) * cycle_len])) for i in range(n_cycles)
    ]
    return float(np.std(per_cycle_means))


def validate_trajectory(
    reference_df: pd.DataFrame,
    estimate_df: pd.DataFrame,
    n_cycles: int = 1,
    rotational_rmse_target_deg: float = DEFAULT_ROTATIONAL_RMSE_TARGET_DEG,
    translational_rmse_target_mm: float = DEFAULT_TRANSLATIONAL_RMSE_TARGET_MM,
) -> ValidationReport:
    """
    Full validation report for an estimate trajectory against a
    reference, per the abstract's target: <=5 deg rotational RMS error,
    <=5 mm translational RMS error.

    "Overall" rotational/translational RMSE pools all 3 rotational (or
    translational) channels together (root-mean-square across all of
    them combined), matching how a single headline RMSE number is
    usually reported, while per-channel detail remains available in
    `channel_metrics` for diagnosing which specific axis drove any
    failure.
    """
    if len(reference_df) != len(estimate_df):
        raise ValueError("reference_df and estimate_df must have the same number of rows (aligned in time)")

    channel_metrics = compute_channel_metrics(reference_df, estimate_df)

    rot_errors = []
    for col in ROTATIONAL_COLUMNS:
        rot_errors.append(reference_df[col].to_numpy() - estimate_df[col].to_numpy())
    overall_rot_rmse = float(np.sqrt(np.mean(np.concatenate(rot_errors) ** 2)))

    trans_errors = []
    for col in TRANSLATIONAL_COLUMNS:
        trans_errors.append(reference_df[col].to_numpy() - estimate_df[col].to_numpy())
    overall_trans_rmse = float(np.sqrt(np.mean(np.concatenate(trans_errors) ** 2)))

    cycle_consistency = compute_cycle_consistency(reference_df, estimate_df, n_cycles=n_cycles)

    return ValidationReport(
        channel_metrics=channel_metrics,
        overall_rotational_rmse_deg=overall_rot_rmse,
        overall_translational_rmse_mm=overall_trans_rmse,
        cycle_consistency_std_deg=cycle_consistency,
        passes_rotational_target=overall_rot_rmse <= rotational_rmse_target_deg,
        passes_translational_target=overall_trans_rmse <= translational_rmse_target_mm,
    )


def format_report(report: ValidationReport) -> str:
    lines = ["Validation Report", "=" * 50]
    for cm in report.channel_metrics:
        unit = "deg" if "deg" in cm.channel else "mm"
        lines.append(f"  {cm.channel:35s} RMSE={cm.rmse:6.2f}{unit}  MAE={cm.mae:6.2f}{unit}  Max={cm.max_error:6.2f}{unit}")
    lines.append("-" * 50)
    lines.append(f"Overall rotational RMSE:    {report.overall_rotational_rmse_deg:.2f} deg "
                 f"(target <= 5.0 deg) -- {'PASS' if report.passes_rotational_target else 'FAIL'}")
    lines.append(f"Overall translational RMSE: {report.overall_translational_rmse_mm:.2f} mm "
                 f"(target <= 5.0 mm) -- {'PASS' if report.passes_translational_target else 'FAIL'}")
    lines.append(f"Cycle-to-cycle consistency (std): {report.cycle_consistency_std_deg:.2f} deg")
    lines.append(f"OVERALL: {'PASS' if report.passes_overall else 'FAIL'}")
    return "\n".join(lines)
