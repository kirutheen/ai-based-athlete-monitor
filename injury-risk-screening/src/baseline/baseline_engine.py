"""
Personal baseline engine.

Core design decision: this system does NOT compare an athlete's mechanics to a
population norm. Instead, each athlete gets their OWN baseline (median, IQR, MAD)
over warm-up events, and later events are scored as deviations from that athlete's normal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WARMUP_EVENTS = 10  # events needed before a baseline is considered established
FEATURE_COLUMNS = (
    "fppa_left",
    "fppa_right",
    "knee_flexion_left",
    "knee_flexion_right",
    "hip_flexion_left",
    "hip_flexion_right",
    "ankle_dorsiflexion_left",
    "ankle_dorsiflexion_right",
    "trunk_lean",
)


@dataclass
class AthleteBaseline:
    athlete_id: str
    n_events: int
    established: bool  # False until WARMUP_EVENTS have been collected
    medians: dict[str, float] = field(default_factory=dict)
    iqrs: dict[str, float] = field(default_factory=dict)  # Q3 - Q1, per feature
    mads: dict[str, float] = field(default_factory=dict)  # Median Absolute Deviation

    def to_dict(self) -> dict:
        return {
            "athlete_id": self.athlete_id,
            "n_events": self.n_events,
            "established": self.established,
            **{f"{k}_median": v for k, v in self.medians.items()},
            **{f"{k}_iqr": v for k, v in self.iqrs.items()},
            **{f"{k}_mad": v for k, v in self.mads.items()},
        }


def compute_baseline(
    athlete_events: pd.DataFrame,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    warmup_events: int = WARMUP_EVENTS,
) -> AthleteBaseline:
    """Compute one athlete's personal baseline (median, IQR, MAD) from event-level feature history."""
    if athlete_events.empty:
        raise ValueError("athlete_events is empty")

    athlete_id = str(athlete_events["athlete_id"].iloc[0])
    n_events = len(athlete_events)
    warmup_slice = athlete_events.iloc[: min(n_events, warmup_events)]

    medians: dict[str, float] = {}
    iqrs: dict[str, float] = {}
    mads: dict[str, float] = {}

    for col in feature_columns:
        if col not in warmup_slice.columns:
            continue
        values = warmup_slice[col].dropna().to_numpy()
        if len(values) == 0:
            medians[col] = float("nan")
            iqrs[col] = float("nan")
            mads[col] = float("nan")
            continue

        med = float(np.median(values))
        medians[col] = med
        q1, q3 = np.percentile(values, [25, 75])
        iqrs[col] = float(q3 - q1)
        mads[col] = float(np.median(np.abs(values - med)))

    return AthleteBaseline(
        athlete_id=athlete_id,
        n_events=n_events,
        established=n_events >= warmup_events,
        medians=medians,
        iqrs=iqrs,
        mads=mads,
    )


def deviation_from_baseline(value: float, median: float, iqr: float) -> float:
    """
    Robust IQR-relative deviation: how many IQRs away from median this value is.
    Returns NaN if baseline IQR is 0 or NaN.
    """
    if iqr is None or iqr != iqr or iqr == 0 or median != median or value != value:
        return float("nan")
    return (value - median) / iqr


def robust_z_score(value: float, median: float, mad: float, iqr: float | None = None) -> float:
    """
    Robust z-score: (value - median) / MAD.
    Falls back to IQR / 1.3489 if MAD is 0 or unestimable.
    Returns NaN if median or value is NaN, or if scale is 0.
    """
    if math.isnan(value) or math.isnan(median):
        return float("nan")

    scale = mad
    if scale is None or math.isnan(scale) or scale == 0:
        if iqr is not None and not math.isnan(iqr) and iqr > 0:
            scale = iqr / 1.3489
        else:
            return float("nan")

    return float((value - median) / scale)
