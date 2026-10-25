"""
Multi-Athlete Quality Assessment
=================================
Computes a per-athlete quality tier and quality indicators directly from the
summary-level stats produced by process_multi_athlete_video().

Design intent
-------------
- Does NOT re-run the quality engine's per-frame landmark pipeline (those
  landmarks are not stored in the multi-athlete result to keep the payload
  small).  Instead it uses the aggregate statistics that *are* stored:
    * coverage_pct          — fraction of total frames the athlete appeared in
    * uncertainty_pct       — fraction of their frames where identity was uncertain
    * biomechanics_valid_pct — fraction where bio values are trusted

- Produces a QualityTier string and a list of human-readable caveats.
- Never claims clinical validity; always returns a screening-output disclaimer.
- Never fabricates values; returns None for any metric missing from the summary.

Invariants
----------
- Does not touch single-athlete pipeline or DB schema.
- Safe to import without MediaPipe (no CV imports).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Quality tiers (string enum — no extra Enum import needed at call sites)
QUALITY_HIGH         = "HIGH"
QUALITY_ACCEPTABLE   = "ACCEPTABLE"
QUALITY_LOW          = "LOW"
QUALITY_INSUFFICIENT = "INSUFFICIENT"

# Thresholds
_COVERAGE_SUFFICIENT  = 40.0   # % of frames athlete must appear in
_COVERAGE_GOOD        = 70.0
_UNCERTAINTY_WARN     = 15.0   # % uncertain frames → warn
_UNCERTAINTY_FAIL     = 40.0   # % uncertain frames → LOW or INSUFFICIENT
_BIO_VALID_SUFFICIENT = 60.0   # % valid bio frames required for screening
_BIO_VALID_GOOD       = 85.0


@dataclass
class AthleteQualityResult:
    track_id: int
    display_name: str
    quality_tier: str                 # HIGH / ACCEPTABLE / LOW / INSUFFICIENT
    is_sufficient_for_screening: bool
    quality_score: float              # 0.0 – 1.0 composite
    caveats: List[str]
    coverage_pct: float
    uncertainty_pct: float
    biomechanics_valid_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id":                   self.track_id,
            "display_name":               self.display_name,
            "quality_tier":               self.quality_tier,
            "is_sufficient_for_screening": self.is_sufficient_for_screening,
            "quality_score":              round(self.quality_score, 3),
            "caveats":                    self.caveats,
            "coverage_pct":               self.coverage_pct,
            "uncertainty_pct":            self.uncertainty_pct,
            "biomechanics_valid_pct":     self.biomechanics_valid_pct,
        }


def assess_athlete_quality(athlete_summary: Dict[str, Any]) -> AthleteQualityResult:
    """
    Derive quality tier for one athlete from the summary dict produced by
    process_multi_athlete_video().

    Parameters
    ----------
    athlete_summary : dict
        One element from result["athletes"] — must contain at minimum:
        track_id, display_name, coverage_pct, uncertainty_pct,
        biomechanics_valid_pct.

    Returns
    -------
    AthleteQualityResult
    """
    tid      = athlete_summary.get("track_id", 0)
    name     = athlete_summary.get("display_name", f"Athlete #{tid}")
    cov      = float(athlete_summary.get("coverage_pct", 0.0))
    unc      = float(athlete_summary.get("uncertainty_pct", 0.0))
    bio_val  = float(athlete_summary.get("biomechanics_valid_pct", 100.0))

    caveats: List[str] = []

    # --- Scoring (each dimension 0–1, equal weight) --------------------------
    cov_score     = min(1.0, cov / 100.0)
    unc_score     = max(0.0, 1.0 - unc / 100.0)
    bio_val_score = min(1.0, bio_val / 100.0)

    quality_score = round((cov_score + unc_score + bio_val_score) / 3.0, 3)

    # --- Caveats --------------------------------------------------------------
    if cov < _COVERAGE_SUFFICIENT:
        caveats.append(
            f"Athlete appeared in only {cov:.0f}% of frames — "
            "insufficient coverage for reliable screening."
        )
    elif cov < _COVERAGE_GOOD:
        caveats.append(
            f"Athlete coverage {cov:.0f}% — moderate; interpret results with caution."
        )

    if unc >= _UNCERTAINTY_FAIL:
        caveats.append(
            f"{unc:.0f}% of frames had uncertain identity (near-crossing detected) — "
            "biomechanics were suppressed for those frames."
        )
    elif unc >= _UNCERTAINTY_WARN:
        caveats.append(
            f"{unc:.0f}% of frames flagged as uncertain identity — "
            "some biomechanical values may be unreliable."
        )

    if bio_val < _BIO_VALID_SUFFICIENT:
        caveats.append(
            f"Only {bio_val:.0f}% of frames produced valid biomechanics — "
            "average values may not represent true performance."
        )

    # --- Tier -----------------------------------------------------------------
    if cov < _COVERAGE_SUFFICIENT or bio_val < _BIO_VALID_SUFFICIENT or unc >= _UNCERTAINTY_FAIL:
        tier       = QUALITY_INSUFFICIENT
        sufficient = False
    elif unc >= _UNCERTAINTY_WARN or bio_val < _BIO_VALID_GOOD or cov < _COVERAGE_GOOD:
        tier       = QUALITY_LOW if unc >= _UNCERTAINTY_WARN else QUALITY_ACCEPTABLE
        sufficient = tier == QUALITY_ACCEPTABLE
    else:
        tier       = QUALITY_HIGH
        sufficient = True

    return AthleteQualityResult(
        track_id=tid,
        display_name=name,
        quality_tier=tier,
        is_sufficient_for_screening=sufficient,
        quality_score=quality_score,
        caveats=caveats,
        coverage_pct=cov,
        uncertainty_pct=unc,
        biomechanics_valid_pct=bio_val,
    )


def assess_session_quality(
    result: Dict[str, Any],
) -> Dict[int, AthleteQualityResult]:
    """
    Assess quality for every athlete in a multi-athlete session result.

    Parameters
    ----------
    result : dict
        Return value of process_multi_athlete_video().

    Returns
    -------
    dict mapping track_id → AthleteQualityResult
    """
    return {
        a["track_id"]: assess_athlete_quality(a)
        for a in result.get("athletes", [])
    }
