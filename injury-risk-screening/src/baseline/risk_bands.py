"""
Risk banding and the persistence rule.

Locked design decisions (docs/PRODUCT_BUILD_SPEC.md):
  - Three bands: Low / Moderate / High, based on how many IQRs an event's
    feature value sits from the athlete's own baseline median (see
    src/baseline/baseline_engine.py for why IQR-relative, not absolute).
  - 1.5x IQR  -> Moderate  (the same multiplier Tukey's boxplot uses to
                 flag a mild outlier)
  - 3.0x IQR  -> High      (Tukey's "far outlier" threshold)
  - A SINGLE event crossing a threshold does not trigger a flag shown to
    a doctor. Pose estimation noise, a single bad step, or an unusual but
    harmless movement can all produce one outlier event. Instead we
    require persistence: 3 of the athlete's last 5 events must be in the
    same band (or higher) before the system raises that band to a
    clinician. This trades a little latency (it takes a few sessions to
    confirm a real drift) for a large reduction in false alarms, which
    matters enormously for a system whose credibility depends on doctors
    trusting its flags rather than dismissing it as "crying wolf".
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

MODERATE_IQR_MULTIPLIER = 1.5
HIGH_IQR_MULTIPLIER = 3.0

PERSISTENCE_WINDOW = 5  # look at the last N events
PERSISTENCE_REQUIRED = 3  # ...and require at least this many in-band


class RiskBand(IntEnum):
    LOW = 0
    MODERATE = 1
    HIGH = 2

    def __str__(self) -> str:
        return self.name.title()


def band_for_deviation(abs_deviation: float) -> RiskBand:
    """
    Map a single event's |deviation| (in IQR units, from
    baseline_engine.deviation_from_baseline) to a single-event risk band.

    This is the PRE-persistence, single-event band -- callers almost
    always want AthleteRiskTracker.update() instead, which applies the
    persistence rule before surfacing a band to a clinician. This
    function is exposed separately because the raw per-event band is
    still useful for debugging/plotting an athlete's full trend line.
    """
    if abs_deviation != abs_deviation:  # NaN: insufficient baseline data
        return RiskBand.LOW
    if abs_deviation >= HIGH_IQR_MULTIPLIER:
        return RiskBand.HIGH
    if abs_deviation >= MODERATE_IQR_MULTIPLIER:
        return RiskBand.MODERATE
    return RiskBand.LOW


@dataclass
class AthleteRiskTracker:
    """
    Stateful per-athlete, per-feature risk tracker implementing the
    persistence rule. One instance should be kept per (athlete, feature)
    pair -- e.g. one for "fppa_left", one for "fppa_right" -- since an
    athlete can legitimately drift on one knee and not the other.
    """

    athlete_id: str
    feature_name: str
    window: deque = field(default_factory=lambda: deque(maxlen=PERSISTENCE_WINDOW))
    confirmed_band: RiskBand = RiskBand.LOW

    def update(self, abs_deviation: float) -> RiskBand:
        """
        Record one new event's deviation and return the CONFIRMED band
        (post-persistence-rule) that should be shown to a clinician.

        Confirmation logic: for each band from HIGH down to MODERATE, if
        at least PERSISTENCE_REQUIRED of the last PERSISTENCE_WINDOW
        events are at-or-above that band, the confirmed band is raised to
        it. The confirmed band never drops below what persisted history
        supports, but it CAN fall back to LOW once enough recent events
        are back in range -- i.e. this also lets a flag clear itself
        once the drift resolves, rather than permanently labeling an
        athlete "high risk" after one bad patch.
        """
        single_event_band = band_for_deviation(abs_deviation)
        self.window.append(single_event_band)

        for band in (RiskBand.HIGH, RiskBand.MODERATE):
            count_at_or_above = sum(1 for b in self.window if b >= band)
            if count_at_or_above >= PERSISTENCE_REQUIRED:
                self.confirmed_band = band
                return self.confirmed_band

        self.confirmed_band = RiskBand.LOW
        return self.confirmed_band

    def events_recorded(self) -> int:
        return len(self.window)

    def has_enough_history(self) -> bool:
        """Whether enough events exist to trust the confirmed band at all."""
        return len(self.window) >= PERSISTENCE_REQUIRED
