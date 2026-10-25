"""
Per-Player State Management & Data Isolation Engine for DriftGuard.

Guarantees data isolation: Each tracked athlete maintains an independent
PlayerState instance. Player A's movements, baseline, and risk flags
NEVER mutate Player B's state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PlayerState:
    """
    Independent state container for a single tracked player.
    """
    tracker_id: int
    jersey_number: Optional[int] = None
    display_name: str = ""
    first_seen: float = 0.0
    last_seen: float = 0.0
    active: bool = True
    missing_frame_count: int = 0

    bbox_history: List[Tuple[float, float, float, float]] = field(default_factory=list)  # (xmin, ymin, xmax, ymax)
    pose_history: List[Optional[np.ndarray]] = field(default_factory=list)  # (33, 4) landmark arrays
    biomechanics_history: List[Dict[str, float]] = field(default_factory=list)
    baseline_history: List[Dict[str, float]] = field(default_factory=list)
    risk_history: List[Dict[str, Any]] = field(default_factory=list)
    abnormal_events: List[Dict[str, Any]] = field(default_factory=list)
    collision_events: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.display_name:
            if self.jersey_number is not None:
                self.display_name = f"Player #{self.jersey_number}"
            else:
                self.display_name = f"Player 0{self.tracker_id}"

    def update_identity(self, jersey_number: Optional[int]) -> None:
        """Update jersey number via confirmed OCR without breaking tracker ID continuity."""
        if jersey_number is not None:
            self.jersey_number = jersey_number
            self.display_name = f"Player #{jersey_number}"

    def add_frame_data(
        self,
        timestamp_s: float,
        bbox: Tuple[float, float, float, float],
        landmarks: Optional[np.ndarray],
        biomechanics: Dict[str, float],
    ) -> None:
        """Add frame observation to player's independent history."""
        self.last_seen = timestamp_s
        self.missing_frame_count = 0
        self.active = True

        self.bbox_history.append(bbox)
        self.pose_history.append(landmarks)
        self.biomechanics_history.append(biomechanics)

    def mark_missing(self, max_timeout_frames: int = 30) -> None:
        """Increment missing frame count; deactivate if missing beyond timeout."""
        self.missing_frame_count += 1
        if self.missing_frame_count >= max_timeout_frames:
            self.active = False

    def add_abnormal_event(self, event: Dict[str, Any]) -> None:
        """Add an abnormal movement event strictly for this player."""
        event["tracker_id"] = self.tracker_id
        event["display_name"] = self.display_name
        self.abnormal_events.append(event)

    def add_collision_event(self, event: Dict[str, Any]) -> None:
        """Add a collision event involving this player."""
        if not any(e.get("event_id") == event.get("event_id") for e in self.collision_events):
            self.collision_events.append(event)

    def get_latest_risk_score(self) -> Dict[str, Any]:
        """Return player's latest independent risk assessment."""
        if self.risk_history:
            return self.risk_history[-1]

        # Calculate default baseline risk if no events recorded
        if not self.biomechanics_history:
            return {
                "score": 90,
                "risk_band": "LOW",
                "reasons": ["Normal Movement Pattern"],
            }

        latest_bio = self.biomechanics_history[-1]
        fppa_raw = latest_bio.get("fppa_left")
        fppa = abs(fppa_raw) if fppa_raw is not None else 0.0
        flex_raw = latest_bio.get("knee_flexion_left")
        flex = flex_raw if flex_raw is not None else 0.0
        lsi_raw = latest_bio.get("valgus_asymmetry_lsi_pct")
        lsi = lsi_raw if lsi_raw is not None else 0.0

        if fppa > 18.0 or lsi > 35.0:
            band = "HIGH"
            score = 45
            reasons = ["Elevated Dynamic Knee Valgus Proxy", "High Bilateral Asymmetry"]
        elif fppa > 12.0 or lsi > 20.0:
            band = "MEDIUM"
            score = 68
            reasons = ["Moderate Dynamic Valgus Deviation"]
        else:
            band = "LOW"
            score = 88
            reasons = ["Normal Movement Mechanics"]

        risk_dict = {
            "score": score,
            "risk_band": band,
            "reasons": reasons,
            "tracker_id": self.tracker_id,
            "display_name": self.display_name,
        }
        self.risk_history.append(risk_dict)
        return risk_dict

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return player summary object for API and dashboard."""
        risk = self.get_latest_risk_score()
        has_collision = len(self.collision_events) > 0
        latest_finding = risk["reasons"][0] if risk["reasons"] else "Normal"

        return {
            "tracker_id": self.tracker_id,
            "jersey_number": self.jersey_number,
            "display_name": self.display_name,
            "active": self.active,
            "first_seen": round(self.first_seen, 2),
            "last_seen": round(self.last_seen, 2),
            "tracked_frames": len(self.pose_history),
            "score": risk["score"],
            "risk_band": risk["risk_band"],
            "main_finding": latest_finding,
            "has_collision": has_collision,
            "abnormal_events_count": len(self.abnormal_events),
            "collision_events_count": len(self.collision_events),
        }
