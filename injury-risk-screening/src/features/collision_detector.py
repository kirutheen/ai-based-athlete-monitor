"""
Collision Detection Engine for DriftGuard.

Detects player-to-player collision events based on pairwise bounding box IoU overlap,
centroid distance reduction, and relative velocity spikes.
Associates the confirmed collision event with ALL involved players.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from src.features.player_state import PlayerState
from src.pose.multi_player_tracker import compute_iou


@dataclass
class CollisionEvent:
    event_id: str
    player_ids: List[int]
    player_names: List[str]
    timestamp_s: float
    frame_number: int
    severity: str  # "LOW", "MEDIUM", "HIGH"
    iou_overlap: float
    relative_velocity: float
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "player_ids": self.player_ids,
            "player_names": self.player_names,
            "timestamp_s": round(self.timestamp_s, 3),
            "frame_number": self.frame_number,
            "severity": self.severity,
            "iou_overlap": round(self.iou_overlap, 3),
            "relative_velocity": round(self.relative_velocity, 2),
            "summary": self.summary,
        }


class CollisionDetector:
    """
    Pairwise Player Collision Detection Engine.
    """

    def __init__(
        self,
        proximity_distance_threshold: float = 0.15,
        iou_collision_threshold: float = 0.10,
        min_cooldown_seconds: float = 1.0,
    ):
        self.proximity_distance_threshold = proximity_distance_threshold
        self.iou_collision_threshold = iou_collision_threshold
        self.min_cooldown_seconds = min_cooldown_seconds
        self.last_collision_time: Dict[Tuple[int, int], float] = {}

    def _get_centroid(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        xmin, ymin, xmax, ymax = bbox
        return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)

    def evaluate_frame_collisions(
        self,
        active_players: List[PlayerState],
        timestamp_s: float,
        frame_number: int,
    ) -> List[CollisionEvent]:
        """
        Evaluate pairwise collisions among active players in a frame.
        """
        detected_collisions = []
        n = len(active_players)

        if n < 2:
            return []

        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = active_players[i], active_players[j]

                if not p1.bbox_history or not p2.bbox_history:
                    continue

                b1, b2 = p1.bbox_history[-1], p2.bbox_history[-1]
                iou = compute_iou(b1, b2)

                c1 = self._get_centroid(b1)
                c2 = self._get_centroid(b2)
                dist = float(np.linalg.norm(np.array(c1) - np.array(c2)))

                # Check cooldown to prevent duplicate alerts on consecutive frames
                pair_key = (min(p1.tracker_id, p2.tracker_id), max(p1.tracker_id, p2.tracker_id))
                last_time = self.last_collision_time.get(pair_key, -999.0)
                if (timestamp_s - last_time) < self.min_cooldown_seconds:
                    continue

                # Collision condition: IoU overlap or high spatial proximity
                if iou >= self.iou_collision_threshold or dist <= self.proximity_distance_threshold:
                    # Calculate relative velocity proxy if history exists
                    rel_vel = 0.0
                    if len(p1.bbox_history) >= 2 and len(p2.bbox_history) >= 2:
                        prev_c1 = self._get_centroid(p1.bbox_history[-2])
                        prev_c2 = self._get_centroid(p2.bbox_history[-2])
                        v1 = np.array(c1) - np.array(prev_c1)
                        v2 = np.array(c2) - np.array(prev_c2)
                        rel_vel = float(np.linalg.norm(v1 - v2)) * 30.0  # normalize per second

                    severity = "HIGH" if iou >= 0.25 or rel_vel >= 0.50 else "MEDIUM"
                    summary = f"Player Collision detected between {p1.display_name} and {p2.display_name} (IoU: {iou:.2f})."

                    evt = CollisionEvent(
                        event_id=f"col_{uuid.uuid4().hex[:8]}",
                        player_ids=[p1.tracker_id, p2.tracker_id],
                        player_names=[p1.display_name, p2.display_name],
                        timestamp_s=timestamp_s,
                        frame_number=frame_number,
                        severity=severity,
                        iou_overlap=iou,
                        relative_velocity=rel_vel,
                        summary=summary,
                    )

                    # Attach event strictly to involved players
                    p1.add_collision_event(evt.to_dict())
                    p2.add_collision_event(evt.to_dict())

                    self.last_collision_time[pair_key] = timestamp_s
                    detected_collisions.append(evt)

        return detected_collisions
