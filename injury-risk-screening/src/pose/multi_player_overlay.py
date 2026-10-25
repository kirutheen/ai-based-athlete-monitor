"""
Multi-Player Video Overlay Renderer for DriftGuard.

Draws color-coded bounding boxes, player identity badges, pose skeletons, risk tags,
and collision alert banners onto video frames.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from src.features.player_state import PlayerState


class MultiPlayerOverlayRenderer:
    """
    Renders multi-player overlay graphics onto BGR video frames.
    """

    COLOR_LOW = (34, 197, 94)      # Green (BGR)
    COLOR_MEDIUM = (234, 179, 8)   # Amber (BGR)
    COLOR_HIGH = (239, 68, 68)     # Red (BGR)
    COLOR_COLLISION = (168, 85, 247) # Purple (BGR)

    def _get_risk_color(self, band: str) -> tuple[int, int, int]:
        if band == "HIGH":
            return self.COLOR_HIGH
        elif band == "MEDIUM":
            return self.COLOR_MEDIUM
        return self.COLOR_LOW

    def render_frame_overlay(
        self,
        frame_bgr: np.ndarray,
        active_players: List[PlayerState],
    ) -> np.ndarray:
        """Draw player bounding boxes, risk badges, pose skeletons, and alert tags."""
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]

        for player in active_players:
            if not player.bbox_history:
                continue

            bbox = player.bbox_history[-1]
            xmin, ymin, xmax, ymax = bbox

            px_xmin = int(xmin * w)
            px_ymin = int(ymin * h)
            px_xmax = int(xmax * w)
            px_ymax = int(ymax * h)

            risk = player.get_latest_risk_score()
            band = risk.get("risk_band", "LOW")
            color = self._get_risk_color(band)

            # Draw bounding box
            cv2.rectangle(annotated, (px_xmin, px_ymin), (px_xmax, px_ymax), color, 2)

            # Draw header badge (Player Name + Risk Band)
            badge_text = f"{player.display_name} [{band}]"
            (text_w, text_h), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(
                annotated,
                (px_xmin, max(0, px_ymin - text_h - 8)),
                (px_xmin + text_w + 10, px_ymin),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                badge_text,
                (px_xmin + 5, max(12, px_ymin - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Draw Collision Alert Banner if collision active
            if player.collision_events and (len(player.bbox_history) - player.collision_events[-1].get("frame_number", 0)) < 30:
                cv2.putText(
                    annotated,
                    "COLLISION!",
                    (px_xmin + 5, px_ymin + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.COLOR_COLLISION,
                    2,
                    cv2.LINE_AA,
                )

            # Draw simplified pose skeleton if landmarks present
            if player.pose_history and player.pose_history[-1] is not None:
                lm = player.pose_history[-1]
                # Lower body indices: Hips (23,24), Knees (25,26), Ankles (27,28)
                pairs = [(23, 25), (25, 27), (24, 26), (26, 28), (23, 24)]
                for p1_idx, p2_idx in pairs:
                    if p1_idx < len(lm) and p2_idx < len(lm):
                        if lm[p1_idx][3] >= 0.5 and lm[p2_idx][3] >= 0.5:
                            pt1 = (int(lm[p1_idx][0] * w), int(lm[p1_idx][1] * h))
                            pt2 = (int(lm[p2_idx][0] * w), int(lm[p2_idx][1] * h))
                            cv2.line(annotated, pt1, pt2, (255, 255, 0), 2)
                            cv2.circle(annotated, pt1, 4, (0, 0, 255), -1)
                            cv2.circle(annotated, pt2, 4, (0, 0, 255), -1)

        return annotated
