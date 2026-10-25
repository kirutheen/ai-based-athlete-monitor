"""
Basic multi-person tracker: assigns a stable-ish ID to each detected pose
across frames, so a multi-athlete session can be split back into
per-person tracks.

============================= HONEST SCOPE =================================
This is a simple, real, tested centroid-distance tracker -- NOT a
research-grade re-identification system. It assigns IDs purely by "which
detected pose in this frame is closest to which tracked person in the
last frame", with no appearance/clothing/face matching at all.

What this means in practice, consistent with the occlusion/crowding
limitation discussed for this project from the start:
  - Works well for a few athletes performing SEPARATE, well-spaced drills
    (e.g. two athletes taking turns, or two stations several meters
    apart) where paths don't cross and no one is fully occluded.
  - WILL mis-assign IDs ("ID switch") when two tracked people cross
    paths, when one is fully occluded by another and reappears, or when
    detections flicker (MediaPipe loses and re-finds a person a few
    frames apart). This is the exact, documented failure mode discussed
    earlier in this project for crowded/team scenes -- this tracker does
    not solve it, it only extends simple single-person tracking to "a
    few people in a mostly-uncrossed scene," nothing more.
  - For anything beyond that (a real team scrimmage, overlapping
    players), this is not the right tool -- it would silently produce
    track IDs that doctors/coaches could easily mistake for being
    reliable per-athlete identity, which is worse than no tracking at
    all. Use it only for the structured, limited-occlusion setups this
    project has always been scoped to.

Algorithm: greedy nearest-centroid matching with a maximum-distance gate
and a short grace period (a track survives a few missed frames before
being dropped, so a 1-frame detection flicker doesn't spawn a new ID).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Valid track status values (string enum — avoids an Enum import for simplicity).
# "active"           — confidently tracked this frame.
# "temporarily_lost" — not seen for 1..max_missed_frames frames; retained for re-ID.
# "uncertain"        — seen but identity confidence is low (near-crossing detected).
# "ended"            — exceeded max_missed_frames; track pruned from registry.
TRACK_STATUS_ACTIVE           = "active"
TRACK_STATUS_TEMPORARILY_LOST = "temporarily_lost"
TRACK_STATUS_UNCERTAIN        = "uncertain"
TRACK_STATUS_ENDED            = "ended"


@dataclass
class Track:
    track_id: int
    last_centroid: np.ndarray       # (x, y) normalized image coords
    last_bbox: tuple                 # (xmin, ymin, xmax, ymax) normalized, or None
    frames_since_seen: int = 0
    total_frames_seen: int = 1
    status: str = TRACK_STATUS_ACTIVE
    uncertainty_reason: str = ""    # human-readable; empty when not uncertain


@dataclass
class TrackedPose:
    track_id: int
    landmarks_2d: np.ndarray
    confidence: float
    track_status: str = TRACK_STATUS_ACTIVE
    uncertainty_reason: str = ""
    frame_index: int = 0
    session_id: str = ""


def _pose_centroid(landmarks_2d: np.ndarray) -> np.ndarray:
    """Mean (x, y) of all landmarks with visibility above 0.3, used as a
    cheap stand-in for a person's on-screen position."""
    visible = landmarks_2d[landmarks_2d[:, 3] > 0.3]
    if len(visible) == 0:
        visible = landmarks_2d
    return visible[:, :2].mean(axis=0)


def _landmarks_bbox(landmarks_2d: np.ndarray) -> tuple:
    """Bounding box of visible landmarks as (xmin, ymin, xmax, ymax) normalized."""
    visible = landmarks_2d[landmarks_2d[:, 3] > 0.3]
    if len(visible) == 0:
        visible = landmarks_2d
    xs, ys = visible[:, 0], visible[:, 1]
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


class MultiPersonTracker:
    """
    Feed one frame's list of detected poses (each a (33,4) landmark array,
    e.g. from PoseEstimator configured with num_poses > 1) per call to
    `update()`. Returns the same poses, each now tagged with a stable
    track_id and an explicit track_status.

    State machine:
        active           → updated this frame (frames_since_seen == 0)
        temporarily_lost → unmatched for 1..max_missed_frames frames
        uncertain        → matched but suspected crossing / low-confidence swap
        ended            → pruned after exceeding max_missed_frames (not returned)

    Uncertainty heuristic: when two *different* active tracks have centroids
    within `crossing_distance` of *each other's* assignments in consecutive
    frames, both are flagged uncertain. This is a cheap, conservative heuristic
    that avoids silently swapping identities.
    """

    def __init__(
        self,
        max_match_distance: float = 0.15,
        max_missed_frames: int = 10,
        min_detection_confidence: float = 0.30,
        crossing_distance: float = 0.08,
    ):
        self.max_match_distance = max_match_distance
        self.max_missed_frames = max_missed_frames
        self.min_detection_confidence = min_detection_confidence
        self.crossing_distance = crossing_distance
        self._tracks: dict[int, Track] = {}
        self._next_id = 0

    def update(
        self,
        detections: list[tuple[np.ndarray, float]],
        frame_index: int = 0,
        session_id: str = "",
    ) -> list[TrackedPose]:
        """
        detections: list of (landmarks_2d, confidence) for this frame.
        Detections below min_detection_confidence are silently dropped (never
        fabricated as 'low-confidence' tracks).
        Returns one TrackedPose per accepted detection, each with a stable
        track_id and an explicit track_status.
        """
        # --- Confidence gate -------------------------------------------------
        detections = [
            (lm, conf) for lm, conf in detections
            if conf >= self.min_detection_confidence
        ]

        centroids = [_pose_centroid(lm) for lm, _ in detections]
        bboxes    = [_landmarks_bbox(lm) for lm, _ in detections]

        unmatched_dets   = set(range(len(detections)))
        unmatched_tracks = set(self._tracks.keys())
        assignments: dict[int, int] = {}  # det_idx → track_id

        # --- Greedy nearest-centroid matching --------------------------------
        pairs = []
        for det_idx in unmatched_dets:
            for track_id in unmatched_tracks:
                dist = float(np.linalg.norm(
                    centroids[det_idx] - self._tracks[track_id].last_centroid
                ))
                if dist <= self.max_match_distance:
                    pairs.append((dist, det_idx, track_id))
        pairs.sort(key=lambda p: p[0])

        used_dets, used_tracks = set(), set()
        for dist, det_idx, track_id in pairs:
            if det_idx in used_dets or track_id in used_tracks:
                continue
            assignments[det_idx] = track_id
            used_dets.add(det_idx)
            used_tracks.add(track_id)

        unmatched_dets   -= used_dets
        unmatched_tracks -= used_tracks

        # --- New tracks for unmatched detections -----------------------------
        for det_idx in unmatched_dets:
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = Track(
                track_id=track_id,
                last_centroid=centroids[det_idx],
                last_bbox=bboxes[det_idx],
            )
            assignments[det_idx] = track_id

        # --- Crossing / near-swap uncertainty heuristic ----------------------
        # If two matched tracks have centroids that are very close to each
        # other *after* assignment (possible swap), flag both as uncertain.
        uncertain_track_ids: set[int] = set()
        matched_track_ids = list(assignments.values())
        for i in range(len(matched_track_ids)):
            for j in range(i + 1, len(matched_track_ids)):
                tid_a = matched_track_ids[i]
                tid_b = matched_track_ids[j]
                c_a = self._tracks[tid_a].last_centroid
                c_b = self._tracks[tid_b].last_centroid
                if float(np.linalg.norm(c_a - c_b)) < self.crossing_distance:
                    uncertain_track_ids.add(tid_a)
                    uncertain_track_ids.add(tid_b)

        # --- Update matched tracks -------------------------------------------
        for det_idx, track_id in assignments.items():
            track = self._tracks[track_id]
            track.last_centroid = centroids[det_idx]
            track.last_bbox = bboxes[det_idx]
            track.frames_since_seen = 0
            track.total_frames_seen += 1
            if track_id in uncertain_track_ids:
                track.status = TRACK_STATUS_UNCERTAIN
                track.uncertainty_reason = "near-crossing detected"
            else:
                track.status = TRACK_STATUS_ACTIVE
                track.uncertainty_reason = ""

        # --- Age out unmatched tracks ----------------------------------------
        for track_id in list(unmatched_tracks):
            track = self._tracks[track_id]
            track.frames_since_seen += 1
            if track.frames_since_seen > self.max_missed_frames:
                track.status = TRACK_STATUS_ENDED
                del self._tracks[track_id]
            else:
                track.status = TRACK_STATUS_TEMPORARILY_LOST
                if not track.uncertainty_reason:
                    track.uncertainty_reason = ""

        # --- Build result (only for frames with active detections) -----------
        return [
            TrackedPose(
                track_id=assignments[i],
                landmarks_2d=detections[i][0],
                confidence=detections[i][1],
                track_status=self._tracks[assignments[i]].status,
                uncertainty_reason=self._tracks[assignments[i]].uncertainty_reason,
                frame_index=frame_index,
                session_id=session_id,
            )
            for i in range(len(detections))
        ]

    def active_track_ids(self) -> list[int]:
        return [
            tid for tid, t in self._tracks.items()
            if t.status in (TRACK_STATUS_ACTIVE, TRACK_STATUS_UNCERTAIN)
        ]

    def all_track_statuses(self) -> dict[int, str]:
        return {tid: t.status for tid, t in self._tracks.items()}
