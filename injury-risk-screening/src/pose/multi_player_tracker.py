"""
Multi-Player Persistent Tracker for DriftGuard.

Detects player candidates in video frames using OpenCV HOG person detection,
runs MediaPipe Pose on each crop to produce real landmarks (remapped to
full-frame coordinates), computes per-player biomechanics, and maintains
persistent tracker_ids using IoU centroid matching.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from src.features.player_state import PlayerState

# ---------------------------------------------------------------------------
# MediaPipe PoseLandmarker (IMAGE mode) — one shared instance per process
# ---------------------------------------------------------------------------
_pose_landmarker: Optional[vision.PoseLandmarker] = None
_POSE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models", "pose_landmarker_full.task",
)


def _get_pose_landmarker() -> Optional[vision.PoseLandmarker]:
    """Return a shared IMAGE-mode PoseLandmarker, or None if the model file is missing."""
    global _pose_landmarker
    if _pose_landmarker is not None:
        return _pose_landmarker
    if not os.path.isfile(_POSE_MODEL_PATH):
        return None  # model not present — caller will skip MediaPipe
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_POSE_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    _pose_landmarker = vision.PoseLandmarker.create_from_options(options)
    return _pose_landmarker


# ---------------------------------------------------------------------------
# HOG person detector — one shared instance per process
# ---------------------------------------------------------------------------
_hog: Optional[cv2.HOGDescriptor] = None


def _get_hog() -> cv2.HOGDescriptor:
    global _hog
    if _hog is None:
        _hog = cv2.HOGDescriptor()
        _hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return _hog


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax) normalized 0..1
    confidence: float
    landmarks: Optional[np.ndarray] = None   # (33, 4) full-frame normalized, or None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def compute_iou(
    boxA: Tuple[float, float, float, float],
    boxB: Tuple[float, float, float, float],
) -> float:
    """Intersection over Union between two normalized bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    denom = boxAArea + boxBArea - interArea
    if denom <= 1e-6:
        return 0.0
    return interArea / denom


def predict_bbox(player: PlayerState) -> Tuple[float, float, float, float]:
    """Predict player bounding box using linear velocity extrapolation."""
    if not player.bbox_history:
        return (0.25, 0.10, 0.75, 0.95)
    if len(player.bbox_history) < 2:
        return player.bbox_history[-1]

    b1 = player.bbox_history[-2]
    b2 = player.bbox_history[-1]
    dx = (b2[0] - b1[0] + b2[2] - b1[2]) / 2.0
    dy = (b2[1] - b1[1] + b2[3] - b1[3]) / 2.0

    xmin = max(0.0, min(1.0, b2[0] + dx))
    ymin = max(0.0, min(1.0, b2[1] + dy))
    xmax = max(0.0, min(1.0, b2[2] + dx))
    ymax = max(0.0, min(1.0, b2[3] + dy))
    return (xmin, ymin, xmax, ymax)


# ---------------------------------------------------------------------------
# Real landmark extraction via MediaPipe Pose on each crop
# ---------------------------------------------------------------------------

def extract_crop_landmarks(
    bbox: Tuple[float, float, float, float],
    frame_bgr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Run MediaPipe Pose on the player's bounding-box crop and remap the
    normalized landmark coordinates back into full-frame space.

    If ``frame_bgr`` is None or MediaPipe fails to detect a pose, returns
    None so the caller can choose to skip rather than fabricate data.
    """
    if frame_bgr is None:
        return None

    h, w = frame_bgr.shape[:2]
    xmin, ymin, xmax, ymax = bbox

    # Convert to pixel coords with a small margin
    margin = 0.02
    px0 = max(0, int((xmin - margin) * w))
    py0 = max(0, int((ymin - margin) * h))
    px1 = min(w, int((xmax + margin) * w))
    py1 = min(h, int((ymax + margin) * h))

    if px1 - px0 < 20 or py1 - py0 < 20:
        return None

    landmarker = _get_pose_landmarker()
    if landmarker is None:
        return None  # model file not present

    crop = frame_bgr[py0:py1, px0:px1]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)

    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return None

    # Remap crop-normalized coords to full-frame-normalized coords
    crop_w = px1 - px0
    crop_h = py1 - py0
    lm_out = np.zeros((33, 4), dtype=np.float32)
    for i, lm in enumerate(result.pose_landmarks[0]):
        full_x = (px0 + lm.x * crop_w) / w
        full_y = (py0 + lm.y * crop_h) / h
        lm_out[i, 0] = float(np.clip(full_x, 0.0, 1.0))
        lm_out[i, 1] = float(np.clip(full_y, 0.0, 1.0))
        lm_out[i, 2] = float(lm.z)
        lm_out[i, 3] = float(lm.visibility) if lm.visibility is not None else 0.0

    return lm_out


# ---------------------------------------------------------------------------
# Biomechanics from real landmarks
# ---------------------------------------------------------------------------

def _compute_bio_dict(lm: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    """
    Compute fppa_left and knee_flexion_left from real landmarks.
    Returns None for each metric when the landmark array is unavailable.
    """
    if lm is None:
        return {"fppa_left": None, "knee_flexion_left": None}

    try:
        from src.features.biomechanics import fppa, knee_flexion_angle
        fppa_val = fppa(lm, side="left")
        kf_val = knee_flexion_angle(lm, side="left")
        return {
            "fppa_left": float(fppa_val) if fppa_val is not None else None,
            "knee_flexion_left": float(kf_val) if kf_val is not None else None,
        }
    except Exception:
        return {"fppa_left": None, "knee_flexion_left": None}


def _detect_by_contours(frame_bgr: np.ndarray) -> List[Detection]:
    """
    Secondary fallback person detector using contour bounding boxes.
    Detects well-separated foreground regions (tall aspect ratio, sufficient area).
    Used when HOG finds fewer than 2 candidates.
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold to handle varying illumination
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)

    # Morphological cleanup to join broken body parts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    min_area = h * w * 0.01  # at least 1% of frame
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh / max(1, bw) < 0.6:  # must be taller than wide
            continue
        xmin_n = max(0.0, x / float(w))
        ymin_n = max(0.0, y / float(h))
        xmax_n = min(1.0, (x + bw) / float(w))
        ymax_n = min(1.0, (y + bh) / float(h))
        candidates.append(Detection(bbox=(xmin_n, ymin_n, xmax_n, ymax_n), confidence=0.60))

    # Sort largest area first
    candidates.sort(key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]), reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Main Tracker
# ---------------------------------------------------------------------------

class MultiPlayerTracker:
    """
    Persistent Multi-Player Tracking Engine.

    Uses OpenCV HOG for person detection and MediaPipe Pose (IMAGE mode) for
    per-crop landmark extraction. Tracks across frames with IoU matching.
    """

    def __init__(self, iou_threshold: float = 0.30, max_missing_frames: int = 30):
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self.next_tracker_id = 1
        self.players: Dict[int, PlayerState] = {}

    def detect_player_candidates(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Detect person bounding boxes using OpenCV HOG + NMS.
        Returns a list of Detection objects with normalized bounding boxes.
        Falls back to a single center-of-frame candidate if HOG finds nothing,
        so that single-athlete sessions still produce at least one track.
        """
        h, w = frame_bgr.shape[:2]
        hog = _get_hog()

        # HOG works best at moderate resolution; downscale if too large
        scale = 1.0
        if w > 800:
            scale = 800.0 / w
            small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        else:
            small = frame_bgr

        rects, weights = hog.detectMultiScale(
            small,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05,
        )

        detections: List[Detection] = []

        if len(rects) > 0:
            # Apply non-maximum suppression via built-in groupRectangles
            rects_xywh = [(int(x), int(y), int(bw), int(bh)) for x, y, bw, bh in rects]
            # NMS: group overlapping rects (groupThreshold=1 keeps every rect that
            # is grouped at least once; use a separate IOU-based NMS for quality)
            rects_np = np.array([[x, y, x + bw, y + bh] for x, y, bw, bh in rects_xywh])
            weights_np = np.array([float(ws[0]) if hasattr(ws, '__len__') else float(ws)
                                   for ws in weights], dtype=np.float32)

            # Simple greedy NMS
            order = np.argsort(-weights_np)
            keep = []
            suppressed = set()
            for i in order:
                if i in suppressed:
                    continue
                keep.append(i)
                for j in order:
                    if j <= i or j in suppressed:
                        continue
                    # IOU in pixel space
                    bA = (rects_np[i, 0], rects_np[i, 1], rects_np[i, 2], rects_np[i, 3])
                    bB = (rects_np[j, 0], rects_np[j, 1], rects_np[j, 2], rects_np[j, 3])
                    xA = max(bA[0], bB[0]); yA = max(bA[1], bB[1])
                    xB = min(bA[2], bB[2]); yB = min(bA[3], bB[3])
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    aA = (bA[2] - bA[0]) * (bA[3] - bA[1])
                    aB = (bB[2] - bB[0]) * (bB[3] - bB[1])
                    denom = aA + aB - inter
                    iou = inter / denom if denom > 0 else 0.0
                    if iou > 0.45:
                        suppressed.add(j)

            inv_scale = 1.0 / scale
            for i in keep:
                x0, y0, x1, y1 = rects_np[i]
                # Scale back to original resolution
                x0 = int(x0 * inv_scale); y0 = int(y0 * inv_scale)
                x1 = int(x1 * inv_scale); y1 = int(y1 * inv_scale)
                xmin_n = max(0.0, x0 / w)
                ymin_n = max(0.0, y0 / h)
                xmax_n = min(1.0, x1 / w)
                ymax_n = min(1.0, y1 / h)
                conf = float(weights_np[i])
                detections.append(Detection(bbox=(xmin_n, ymin_n, xmax_n, ymax_n), confidence=conf))

        # Contour-based fallback: runs when HOG finds fewer than 2 candidates
        # (needed for low-resolution, synthetic, or extreme-angle frames where
        # HOG gradient patterns are absent)
        if len(detections) < 2:
            contour_dets = _detect_by_contours(frame_bgr)
            # Add contour detections that don't overlap existing HOG detections
            for cd in contour_dets:
                overlap = any(compute_iou(cd.bbox, d.bbox) > 0.3 for d in detections)
                if not overlap:
                    detections.append(cd)

        # Ultimate fallback: guarantee at least one candidate
        if not detections:
            detections.append(Detection(bbox=(0.25, 0.10, 0.75, 0.95), confidence=0.50))

        return detections

    def update_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_s: float,
        detected_landmarks_list: Optional[List[np.ndarray]] = None,
        candidate_jerseys: Optional[Dict[int, int]] = None,
    ) -> List[PlayerState]:
        """
        Processes a frame with motion prediction, IoU matching, and re-identification.
        Uses real MediaPipe landmarks and biomechanics instead of hardcoded values.
        """
        detections = self.detect_player_candidates(frame_bgr)

        # Extract real landmarks for each detection (unless already provided)
        for det_idx, det in enumerate(detections):
            if detected_landmarks_list and det_idx < len(detected_landmarks_list):
                det.landmarks = detected_landmarks_list[det_idx]
            else:
                det.landmarks = extract_crop_landmarks(det.bbox, frame_bgr)

        active_player_ids = [pid for pid, p in self.players.items() if p.active]
        inactive_player_ids = [
            pid for pid, p in self.players.items()
            if not p.active and p.missing_frame_count <= self.max_missing_frames
        ]

        matched_detections: set = set()
        matched_players: set = set()

        # Phase A: Match active players using motion-predicted bbox
        for pid in active_player_ids:
            player = self.players[pid]
            pred_bbox = predict_bbox(player)

            best_iou = 0.0
            best_det_idx = -1

            for det_idx, det in enumerate(detections):
                if det_idx in matched_detections:
                    continue
                iou = compute_iou(pred_bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_iou >= self.iou_threshold and best_det_idx >= 0:
                det = detections[best_det_idx]
                lm = det.landmarks  # may be None if MediaPipe found no pose
                bio_dict = _compute_bio_dict(lm)
                player.add_frame_data(timestamp_s, det.bbox, lm, bio_dict)

                matched_players.add(pid)
                matched_detections.add(best_det_idx)

        # Phase B: Re-identify temporarily occluded / missed tracks
        for pid in inactive_player_ids:
            if pid in matched_players:
                continue
            player = self.players[pid]
            pred_bbox = predict_bbox(player)

            best_score = 0.0
            best_det_idx = -1

            for det_idx, det in enumerate(detections):
                if det_idx in matched_detections:
                    continue
                iou = compute_iou(pred_bbox, det.bbox)
                jersey_match = (
                    candidate_jerseys is not None
                    and det_idx in candidate_jerseys
                    and player.jersey_number is not None
                    and candidate_jerseys[det_idx] == player.jersey_number
                )
                score = iou + (0.5 if jersey_match else 0.0)

                if (iou >= (self.iou_threshold * 0.7) or jersey_match) and score > best_score:
                    best_score = score
                    best_det_idx = det_idx

            if best_det_idx >= 0:
                det = detections[best_det_idx]
                player.active = True
                player.missing_frame_count = 0
                lm = det.landmarks
                bio_dict = _compute_bio_dict(lm)
                player.add_frame_data(timestamp_s, det.bbox, lm, bio_dict)

                matched_players.add(pid)
                matched_detections.add(best_det_idx)

        # Phase C: Mark unmatched active players as missing
        for pid in active_player_ids:
            if pid not in matched_players:
                self.players[pid].mark_missing(self.max_missing_frames)

        # Phase D: Create new PlayerState for remaining unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_detections:
                new_id = self.next_tracker_id
                self.next_tracker_id += 1

                p = PlayerState(
                    tracker_id=new_id,
                    first_seen=timestamp_s,
                    last_seen=timestamp_s,
                    active=True,
                )
                lm = det.landmarks
                bio_dict = _compute_bio_dict(lm)
                p.add_frame_data(timestamp_s, det.bbox, lm, bio_dict)

                self.players[new_id] = p

        return [p for p in self.players.values() if p.active]
