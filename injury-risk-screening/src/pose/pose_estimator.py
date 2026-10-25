"""MediaPipe Pose Landmarker wrapper with confidence scoring, skeleton
overlay, and temporal smoothing.

Requires a MediaPipe Pose Landmarker model bundle (a `.task` file). MediaPipe
does not ship this file inside the `mediapipe` pip package -- download one of
the official bundles once and point `model_asset_path` at it. See
scripts/download_pose_model.py, or download directly:

    lite (fastest, least accurate):
        https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
    full (balanced -- recommended default):
        https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
    heavy (most accurate, slowest):
        https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections

NUM_LANDMARKS = 33

# (start, end) index pairs describing the BlazePose skeleton, used for the
# debug overlay.
POSE_CONNECTIONS: Tuple[Tuple[int, int], ...] = tuple(
    (c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS
)


@dataclass
class PoseResult:
    """Result of running the pose model on a single frame."""

    landmarks_2d: Optional[np.ndarray]   # (33, 4) -> x, y, z, visibility; normalized
                                          # image coords; None if no pose detected
    landmarks_3d: Optional[np.ndarray]   # (33, 4) -> x, y, z, visibility; metric world
                                          # coords (meters); None if unavailable
    confidence: float                    # 0.0-1.0, mean per-landmark visibility;
                                          # 0.0 if no pose detected
    annotated_frame: np.ndarray          # BGR frame with skeleton drawn (debug)
    detected: bool                       # whether any pose was found at all


class PoseEstimator:
    """Wraps MediaPipe's Pose Landmarker task for per-frame pose extraction."""

    def __init__(
        self,
        model_asset_path: str,
        running_mode: str = "VIDEO",
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        mode_map = {
            "IMAGE": vision.RunningMode.IMAGE,
            "VIDEO": vision.RunningMode.VIDEO,
        }
        if running_mode not in mode_map:
            raise ValueError(f"running_mode must be one of {list(mode_map)}")
        self._running_mode = mode_map[running_mode]

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path),
            running_mode=self._running_mode,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None) -> PoseResult:
        """Run pose detection on a single BGR frame.

        Parameters
        ----------
        frame_bgr:
            An HxWx3 uint8 BGR image, as returned by OpenCV.
        timestamp_ms:
            Monotonically increasing timestamp in milliseconds. Required (and
            must strictly increase) when running_mode="VIDEO". If omitted, or
            if it does not increase, an internal counter takes over so the
            call never crashes on a bad timestamp.
        """
        if timestamp_ms is None or timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._running_mode == vision.RunningMode.VIDEO:
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)

        return self._to_pose_result(result, frame_bgr)

    def _to_pose_result(self, result, frame_bgr: np.ndarray) -> PoseResult:
        annotated = frame_bgr.copy()

        if not result.pose_landmarks:
            return PoseResult(
                landmarks_2d=None,
                landmarks_3d=None,
                confidence=0.0,
                annotated_frame=annotated,
                detected=False,
            )

        # num_poses defaults to 1 for this MVP; take the first detected pose.
        lm2d = result.pose_landmarks[0]
        landmarks_2d = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility if lm.visibility is not None else np.nan] for lm in lm2d],
            dtype=np.float32,
        )

        landmarks_3d = None
        if result.pose_world_landmarks:
            lm3d = result.pose_world_landmarks[0]
            landmarks_3d = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility if lm.visibility is not None else np.nan] for lm in lm3d],
                dtype=np.float32,
            )

        vis_values = landmarks_2d[:, 3]
        valid_vis = vis_values[~np.isnan(vis_values)]
        # Fall back to 1.0 (rather than NaN) when the model doesn't emit visibility
        # scores for this configuration, so downstream persistence logic never has
        # to special-case NaN.
        confidence = float(np.mean(valid_vis)) if valid_vis.size else 1.0

        h, w = frame_bgr.shape[:2]
        self._draw_skeleton(annotated, landmarks_2d, w, h)

        return PoseResult(
            landmarks_2d=landmarks_2d,
            landmarks_3d=landmarks_3d,
            confidence=confidence,
            annotated_frame=annotated,
            detected=True,
        )

    def process_multi(
        self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None
    ) -> List["PoseResult"]:
        """Run pose detection on a single BGR frame and return ALL detected poses.

        Returns one PoseResult per detected person (up to num_poses as set in
        __init__). Returns an empty list when no person is found. The
        annotated_frame in each result shows the skeleton for that person only;
        callers should draw onto their own canvas if they need overlays.

        Parameters mirror process(); see its docstring.
        """
        if timestamp_ms is None or timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._running_mode == vision.RunningMode.VIDEO:
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)

        return self._to_pose_result_multi(result, frame_bgr)

    def _to_pose_result_multi(self, result, frame_bgr: np.ndarray) -> List["PoseResult"]:
        """Convert a raw MediaPipe result into one PoseResult per detected person."""
        if not result.pose_landmarks:
            return []

        out: List[PoseResult] = []
        h, w = frame_bgr.shape[:2]
        num_people = len(result.pose_landmarks)

        for person_idx in range(num_people):
            lm2d = result.pose_landmarks[person_idx]
            landmarks_2d = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility if lm.visibility is not None else np.nan]
                 for lm in lm2d],
                dtype=np.float32,
            )

            landmarks_3d = None
            if result.pose_world_landmarks and person_idx < len(result.pose_world_landmarks):
                lm3d = result.pose_world_landmarks[person_idx]
                landmarks_3d = np.array(
                    [[lm.x, lm.y, lm.z, lm.visibility if lm.visibility is not None else np.nan]
                     for lm in lm3d],
                    dtype=np.float32,
                )

            vis_values = landmarks_2d[:, 3]
            valid_vis = vis_values[~np.isnan(vis_values)]
            confidence = float(np.mean(valid_vis)) if valid_vis.size else 1.0

            # Draw skeleton on a per-person copy for debug purposes.
            annotated = frame_bgr.copy()
            self._draw_skeleton(annotated, landmarks_2d, w, h)

            out.append(PoseResult(
                landmarks_2d=landmarks_2d,
                landmarks_3d=landmarks_3d,
                confidence=confidence,
                annotated_frame=annotated,
                detected=True,
            ))

        return out

    @staticmethod
    def _draw_skeleton(frame: np.ndarray, landmarks_2d: np.ndarray, w: int, h: int) -> None:
        points = [(int(x * w), int(y * h)) for x, y, _, _ in landmarks_2d]
        for start, end in POSE_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
        for x, y in points:
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

def smooth_moving_average(
    landmark_sequence: Sequence[Optional[np.ndarray]], window: int = 5
) -> List[Optional[np.ndarray]]:
    """Smooth a sequence of per-frame landmark arrays with a trailing moving
    average over the last `window` frames. Frames with no detection (None)
    are passed through unchanged and do not enter the averaging buffer.
    """
    if window < 1:
        raise ValueError("window must be >= 1")

    smoothed: List[Optional[np.ndarray]] = []
    buffer: Deque[np.ndarray] = collections.deque(maxlen=window)

    for landmarks in landmark_sequence:
        if landmarks is None:
            smoothed.append(None)
            continue
        buffer.append(landmarks)
        smoothed.append(np.mean(np.stack(buffer, axis=0), axis=0))

    return smoothed


class OneEuroFilter:
    """One-Euro filter for a single scalar signal.

    Adaptive low-pass filter that cuts jitter at low speeds while staying
    responsive during fast motion -- a good fit for landing/cutting frames
    where a stable baseline signal matters but genuinely fast movement must
    not be lagged out. See Casiez et al., 2012.
    """

    def __init__(self, freq: float, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            return x

        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff, self.freq)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, self.freq)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class LandmarkOneEuroFilter:
    """Applies an independent OneEuroFilter to each (landmark, coordinate)
    channel of a (33, 4) landmark array -- i.e. 33*3 = 99 filters for x, y, z.
    Visibility (column 3) is passed through unsmoothed.
    """

    def __init__(
        self,
        num_landmarks: int = NUM_LANDMARKS,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.3,
        d_cutoff: float = 1.0,
    ):
        self._filters = [
            [OneEuroFilter(freq, min_cutoff, beta, d_cutoff) for _ in range(3)]  # x, y, z
            for _ in range(num_landmarks)
        ]

    def __call__(self, landmarks: np.ndarray) -> np.ndarray:
        out = landmarks.copy()
        for i in range(landmarks.shape[0]):
            for c in range(3):
                out[i, c] = self._filters[i][c](float(landmarks[i, c]))
        return out


def smooth_one_euro(
    landmark_sequence: Sequence[Optional[np.ndarray]],
    freq: float = 30.0,
    min_cutoff: float = 1.0,
    beta: float = 0.3,
    d_cutoff: float = 1.0,
) -> List[Optional[np.ndarray]]:
    """Smooth a sequence of per-frame landmark arrays with a One-Euro filter
    per coordinate channel. A None entry (no detection that frame) is passed
    through as None; the filter simply resumes with the next real detection.
    """
    filt = LandmarkOneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
    smoothed: List[Optional[np.ndarray]] = []
    for landmarks in landmark_sequence:
        if landmarks is None:
            smoothed.append(None)
            continue
        smoothed.append(filt(landmarks))
    return smoothed


def smooth(
    landmark_sequence: Sequence[Optional[np.ndarray]], method: str = "one_euro", **kwargs
) -> List[Optional[np.ndarray]]:
    """Convenience dispatcher: smooth(landmark_sequence, method="moving_average"|"one_euro", **kwargs)."""
    if method == "moving_average":
        return smooth_moving_average(landmark_sequence, **kwargs)
    if method == "one_euro":
        return smooth_one_euro(landmark_sequence, **kwargs)
    raise ValueError(f"Unknown smoothing method: {method!r}")
