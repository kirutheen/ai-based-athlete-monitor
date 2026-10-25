"""
Landmark index/name helpers shared by the feature-extraction modules.

MediaPipe's PoseLandmarker emits 33 landmarks in a fixed order (the
BlazePose topology). We only need a handful of them for knee/hip/ankle
biomechanics, so this module gives them readable names instead of
scattering magic integers like `landmarks[25]` through the codebase.

Index values match `mediapipe.tasks.python.components.containers.PoseLandmark`
(the same enum used in src/pose/pose_estimator.py), duplicated here as plain
ints so this module has no mediapipe import and can be unit-tested/used
without the mediapipe dependency installed (e.g. inside the FastAPI service
or in a notebook that only reads already-extracted CSV/Parquet landmark
data).
"""

from __future__ import annotations

from enum import IntEnum


class LM(IntEnum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


class Provenance:
    MEASURED = "MEASURED"
    ESTIMATED = "ESTIMATED"
    SIMULATED = "SIMULATED"
    UNAVAILABLE = "UNAVAILABLE"



# Columns written by scripts/run_pose_on_video.py for a given landmark name,
# e.g. "left_knee_x". Kept here so feature code and the CLI script agree on
# naming without importing each other.
AXES = ("x", "y", "z", "v")


def landmark_columns(name: str) -> list[str]:
    """Return the 4 dataframe column names for a landmark, e.g. 'left_knee'."""
    return [f"{name}_{axis}" for axis in AXES]


# All 33 MediaPipe BlazePose landmark names, lowercased, in MediaPipe's
# fixed landmark order (index 0..32). This is the canonical source for
# that ordering across the project -- scripts/run_pose_on_video.py derives
# its CSV/Parquet column names from mediapipe's own PoseLandmark enum
# directly (since it already has mediapipe imported), while feature/
# pipeline code that should NOT need mediapipe installed (e.g.
# scripts/process_session.py, which only reads already-extracted pose
# data) imports this plain list instead. The two must stay in sync; this
# order matches mediapipe.tasks.python.components.containers.PoseLandmark.
ALL_LANDMARK_NAMES: tuple[str, ...] = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
)
