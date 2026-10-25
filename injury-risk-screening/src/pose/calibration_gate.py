"""
Pre-Test Quality & Calibration Gate for DriftGuard.

Enforces camera positioning, body visibility, framing bounds, landmark confidence,
and orientation checks before allowing movement screening to proceed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class CalibrationStage(str, Enum):
    CAMERA_CHECK = "CAMERA_CHECK"
    BODY_VISIBILITY = "BODY_VISIBILITY"
    LANDMARK_QUALITY = "LANDMARK_QUALITY"
    ORIENTATION_CHECK = "ORIENTATION_CHECK"
    CALIBRATION = "CALIBRATION"
    READY = "READY"
    FAILED = "FAILED"


@dataclass
class CalibrationGateResult:
    is_ready: bool
    current_stage: CalibrationStage
    passed_stages: List[CalibrationStage]
    failed_checks: List[str]
    feedback_messages: List[str]
    estimated_distance_m: Optional[float]
    lighting_quality_index: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "current_stage": self.current_stage.value,
            "passed_stages": [s.value for s in self.passed_stages],
            "failed_checks": self.failed_checks,
            "feedback_messages": self.feedback_messages,
            "estimated_distance_m": round(self.estimated_distance_m, 2) if self.estimated_distance_m else None,
            "lighting_quality_index": round(self.lighting_quality_index, 2),
        }


class CalibrationGate:
    """
    Evaluates frame landmarks during camera setup to ensure valid trial recording.
    """

    MIN_VISIBILITY = 0.55
    MIN_KEY_POINTS = 6  # Shoulders (11,12), Hips (23,24), Knees (25,26), Ankles (27,28)

    def evaluate_calibration_frame(
        self,
        frame_landmarks: Optional[np.ndarray],
        frame_width: int = 1280,
        frame_height: int = 720,
    ) -> CalibrationGateResult:
        """
        Evaluate a single calibration frame (landmarks shape (33,4): x, y, z, v).
        """
        passed_stages = []
        failed_checks = []
        feedback = []

        if frame_landmarks is None or len(frame_landmarks) < 29:
            return CalibrationGateResult(
                is_ready=False,
                current_stage=CalibrationStage.CAMERA_CHECK,
                passed_stages=[],
                failed_checks=["NO_POSE_DETECTED"],
                feedback_messages=["No subject detected in camera view. Step into the frame."],
                estimated_distance_m=None,
                lighting_quality_index=0.0,
            )

        passed_stages.append(CalibrationStage.CAMERA_CHECK)

        # 1. Body Visibility Check
        key_indices = [11, 12, 23, 24, 25, 26, 27, 28]
        visible_keypoints = sum(
            1 for i in key_indices if i < len(frame_landmarks) and frame_landmarks[i][3] >= self.MIN_VISIBILITY
        )

        if visible_keypoints < len(key_indices) - 1:
            failed_checks.append("INCOMPLETE_BODY_VISIBILITY")
            feedback.append("Move farther from camera so full lower body (hips to feet) is visible.")

        # Check framing bounds (is subject cut off at edge of screen?)
        x_coords = [frame_landmarks[i][0] for i in key_indices if i < len(frame_landmarks)]
        y_coords = [frame_landmarks[i][1] for i in key_indices if i < len(frame_landmarks)]

        if min(x_coords) < 0.05 or max(x_coords) > 0.95 or max(y_coords) > 0.95:
            failed_checks.append("SUBJECT_NEAR_FRAME_EDGE")
            feedback.append("Center yourself in the camera frame.")

        if "INCOMPLETE_BODY_VISIBILITY" not in failed_checks and "SUBJECT_NEAR_FRAME_EDGE" not in failed_checks:
            passed_stages.append(CalibrationStage.BODY_VISIBILITY)

        # 2. Landmark Quality Check
        mean_vis = float(np.mean([frame_landmarks[i][3] for i in key_indices]))
        if mean_vis < self.MIN_VISIBILITY:
            failed_checks.append("LOW_LANDMARK_CONFIDENCE")
            feedback.append("Improve lighting or step away from harsh shadows.")
        else:
            passed_stages.append(CalibrationStage.LANDMARK_QUALITY)

        # 3. Orientation Check (Is subject facing camera / side-on appropriately?)
        left_hip, right_hip = frame_landmarks[23], frame_landmarks[24]
        hip_width_norm = abs(left_hip[0] - right_hip[0])

        if hip_width_norm < 0.03:
            failed_checks.append("POOR_SUBJECT_ORIENTATION")
            feedback.append("Turn slightly toward the camera for clearer joint tracking.")
        else:
            passed_stages.append(CalibrationStage.ORIENTATION_CHECK)

        # 4. Calibration & Distance Estimation
        # Estimate subject distance using interpupillary/shoulder width proxy
        left_sh, right_sh = frame_landmarks[11], frame_landmarks[12]
        shoulder_dist = float(np.linalg.norm(left_sh[:2] - right_sh[:2]))
        estimated_distance_m = float(1.2 / max(0.05, shoulder_dist))  # rough distance proxy in meters

        lighting_index = float(mean_vis)

        if len(failed_checks) == 0:
            passed_stages.append(CalibrationStage.CALIBRATION)
            passed_stages.append(CalibrationStage.READY)
            current_stage = CalibrationStage.READY
            feedback.append("Camera setup verified. Ready to record movement.")
            is_ready = True
        else:
            current_stage = passed_stages[-1] if passed_stages else CalibrationStage.CAMERA_CHECK
            is_ready = False

        return CalibrationGateResult(
            is_ready=is_ready,
            current_stage=current_stage,
            passed_stages=passed_stages,
            failed_checks=failed_checks,
            feedback_messages=feedback,
            estimated_distance_m=estimated_distance_m,
            lighting_quality_index=lighting_index,
        )
