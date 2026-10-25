"""
Real Measurement Quality Engine for DriftGuard.

Calculates empirical measurement confidence scores based on:
1. Pose tracking landmark confidence & visibility ratio
2. Signal continuity & missing frame ratio
3. Sudden coordinate jump noise detection
4. Body visibility & framing ratio
5. Protocol frame length requirements

Disentangles Measurement Quality from Movement Risk/Deviation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class QualityCategory(str, Enum):
    HIGH = "HIGH"
    ACCEPTABLE = "ACCEPTABLE"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass
class QualityAssessmentResult:
    quality_score: float  # 0.0 to 1.0
    quality_category: QualityCategory
    is_sufficient_for_screening: bool
    quality_reasons: List[str]
    failed_checks: List[str]
    recommendation: str
    tracking_confidence_mean: float
    missing_frame_ratio: float
    signal_noise_ratio: float
    framing_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality_score": round(self.quality_score, 3),
            "quality_category": self.quality_category.value,
            "is_sufficient_for_screening": self.is_sufficient_for_screening,
            "quality_reasons": self.quality_reasons,
            "failed_checks": self.failed_checks,
            "recommendation": self.recommendation,
            "tracking_confidence_mean": round(self.tracking_confidence_mean, 3),
            "missing_frame_ratio": round(self.missing_frame_ratio, 3),
            "signal_noise_ratio": round(self.signal_noise_ratio, 3),
            "framing_ratio": round(self.framing_ratio, 3),
        }


class MeasurementQualityEngine:
    """
    Evaluates tracking data quality across a video sequence or frame matrix.
    """

    MIN_ACCEPTABLE_FRAMES = 15
    JUMP_THRESHOLD_PX = 0.15  # normalized coordinate jump between adjacent frames

    def evaluate_sequence(
        self,
        landmarks_series: List[Optional[np.ndarray]],
        min_visibility_threshold: float = 0.5,
        expected_min_frames: int = 15,
    ) -> QualityAssessmentResult:
        """
        Evaluate a sequence of frame landmarks (each shape (33, 4) x, y, z, v).
        """
        if not landmarks_series or len(landmarks_series) < self.MIN_ACCEPTABLE_FRAMES:
            return QualityAssessmentResult(
                quality_score=0.0,
                quality_category=QualityCategory.INSUFFICIENT,
                is_sufficient_for_screening=False,
                quality_reasons=["Recording is too short or contains no usable frames."],
                failed_checks=["MINIMUM_FRAME_COUNT"],
                recommendation="INSUFFICIENT MEASUREMENT QUALITY — REPEAT TEST",
                tracking_confidence_mean=0.0,
                missing_frame_ratio=1.0,
                signal_noise_ratio=1.0,
                framing_ratio=0.0,
            )

        total_frames = len(landmarks_series)
        valid_frames = 0
        confidences = []
        jumps = []
        prev_knee_pos = None

        # Key lower-body landmark indices: Hips (23,24), Knees (25,26), Ankles (27,28)
        key_indices = [23, 24, 25, 26, 27, 28]

        for frame in landmarks_series:
            if frame is None or len(frame) < 29:
                continue

            # Check visibility of key lower body landmarks
            key_visibilities = [frame[i][3] for i in key_indices if i < len(frame)]
            if not key_visibilities:
                continue

            mean_vis = float(np.mean(key_visibilities))
            confidences.append(mean_vis)

            if mean_vis >= min_visibility_threshold:
                valid_frames += 1

            # Check for sudden joint jumps (signal noise/occlusion artifacts)
            curr_knee = frame[25][:2]  # left knee x, y
            if prev_knee_pos is not None:
                dist = float(np.linalg.norm(curr_knee - prev_knee_pos))
                if dist > self.JUMP_THRESHOLD_PX:
                    jumps.append(dist)
            prev_knee_pos = curr_knee

        missing_frame_ratio = 1.0 - (valid_frames / max(1, total_frames))
        mean_confidence = float(np.mean(confidences)) if confidences else 0.0
        noise_ratio = len(jumps) / max(1, total_frames)
        framing_ratio = valid_frames / max(1, total_frames)

        failed_checks = []
        quality_reasons = []

        if missing_frame_ratio > 0.40:
            failed_checks.append("EXCESSIVE_MISSING_LANDMARKS")
            quality_reasons.append(f"High occlusion / missing landmark ratio ({missing_frame_ratio*100:.1f}%).")

        if mean_confidence < 0.60:
            failed_checks.append("LOW_TRACKING_CONFIDENCE")
            quality_reasons.append(f"Low overall landmark confidence ({mean_confidence*100:.1f}%).")

        if noise_ratio > 0.15:
            failed_checks.append("HIGH_SIGNAL_NOISE")
            quality_reasons.append(f"Frequent landmark tracking jumps ({noise_ratio*100:.1f}% frames).")

        if total_frames < expected_min_frames:
            failed_checks.append("SHORT_RECORDING_DURATION")
            quality_reasons.append(f"Trial length ({total_frames} frames) below protocol requirement ({expected_min_frames}).")

        # Composite Quality Score (0.0 to 1.0)
        score = (
            (1.0 - missing_frame_ratio) * 0.40
            + mean_confidence * 0.40
            + (1.0 - min(1.0, noise_ratio * 3)) * 0.20
        )
        score = max(0.0, min(1.0, float(score)))

        # Categorize
        if score >= 0.80 and not failed_checks:
            category = QualityCategory.HIGH
            is_sufficient = True
            recommendation = "Measurement quality is excellent. Proceed with screening evaluation."
        elif score >= 0.60 and len(failed_checks) <= 1:
            category = QualityCategory.ACCEPTABLE
            is_sufficient = True
            recommendation = "Measurement quality is acceptable for baseline screening."
        elif score >= 0.40:
            category = QualityCategory.LOW
            is_sufficient = True
            recommendation = "Measurement quality is low. Interpret screening findings with caution."
        else:
            category = QualityCategory.INSUFFICIENT
            is_sufficient = False
            recommendation = "INSUFFICIENT MEASUREMENT QUALITY — REPEAT TEST"

        return QualityAssessmentResult(
            quality_score=score,
            quality_category=category,
            is_sufficient_for_screening=is_sufficient,
            quality_reasons=quality_reasons if quality_reasons else ["High tracking continuity across all frames."],
            failed_checks=failed_checks,
            recommendation=recommendation,
            tracking_confidence_mean=mean_confidence,
            missing_frame_ratio=missing_frame_ratio,
            signal_noise_ratio=noise_ratio,
            framing_ratio=framing_ratio,
        )
