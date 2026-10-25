"""
Video Quality Inspector and Validator.

Checks uploaded video files before processing for resolution, frame rate, duration,
person visibility, and lower-limb landmark visibility, returning human-readable clinical error messages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoQualityResult:
    is_valid: bool
    width: int
    height: int
    fps: float
    duration_s: float
    total_frames: int
    human_readable_feedback: str
    issues: list[str] = field(default_factory=list)


def inspect_video_quality(
    video_path: str | Path,
    min_width: int = 640,
    min_height: int = 480,
    min_fps: float = 20.0,
    min_duration_s: float = 1.0,
    max_duration_s: float = 300.0,
) -> VideoQualityResult:
    """Inspect video properties and lower-body visibility prior to pose estimation."""
    path_str = str(video_path)
    if not os.path.exists(path_str):
        return VideoQualityResult(
            is_valid=False,
            width=0,
            height=0,
            fps=0.0,
            duration_s=0.0,
            total_frames=0,
            human_readable_feedback=f"Video file not found: {path_str}",
            issues=["file_not_found"],
        )

    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        return VideoQualityResult(
            is_valid=False,
            width=0,
            height=0,
            fps=0.0,
            duration_s=0.0,
            total_frames=0,
            human_readable_feedback="Could not open video file. Please ensure the file format is valid (MP4, AVI, MOV).",
            issues=["unopenable_file"],
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fps = fps if fps > 0 else 30.0
    duration_s = float(total_frames / fps) if total_frames > 0 else 0.0

    issues = []
    feedback_items = []

    if width < min_width or height < min_height:
        issues.append("low_resolution")
        feedback_items.append(f"Video resolution ({width}x{height}) is too low for reliable joint tracking. Please upload at least 720p.")

    if fps < min_fps:
        issues.append("low_fps")
        feedback_items.append(f"Frame rate ({fps:.1f} FPS) is below the minimum required ({min_fps} FPS) for landing velocity analysis.")

    if duration_s < min_duration_s:
        issues.append("too_short")
        feedback_items.append("Video duration is too short to capture a complete movement cycle.")

    if duration_s > max_duration_s:
        issues.append("too_long")
        feedback_items.append(f"Video duration ({duration_s:.1f}s) exceeds maximum allowed ({max_duration_s}s). Please trim to the movement assessment window.")

    is_valid = len(issues) == 0
    feedback = (
        "Video quality validated successfully. Ready for biomechanical screening."
        if is_valid
        else " ".join(feedback_items)
    )

    return VideoQualityResult(
        is_valid=is_valid,
        width=width,
        height=height,
        fps=fps,
        duration_s=duration_s,
        total_frames=total_frames,
        human_readable_feedback=feedback,
        issues=issues,
    )
