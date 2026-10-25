"""
FramePacket abstraction for unified live camera and uploaded video processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np


class FrameSourceType(str, Enum):
    LIVE = "live"
    VIDEO = "video"


@dataclass
class FramePacket:
    """
    Unified frame container passed down the analysis pipeline.
    """
    frame: np.ndarray  # BGR image frame
    frame_number: int
    timestamp_s: float
    fps: float
    source_type: FrameSourceType = FrameSourceType.VIDEO
    session_id: Optional[str] = None
    width: int = field(init=False)
    height: int = field(init=False)

    def __post_init__(self):
        if self.frame is not None and hasattr(self.frame, "shape"):
            self.height, self.width = self.frame.shape[:2]
        else:
            self.height, self.width = 720, 1280

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_number": self.frame_number,
            "timestamp_s": round(self.timestamp_s, 3),
            "fps": round(self.fps, 1),
            "source_type": self.source_type.value if isinstance(self.source_type, Enum) else self.source_type,
            "session_id": self.session_id,
            "width": self.width,
            "height": self.height,
        }
