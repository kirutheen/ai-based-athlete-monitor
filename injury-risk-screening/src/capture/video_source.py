"""Video ingestion utilities.

Wraps OpenCV's VideoCapture to provide a simple, resource-safe iterator over
frames from either a video file or a live camera, with optional resampling to
a fixed target frame rate (dropping or duplicating source frames as needed).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional, Union

import cv2
import numpy as np


@dataclass
class Frame:
    """A single video frame with its timing/index metadata."""

    image: np.ndarray          # BGR image, as returned by OpenCV
    timestamp: float           # seconds, on the OUTPUT timeline (0.0 at start)
    frame_index: int           # index on the OUTPUT timeline (after fps resampling)
    source_frame_index: int    # index of the underlying source frame (may repeat
                                # if a frame was duplicated to hit the target fps)


class VideoSourceError(RuntimeError):
    """Raised when a video source cannot be opened or read."""


class VideoSource:
    """Reads frames from a video file or camera, optionally resampled to a fixed FPS.

    Parameters
    ----------
    source:
        Path to a video file, or an integer camera index (e.g. 0 for the default
        webcam).
    target_fps:
        If given, frames are dropped or duplicated so the *output* stream runs at
        this frame rate, regardless of the source's native frame rate. If None,
        every native frame is emitted once, as-is.
    loop:
        If True (file sources only), restart from the beginning when the source is
        exhausted instead of stopping iteration. Ignored for camera sources.
    """

    def __init__(
        self,
        source: Union[str, int],
        target_fps: Optional[float] = None,
        loop: bool = False,
    ) -> None:
        self.source = source
        self.target_fps = target_fps
        self.loop = loop and isinstance(source, str)
        self._is_camera = isinstance(source, int)

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise VideoSourceError(f"Could not open video source: {source!r}")

        native_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if not native_fps or native_fps <= 1e-3 or native_fps != native_fps:  # NaN check
            # Cameras (and some containers) misreport FPS; fall back to a sane default.
            native_fps = 30.0
        self.native_fps: float = float(native_fps)

        self.width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_count = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # Cameras/streams report 0 or -1 for frame count.
        self.frame_count: Optional[int] = (
            int(frame_count) if frame_count and frame_count > 0 else None
        )

    @property
    def output_fps(self) -> float:
        return self.target_fps if self.target_fps else self.native_fps

    def frames(self) -> Iterator[Frame]:
        """Yield Frame objects at `output_fps`, dropping/duplicating source frames
        as needed. For a live camera with no target_fps, frames are emitted as
        they arrive and timestamped against native frame index / native fps.
        """
        if self.target_fps is None:
            yield from self._frames_native()
        else:
            yield from self._frames_resampled()

    def _read_raw(self):
        ret, frame = self._cap.read()
        if not ret and self.loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
        return ret, frame

    def _frames_native(self) -> Iterator[Frame]:
        idx = 0
        start_time = time.time()
        while True:
            ret, frame = self._read_raw()
            if not ret:
                return
            ts = (time.time() - start_time) if self._is_camera else idx / self.native_fps
            yield Frame(image=frame, timestamp=ts, frame_index=idx, source_frame_index=idx)
            idx += 1

    def _frames_resampled(self) -> Iterator[Frame]:
        step = 1.0 / self.target_fps
        next_emit_time = 0.0
        out_idx = 0
        src_idx = -1
        current_frame = None

        while True:
            ret, frame = self._read_raw()
            if not ret:
                break
            src_idx += 1
            current_frame = frame
            current_source_time = src_idx / self.native_fps

            # Emit as many output frames as fit before the next source frame arrives
            # (this duplicates `current_frame` if target_fps > native_fps, and
            # naturally drops source frames if target_fps < native_fps).
            while next_emit_time <= current_source_time:
                yield Frame(
                    image=current_frame,
                    timestamp=next_emit_time,
                    frame_index=out_idx,
                    source_frame_index=src_idx,
                )
                out_idx += 1
                next_emit_time = out_idx * step

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __iter__(self) -> Iterator[Frame]:
        return self.frames()
