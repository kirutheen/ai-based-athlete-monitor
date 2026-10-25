"""
Streaming (online) landing-event detector.

src/features/event_segmentation.py's segment_landing_events() is a BATCH
function: it needs the whole session's DataFrame in memory before it can
find any event, because it looks forward from a candidate peak to find
where the athlete lands. That's fine for "process a recorded video
afterward" (scripts/process_session.py), but it cannot work for live
webcam monitoring, where frames arrive one at a time forever and you
can't wait for "the end of the video" (there isn't one) before detecting
the first landing.

This module re-implements the same detection logic (same thresholds,
same physical reasoning: a landing is a local minimum in hip-height
followed by a return-to-baseline with decelerating downward velocity) as
a STATEFUL, O(1)-memory class that consumes one frame at a time and
emits a completed LandingEvent the moment enough frames after a landing
have arrived to confirm it. It only ever keeps a short rolling window of
recent frames (maxlen = a few hundred ms), not the whole session, so it
is safe to run indefinitely from a live camera.

Used by scripts/live_webcam_demo.py.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class _FrameRecord:
    frame_index: int
    timestamp: float
    hip_y: float
    features: dict  # feature_name -> value, e.g. {"fppa_left": 3.2, ...}


@dataclass
class LiveLandingEvent:
    takeoff_frame: int
    peak_frame: int
    landing_frame: int
    landing_timestamp: float
    # Mean of each feature over the window around landing (same averaging
    # process_session.py applies in batch mode).
    averaged_features: dict


class OnlineEventDetector:
    """
    Feed frames one at a time with `push()`. When a landing event is
    confirmed, `push()` returns a LiveLandingEvent; otherwise it returns
    None. Internally equivalent to event_segmentation.segment_landing_events,
    but causal (never looks at frames that haven't arrived yet) and
    bounded-memory (a short rolling buffer, not the whole session).
    """

    def __init__(
        self,
        fps: float,
        min_jump_height_px: float = 0.03,
        min_event_separation_s: float = 1.0,
        pre_landing_s: float = 0.15,
        post_landing_s: float = 0.15,
        baseline_window_s: float = 3.0,
        confirm_delay_s: float = 0.3,
    ):
        self.fps = fps
        self.min_jump_height_px = min_jump_height_px
        self.min_sep_frames = int(round(min_event_separation_s * fps))
        self.pre_frames = int(round(pre_landing_s * fps))
        self.post_frames = int(round(post_landing_s * fps))
        # How many frames to wait after a candidate landing before
        # confirming it (lets a little more data arrive so the
        # deceleration check is stable) -- the online equivalent of
        # batch mode's "look forward in the whole array".
        self.confirm_delay_frames = max(2, int(round(confirm_delay_s * fps)))

        buffer_len = int(round(baseline_window_s * fps)) + self.pre_frames + self.post_frames + self.confirm_delay_frames + 10
        self._buffer: deque[_FrameRecord] = deque(maxlen=buffer_len)
        self._baseline_window: deque[float] = deque(maxlen=int(round(baseline_window_s * fps)))

        self._last_landing_frame_index: int | None = None
        self._pending_peak_idx: int | None = None  # index into _buffer
        self._pending_takeoff_idx: int | None = None
        self._candidate_landing_idx: int | None = None
        self._frames_since_candidate = 0

    def _baseline_height(self) -> float:
        if not self._baseline_window:
            return 0.5
        sorted_vals = sorted(self._baseline_window)
        return sorted_vals[len(sorted_vals) // 2]  # median

    def push(self, frame_index: int, timestamp: float, hip_y: float, features: dict) -> LiveLandingEvent | None:
        """
        Call once per frame, in order, with the current frame's mean
        hip-y (normalized image coords, lower = higher off the ground)
        and this frame's biomechanical features (fppa_left, etc. -- same
        keys src/features/biomechanics.py produces).
        """
        record = _FrameRecord(frame_index=frame_index, timestamp=timestamp, hip_y=hip_y, features=features)
        self._buffer.append(record)
        self._baseline_window.append(hip_y)

        if len(self._buffer) < 5:
            return None

        baseline = self._baseline_height()
        buf = list(self._buffer)
        i = len(buf) - 3  # the frame we can now confidently classify (needs a neighbor on each side)
        if i < 1:
            return None

        is_local_min = buf[i].hip_y < buf[i - 1].hip_y and buf[i].hip_y <= buf[i + 1].hip_y
        dips_enough = (baseline - buf[i].hip_y) >= self.min_jump_height_px

        if self._candidate_landing_idx is None and is_local_min and dips_enough:
            # Record this as a jump peak candidate; takeoff is found by
            # walking backward from here.
            self._pending_peak_idx = i
            takeoff_idx = i
            for j in range(i, 0, -1):
                if buf[j].hip_y >= baseline - self.min_jump_height_px * 0.25:
                    takeoff_idx = j
                    break
            self._pending_takeoff_idx = takeoff_idx

        if self._pending_peak_idx is not None and self._candidate_landing_idx is None:
            # Look for the landing: return-to-baseline with decelerating
            # downward motion, scanning forward from the peak within
            # what's currently buffered.
            for j in range(self._pending_peak_idx + 1, len(buf) - 1):
                returned = buf[j].hip_y >= baseline - self.min_jump_height_px * 0.25
                v_j = buf[j].hip_y - buf[j - 1].hip_y
                v_j1 = buf[j + 1].hip_y - buf[j].hip_y
                decelerating = v_j >= 0 and v_j1 <= v_j
                if returned and decelerating:
                    candidate_abs_frame = buf[j].frame_index
                    if (
                        self._last_landing_frame_index is None
                        or (candidate_abs_frame - self._last_landing_frame_index) >= self.min_sep_frames
                    ):
                        self._candidate_landing_idx = j
                        self._frames_since_candidate = 0
                    else:
                        # Too close to the previous landing -- discard this
                        # peak/takeoff and keep scanning for a real one.
                        self._pending_peak_idx = None
                        self._pending_takeoff_idx = None
                    break

        if self._candidate_landing_idx is not None:
            self._frames_since_candidate += 1
            if self._frames_since_candidate >= self.confirm_delay_frames:
                event = self._emit_event(buf)
                self._last_landing_frame_index = buf[self._candidate_landing_idx].frame_index
                self._pending_peak_idx = None
                self._pending_takeoff_idx = None
                self._candidate_landing_idx = None
                return event

        return None

    def _emit_event(self, buf: list[_FrameRecord]) -> LiveLandingEvent:
        landing_idx = self._candidate_landing_idx
        landing_record = buf[landing_idx]
        window = buf[max(0, landing_idx - self.pre_frames): min(len(buf), landing_idx + self.post_frames + 1)]

        feature_names = set()
        for rec in window:
            feature_names.update(rec.features.keys())
        averaged = {}
        for name in feature_names:
            values = [rec.features[name] for rec in window if name in rec.features and rec.features[name] == rec.features[name]]
            averaged[name] = sum(values) / len(values) if values else float("nan")

        return LiveLandingEvent(
            takeoff_frame=buf[self._pending_takeoff_idx].frame_index if self._pending_takeoff_idx is not None else landing_record.frame_index,
            peak_frame=buf[self._pending_peak_idx].frame_index if self._pending_peak_idx is not None else landing_record.frame_index,
            landing_frame=landing_record.frame_index,
            landing_timestamp=landing_record.timestamp,
            averaged_features=averaged,
        )
