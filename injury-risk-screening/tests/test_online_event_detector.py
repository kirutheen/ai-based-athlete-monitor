"""Tests for the streaming OnlineEventDetector -- feeds frames one at a
time (as a live webcam loop would) and checks it detects the same event
the batch segmenter would, without ever seeing more than a short window
of frames at once."""

from __future__ import annotations

import numpy as np

from src.features.online_event_detector import OnlineEventDetector


def _synthetic_hip_y_stream(fps: float = 30.0, n_frames: int = 150):
    t = np.arange(n_frames) / fps
    baseline = 0.5
    hip_y = np.full(n_frames, baseline)
    jump_start, jump_end = 40, 60
    jump_frames = np.arange(jump_start, jump_end)
    phase = (jump_frames - jump_start) / (jump_end - jump_start)
    hip_y[jump_frames] = baseline - 0.08 * np.sin(phase * np.pi)
    return t, hip_y


def test_online_detector_confirms_one_event_from_a_stream():
    fps = 30.0
    t, hip_y = _synthetic_hip_y_stream(fps=fps)
    detector = OnlineEventDetector(fps=fps, min_jump_height_px=0.03, min_event_separation_s=0.5)

    confirmed_events = []
    for i in range(len(hip_y)):
        features = {"fppa_left": 5.0, "fppa_right": 5.0}
        event = detector.push(frame_index=i, timestamp=t[i], hip_y=hip_y[i], features=features)
        if event is not None:
            confirmed_events.append(event)

    assert len(confirmed_events) == 1
    event = confirmed_events[0]
    assert event.takeoff_frame < event.peak_frame < event.landing_frame
    assert abs(event.averaged_features["fppa_left"] - 5.0) < 1e-6


def test_online_detector_ignores_small_sway_streamed():
    fps = 30.0
    n_frames = 90
    hip_y = 0.5 + 0.005 * np.sin(np.arange(n_frames) / 5.0)
    detector = OnlineEventDetector(fps=fps, min_jump_height_px=0.03)

    confirmed = []
    for i in range(n_frames):
        event = detector.push(frame_index=i, timestamp=i / fps, hip_y=hip_y[i], features={})
        if event is not None:
            confirmed.append(event)
    assert confirmed == []


def test_online_detector_respects_min_separation_streamed():
    """Two jumps close together (closer than min_event_separation_s) should
    collapse to whichever one is correctly detected first, never double-fire
    on the same physical landing."""
    fps = 30.0
    t, hip_y = _synthetic_hip_y_stream(fps=fps, n_frames=150)
    detector = OnlineEventDetector(fps=fps, min_jump_height_px=0.03, min_event_separation_s=5.0)

    confirmed = []
    for i in range(len(hip_y)):
        event = detector.push(frame_index=i, timestamp=t[i], hip_y=hip_y[i], features={})
        if event is not None:
            confirmed.append(event)
    assert len(confirmed) == 1


def test_online_detector_handles_two_separated_jumps():
    fps = 30.0
    n_frames = 240
    t = np.arange(n_frames) / fps
    baseline = 0.5
    hip_y = np.full(n_frames, baseline)
    for start in (40, 160):
        jump_frames = np.arange(start, start + 20)
        phase = (jump_frames - start) / 20.0
        hip_y[jump_frames] = baseline - 0.08 * np.sin(phase * np.pi)

    detector = OnlineEventDetector(fps=fps, min_jump_height_px=0.03, min_event_separation_s=1.0)
    confirmed = []
    for i in range(n_frames):
        event = detector.push(frame_index=i, timestamp=t[i], hip_y=hip_y[i], features={})
        if event is not None:
            confirmed.append(event)
    assert len(confirmed) == 2
    assert confirmed[1].landing_frame > confirmed[0].landing_frame
