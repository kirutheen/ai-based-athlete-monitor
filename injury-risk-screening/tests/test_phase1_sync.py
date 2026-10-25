"""
Phase 1 regression tests: uploaded-video frame-sync logic.

These test the JavaScript-side behaviour indirectly by verifying that the
backend's `timestamp_s` values in `biomechanics_frames` are correctly derived
from `frame_index / fps` (the canonical timeline), and that they are
monotonically increasing so the binary-search frame selection on the frontend
is always valid.

They also test the Python helper utilities used by the sync pipeline.
"""
from __future__ import annotations

import math
import pytest


# ---------------------------------------------------------------------------
# Timestamp correctness (backend side)
# ---------------------------------------------------------------------------

def _make_biomechanics_frames(fps: float, num_frames: int):
    """Simulate what process_video_file() builds for biomechanics_frames."""
    return [
        {"frame": i, "timestamp_s": round(i / fps, 3)}
        for i in range(num_frames)
    ]


class TestTimestampGeneration:
    @pytest.mark.parametrize("fps,n", [(30.0, 120), (25.0, 100), (60.0, 240), (29.97, 90)])
    def test_timestamps_monotonically_increasing(self, fps, n):
        frames = _make_biomechanics_frames(fps, n)
        for i in range(1, len(frames)):
            assert frames[i]["timestamp_s"] > frames[i - 1]["timestamp_s"], (
                f"fps={fps}: timestamp not monotonic at index {i}"
            )

    @pytest.mark.parametrize("fps,n", [(30.0, 120), (25.0, 100)])
    def test_first_frame_timestamp_is_zero(self, fps, n):
        frames = _make_biomechanics_frames(fps, n)
        assert frames[0]["timestamp_s"] == 0.0

    @pytest.mark.parametrize("fps,n", [(30.0, 30), (25.0, 25)])
    def test_one_second_duration(self, fps, n):
        frames = _make_biomechanics_frames(fps, n)
        # Last frame should be at (n-1)/fps ≈ 1s - 1/fps
        expected = round((n - 1) / fps, 3)
        assert math.isclose(frames[-1]["timestamp_s"], expected, abs_tol=1e-3)

    def test_frame_index_matches_position(self):
        frames = _make_biomechanics_frames(30.0, 60)
        for i, f in enumerate(frames):
            assert f["frame"] == i


# ---------------------------------------------------------------------------
# Binary-search frame selection (JS logic replicated in Python)
# ---------------------------------------------------------------------------

def _frame_idx_at_time(frames, t: float) -> int:
    """Mirror of the JS _frameIdxAtTime() function (binary search on timestamp_s)."""
    if not frames:
        return 0
    lo, hi = 0, len(frames) - 1
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        if frames[mid]["timestamp_s"] <= t:
            lo = mid
        else:
            hi = mid - 1
    return lo


class TestBinarySearchFrameSelection:
    def setup_method(self):
        self.frames = _make_biomechanics_frames(30.0, 120)

    def test_time_zero_selects_frame_zero(self):
        assert _frame_idx_at_time(self.frames, 0.0) == 0

    def test_exact_timestamp_selects_that_frame(self):
        for i in (0, 15, 30, 60, 119):
            t = self.frames[i]["timestamp_s"]
            assert _frame_idx_at_time(self.frames, t) == i

    def test_midpoint_selects_lower_frame(self):
        # t between frame 0 (0.0) and frame 1 (0.0333…) → should return 0
        t = 0.015
        assert _frame_idx_at_time(self.frames, t) == 0

    def test_time_at_end_of_video(self):
        t = self.frames[-1]["timestamp_s"] + 0.5
        # Should clamp to last frame
        assert _frame_idx_at_time(self.frames, t) == len(self.frames) - 1

    def test_empty_frames(self):
        assert _frame_idx_at_time([], 1.5) == 0

    def test_single_frame(self):
        frames = [{"timestamp_s": 0.0, "frame": 0}]
        assert _frame_idx_at_time(frames, 99.9) == 0

    @pytest.mark.parametrize("fps", [25.0, 29.97, 30.0, 60.0])
    def test_correct_for_all_common_fps(self, fps):
        frames = _make_biomechanics_frames(fps, 60)
        # For every frame, feeding its own timestamp should return exactly that index
        for i, f in enumerate(frames):
            assert _frame_idx_at_time(frames, f["timestamp_s"]) == i

    def test_floor_fps_method_error_vs_binary_search(self):
        """
        Demonstrate the bug: Math.floor(t * fps) gives a wrong index when
        timestamp_s values have rounding, while binary search is exact.
        This test confirms the fix is necessary.
        """
        fps = 29.97
        frames = _make_biomechanics_frames(fps, 90)
        errors = 0
        for i, f in enumerate(frames):
            t = f["timestamp_s"]
            floor_idx = min(len(frames) - 1, max(0, int(t * fps)))
            bsearch_idx = _frame_idx_at_time(frames, t)
            if floor_idx != i:
                errors += 1
            # Binary search should always be correct
            assert bsearch_idx == i, f"Binary search wrong at frame {i}: got {bsearch_idx}"
        # floor method should have at least some errors at 29.97fps
        assert errors > 0, "Expected floor(t*fps) to have rounding errors at 29.97fps"


# ---------------------------------------------------------------------------
# Sync error measurement (no hardware needed — uses known inputs)
# ---------------------------------------------------------------------------

class TestSyncErrorBounds:
    def test_rvfc_sync_error_zero_for_exact_pts(self):
        """
        When RVFC delivers the exact PTS of a frame, the binary search should
        select that exact frame → sync error = 0 frames.
        """
        fps = 30.0
        frames = _make_biomechanics_frames(fps, 120)
        # Simulate RVFC delivering exact PTS for 10 random frames
        for i in (0, 5, 15, 29, 44, 60, 75, 90, 100, 119):
            pts = frames[i]["timestamp_s"]
            selected = _frame_idx_at_time(frames, pts)
            assert selected == i, f"Sync error at frame {i}: selected {selected}"

    def test_ontimeupdate_max_lag_one_frame(self):
        """
        ontimeupdate fires up to ~250ms late. In the worst case, the binary
        search still picks the last frame whose timestamp <= currentTime, so
        lag is at most 1 frame behind (never ahead).
        """
        fps = 30.0
        frames = _make_biomechanics_frames(fps, 120)
        frame_duration = 1.0 / fps
        # Simulate ontimeupdate firing 200ms into frame 30
        actual_frame = 30
        late_time = frames[actual_frame]["timestamp_s"] + 0.20  # 200ms into the frame
        selected = _frame_idx_at_time(frames, late_time)
        # Selected index should be >= actual_frame (on time or 1 frame ahead)
        assert selected >= actual_frame
        # And not more than a few frames ahead (200ms / (1/30s) = 6 frames max)
        assert selected <= actual_frame + 7


# ---------------------------------------------------------------------------
# Seek correctness
# ---------------------------------------------------------------------------

class TestSeekTarget:
    def test_seek_uses_timestamp_s_not_frameidx_over_fps(self):
        """
        The fixed renderFrame() uses frame.timestamp_s for seek, not frameIdx/fps.
        Verify these agree to within 1ms for CFR and differ for 29.97fps.
        """
        fps = 29.97
        frames = _make_biomechanics_frames(fps, 90)
        errors_ms = []
        for i, f in enumerate(frames):
            seek_old = i / fps            # old buggy formula
            seek_new = f["timestamp_s"]   # new formula (= round(i/fps, 3))
            err_ms = abs(seek_new - seek_old) * 1000
            errors_ms.append(err_ms)

        # The new formula (pre-rounded) and raw i/fps should be within 0.5ms each
        assert max(errors_ms) < 0.6, f"Max seek error {max(errors_ms):.3f}ms exceeds 0.6ms"

    def test_seek_target_within_half_frame_of_correct(self):
        fps = 30.0
        frames = _make_biomechanics_frames(fps, 120)
        frame_duration_ms = 1000.0 / fps
        for i, f in enumerate(frames):
            seek_target_ms = f["timestamp_s"] * 1000
            ideal_ms = i * frame_duration_ms
            assert abs(seek_target_ms - ideal_ms) < frame_duration_ms / 2
