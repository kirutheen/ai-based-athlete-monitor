"""Tests for src/capture/multi_person_tracker.py -- including a test that
documents its known limitation (ID switch on crossing paths) rather than
hiding it."""

from __future__ import annotations

import numpy as np

from src.capture.multi_person_tracker import MultiPersonTracker


def _pose_at(x: float, y: float) -> np.ndarray:
    """A (33,4) landmark array centered at (x, y), all visible."""
    lm = np.zeros((33, 4), dtype=np.float64)
    lm[:, 0] = x
    lm[:, 1] = y
    lm[:, 3] = 1.0
    return lm


def test_two_well_separated_people_keep_stable_ids_across_frames():
    tracker = MultiPersonTracker()

    # Frame 1: person A at (0.2, 0.5), person B at (0.8, 0.5)
    result1 = tracker.update([(_pose_at(0.2, 0.5), 0.9), (_pose_at(0.8, 0.5), 0.9)])
    id_a, id_b = result1[0].track_id, result1[1].track_id
    assert id_a != id_b

    # Frame 2: both move slightly but stay well separated and don't cross
    result2 = tracker.update([(_pose_at(0.22, 0.5), 0.9), (_pose_at(0.78, 0.5), 0.9)])
    assert result2[0].track_id == id_a
    assert result2[1].track_id == id_b

    # Frame 3: same again
    result3 = tracker.update([(_pose_at(0.25, 0.5), 0.9), (_pose_at(0.75, 0.5), 0.9)])
    assert result3[0].track_id == id_a
    assert result3[1].track_id == id_b


def test_new_person_entering_gets_a_new_id():
    tracker = MultiPersonTracker()
    result1 = tracker.update([(_pose_at(0.2, 0.5), 0.9)])
    id_a = result1[0].track_id

    result2 = tracker.update([(_pose_at(0.22, 0.5), 0.9), (_pose_at(0.9, 0.5), 0.9)])
    ids = {r.track_id for r in result2}
    assert id_a in ids
    assert len(ids) == 2


def test_track_survives_a_brief_missed_detection():
    tracker = MultiPersonTracker(max_missed_frames=3)
    result1 = tracker.update([(_pose_at(0.5, 0.5), 0.9)])
    id_a = result1[0].track_id

    tracker.update([])  # missed one frame (e.g. brief occlusion)
    tracker.update([])  # missed another

    result2 = tracker.update([(_pose_at(0.5, 0.5), 0.9)])
    assert result2[0].track_id == id_a, "a brief gap should not spawn a new ID"


def test_track_is_dropped_after_too_many_missed_frames():
    tracker = MultiPersonTracker(max_missed_frames=2)
    result1 = tracker.update([(_pose_at(0.5, 0.5), 0.9)])
    id_a = result1[0].track_id

    for _ in range(5):
        tracker.update([])

    result2 = tracker.update([(_pose_at(0.5, 0.5), 0.9)])
    assert result2[0].track_id != id_a, "track should have been dropped and reassigned a new ID"


def test_KNOWN_LIMITATION_crossing_paths_can_swap_ids():
    """
    This test documents the tracker's known, accepted limitation rather
    than hiding it: two people on a direct collision course that pass
    through (approximately) the same point WILL have their identities
    swapped by this centroid-only tracker, because at the crossing frame
    both detections are equidistant from both tracks and greedy matching
    has no appearance information to break the tie correctly.

    This is the exact "ID switch in a crowd" failure mode this project
    has always said a 2D, non-appearance-based tracker cannot solve.
    This test exists so that limitation is enforced and visible in CI,
    not quietly forgotten.
    """
    tracker = MultiPersonTracker()
    result1 = tracker.update([(_pose_at(0.2, 0.5), 0.9), (_pose_at(0.8, 0.5), 0.9)])
    id_a_start, id_b_start = result1[0].track_id, result1[1].track_id

    # Both walk toward each other and cross at the midpoint.
    result2 = tracker.update([(_pose_at(0.5, 0.5), 0.9), (_pose_at(0.5, 0.5), 0.9)])
    # After crossing, A continues to where B started and vice versa.
    result3 = tracker.update([(_pose_at(0.8, 0.5), 0.9), (_pose_at(0.2, 0.5), 0.9)])

    ids_after = {result3[0].track_id, result3[1].track_id}
    # Both original IDs still exist (no track was dropped)...
    assert ids_after == {id_a_start, id_b_start}
    # ...but WHICH physical person holds which ID is no longer
    # guaranteed to match frame 1 -- this assertion intentionally is not
    # "the ids are correct", because they may not be. The point of this
    # test is that the swap is a known, reproducible, accepted behavior,
    # not a crash or undefined state.
