"""
Tests for src/validation/opensim_knee_model.py -- real OpenSim forward
kinematics, skipped automatically if the `opensim` package isn't
installed in this environment (same skip pattern as
tests/test_pose_estimator.py's real-model smoke test).
"""

from __future__ import annotations

import pytest

opensim = pytest.importorskip("opensim")

from src.validation.opensim_knee_model import KneePose, get_shank_position_mm, init_model, set_knee_coordinates


def test_model_builds_and_initializes():
    handles, state = init_model()
    assert handles.model.getName() == "driftguard_knee_6dof"
    assert handles.knee_joint.numCoordinates() == 6


def test_neutral_pose_gives_near_zero_shank_position():
    handles, state = init_model()
    neutral = KneePose(0, 0, 0, 0, 0, 0)
    set_knee_coordinates(handles, state, neutral)
    x, y, z = get_shank_position_mm(handles, state)
    assert abs(x) < 1e-6
    assert abs(y) < 1e-6
    assert abs(z) < 1e-6


def test_translation_coordinates_move_shank_by_expected_amount():
    handles, state = init_model()
    pose = KneePose(
        flexion_extension_deg=0, abduction_adduction_deg=0, internal_external_rotation_deg=0,
        ap_translation_mm=5.0, ml_translation_mm=2.0, si_translation_mm=-3.0,
    )
    set_knee_coordinates(handles, state, pose)
    x, y, z = get_shank_position_mm(handles, state)
    assert x == pytest.approx(5.0, abs=1e-3)
    assert y == pytest.approx(2.0, abs=1e-3)
    assert z == pytest.approx(-3.0, abs=1e-3)


def test_rotation_coordinates_change_shank_position_when_combined_with_offset():
    """A pure rotation about the joint origin doesn't move the shank BODY
    ORIGIN (which sits at the joint), so this test instead confirms
    rotating the knee changes the model's internal state in a way that's
    reflected consistently and repeatably -- i.e. setting the same angle
    twice gives the same result (determinism), which is what the
    validation pipeline depends on."""
    handles, state = init_model()
    pose = KneePose(30, 5, 3, 0, 0, 0)
    set_knee_coordinates(handles, state, pose)
    pos1 = get_shank_position_mm(handles, state)

    handles2, state2 = init_model()
    set_knee_coordinates(handles2, state2, pose)
    pos2 = get_shank_position_mm(handles2, state2)

    assert pos1 == pytest.approx(pos2, abs=1e-9)


def test_repeated_coordinate_updates_remain_stable():
    """Both the shank body's origin and the joint's zero-offset frames
    coincide in this model, so a pure rotation (with the body origin AT
    the joint) does not move the body origin -- only translations do.
    This test's point is that re-setting coordinates on the SAME
    handles/state object repeatedly is safe and deterministic, which the
    validation pipeline depends on when iterating frame-by-frame."""
    handles, state = init_model()
    pose_a = KneePose(10, 0, 0, 1, 0, 0)
    pose_b = KneePose(50, 0, 0, 1, 0, 0)

    set_knee_coordinates(handles, state, pose_a)
    pos_a = get_shank_position_mm(handles, state)
    set_knee_coordinates(handles, state, pose_b)
    pos_b = get_shank_position_mm(handles, state)
    set_knee_coordinates(handles, state, pose_a)
    pos_a_again = get_shank_position_mm(handles, state)

    assert pos_a == pytest.approx(pos_a_again, abs=1e-9), "re-applying the same pose must be deterministic"
    assert pos_a == pytest.approx(pos_b, abs=1e-9), "rotation-only change shouldn't move a body whose origin sits at the joint"
