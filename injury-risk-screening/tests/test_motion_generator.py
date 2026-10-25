"""Tests for src/validation/motion_generator.py."""

from __future__ import annotations

import pytest

from src.validation.motion_generator import MOVEMENT_TYPES, generate_reference_trajectory, trajectory_row_to_pose


@pytest.mark.parametrize("movement_type", MOVEMENT_TYPES)
def test_generate_reference_trajectory_has_expected_shape(movement_type):
    df = generate_reference_trajectory(movement_type, duration_s=2.0, fps=100.0)
    assert len(df) == 200
    for col in (
        "time_s", "flexion_extension_deg", "abduction_adduction_deg",
        "internal_external_rotation_deg", "ap_translation_mm", "ml_translation_mm", "si_translation_mm",
    ):
        assert col in df.columns
    assert df["time_s"].is_monotonic_increasing


def test_generate_reference_trajectory_rejects_unknown_movement():
    with pytest.raises(ValueError):
        generate_reference_trajectory("tap_dance")


def test_gait_flexion_stays_in_plausible_clinical_range():
    df = generate_reference_trajectory("gait", duration_s=1.0, fps=100.0)
    # Normal-speed gait knee flexion is commonly reported roughly within
    # 0-65 degrees; this is a sanity bound on the simulated profile, not
    # a claim this matches any specific real patient's gait.
    assert df["flexion_extension_deg"].min() >= -1
    assert df["flexion_extension_deg"].max() <= 70


def test_sit_to_stand_flexion_decreases_overall():
    df = generate_reference_trajectory("sit_to_stand", duration_s=2.0, fps=50.0)
    assert df["flexion_extension_deg"].iloc[0] > df["flexion_extension_deg"].iloc[-1]


def test_trajectory_row_to_pose_roundtrip():
    df = generate_reference_trajectory("gait", duration_s=1.0, fps=10.0)
    pose = trajectory_row_to_pose(df.iloc[5])
    assert pose.flexion_extension_deg == pytest.approx(df.iloc[5]["flexion_extension_deg"])
    assert pose.si_translation_mm == pytest.approx(df.iloc[5]["si_translation_mm"])
