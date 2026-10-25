"""Tests for src/validation/trajectory_metrics.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.validation.motion_generator import generate_reference_trajectory
from src.validation.noise_model import NoiseProfile, inject_noise
from src.validation.trajectory_metrics import (
    compute_channel_metrics,
    compute_cycle_consistency,
    validate_trajectory,
)

ZERO_NOISE = NoiseProfile(
    name="zero", rotational_std_deg=0, rotational_bias_deg=0, rotational_drift_deg_per_s=0,
    translational_std_mm=0, translational_bias_mm=0, translational_drift_mm_per_s=0,
)
SMALL_NOISE = NoiseProfile(
    name="small", rotational_std_deg=0.5, rotational_bias_deg=0.1, rotational_drift_deg_per_s=0,
    translational_std_mm=0.5, translational_bias_mm=0.1, translational_drift_mm_per_s=0,
)
HUGE_NOISE = NoiseProfile(
    name="huge", rotational_std_deg=20, rotational_bias_deg=10, rotational_drift_deg_per_s=0,
    translational_std_mm=50, translational_bias_mm=20, translational_drift_mm_per_s=0,
)


def test_identical_trajectories_have_zero_error():
    ref = generate_reference_trajectory("gait", duration_s=1.0, fps=50.0)
    report = validate_trajectory(ref, ref.copy())
    assert report.overall_rotational_rmse_deg == pytest.approx(0.0, abs=1e-9)
    assert report.overall_translational_rmse_mm == pytest.approx(0.0, abs=1e-9)
    assert report.passes_overall is True


def test_small_noise_passes_default_target():
    ref = generate_reference_trajectory("gait", duration_s=1.0, fps=50.0)
    estimate = inject_noise(ref, SMALL_NOISE, random_state=1)
    report = validate_trajectory(ref, estimate)
    assert report.passes_rotational_target is True
    assert report.passes_translational_target is True


def test_huge_noise_fails_default_target():
    ref = generate_reference_trajectory("gait", duration_s=1.0, fps=50.0)
    estimate = inject_noise(ref, HUGE_NOISE, random_state=1)
    report = validate_trajectory(ref, estimate)
    assert report.passes_rotational_target is False
    assert report.passes_translational_target is False
    assert report.passes_overall is False


def test_mismatched_lengths_raise():
    ref = generate_reference_trajectory("gait", duration_s=1.0, fps=50.0)
    short_estimate = ref.iloc[:10].copy()
    with pytest.raises(ValueError):
        validate_trajectory(ref, short_estimate)


def test_channel_metrics_cover_all_six_dof():
    ref = generate_reference_trajectory("gait", duration_s=1.0, fps=50.0)
    estimate = inject_noise(ref, SMALL_NOISE, random_state=1)
    metrics = compute_channel_metrics(ref, estimate)
    channels = {m.channel for m in metrics}
    assert channels == {
        "flexion_extension_deg", "abduction_adduction_deg", "internal_external_rotation_deg",
        "ap_translation_mm", "ml_translation_mm", "si_translation_mm",
    }
    for m in metrics:
        assert m.max_error >= m.mae >= 0
        assert m.rmse >= 0


def test_cycle_consistency_zero_for_single_cycle():
    ref = generate_reference_trajectory("gait", duration_s=1.0, fps=50.0)
    estimate = inject_noise(ref, SMALL_NOISE, random_state=1)
    consistency = compute_cycle_consistency(ref, estimate, n_cycles=1)
    assert consistency == 0.0


def test_cycle_consistency_detects_increasing_drift_across_cycles():
    """Concatenate 3 identical reference cycles, but inject IMU-style
    drifting noise (so later cycles have larger mean error than earlier
    ones) -- cycle consistency should be nonzero, correctly detecting
    the inconsistency."""
    from src.validation.noise_model import NOISE_PRESETS

    cycles = []
    for i in range(3):
        c = generate_reference_trajectory("gait", duration_s=1.0, fps=50.0)
        c["time_s"] = c["time_s"] + i * 1.0
        cycles.append(c)
    ref = pd.concat(cycles, ignore_index=True)
    estimate = inject_noise(ref, NOISE_PRESETS["imu"], random_state=1)

    consistency = compute_cycle_consistency(ref, estimate, n_cycles=3)
    assert consistency > 0.0
