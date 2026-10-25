"""Tests for src/validation/noise_model.py."""

from __future__ import annotations

import numpy as np

from src.validation.motion_generator import generate_reference_trajectory
from src.validation.noise_model import NOISE_PRESETS, inject_noise


def test_inject_noise_preserves_shape_and_columns():
    ref = generate_reference_trajectory("gait", duration_s=1.0, fps=50.0)
    noisy = inject_noise(ref, NOISE_PRESETS["camera"])
    assert noisy.shape == ref.shape
    assert list(noisy.columns) == list(ref.columns)


def test_inject_noise_is_deterministic_with_fixed_seed():
    ref = generate_reference_trajectory("gait", duration_s=1.0, fps=50.0)
    noisy1 = inject_noise(ref, NOISE_PRESETS["camera"], random_state=7)
    noisy2 = inject_noise(ref, NOISE_PRESETS["camera"], random_state=7)
    assert np.allclose(noisy1["flexion_extension_deg"], noisy2["flexion_extension_deg"])


def test_inject_noise_different_seeds_differ():
    ref = generate_reference_trajectory("gait", duration_s=1.0, fps=50.0)
    noisy1 = inject_noise(ref, NOISE_PRESETS["camera"], random_state=1)
    noisy2 = inject_noise(ref, NOISE_PRESETS["camera"], random_state=2)
    assert not np.allclose(noisy1["flexion_extension_deg"], noisy2["flexion_extension_deg"])


def test_zero_noise_profile_leaves_reference_unchanged():
    from src.validation.noise_model import NoiseProfile

    zero_profile = NoiseProfile(
        name="zero", rotational_std_deg=0, rotational_bias_deg=0, rotational_drift_deg_per_s=0,
        translational_std_mm=0, translational_bias_mm=0, translational_drift_mm_per_s=0,
    )
    ref = generate_reference_trajectory("gait", duration_s=1.0, fps=50.0)
    noisy = inject_noise(ref, zero_profile)
    assert np.allclose(noisy["flexion_extension_deg"], ref["flexion_extension_deg"])
    assert np.allclose(noisy["ap_translation_mm"], ref["ap_translation_mm"])


def test_imu_drift_grows_with_time():
    """The IMU preset has a nonzero drift term -- error should grow over
    a longer duration, a real and important property of IMU integration
    drift that a validation pipeline must be able to detect."""
    ref_short = generate_reference_trajectory("gait", duration_s=1.0, fps=50.0)
    ref_long = generate_reference_trajectory("gait", duration_s=10.0, fps=50.0)

    noisy_short = inject_noise(ref_short, NOISE_PRESETS["imu"], random_state=42)
    noisy_long = inject_noise(ref_long, NOISE_PRESETS["imu"], random_state=42)

    late_error_short = abs((noisy_short["ap_translation_mm"] - ref_short["ap_translation_mm"]).iloc[-1])
    late_error_long = abs((noisy_long["ap_translation_mm"] - ref_long["ap_translation_mm"]).iloc[-1])
    assert late_error_long > late_error_short, "drift-induced error should grow with elapsed time"


def test_all_presets_produce_valid_output():
    ref = generate_reference_trajectory("sit_to_stand", duration_s=1.0, fps=30.0)
    for name, profile in NOISE_PRESETS.items():
        noisy = inject_noise(ref, profile)
        assert not noisy.isna().any().any(), f"preset {name} produced NaNs"
