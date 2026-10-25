"""Tests for src/validation/depth_simulation.py."""

from __future__ import annotations

import numpy as np

from src.validation.depth_simulation import (
    back_project_with_depth,
    simulate_depth_assisted_reconstruction,
    simulated_depth_noise_std_mm,
)
from src.validation.stereo_calibration import make_camera, project_point


def _camera():
    return make_camera("frontal", position_m=(0.0, 1.0, -3.0), look_at_m=(0.0, 1.0, 0.0))


def test_back_projection_recovers_point_given_true_depth():
    camera = _camera()
    true_point = np.array([0.05, 1.1, 0.02])
    pixel = project_point(true_point, camera)
    true_depth = float(camera.rotation[2] @ true_point + camera.translation[2])

    recovered = back_project_with_depth(pixel, true_depth, camera)
    assert np.allclose(recovered, true_point, atol=1e-6)


def test_depth_noise_grows_with_distance():
    near = simulated_depth_noise_std_mm(1.0)
    far = simulated_depth_noise_std_mm(4.0)
    assert far > near


def test_depth_assisted_reconstruction_beats_monocular_guess():
    """The whole point of the comparison: a depth sensor (even a noisy
    one) should reconstruct a point far more accurately than a
    monocular pipeline guessing a population-average depth, whenever
    the true point's depth differs meaningfully from that guess."""
    camera = _camera()
    true_point = np.array([0.05, 1.1, 1.5])  # true depth very different from assumed 3.0m guess

    result = simulate_depth_assisted_reconstruction(true_point, camera, random_state=0)
    assert result.depth_assisted_reconstruction_error_mm < result.monocular_reconstruction_error_mm


def test_depth_assisted_error_is_small_and_bounded():
    camera = _camera()
    # Point close to the camera (true depth ~1.5m from the camera along
    # its forward axis) so the simulated depth-noise std stays small --
    # consistent with the published close-range accuracy specs this
    # module uses.
    true_point = np.array([0.0, 1.0, -1.5])
    errors = []
    for seed in range(20):
        result = simulate_depth_assisted_reconstruction(true_point, camera, random_state=seed)
        errors.append(result.depth_assisted_reconstruction_error_mm)
    assert max(errors) < 50.0


def test_result_reports_the_simulated_noise_used():
    camera = _camera()
    true_point = np.array([0.0, 1.0, 2.0])
    result = simulate_depth_assisted_reconstruction(true_point, camera, random_state=1)
    assert result.simulated_depth_noise_mm > 0
