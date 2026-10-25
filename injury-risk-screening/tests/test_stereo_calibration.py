"""Tests for src/validation/stereo_calibration.py.

Proves the triangulation math itself is correct using synthetic cameras
and synthetic 3D points -- see the module docstring for what this does
and does not prove about a real two-camera rig.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.validation.stereo_calibration import (
    make_camera,
    project_point,
    triangulate_point,
    triangulate_trajectory,
)


@pytest.fixture
def frontal_and_sagittal_cameras():
    """A frontal camera (facing the subject along +Z) and a sagittal
    (side-on) camera at 90 degrees -- the standard two-view setup this
    project's abstract describes for multi-camera coverage."""
    frontal = make_camera("frontal", position_m=(0.0, 1.0, -3.0), look_at_m=(0.0, 1.0, 0.0))
    sagittal = make_camera("sagittal", position_m=(3.0, 1.0, 0.0), look_at_m=(0.0, 1.0, 0.0))
    return frontal, sagittal


def test_triangulation_recovers_known_point_exactly(frontal_and_sagittal_cameras):
    frontal, sagittal = frontal_and_sagittal_cameras
    true_point = np.array([0.05, 1.1, 0.02])  # near-knee-height point, meters

    p1 = project_point(true_point, frontal)
    p2 = project_point(true_point, sagittal)
    recovered = triangulate_point(p1, frontal, p2, sagittal)

    assert np.allclose(recovered, true_point, atol=1e-9)


def test_triangulation_recovers_many_points(frontal_and_sagittal_cameras):
    frontal, sagittal = frontal_and_sagittal_cameras
    rng = np.random.default_rng(0)
    points = rng.uniform(low=[-0.3, 0.5, -0.3], high=[0.3, 1.5, 0.3], size=(50, 3))

    for point in points:
        p1 = project_point(point, frontal)
        p2 = project_point(point, sagittal)
        recovered = triangulate_point(p1, frontal, p2, sagittal)
        assert np.allclose(recovered, point, atol=1e-8)


def test_triangulate_trajectory_matches_pointwise(frontal_and_sagittal_cameras):
    frontal, sagittal = frontal_and_sagittal_cameras
    rng = np.random.default_rng(1)
    points = rng.uniform(low=[-0.2, 0.8, -0.2], high=[0.2, 1.3, 0.2], size=(20, 3))

    points_2d_frontal = np.array([project_point(p, frontal) for p in points])
    points_2d_sagittal = np.array([project_point(p, sagittal) for p in points])

    recovered = triangulate_trajectory(points_2d_frontal, frontal, points_2d_sagittal, sagittal)
    assert np.allclose(recovered, points, atol=1e-8)


def test_pixel_noise_degrades_triangulation_accuracy_predictably(frontal_and_sagittal_cameras):
    """Simulates realistic 2D-detection pixel noise (as if from a real
    pose estimator) and shows triangulation error grows with it -- this
    is the honest "sample" proof-of-concept: showing HOW MUCH 3D error a
    given amount of 2D pixel noise produces, not claiming a real rig
    would hit this exact number."""
    frontal, sagittal = frontal_and_sagittal_cameras
    true_point = np.array([0.05, 1.1, 0.02])
    p1 = project_point(true_point, frontal)
    p2 = project_point(true_point, sagittal)

    rng = np.random.default_rng(2)
    errors_small_noise = []
    errors_large_noise = []
    for _ in range(100):
        noisy_p1 = p1 + rng.normal(0, 0.5, size=2)   # 0.5 px std -- good detector
        noisy_p2 = p2 + rng.normal(0, 0.5, size=2)
        recovered = triangulate_point(noisy_p1, frontal, noisy_p2, sagittal)
        errors_small_noise.append(np.linalg.norm(recovered - true_point))

        noisy_p1 = p1 + rng.normal(0, 5.0, size=2)   # 5 px std -- noisy detector
        noisy_p2 = p2 + rng.normal(0, 5.0, size=2)
        recovered = triangulate_point(noisy_p1, frontal, noisy_p2, sagittal)
        errors_large_noise.append(np.linalg.norm(recovered - true_point))

    assert np.mean(errors_small_noise) < np.mean(errors_large_noise)


def test_point_behind_camera_returns_nan():
    camera = make_camera("behind_test", position_m=(0.0, 1.0, -3.0), look_at_m=(0.0, 1.0, 0.0))
    behind_point = np.array([0.0, 1.0, -10.0])  # further behind the camera than its own position
    projected = project_point(behind_point, camera)
    assert np.isnan(projected).any()


def test_degenerate_single_camera_setup_is_not_used_for_triangulation(frontal_and_sagittal_cameras):
    """Sanity check documenting WHY two views are required: a single
    camera's projection alone is a 2D ray, not a 3D point -- two
    different viewing angles are what makes triangulation possible at
    all. (Not a failure case, just documents the geometric reason two
    cameras are needed.)"""
    frontal, _ = frontal_and_sagittal_cameras
    true_point = np.array([0.05, 1.1, 0.02])
    p1 = project_point(true_point, frontal)
    # Using the SAME camera twice (no second viewing angle) should fail
    # to uniquely recover the point -- the system is degenerate.
    with np.errstate(invalid="ignore", divide="ignore"):
        recovered = triangulate_point(p1, frontal, p1, frontal)
    # Recovered point will NOT match the true depth since there's no
    # second vantage point to constrain it.
    assert not np.allclose(recovered, true_point, atol=1e-6)
