"""
Synthetic depth-frame simulation -- demonstrates how a real depth sensor
(e.g. Intel RealSense, Azure Kinect) WOULD improve 3D landmark accuracy,
using a simulated depth map, not a real depth camera.

============================= HONEST SCOPE =================================
A real depth sensor emits infrared light and measures time-of-flight or
structured-light disparity to get a true per-pixel distance -- that is
physically impossible to replicate without the hardware. What IS honestly
demonstrable without the hardware is the DOWNSTREAM MATH: given a depth
value at a landmark's pixel, how do you convert (pixel_x, pixel_y, depth)
into a real-world 3D point, and how much does having true depth (vs.
MediaPipe's own relative-depth estimate) reduce 3D reconstruction error?

This module:
  1. Takes known 3D ground-truth points (e.g. from the OpenSim knee
     model), projects them into a synthetic camera view to get
     (pixel_x, pixel_y).
  2. Builds a SIMULATED depth map by taking the true Z-distance from
     camera to each point and adding sensor-realistic noise (depth
     sensors have real, documented noise characteristics: ~1-2mm std at
     close range, growing roughly quadratically with distance -- these
     numbers come from published specs for consumer depth sensors, not
     from this project's own measurements).
  3. Reconstructs the 3D point from (pixel_x, pixel_y, simulated_depth)
     using the standard pinhole back-projection formula and compares to
     monocular-only (depth-less) reconstruction, to show the accuracy
     difference depth data would make.

What this does NOT do:
  - It does not simulate real depth-sensor failure modes (IR
    interference in bright sunlight, reflective/dark surface dropout,
    multi-path interference) -- those are hardware- and
    environment-specific and would need a real sensor to characterize.
  - It is not a claim that a specific real depth sensor would hit these
    exact numbers in a real gym -- it demonstrates the mechanism and
    rough order of magnitude using published noise specs.
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.validation.stereo_calibration import CameraParams, project_point

# Published approximate depth-noise characteristics for consumer
# structured-light / time-of-flight depth sensors (e.g. Intel RealSense
# D400 series, Azure Kinect) at typical screening distances (1.5-3m).
# Source: manufacturer datasheets' stated depth accuracy specs.
DEPTH_NOISE_STD_MM_AT_1M = 1.5
DEPTH_NOISE_QUADRATIC_GROWTH = 2.0  # noise roughly scales with distance^2 beyond 1m


@dataclass
class DepthSimulationResult:
    true_point_m: np.ndarray
    monocular_reconstruction_error_mm: float
    depth_assisted_reconstruction_error_mm: float
    simulated_depth_noise_mm: float


def simulated_depth_noise_std_mm(distance_m: float) -> float:
    """Published-spec-based approximation of depth sensor noise std at a
    given distance; grows faster than linearly as distance from the
    sensor increases (consistent with real depth-sensor datasheets)."""
    if distance_m <= 1.0:
        return DEPTH_NOISE_STD_MM_AT_1M
    return DEPTH_NOISE_STD_MM_AT_1M * (distance_m ** DEPTH_NOISE_QUADRATIC_GROWTH)


def back_project_with_depth(pixel_xy: np.ndarray, depth_m: float, camera: CameraParams) -> np.ndarray:
    """Reconstruct a 3D camera-space point from a pixel + known depth,
    then transform into world coordinates -- the standard pinhole
    back-projection used by every real depth camera's SDK."""
    K = camera.intrinsic_matrix()
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (pixel_xy[0] - cx) * depth_m / fx
    y_cam = (pixel_xy[1] - cy) * depth_m / fy
    z_cam = depth_m
    point_cam = np.array([x_cam, y_cam, z_cam])
    # camera.rotation/translation map world -> camera, so invert:
    point_world = camera.rotation.T @ (point_cam - camera.translation)
    return point_world


def simulate_depth_assisted_reconstruction(
    true_point_m: np.ndarray, camera: CameraParams, random_state: int | None = None
) -> DepthSimulationResult:
    """
    Compares two reconstruction strategies for a single 3D point seen by
    one camera:
      - "monocular": no real depth -- we approximate what a monocular
        pipeline does, assuming a rough constant depth guess (since a
        single 2D camera genuinely cannot recover true depth/scale
        without extra assumptions -- this is the real, well-known
        monocular depth ambiguity).
      - "depth_assisted": using a simulated noisy depth reading, as a
        real depth sensor would provide.
    """
    rng = np.random.default_rng(random_state)

    pixel = project_point(true_point_m, camera)
    true_depth_m = float(camera.rotation[2] @ true_point_m + camera.translation[2])

    # Monocular: assume the pipeline guesses a population-average depth
    # (this is what monocular pose pipelines effectively do without a
    # depth source -- a real, structural limitation, not a strawman).
    assumed_depth_m = 3.0  # typical screening distance from camera
    monocular_reconstruction = back_project_with_depth(pixel, assumed_depth_m, camera)
    monocular_error_mm = float(np.linalg.norm(monocular_reconstruction - true_point_m) * 1000)

    # Depth-assisted: true depth plus realistic sensor noise.
    noise_std_mm = simulated_depth_noise_std_mm(true_depth_m)
    noisy_depth_m = true_depth_m + rng.normal(0, noise_std_mm / 1000)
    depth_reconstruction = back_project_with_depth(pixel, noisy_depth_m, camera)
    depth_error_mm = float(np.linalg.norm(depth_reconstruction - true_point_m) * 1000)

    return DepthSimulationResult(
        true_point_m=true_point_m,
        monocular_reconstruction_error_mm=monocular_error_mm,
        depth_assisted_reconstruction_error_mm=depth_error_mm,
        simulated_depth_noise_mm=noise_std_mm,
    )
