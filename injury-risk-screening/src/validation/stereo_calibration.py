"""
Two-camera (stereo) calibration and triangulation -- real photogrammetry
math (the Direct Linear Transform method), validated here against
SYNTHETIC camera views of a known 3D point, not real camera footage.

============================= HONEST SCOPE =================================
This answers the "multi-camera calibration" gap honestly: the
TRIANGULATION MATH is real, standard, and testable without any physical
camera -- you can prove the algorithm correctly recovers a 3D point from
two 2D projections using nothing but arithmetic. That's what this module
and its tests do.

What this does NOT do:
  - It does not calibrate two REAL cameras. Real calibration requires
    photographing a real checkerboard pattern with your actual two
    cameras and solving for their real intrinsic/extrinsic parameters
    (typically via OpenCV's `cv2.calibrateCamera` /
    `cv2.stereoCalibrate`) -- that step needs the physical cameras in
    hand and cannot be simulated meaningfully; a synthetic calibration
    would just be checking the math against itself.
  - Real camera lenses have distortion (radial/tangential) that a real
    calibration corrects for. This module assumes an idealized pinhole
    camera with no distortion -- fine for proving the triangulation
    math, not sufficient for a real deployment without adding distortion
    correction (OpenCV provides this once real calibration images
    exist).
  - It does not say anything about how well this would perform on real
    pose-estimation 2D points, whose noise characteristics differ from
    the synthetic Gaussian pixel noise used in the tests.

How to turn this into a real two-camera setup once you have the
hardware: print a checkerboard, photograph it from both cameras
simultaneously at various angles, run `cv2.stereoCalibrate` on those
image pairs to get real `CameraParams` for each camera (replacing the
synthetic ones constructed in tests), and everything else in this module
(`triangulate_point`) works unchanged on the resulting real parameters.
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraParams:
    """Idealized pinhole camera: a 3x3 intrinsic matrix (focal lengths +
    principal point) plus a rotation and translation placing it in a
    shared world coordinate frame."""

    name: str
    focal_length_px: float
    principal_point_px: tuple[float, float]
    rotation: np.ndarray  # 3x3, world-to-camera
    translation: np.ndarray  # (3,), world-to-camera, in meters

    def intrinsic_matrix(self) -> np.ndarray:
        fx = fy = self.focal_length_px
        cx, cy = self.principal_point_px
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    def projection_matrix(self) -> np.ndarray:
        """3x4 camera projection matrix P = K [R | t], mapping a
        homogeneous world point to a homogeneous image point."""
        K = self.intrinsic_matrix()
        Rt = np.hstack([self.rotation, self.translation.reshape(3, 1)])
        return K @ Rt


def make_camera(
    name: str,
    position_m: tuple[float, float, float],
    look_at_m: tuple[float, float, float],
    focal_length_px: float = 1000.0,
    principal_point_px: tuple[float, float] = (640.0, 360.0),
) -> CameraParams:
    """
    Build a CameraParams for a camera at `position_m` looking toward
    `look_at_m`, with a conventional "up" of +Y. Used to construct the
    two synthetic cameras (e.g. frontal + sagittal) for testing.
    """
    position = np.array(position_m, dtype=np.float64)
    target = np.array(look_at_m, dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0])

    forward = target - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    # World-to-camera rotation: camera's local axes (right, -up, forward)
    # expressed as rows, matching standard computer-vision convention
    # (x-right, y-down, z-forward).
    R = np.vstack([right, -true_up, forward])
    t = -R @ position

    return CameraParams(
        name=name,
        focal_length_px=focal_length_px,
        principal_point_px=principal_point_px,
        rotation=R,
        translation=t,
    )


def project_point(point_3d_m: np.ndarray, camera: CameraParams) -> np.ndarray:
    """Project a 3D world point (meters) into this camera's 2D pixel
    coordinates. Returns NaN if the point is behind the camera."""
    P = camera.projection_matrix()
    homog = np.append(point_3d_m, 1.0)
    projected = P @ homog
    if projected[2] <= 0:
        return np.array([np.nan, np.nan])
    return projected[:2] / projected[2]


def triangulate_point(
    point_2d_cam1: np.ndarray, camera1: CameraParams, point_2d_cam2: np.ndarray, camera2: CameraParams
) -> np.ndarray:
    """
    Reconstruct a 3D world point (meters) from its 2D pixel observation
    in two calibrated cameras, via the Direct Linear Transform (DLT):
    for each camera, the projection equation gives two linear
    constraints on the unknown 3D point; stack all 4 constraints (2 per
    camera) and solve via SVD for the least-squares 3D point.
    """
    P1 = camera1.projection_matrix()
    P2 = camera2.projection_matrix()

    def rows_for(point_2d: np.ndarray, P: np.ndarray) -> np.ndarray:
        x, y = point_2d
        return np.vstack([x * P[2] - P[0], y * P[2] - P[1]])

    A = np.vstack([rows_for(point_2d_cam1, P1), rows_for(point_2d_cam2, P2)])
    _, _, vt = np.linalg.svd(A)
    homog_point = vt[-1]
    return homog_point[:3] / homog_point[3]


def triangulate_trajectory(
    points_2d_cam1: np.ndarray, camera1: CameraParams, points_2d_cam2: np.ndarray, camera2: CameraParams
) -> np.ndarray:
    """Vectorized convenience wrapper: triangulate_point for each row of
    two (N, 2) pixel-coordinate arrays, returning an (N, 3) array of 3D points."""
    n = len(points_2d_cam1)
    out = np.zeros((n, 3))
    for i in range(n):
        out[i] = triangulate_point(points_2d_cam1[i], camera1, points_2d_cam2[i], camera2)
    return out
