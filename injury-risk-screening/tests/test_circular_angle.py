"""
Unit tests for Circular Angle Difference utility.
"""

from __future__ import annotations

import math
import pytest

from src.features.biomechanics import circular_angle_diff


def test_circular_angle_diff_normal():
    # Simple angle difference
    diff = circular_angle_diff(45.0, 30.0)
    assert pytest.approx(diff, 1e-5) == 15.0


def test_circular_angle_diff_boundary_wrap():
    # Boundary wrap: +179 deg to -179 deg is 2 deg difference, not 358 deg
    diff = circular_angle_diff(179.0, -179.0)
    assert pytest.approx(diff, 1e-5) == -2.0

    diff2 = circular_angle_diff(-179.0, 179.0)
    assert pytest.approx(diff2, 1e-5) == 2.0


def test_circular_angle_diff_nan():
    diff = circular_angle_diff(float("nan"), 45.0)
    assert math.isnan(diff)
