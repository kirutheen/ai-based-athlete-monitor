"""Tests for the hardware-agnostic parts of src/sensors/thermal_capture.py.

Does NOT test ThermalCameraDriver.capture_frame (that's the hardware-
dependent stub, which correctly raises NotImplementedError -- tested
below as a guard so no one silently removes that honesty check)."""

from __future__ import annotations

import numpy as np
import pytest

from src.sensors.thermal_capture import (
    ThermalCameraDriver,
    ThermalFrame,
    compute_bilateral_asymmetry,
    extract_knee_roi_temperature,
)


def test_capture_frame_stub_raises_not_implemented():
    driver = ThermalCameraDriver()
    with pytest.raises(NotImplementedError):
        driver.capture_frame()


def test_extract_knee_roi_temperature_averages_region():
    temps = np.full((100, 100), 30.0)
    temps[40:50, 40:50] = 34.0  # a warmer patch around the "knee"
    frame = ThermalFrame(temperature_c=temps, width=100, height=100)
    result = extract_knee_roi_temperature(frame, knee_pixel_x=45, knee_pixel_y=45, roi_radius_px=5)
    assert result == pytest.approx(34.0, abs=0.5)


def test_extract_knee_roi_temperature_rejects_out_of_bounds():
    temps = np.full((10, 10), 30.0)
    frame = ThermalFrame(temperature_c=temps, width=10, height=10)
    with pytest.raises(ValueError):
        extract_knee_roi_temperature(frame, knee_pixel_x=500, knee_pixel_y=500, roi_radius_px=5)


def test_bilateral_asymmetry_flags_above_threshold():
    result = compute_bilateral_asymmetry(left_temp_c=31.0, right_temp_c=30.3)
    assert result.asymmetry_c == pytest.approx(0.7)
    assert result.notable is True


def test_bilateral_asymmetry_not_notable_when_small():
    result = compute_bilateral_asymmetry(left_temp_c=30.2, right_temp_c=30.0)
    assert result.notable is False
