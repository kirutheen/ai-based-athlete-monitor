"""
Infrared thermography (IRT) capture for non-invasive knee-temperature /
inflammation monitoring.

============================= HONEST FLAG =================================
Like src/sensors/imu_fusion.py, this is an interface + processing-logic
skeleton, not a tested working implementation, because:

1. The actual frame-capture call (`capture_frame` below) depends on the
   specific thermal camera's SDK/driver (e.g. a FLIR Lepton module talks
   over SPI/I2C to a microcontroller or a USB-attached breakout board
   with its own vendor library) -- there is no hardware-agnostic way to
   write this without knowing the exact part purchased, and no way to
   test it without the part in hand.
2. The bilateral-asymmetry threshold (>0.5C) is a real, published
   physiological reference point for inflammation asymmetry, but
   converting a thermal camera's raw radiometric output into an accurate
   skin-surface temperature in Celsius requires per-device calibration
   (emissivity setting, distance compensation, ambient temperature
   compensation) that can only be done against the real sensor.

What IS real below: the region-of-interest extraction logic (given an
already-captured thermal frame as a plain 2D numpy array of Celsius
values, finding the knee region using the pose landmarks from
src/pose/pose_estimator.py to localize where to sample) and the
asymmetry-flagging logic, which only need calibrated temperature data as
input -- so this module is ready to receive real sensor data the moment
hardware arrives, without further design work.
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BILATERAL_ASYMMETRY_THRESHOLD_C = 0.5  # physiologically notable, per published IRT literature


@dataclass
class ThermalFrame:
    """A thermal camera frame already converted to calibrated Celsius
    values per pixel. Producing this from a raw sensor reading is the
    hardware-dependent part (see capture_frame below); everything that
    consumes a ThermalFrame is hardware-agnostic and testable today."""

    temperature_c: np.ndarray  # (H, W) float array, degrees Celsius
    width: int
    height: int


class ThermalCameraDriver:
    """
    Vendor-specific driver interface. A real implementation subclasses
    this for the specific thermal camera purchased (e.g.
    `FlirLeptonDriver(ThermalCameraDriver)`), implementing capture_frame
    using that vendor's SDK.
    """

    def capture_frame(self) -> ThermalFrame:
        raise NotImplementedError(
            "No thermal camera hardware/SDK connected. Subclass ThermalCameraDriver with "
            "the real device's vendor SDK once hardware is purchased -- see module docstring "
            "for why this can't be written generically in advance."
        )


def extract_knee_roi_temperature(
    frame: ThermalFrame,
    knee_pixel_x: int,
    knee_pixel_y: int,
    roi_radius_px: int = 15,
) -> float:
    """
    Mean temperature (Celsius) in a square region of interest around a
    knee's pixel location in an ALREADY-CALIBRATED thermal frame.

    knee_pixel_x/y should come from projecting the pose landmarker's
    knee landmark (src/pose/pose_estimator.py, LM.LEFT_KNEE /
    LM.RIGHT_KNEE) into the thermal camera's pixel space -- which
    requires the thermal and RGB cameras to be extrinsically calibrated
    (a fixed, known offset/rotation between the two lenses, established
    once at setup time with a calibration checkerboard visible to both
    cameras). That extrinsic calibration is a physical setup step, not
    something this function can do; it assumes knee_pixel_x/y are
    already correct for this thermal camera's frame.

    This function itself needs no hardware and is fully testable with a
    synthetic ThermalFrame (see tests/test_thermal_capture.py).
    """
    y0, y1 = max(0, knee_pixel_y - roi_radius_px), min(frame.height, knee_pixel_y + roi_radius_px)
    x0, x1 = max(0, knee_pixel_x - roi_radius_px), min(frame.width, knee_pixel_x + roi_radius_px)
    roi = frame.temperature_c[y0:y1, x0:x1]
    if roi.size == 0:
        raise ValueError("ROI is empty -- knee pixel coordinates are outside the thermal frame")
    return float(np.mean(roi))


@dataclass
class BilateralAsymmetryResult:
    left_temp_c: float
    right_temp_c: float
    asymmetry_c: float
    notable: bool  # True if asymmetry exceeds BILATERAL_ASYMMETRY_THRESHOLD_C


def compute_bilateral_asymmetry(left_temp_c: float, right_temp_c: float) -> BilateralAsymmetryResult:
    """
    Compare left/right knee temperatures from an already-calibrated
    thermal frame. A notable asymmetry (>0.5C) is a real, published
    inflammation signal but is NOT itself a diagnosis -- the output here
    should be surfaced to a clinician as one more data point alongside
    the FPPA/flexion flags, through the same rule_explainer.py pattern,
    never as a standalone automated "this knee is inflamed" claim.
    """
    asymmetry = abs(left_temp_c - right_temp_c)
    return BilateralAsymmetryResult(
        left_temp_c=left_temp_c,
        right_temp_c=right_temp_c,
        asymmetry_c=asymmetry,
        notable=asymmetry > BILATERAL_ASYMMETRY_THRESHOLD_C,
    )
