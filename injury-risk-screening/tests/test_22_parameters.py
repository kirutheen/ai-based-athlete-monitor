"""
Comprehensive test suite for Phase 5: 22 Biomechanical Parameters Specification,
Calculation Engine, Level 2 Gating, and API Endpoints.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.features.biomechanics import fppa, knee_flexion_angle
from src.features.landmarks import LM, ALL_LANDMARK_NAMES
from src.features.parameters_22 import (
    PARAMETERS_22_SPECS,
    PARAMETERS_22_MAP,
    compute_22_parameters,
    get_parameter_specs,
)
from src.features.event_segmentation import LandingEvent

client = TestClient(app)


def _build_test_frame(vis: float = 0.9) -> np.ndarray:
    """Helper to build a 33x4 landmark array with realistic 2D coordinates."""
    lm = np.zeros((33, 4), dtype=np.float64)
    lm[:, 3] = vis

    lm[LM.LEFT_SHOULDER] = [0.4, 0.2, 0.0, vis]
    lm[LM.RIGHT_SHOULDER] = [0.6, 0.2, 0.0, vis]
    lm[LM.LEFT_HIP] = [0.42, 0.5, 0.0, vis]
    lm[LM.RIGHT_HIP] = [0.58, 0.5, 0.0, vis]
    lm[LM.LEFT_KNEE] = [0.43, 0.7, 0.0, vis]
    lm[LM.RIGHT_KNEE] = [0.57, 0.7, 0.0, vis]
    lm[LM.LEFT_ANKLE] = [0.42, 0.9, 0.0, vis]
    lm[LM.RIGHT_ANKLE] = [0.58, 0.9, 0.0, vis]
    lm[LM.LEFT_HEEL] = [0.42, 0.92, 0.0, vis]
    lm[LM.RIGHT_HEEL] = [0.58, 0.92, 0.0, vis]
    lm[LM.LEFT_FOOT_INDEX] = [0.42, 0.96, 0.0, vis]
    lm[LM.RIGHT_FOOT_INDEX] = [0.58, 0.96, 0.0, vis]

    return lm


def test_22_parameters_registry_completeness():
    """Verify exactly 22 parameters exist, numbered 1 to 22 with unique names."""
    assert len(PARAMETERS_22_SPECS) == 22
    param_ids = [s.param_id for s in PARAMETERS_22_SPECS]
    assert param_ids == list(range(1, 23))

    expected_names = [
        "fppa_right",
        "knee_flexion_left",
        "knee_flexion_right",
        "hip_flexion_left",
        "hip_flexion_right",
        "hip_abduction_left",
        "hip_abduction_right",
        "ankle_dorsiflexion_left",
        "ankle_dorsiflexion_right",
        "trunk_lean",
        "trunk_flexion_deg",
        "pelvic_obliquity_deg",
        "knee_valgus_LSI",
        "knee_flexion_LSI",
        "knee_valgus_at_initial_contact",
        "knee_flexion_at_initial_contact",
        "valgus_duration",
        "com_displacement",
        "knee_ap_translation_mm",
        "knee_ml_translation_mm",
        "knee_si_translation_mm",
        "knee_int_ext_rot_deg",
    ]
    actual_names = [s.name for s in PARAMETERS_22_SPECS]
    assert actual_names == expected_names


def test_classification_labels_exact_match():
    """Every parameter must have a classification matching measured, estimate, proxy, or model-derived."""
    valid_classes = {"measured", "estimate", "proxy", "model-derived"}
    for spec in PARAMETERS_22_SPECS:
        assert spec.classification in valid_classes, f"Invalid classification for {spec.name}: {spec.classification}"

    assert PARAMETERS_22_MAP["valgus_duration"].classification == "measured"
    assert PARAMETERS_22_MAP["fppa_right"].classification == "proxy"
    assert PARAMETERS_22_MAP["knee_flexion_left"].classification == "estimate"
    assert PARAMETERS_22_MAP["knee_valgus_LSI"].classification == "model-derived"
    assert PARAMETERS_22_MAP["knee_ap_translation_mm"].classification == "estimate"
    assert PARAMETERS_22_MAP["knee_int_ext_rot_deg"].classification == "proxy"


def test_level_2_gating_returns_not_available_without_depth():
    """Level 2 parameters (19-22) must strictly return 'not available' on monocular video without depth/calibration."""
    df = pd.DataFrame({
        "fppa_left": [5.0, 6.0, 7.0],
        "fppa_right": [4.0, 5.0, 6.0],
        "knee_flexion_left": [30.0, 45.0, 60.0],
        "knee_flexion_right": [32.0, 44.0, 58.0],
        "trunk_lean": [3.0, 4.0, 3.5],
        "knee_ap_translation_mm": [4.0, 4.5, 4.2],  # Even if arbitrary numbers present in DF
    })

    # Call with has_depth_input=False and has_stereo_calibration=False
    results = compute_22_parameters(df, has_depth_input=False, has_stereo_calibration=False)

    for level2_name in ["knee_ap_translation_mm", "knee_ml_translation_mm", "knee_si_translation_mm", "knee_int_ext_rot_deg"]:
        res = results[level2_name]
        assert res["level"] == 2
        assert res["status"] == "not available"
        assert res["value"] == "not available"
        assert "depth sensor" in res["notes"].lower() or "calibration" in res["notes"].lower()


def test_level_2_can_return_values_when_depth_or_stereo_available():
    """When depth or stereo calibration is available, Level 2 parameters report available status."""
    df = pd.DataFrame({
        "knee_ap_translation_mm": [3.5, 4.0, 4.5],
        "knee_ml_translation_mm": [1.5, 2.0, 2.2],
        "knee_si_translation_mm": [1.0, 1.2, 1.4],
        "knee_int_ext_rot_deg": [5.0, 6.0, 5.5],
    })

    results = compute_22_parameters(df, has_depth_input=True)
    assert results["knee_ap_translation_mm"]["status"] == "available"
    assert results["knee_ap_translation_mm"]["value"] == pytest.approx(4.0, abs=0.1)


def test_missing_or_nan_landmarks_do_not_fabricate_values():
    """Missing or low-confidence data must produce 'not available' or NaN, never a fabricated default."""
    empty_df = pd.DataFrame()
    results = compute_22_parameters(empty_df, events=None, has_depth_input=False)

    assert results["fppa_right"]["value"] == "not available"
    assert results["fppa_right"]["status"] == "not available"
    assert results["knee_flexion_left"]["value"] == "not available"
    assert results["pelvic_obliquity_deg"]["value"] == "not available"


def test_fppa_left_preservation():
    """Verify that original fppa_left logic and output are completely preserved."""
    lm = _build_test_frame(vis=1.0)
    lm[LM.LEFT_HIP] = [0.42, 0.5, 0.0, 1.0]
    lm[LM.LEFT_KNEE] = [0.42, 0.7, 0.0, 1.0]
    lm[LM.LEFT_ANKLE] = [0.42, 0.9, 0.0, 1.0]
    lm[LM.RIGHT_HIP] = [0.58, 0.5, 0.0, 1.0]
    lm[LM.RIGHT_KNEE] = [0.58, 0.7, 0.0, 1.0]
    lm[LM.RIGHT_ANKLE] = [0.58, 0.9, 0.0, 1.0]

    # Collinear leg -> 0 deg FPPA
    val_l = fppa(lm, "left")
    assert val_l == pytest.approx(0.0, abs=1e-5)

    # Right side calculation
    val_r = fppa(lm, "right")
    assert val_r == pytest.approx(0.0, abs=1e-5)


def test_api_parameters_22_specs_endpoint():
    """GET /biomechanics/parameters_22/specs returns the canonical 22 specifications."""
    res = client.get("/biomechanics/parameters_22/specs")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) == 22
    assert data[0]["name"] == "fppa_right"
    assert data[18]["name"] == "knee_ap_translation_mm"
    assert data[18]["level"] == 2
    assert data[18]["requires_depth_or_calibration"] is True


def test_api_session_parameters_22_endpoint():
    """GET /sessions/{session_id}/parameters_22 returns 22 parameters for a session."""
    # Register demo athlete & session
    ath_res = client.post("/athletes", json={
        "display_name": "Param22 Athlete",
        "sport": "Basketball",
        "dominant_leg": "right"
    })
    ath_id = ath_res.json()["athlete_id"]

    ses_res = client.post(
        "/sessions",
        json={
            "athlete_id": ath_id,
            "recorded_at": "2026-10-01T10:00:00Z",
            "camera_setup": "monocular_front",
            "fps": 30.0,
        },
    )
    ses_id = ses_res.json()["session_id"]

    # Ingest event
    client.post(f"/sessions/{ses_id}/events", json=[{
        "event_index": 0,
        "takeoff_frame": 10,
        "peak_frame": 20,
        "landing_frame": 30,
        "landing_timestamp_s": 1.0,
        "fppa_left": 12.0,
        "fppa_right": 8.0,
        "knee_flexion_left": 45.0,
        "knee_flexion_right": 50.0,
        "trunk_lean": 4.5,
    }])

    res = client.get(f"/sessions/{ses_id}/parameters_22")
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == ses_id
    params = data["parameters"]
    assert len(params) == 22
    assert params["fppa_right"]["value"] == 8.0
    assert params["knee_flexion_left"]["value"] == 45.0
    # Level 2 must be not available
    assert params["knee_ap_translation_mm"]["value"] == "not available"
    assert params["knee_ap_translation_mm"]["status"] == "not available"
