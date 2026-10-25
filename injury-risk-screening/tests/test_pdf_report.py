"""
Unit tests for clinical PDF report generation module.
"""

import os
import tempfile
import pytest

from src.reports.pdf_generator import generate_clinical_pdf_report


def test_generate_clinical_pdf_report_creates_valid_file():
    session_data = {
        "session_id": "ses_test123",
        "recorded_at": "2026-10-05T12:00:00Z",
        "camera_setup": "frontal+sagittal",
        "overall_risk_band": "High",
        "metrics": {
            "fppa_left_deg": 18.4,
            "fppa_right_deg": 5.2,
            "knee_flexion_left_deg": 40.1,
            "knee_flexion_right_deg": 48.0,
            "valgus_asymmetry_lsi_pct": 65.5,
            "flexion_asymmetry_lsi_pct": 16.4,
            "trunk_lean_deg": 6.8,
        },
    }

    athlete_data = {
        "athlete_id": "ath_test456",
        "display_name": "Test Athlete",
        "sport": "Basketball",
        "dominant_leg": "right",
    }

    flags_data = [
        {
            "feature_name": "fppa_left",
            "band": "High",
            "deviation_iqr": 3.2,
            "headline": "Elevated Left FPPA Valgus Proxy",
            "detail": "Left knee valgus reached 18.4 deg during landing phase.",
            "citation": "Hewett et al. (2005)",
        }
    ]

    notes_data = [
        {
            "note_type": "recommendations",
            "clinician_name": "Dr. Smith",
            "content": "Prescribe hip abductor and gluteus medius strengthening exercises.",
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "test_report.pdf")
        res_path = generate_clinical_pdf_report(
            session_data=session_data,
            athlete_data=athlete_data,
            flags_data=flags_data,
            notes_data=notes_data,
            output_path=pdf_path,
        )

        assert os.path.exists(res_path)
        assert os.path.getsize(res_path) > 1000  # Generated PDF file should be > 1 KB
