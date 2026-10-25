"""Phase 3 regression tests — quality assessment, report generation, API endpoint."""
import pytest

# ---------------------------------------------------------------------------
# Import the new modules
# ---------------------------------------------------------------------------
from src.pose.multi_athlete_quality import (
    assess_athlete_quality,
    assess_session_quality,
    AthleteQualityResult,
    QUALITY_HIGH,
    QUALITY_ACCEPTABLE,
    QUALITY_LOW,
    QUALITY_INSUFFICIENT,
)
from src.pose.multi_athlete_report import (
    generate_multi_athlete_report,
    SCREENING_DISCLAIMER,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _athlete(
    track_id=1,
    display_name="Athlete #1",
    coverage_pct=90.0,
    uncertainty_pct=0.0,
    biomechanics_valid_pct=100.0,
    avg_fppa_left=-8.5,
    avg_knee_flex_left=52.0,
    peak_fppa_left=14.0,
    peak_knee_flex_left=65.0,
    color="#38BDF8",
    first_seen_s=0.0,
    last_seen_s=5.0,
    frames_tracked=150,
):
    return {
        "track_id": track_id,
        "session_id": "sess-test",
        "display_name": display_name,
        "color": color,
        "first_seen_s": first_seen_s,
        "last_seen_s": last_seen_s,
        "frames_tracked": frames_tracked,
        "coverage_pct": coverage_pct,
        "uncertainty_pct": uncertainty_pct,
        "biomechanics_valid_pct": biomechanics_valid_pct,
        "avg_fppa_left": avg_fppa_left,
        "avg_knee_flex_left": avg_knee_flex_left,
        "peak_fppa_left": peak_fppa_left,
        "peak_knee_flex_left": peak_knee_flex_left,
    }


def _result(athletes=None):
    """Minimal process_multi_athlete_video() result dict."""
    if athletes is None:
        athletes = [_athlete()]
    return {
        "session_id": "sess-test",
        "source_video": "clip.mp4",
        "total_frames": 150,
        "fps": 30.0,
        "duration_sec": 5.0,
        "max_athletes": 5,
        "athletes": athletes,
        "frames": [],
    }


# ---------------------------------------------------------------------------
# AthleteQuality: tier assignment
# ---------------------------------------------------------------------------

class TestAthleteQualityTiers:
    def test_high_quality_all_good(self):
        q = assess_athlete_quality(_athlete(
            coverage_pct=90.0, uncertainty_pct=0.0, biomechanics_valid_pct=100.0
        ))
        assert q.quality_tier == QUALITY_HIGH
        assert q.is_sufficient_for_screening is True
        assert q.quality_score > 0.8

    def test_insufficient_low_coverage(self):
        q = assess_athlete_quality(_athlete(coverage_pct=20.0))
        assert q.quality_tier == QUALITY_INSUFFICIENT
        assert q.is_sufficient_for_screening is False

    def test_insufficient_high_uncertainty(self):
        q = assess_athlete_quality(_athlete(uncertainty_pct=55.0, biomechanics_valid_pct=45.0))
        assert q.quality_tier == QUALITY_INSUFFICIENT
        assert q.is_sufficient_for_screening is False

    def test_low_moderate_uncertainty(self):
        q = assess_athlete_quality(_athlete(
            coverage_pct=80.0, uncertainty_pct=20.0, biomechanics_valid_pct=80.0
        ))
        assert q.quality_tier in (QUALITY_LOW, QUALITY_ACCEPTABLE)

    def test_acceptable_coverage_ok_bio_ok(self):
        q = assess_athlete_quality(_athlete(
            coverage_pct=75.0, uncertainty_pct=5.0, biomechanics_valid_pct=90.0
        ))
        assert q.quality_tier in (QUALITY_ACCEPTABLE, QUALITY_HIGH)

    def test_quality_score_bounded_0_to_1(self):
        for cov, unc, bio in [(0, 100, 0), (100, 0, 100), (50, 30, 70)]:
            q = assess_athlete_quality(_athlete(
                coverage_pct=cov, uncertainty_pct=unc, biomechanics_valid_pct=bio
            ))
            assert 0.0 <= q.quality_score <= 1.0

    def test_caveats_present_for_low_coverage(self):
        q = assess_athlete_quality(_athlete(coverage_pct=20.0))
        assert len(q.caveats) > 0
        assert any("coverage" in c.lower() for c in q.caveats)

    def test_no_caveats_for_perfect_data(self):
        q = assess_athlete_quality(_athlete(
            coverage_pct=95.0, uncertainty_pct=0.0, biomechanics_valid_pct=100.0
        ))
        assert q.caveats == []

    def test_uncertainty_caveat_present(self):
        q = assess_athlete_quality(_athlete(uncertainty_pct=25.0))
        assert any("uncertain" in c.lower() for c in q.caveats)

    def test_result_has_all_fields(self):
        q = assess_athlete_quality(_athlete())
        assert isinstance(q, AthleteQualityResult)
        assert hasattr(q, "track_id")
        assert hasattr(q, "quality_tier")
        assert hasattr(q, "is_sufficient_for_screening")
        assert hasattr(q, "quality_score")
        assert hasattr(q, "caveats")
        assert hasattr(q, "coverage_pct")
        assert hasattr(q, "uncertainty_pct")
        assert hasattr(q, "biomechanics_valid_pct")


class TestAssessSessionQuality:
    def test_returns_dict_keyed_by_track_id(self):
        result = _result(athletes=[_athlete(track_id=1), _athlete(track_id=2)])
        qmap = assess_session_quality(result)
        assert set(qmap.keys()) == {1, 2}

    def test_empty_athletes_returns_empty_dict(self):
        result = _result(athletes=[])
        qmap = assess_session_quality(result)
        assert qmap == {}

    def test_each_value_is_quality_result(self):
        result = _result(athletes=[_athlete(track_id=3)])
        qmap = assess_session_quality(result)
        assert isinstance(qmap[3], AthleteQualityResult)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

class TestGenerateMultiAthleteReport:
    def test_report_has_required_top_level_keys(self):
        report = generate_multi_athlete_report(_result())
        for key in ("disclaimer", "provenance", "session_quality_summary", "athletes"):
            assert key in report, f"Missing top-level key: {key}"

    def test_disclaimer_is_present_and_non_empty(self):
        report = generate_multi_athlete_report(_result())
        assert report["disclaimer"] == SCREENING_DISCLAIMER
        assert len(report["disclaimer"]) > 50

    def test_disclaimer_mentions_not_diagnosis(self):
        assert "NOT A CLINICAL DIAGNOSIS" in SCREENING_DISCLAIMER
        assert "screening" in SCREENING_DISCLAIMER.lower()

    def test_provenance_has_required_fields(self):
        report = generate_multi_athlete_report(_result())
        prov = report["provenance"]
        for key in ("algorithm_version", "pose_model", "tracker",
                    "session_id", "source_video", "total_frames",
                    "fps", "duration_sec", "generated_at_utc"):
            assert key in prov, f"Missing provenance field: {key}"

    def test_athlete_block_has_required_fields(self):
        report = generate_multi_athlete_report(_result())
        assert len(report["athletes"]) == 1
        a = report["athletes"][0]
        for key in ("track_id", "display_name", "quality", "biomechanics", "interpretation"):
            assert key in a, f"Missing athlete block key: {key}"

    def test_biomechanics_params_19_22_are_not_available(self):
        report = generate_multi_athlete_report(_result())
        bio = report["athletes"][0]["biomechanics"]
        for key in ("param_19_knee_valgus_3d", "param_20_hip_adduction_3d",
                    "param_21_trunk_lean_3d", "param_22_pelvic_drop_3d"):
            assert bio[key] == "not_available", f"{key} should be not_available"

    def test_biomechanics_measurement_source(self):
        report = generate_multi_athlete_report(_result())
        bio = report["athletes"][0]["biomechanics"]
        assert bio["measurement_source"] == "POSE_DERIVED"
        assert bio["measurement_status"] == "ESTIMATED_PROXY"
        assert bio["validation_status"] == "NOT_VALIDATED"

    def test_session_quality_summary_present(self):
        report = generate_multi_athlete_report(_result())
        sqs = report["session_quality_summary"]
        assert "athletes_detected" in sqs
        assert "athletes_sufficient_for_screening" in sqs
        assert "any_athlete_sufficient" in sqs
        assert "session_recommendation" in sqs

    def test_interpretation_present_and_non_empty(self):
        report = generate_multi_athlete_report(_result())
        interp = report["athletes"][0]["interpretation"]
        assert isinstance(interp, str) and len(interp) > 10

    def test_insufficient_quality_adds_warning_to_bio(self):
        """Low-coverage athlete should get a _quality_warning in bio block."""
        report = generate_multi_athlete_report(_result(athletes=[
            _athlete(coverage_pct=10.0, biomechanics_valid_pct=10.0)
        ]))
        bio = report["athletes"][0]["biomechanics"]
        assert "_quality_warning" in bio

    def test_no_quality_warning_for_high_quality(self):
        report = generate_multi_athlete_report(_result(athletes=[
            _athlete(coverage_pct=95.0, uncertainty_pct=0.0, biomechanics_valid_pct=100.0)
        ]))
        bio = report["athletes"][0]["biomechanics"]
        assert "_quality_warning" not in bio

    def test_empty_athletes_report_still_valid(self):
        report = generate_multi_athlete_report(_result(athletes=[]))
        assert report["athletes"] == []
        assert report["session_quality_summary"]["athletes_detected"] == 0
        assert report["session_quality_summary"]["any_athlete_sufficient"] is False

    def test_none_bio_values_remain_none(self):
        """Missing biomechanical values must not be fabricated."""
        report = generate_multi_athlete_report(_result(athletes=[
            _athlete(avg_fppa_left=None, avg_knee_flex_left=None)
        ]))
        bio = report["athletes"][0]["biomechanics"]
        assert bio["avg_fppa_left"] is None
        assert bio["avg_knee_flex_left"] is None

    def test_to_dict_on_quality_result(self):
        q = assess_athlete_quality(_athlete())
        d = q.to_dict()
        assert isinstance(d, dict)
        assert "quality_tier" in d
        assert "quality_score" in d


# ---------------------------------------------------------------------------
# API endpoint import regression
# ---------------------------------------------------------------------------

class TestPhase3APIImport:
    def test_api_imports_without_error(self):
        from src.api.main import app
        assert app is not None

    def test_report_endpoint_registered(self):
        from src.api.main import app
        routes = [r.path for r in app.routes]
        assert "/sessions/{session_id}/multi-athlete/report" in routes, (
            "multi-athlete/report endpoint must be registered"
        )

    def test_multi_athlete_report_module_importable(self):
        from src.pose.multi_athlete_report import generate_multi_athlete_report, SCREENING_DISCLAIMER
        assert callable(generate_multi_athlete_report)
        assert isinstance(SCREENING_DISCLAIMER, str)

    def test_multi_athlete_quality_module_importable(self):
        from src.pose.multi_athlete_quality import assess_session_quality, assess_athlete_quality
        assert callable(assess_session_quality)
        assert callable(assess_athlete_quality)
