"""Phase 4 regression tests — multi-athlete validation protocol."""
import math
import pytest

from src.validation.multi_athlete_validator import (
    MultiAthleteValidator,
    MetricValidationRequest,
    AthleteValidationRollup,
    ValidationSessionReport,
    VALIDATION_DISCLAIMER,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _req(athlete_id="1", metric="fppa_left", n=20, error=1.0, unit="deg"):
    """Perfect reference + small Gaussian-like noise in estimated."""
    ref = [float(i) for i in range(n)]
    est = [r + error for r in ref]
    return MetricValidationRequest(
        athlete_id=athlete_id,
        metric_name=metric,
        estimated_values=est,
        reference_values=ref,
        unit=unit,
    )


def _req_exact(athlete_id="1", metric="fppa_left", n=20):
    """Perfect match — RMSE=0, should always pass threshold."""
    vals = [float(i) for i in range(n)]
    return MetricValidationRequest(
        athlete_id=athlete_id,
        metric_name=metric,
        estimated_values=list(vals),
        reference_values=list(vals),
        unit="deg",
    )


def _req_bad(athlete_id="1", metric="fppa_left", n=20):
    """Large error — should fail threshold."""
    ref = [float(i) for i in range(n)]
    est = [r + 50.0 for r in ref]   # 50° error — way above 5° threshold
    return MetricValidationRequest(
        athlete_id=athlete_id,
        metric_name=metric,
        estimated_values=est,
        reference_values=ref,
        unit="deg",
    )


def _req_too_short(athlete_id="1", metric="fppa_left"):
    """Only 2 pairs — engine returns NOT_VALIDATED."""
    return MetricValidationRequest(
        athlete_id=athlete_id,
        metric_name=metric,
        estimated_values=[1.0, 2.0],
        reference_values=[1.0, 2.0],
        unit="deg",
    )


validator = MultiAthleteValidator()


# ---------------------------------------------------------------------------
# Core validator tests
# ---------------------------------------------------------------------------

class TestMultiAthleteValidatorCore:
    def test_single_athlete_single_metric_passes(self):
        report = validator.run([_req_exact()])
        assert report.total_requests == 1
        assert report.athletes_validated == 1
        rollup = report.athlete_rollups[0]
        assert rollup.metrics_passed == 1
        assert rollup.metrics_failed == 0
        assert rollup.all_passed is True

    def test_single_athlete_bad_metric_fails(self):
        report = validator.run([_req_bad()])
        rollup = report.athlete_rollups[0]
        assert rollup.metrics_failed == 1
        assert rollup.all_passed is False

    def test_two_athletes_independent(self):
        """Each athlete's metrics must be evaluated independently."""
        report = validator.run([
            _req_exact(athlete_id="1", metric="fppa_left"),
            _req_bad(athlete_id="2", metric="fppa_left"),
        ])
        assert report.athletes_validated == 2
        by_id = {r.athlete_id: r for r in report.athlete_rollups}
        assert by_id["1"].all_passed is True
        assert by_id["2"].all_passed is False

    def test_athlete_ids_never_mixed(self):
        """Results for athlete '1' must not appear under athlete '2'."""
        report = validator.run([
            _req_exact(athlete_id="1", metric="fppa_left"),
            _req_exact(athlete_id="2", metric="knee_flexion_left"),
        ])
        by_id = {r.athlete_id: r for r in report.athlete_rollups}
        assert "fppa_left" in {m["metric_name"] for m in by_id["1"].metric_results}
        assert "knee_flexion_left" in {m["metric_name"] for m in by_id["2"].metric_results}
        # Cross-check: athlete 1 has no knee_flexion, athlete 2 has no fppa
        assert "knee_flexion_left" not in {m["metric_name"] for m in by_id["1"].metric_results}
        assert "fppa_left" not in {m["metric_name"] for m in by_id["2"].metric_results}

    def test_multiple_metrics_per_athlete(self):
        report = validator.run([
            _req_exact(athlete_id="1", metric="fppa_left"),
            _req_exact(athlete_id="1", metric="knee_flexion_left"),
            _req_exact(athlete_id="1", metric="hip_angle"),
        ])
        rollup = report.athlete_rollups[0]
        assert rollup.metrics_validated == 3
        assert rollup.metrics_passed == 3
        assert rollup.all_passed is True

    def test_insufficient_data_not_counted_as_pass_or_fail(self):
        report = validator.run([_req_too_short()])
        rollup = report.athlete_rollups[0]
        assert rollup.metrics_insufficient == 1
        assert rollup.metrics_passed == 0
        assert rollup.metrics_failed == 0
        assert rollup.all_passed is False  # insufficient = not fully validated

    def test_session_passed_only_if_all_athletes_passed(self):
        report = validator.run([
            _req_exact(athlete_id="1"),
            _req_bad(athlete_id="2"),
        ])
        assert report.session_passed is False

    def test_session_passed_when_all_pass(self):
        report = validator.run([
            _req_exact(athlete_id="1"),
            _req_exact(athlete_id="2"),
        ])
        assert report.session_passed is True

    def test_empty_requests_returns_valid_empty_report(self):
        report = validator.run([])
        assert report.total_requests == 0
        assert report.athletes_validated == 0
        assert report.session_passed is False
        assert report.athlete_rollups == []


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------

class TestValidationSessionReport:
    def test_to_dict_has_required_keys(self):
        report = validator.run([_req_exact()])
        d = report.to_dict()
        for key in ("disclaimer", "provenance", "session_passed",
                    "total_requests", "athletes_validated",
                    "athletes_all_passed", "athletes"):
            assert key in d, f"Missing key: {key}"

    def test_disclaimer_is_research_only(self):
        report = validator.run([_req_exact()])
        d = report.to_dict()
        assert "RESEARCH" in d["disclaimer"].upper()
        assert "NOT" in d["disclaimer"].upper()

    def test_provenance_has_required_fields(self):
        report = validator.run([_req_exact()])
        prov = report.to_dict()["provenance"]
        for key in ("engine_version", "angle_rmse_threshold_deg",
                    "generated_at_utc", "total_requests"):
            assert key in prov, f"Missing provenance field: {key}"

    def test_athlete_rollup_has_metric_results(self):
        report = validator.run([_req_exact()])
        a = report.to_dict()["athletes"][0]
        assert "metric_results" in a
        assert len(a["metric_results"]) == 1
        m = a["metric_results"][0]
        assert "metric_name" in m
        assert "rmse" in m or "validation_status" in m

    def test_athlete_rollup_to_dict_keys(self):
        report = validator.run([_req_exact()])
        a = report.to_dict()["athletes"][0]
        for key in ("athlete_id", "metrics_validated", "metrics_passed",
                    "metrics_failed", "metrics_insufficient", "all_passed"):
            assert key in a, f"Missing rollup key: {key}"

    def test_disclaimer_constant_non_empty(self):
        assert isinstance(VALIDATION_DISCLAIMER, str)
        assert len(VALIDATION_DISCLAIMER) > 50


# ---------------------------------------------------------------------------
# API endpoint registration
# ---------------------------------------------------------------------------

class TestPhase4APIEndpoints:
    def test_api_imports_without_error(self):
        from src.api.main import app
        assert app is not None

    def test_multi_athlete_validation_endpoint_registered(self):
        from src.api.main import app
        routes = [r.path for r in app.routes]
        assert "/validation/multi-athlete/compare" in routes, (
            "POST /validation/multi-athlete/compare must be registered"
        )

    def test_existing_single_validation_endpoint_unchanged(self):
        from src.api.main import app
        routes = [r.path for r in app.routes]
        assert "/validation/compare" in routes, (
            "Original /validation/compare must still exist"
        )

    def test_schema_imports(self):
        from src.api.schemas import MultiAthleteValidationRequest, MetricValidationEntry
        assert MultiAthleteValidationRequest is not None
        assert MetricValidationEntry is not None

    def test_validator_module_importable(self):
        from src.validation.multi_athlete_validator import (
            MultiAthleteValidator,
            MetricValidationRequest,
            ValidationSessionReport,
        )
        assert callable(MultiAthleteValidator)
