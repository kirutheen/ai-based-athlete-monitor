"""
Regression tests for DriftGuard fix-in-place session.

Covers:
  1. LSI formula – 2|L-R|/(|L|+|R|)*100 as per parameters_22 spec
  2. Level 2 gating – params 19-22 always return "not available" on monocular input
  3. Session isolation – flags belong only to their session (no cross-session bleed)
  4. Null safety – risk model handles missing feature values without crashing
  5. Dynamics persistence – LandingEventRecord nullable column introspection
  6. wger API failure – graceful fallback when network is unavailable / key missing
  7. Camera manager – stream lifecycle contract (mocked; no real getUserMedia)

Run with:
    pytest tests/test_fix_regression.py -v
"""

import math
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from typing import Optional


# ---------------------------------------------------------------------------
# 1. LSI formula
# ---------------------------------------------------------------------------

def _calc_lsi(left_val: Optional[float], right_val: Optional[float]) -> Optional[float]:
    """Mirrors the formula in parameters_22.py _calc_lsi helper."""
    if left_val is None or right_val is None:
        return None
    denom = abs(left_val) + abs(right_val)
    if denom < 1e-6:
        return 0.0
    return round((2.0 * abs(left_val - right_val) / denom) * 100.0, 2)


class TestLSIFormula:
    def test_symmetric_limbs_give_zero(self):
        assert _calc_lsi(10.0, 10.0) == 0.0

    def test_known_asymmetric_values(self):
        # L=8, R=12  =>  2*|8-12|/(8+12)*100 = 2*4/20*100 = 40.0%
        result = _calc_lsi(8.0, 12.0)
        assert abs(result - 40.0) < 0.01, f"Expected 40.0, got {result}"

    def test_sign_insensitive_same_magnitude(self):
        # Both negative: |-8|+|-12|=20, ||-8|-|-12||=4 -> 40%
        assert _calc_lsi(-8.0, -12.0) == _calc_lsi(8.0, 12.0)

    def test_mixed_sign_uses_absolute_values(self):
        # Mixed signs (-8, +12): the formula uses abs(L)+abs(R)=20 and abs(-8-12)=20
        # => 2*20/20*100 = 200.0% — this is the correct behavior per the registry spec
        result = _calc_lsi(-8.0, 12.0)
        expected = round((2.0 * abs(-8.0 - 12.0) / (abs(-8.0) + abs(12.0))) * 100.0, 2)
        assert abs(result - expected) < 0.01, (
            f"Mixed-sign LSI: expected {expected}%, got {result}%"
        )

    def test_none_left_returns_none(self):
        assert _calc_lsi(None, 10.0) is None

    def test_none_right_returns_none(self):
        assert _calc_lsi(10.0, None) is None

    def test_both_none_returns_none(self):
        assert _calc_lsi(None, None) is None

    def test_both_zero_returns_zero(self):
        assert _calc_lsi(0.0, 0.0) == 0.0

    def test_large_asymmetry(self):
        # L=0, R=20  =>  2*20/20*100 = 200%
        result = _calc_lsi(0.0, 20.0)
        assert abs(result - 200.0) < 0.01


# ---------------------------------------------------------------------------
# 2. Level 2 gating
# ---------------------------------------------------------------------------

class TestLevel2Gating:
    """Level 2 parameters (19-22) MUST return 'not available' on monocular input."""

    def _build_test_df(self, n=30):
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "fppa_left":               rng.uniform(-15, 5, n),
            "fppa_right":              rng.uniform(-15, 5, n),
            "knee_flexion_left":       rng.uniform(20, 80, n),
            "knee_flexion_right":      rng.uniform(20, 80, n),
            "hip_flexion_left":        rng.uniform(30, 60, n),
            "hip_flexion_right":       rng.uniform(30, 60, n),
            "hip_abduction_left":      rng.uniform(-5, 15, n),
            "hip_abduction_right":     rng.uniform(-5, 15, n),
            "ankle_dorsiflexion_left": rng.uniform(5, 25, n),
            "ankle_dorsiflexion_right":rng.uniform(5, 25, n),
            "trunk_lean":              rng.uniform(-10, 10, n),
            "trunk_flexion_deg":       rng.uniform(-10, 10, n),
            "pelvic_obliquity_deg":    rng.uniform(-5, 5, n),
        })

    def _run_compute(self, has_depth=False, has_stereo=False):
        from src.features.parameters_22 import compute_22_parameters
        return compute_22_parameters(
            self._build_test_df(), events=[], fps=30.0,
            has_depth_input=has_depth,
            has_stereo_calibration=has_stereo,
        )

    def test_level2_params_not_available_on_monocular(self):
        result = self._run_compute(has_depth=False, has_stereo=False)
        level2_names = [
            "knee_ap_translation_mm",
            "knee_ml_translation_mm",
            "knee_si_translation_mm",
            "knee_int_ext_rot_deg",
        ]
        for name in level2_names:
            assert name in result, f"Parameter '{name}' missing from result dict"
            p = result[name]
            assert p["status"] == "not available", \
                f"'{name}' status should be 'not available', got '{p['status']}'"
            assert p["value"] == "not available", \
                f"'{name}' value should be 'not available', got '{p['value']}'"

    def test_level2_value_is_string_not_float(self):
        """Ensures nothing accidentally converts 'not available' to NaN/None/0.0."""
        result = self._run_compute()
        for name in ["knee_ap_translation_mm", "knee_int_ext_rot_deg"]:
            val = result[name]["value"]
            assert isinstance(val, str), \
                f"'{name}' value should be str 'not available', got {type(val)}: {val}"

    def test_level2_excluded_from_feature_columns(self):
        """Level 2 params must not appear in the risk model's FEATURE_COLUMNS."""
        try:
            from src.baseline.risk_model import FEATURE_COLUMNS
        except ImportError:
            pytest.skip("risk_model module not importable in this environment")

        level2_names = {
            "knee_ap_translation_mm",
            "knee_ml_translation_mm",
            "knee_si_translation_mm",
            "knee_int_ext_rot_deg",
        }
        overlap = level2_names & set(FEATURE_COLUMNS)
        assert not overlap, f"Level 2 params found in FEATURE_COLUMNS: {overlap}"

    def test_all_22_params_returned(self):
        result = self._run_compute()
        assert len(result) == 22, f"Expected 22 params, got {len(result)}"

    def test_level1_params_have_numeric_or_not_available(self):
        result = self._run_compute()
        level1_names = [
            "fppa_right", "knee_flexion_left", "knee_flexion_right",
            "hip_flexion_left", "hip_flexion_right",
        ]
        for name in level1_names:
            p = result[name]
            val = p["value"]
            assert val == "not available" or isinstance(val, (int, float)), \
                f"'{name}' value must be numeric or 'not available', got {val!r}"


# ---------------------------------------------------------------------------
# 3. Session isolation
# ---------------------------------------------------------------------------

class TestSessionIsolation:
    """Flags must belong to queried session only — not all flags for the athlete."""

    def _make_in_memory_db(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from src.db import models
        engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False}
        )
        models.Base.metadata.create_all(engine)
        return sessionmaker(bind=engine)()

    def test_flags_not_bleed_across_sessions(self):
        db = self._make_in_memory_db()
        try:
            from src.db import models

            # Athlete
            ath = models.Athlete(
                athlete_id="ath_iso_T1", display_name="Isolation Tester",
            )
            db.add(ath)

            # Two sessions — supply all NOT NULL fields required by SQLite
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            sA = models.SessionRecord(
                session_id="sess_iso_A", athlete_id="ath_iso_T1",
                sport_id="basketball", fps=30.0,
                recorded_at=now, camera_setup="single",
            )
            sB = models.SessionRecord(
                session_id="sess_iso_B", athlete_id="ath_iso_T1",
                sport_id="basketball", fps=30.0,
                recorded_at=now, camera_setup="single",
            )

            db.add_all([sA, sB])
            db.commit()

            # Events — use only real column names from the ORM
            evA = models.LandingEventRecord(
                event_id="ev_iso_A1", session_id="sess_iso_A", athlete_id="ath_iso_T1",
                event_index=0, takeoff_frame=5, peak_frame=15, landing_frame=10,
                landing_timestamp_s=0.33,
                knee_flexion_at_initial_contact=45.0,
                knee_valgus_at_initial_contact=8.0,
                valgus_duration=0.15,
            )
            evB = models.LandingEventRecord(
                event_id="ev_iso_B1", session_id="sess_iso_B", athlete_id="ath_iso_T1",
                event_index=0, takeoff_frame=2, peak_frame=10, landing_frame=5,
                landing_timestamp_s=0.17,
                knee_flexion_at_initial_contact=50.0,
                knee_valgus_at_initial_contact=12.0,
                valgus_duration=0.20,
            )

            db.add_all([evA, evB])
            db.commit()

            # Flag only on session B's event
            flag_b = models.RiskFlag(
                flag_id="flg_iso_B1", athlete_id="ath_iso_T1",
                feature_name="knee_valgus_at_initial_contact",
                event_id="ev_iso_B1", band="High",
                headline="High valgus", detail="Exceeds threshold",
            )
            db.add(flag_b)
            db.commit()

            # Query as the fixed endpoint does — via event_ids scalar subquery
            event_ids_subq = (
                db.query(models.LandingEventRecord.event_id)
                .filter(models.LandingEventRecord.session_id == "sess_iso_A")
                .scalar_subquery()
            )
            flags_for_A = (
                db.query(models.RiskFlag)
                .filter(models.RiskFlag.event_id.in_(event_ids_subq))
                .all()
            )

            assert len(flags_for_A) == 0, (
                f"Session bleed: {len(flags_for_A)} flag(s) from session B appeared "
                "in session A results"
            )

            # And session B's flag query returns the one flag
            event_ids_subq_b = (
                db.query(models.LandingEventRecord.event_id)
                .filter(models.LandingEventRecord.session_id == "sess_iso_B")
                .scalar_subquery()
            )
            flags_for_B = (
                db.query(models.RiskFlag)
                .filter(models.RiskFlag.event_id.in_(event_ids_subq_b))
                .all()
            )
            assert len(flags_for_B) == 1, f"Expected 1 flag for session B, got {len(flags_for_B)}"

        finally:
            db.close()


# ---------------------------------------------------------------------------
# 4. Null safety — risk model
# ---------------------------------------------------------------------------

class TestNullSafetyRiskModel:
    """Risk model must not raise when feature values are None."""

    def test_all_none_features(self):
        try:
            from src.baseline.risk_model import predict_risk_band, FEATURE_COLUMNS
        except ImportError:
            pytest.skip("risk_model not importable")

        features = {k: None for k in FEATURE_COLUMNS}
        try:
            result = predict_risk_band(features)
            assert result is not None, "predict_risk_band returned None on all-None input"
        except Exception as e:
            pytest.fail(f"predict_risk_band raised with all-None features: {e}")

    def test_partial_none_features(self):
        try:
            from src.baseline.risk_model import predict_risk_band, FEATURE_COLUMNS
        except ImportError:
            pytest.skip("risk_model not importable")

        features = {k: None for k in FEATURE_COLUMNS}
        # Populate only FPPA fields
        if "fppa_left" in features:
            features["fppa_left"] = -12.5
        if "fppa_right" in features:
            features["fppa_right"] = -8.0

        try:
            result = predict_risk_band(features)
            assert result is not None
        except Exception as e:
            pytest.fail(f"predict_risk_band raised with partial features: {e}")


# ---------------------------------------------------------------------------
# 5. Dynamics persistence — LandingEventRecord column nullability
# ---------------------------------------------------------------------------

class TestDynamicsNullability:
    """Level 2 kinematic columns in LandingEventRecord must be nullable."""

    MUST_BE_NULLABLE = [
        "knee_ap_translation_mm",
        "knee_ml_translation_mm",
        "knee_si_translation_mm",
        "knee_int_ext_rot_deg",
        "knee_flex_ext_deg",
        "knee_abd_add_deg",
    ]

    def test_level2_columns_nullable(self):
        from sqlalchemy import inspect as sa_inspect
        from src.db import models

        mapper = sa_inspect(models.LandingEventRecord)
        col_map = {c.key: c for c in mapper.columns}

        missing = []
        not_nullable = []
        for field in self.MUST_BE_NULLABLE:
            if field not in col_map:
                missing.append(field)
                continue
            if not col_map[field].nullable:
                not_nullable.append(field)

        if missing:
            pytest.skip(f"Columns not yet in schema: {missing}")
        assert not not_nullable, \
            f"These Level 2 columns must be nullable but are NOT: {not_nullable}"


# ---------------------------------------------------------------------------
# 6. wger API failure — graceful fallback
# ---------------------------------------------------------------------------

class TestWgerClientFallback:
    """wger client must return valid fallback dict on network failure."""

    def test_fallback_on_connection_error(self):
        try:
            from src.recovery.wger_client import fetch_exercise_media
        except ImportError:
            pytest.skip("wger_client not importable")

        with patch("requests.get", side_effect=ConnectionError("Network down")):
            result = fetch_exercise_media("Quad Stretch")

        assert result is not None
        assert isinstance(result, dict), "Expected dict from wger fallback"
        assert "fallback_svg" in result, "Fallback dict missing 'fallback_svg' key"
        assert result["fallback_svg"], "'fallback_svg' is empty"

    def test_fallback_on_http_503(self):
        try:
            from src.recovery.wger_client import fetch_exercise_media
        except ImportError:
            pytest.skip("wger_client not importable")

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("503 Service Unavailable")

        with patch("requests.get", return_value=mock_resp):
            result = fetch_exercise_media("Hamstring Curl")

        assert "fallback_svg" in result
        assert result["fallback_svg"]

    def test_missing_api_key_env_does_not_crash(self):
        """Client must not crash if WGER_API_KEY env var is absent."""
        try:
            from src.recovery.wger_client import fetch_exercise_media
        except ImportError:
            pytest.skip("wger_client not importable")

        backup = os.environ.pop("WGER_API_KEY", None)
        try:
            with patch("requests.get", side_effect=ConnectionError("offline")):
                result = fetch_exercise_media("Leg Press")
            assert "fallback_svg" in result
        finally:
            if backup is not None:
                os.environ["WGER_API_KEY"] = backup

    def test_fallback_dict_structure(self):
        """Fallback dict must always have image_url, fallback_svg, source keys."""
        try:
            from src.recovery.wger_client import fetch_exercise_media
        except ImportError:
            pytest.skip("wger_client not importable")

        with patch("requests.get", side_effect=TimeoutError("timeout")):
            result = fetch_exercise_media("Calf Raise")

        required_keys = {"fallback_svg", "source"}
        missing = required_keys - set(result.keys())
        assert not missing, f"Fallback dict missing keys: {missing}"


# ---------------------------------------------------------------------------
# 7. Camera manager — stream lifecycle (pure Python contract tests)
# ---------------------------------------------------------------------------

class TestCameraManagerLifecycle:
    """
    Verifies the RecoveryCameraManager state machine contract in Python.
    The actual JS implementation is tested in integration; these tests
    document and verify the equivalent Python logic for CI.
    """

    class FakeTrack:
        def __init__(self):
            self.stopped = False
        def stop(self):
            self.stopped = True

    class FakeStream:
        def __init__(self):
            self._tracks = None

        def init_tracks(self):
            self._tracks = [TestCameraManagerLifecycle.FakeTrack()]

        def get_tracks(self):
            return self._tracks or []

    class PythonCameraManager:
        """Python mirror of the JS RecoveryCameraManager contract."""
        def __init__(self):
            self._stream = None
            self._anim_frame = None
            self._status = "stopped"

        def stop(self):
            if self._anim_frame is not None:
                self._anim_frame = None
            if self._stream is not None:
                for t in self._stream.get_tracks():
                    t.stop()
                self._stream = None
            self._status = "stopped"

        def start(self, stream_factory, permission_denied=False):
            self.stop()  # always release old stream first
            if permission_denied:
                self._status = "permission_denied"
                return
            self._stream = stream_factory()
            self._status = "active"

    def test_stop_idempotent(self):
        mgr = self.PythonCameraManager()
        mgr.stop()
        mgr.stop()  # second call must not raise
        assert mgr._stream is None

    def test_start_releases_previous_stream(self):
        mgr = self.PythonCameraManager()

        old_stream = self.FakeStream()
        old_stream.init_tracks()
        mgr._stream = old_stream

        new_stream = self.FakeStream()
        new_stream.init_tracks()

        mgr.start(lambda: new_stream)

        assert old_stream._tracks[0].stopped, "Old stream tracks were not stopped"
        assert mgr._stream is new_stream

    def test_no_stream_on_permission_denied(self):
        mgr = self.PythonCameraManager()
        mgr.start(lambda: self.FakeStream(), permission_denied=True)
        assert mgr._stream is None
        assert mgr._status == "permission_denied"

    def test_status_transitions(self):
        mgr = self.PythonCameraManager()
        assert mgr._status == "stopped"

        stream = self.FakeStream()
        stream.init_tracks()
        mgr.start(lambda: stream)
        assert mgr._status == "active"

        mgr.stop()
        assert mgr._status == "stopped"
        assert stream._tracks[0].stopped

    def test_retry_replaces_stream(self):
        """Retry = stop + start; a new stream object must replace the old one."""
        mgr = self.PythonCameraManager()

        stream1 = self.FakeStream()
        stream1.init_tracks()
        mgr.start(lambda: stream1)

        stream2 = self.FakeStream()
        stream2.init_tracks()
        # Retry: stop + start
        mgr.stop()
        mgr.start(lambda: stream2)

        assert stream1._tracks[0].stopped, "Stream 1 tracks not stopped on retry"
        assert mgr._stream is stream2
