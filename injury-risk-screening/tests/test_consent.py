"""Tests for src/governance/consent.py."""

from __future__ import annotations

from datetime import datetime, timedelta, UTC

import pytest

from src.governance.consent import (
    ConsentRegistry,
    ConsentRequiredError,
    decide_raw_video_retention,
)


def test_require_consent_raises_when_none_recorded():
    registry = ConsentRegistry()
    with pytest.raises(ConsentRequiredError):
        registry.require_consent("ath_1")


def test_record_and_require_consent_succeeds():
    registry = ConsentRegistry()
    registry.record_consent("ath_1", consent_version="v1.0", given_by="guardian")
    record = registry.require_consent("ath_1")
    assert record.athlete_id == "ath_1"
    assert record.given_by == "guardian"
    assert record.is_active


def test_withdrawn_consent_blocks_further_processing():
    registry = ConsentRegistry()
    registry.record_consent("ath_1", consent_version="v1.0")
    registry.withdraw_consent("ath_1")
    assert registry.has_consent("ath_1") is False
    with pytest.raises(ConsentRequiredError):
        registry.require_consent("ath_1")


def test_withdraw_without_prior_consent_raises():
    registry = ConsentRegistry()
    with pytest.raises(ConsentRequiredError):
        registry.withdraw_consent("ath_never_consented")


def test_record_consent_rejects_invalid_given_by():
    registry = ConsentRegistry()
    with pytest.raises(ValueError):
        registry.record_consent("ath_1", consent_version="v1.0", given_by="someone_else")


def test_retention_keeps_recent_unflagged_session():
    now = datetime.now(UTC)
    decision = decide_raw_video_retention(
        session_recorded_at=now - timedelta(days=5), session_is_flagged=False, retention_days=30, now=now
    )
    assert decision.should_delete_raw_video is False


def test_retention_deletes_old_unflagged_session():
    now = datetime.now(UTC)
    decision = decide_raw_video_retention(
        session_recorded_at=now - timedelta(days=45), session_is_flagged=False, retention_days=30, now=now
    )
    assert decision.should_delete_raw_video is True


def test_retention_keeps_old_flagged_session():
    now = datetime.now(UTC)
    decision = decide_raw_video_retention(
        session_recorded_at=now - timedelta(days=90), session_is_flagged=True, retention_days=30, now=now
    )
    assert decision.should_delete_raw_video is False
    assert "flag" in decision.reason.lower()
