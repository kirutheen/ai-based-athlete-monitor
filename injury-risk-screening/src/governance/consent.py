"""
Consent recording and data-minimization helpers for DPDP Act compliance.

============================= HONEST SCOPE =================================
This module gives the SYSTEM a way to enforce "don't process this
athlete's data until consent is on record" and "delete raw video after
the retention window unless flagged" -- real, testable code.

What this module does NOT do, and what no code can do:
  - It does not make your consent FORM legally sufficient. The actual
    wording of what athletes/guardians are asked to agree to -- what
    data is collected, how long it's kept, who can see it, how to
    withdraw -- is a legal/policy document, not a function. Have the
    hospital's compliance team or a lawyer review the actual consent
    text before any real athlete is recorded.
  - It does not obtain consent from anyone. `record_consent()` only
    timestamps and stores a fact your intake process already
    established (a form was signed, a guardian agreed) -- it is a
    ledger, not a consent-gathering UI or process.
  - It does not itself satisfy IEC (Institutional Ethics Committee)
    approval or CDSCO classification -- those are institutional
    approvals this code has no way to obtain. See
    docs/PRODUCT_BUILD_SPEC.md and docs/REGULATORY_CHECKLIST.md for
    what those actually require.
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, UTC

DEFAULT_RAW_VIDEO_RETENTION_DAYS = 30  # data-minimization default: delete raw video after this many
                                        # days unless the session was flagged (see schema.sql's
                                        # raw_video_deleted_at column)


class ConsentRequiredError(RuntimeError):
    """Raised when an operation on an athlete's data is attempted before
    consent has been recorded for them."""


@dataclass
class ConsentRecord:
    athlete_id: str
    consent_given_at: datetime
    consent_version: str  # which version of the consent form text was agreed to
    given_by: str  # "self" or "guardian" -- who actually consented
    withdrawn_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        return self.withdrawn_at is None


class ConsentRegistry:
    """
    In-memory consent ledger for use by the API layer (src/api/main.py)
    before accepting any session/event data for an athlete. A real
    deployment persists this in the `athletes.consent_given_at` column
    (see src/db/schema.sql) -- this class is the same logic, usable
    standalone and independently testable.
    """

    def __init__(self):
        self._records: dict[str, ConsentRecord] = {}

    def record_consent(self, athlete_id: str, consent_version: str, given_by: str = "self") -> ConsentRecord:
        if given_by not in ("self", "guardian"):
            raise ValueError("given_by must be 'self' or 'guardian'")
        record = ConsentRecord(
            athlete_id=athlete_id,
            consent_given_at=datetime.now(UTC),
            consent_version=consent_version,
            given_by=given_by,
        )
        self._records[athlete_id] = record
        return record

    def withdraw_consent(self, athlete_id: str) -> None:
        record = self._records.get(athlete_id)
        if record is None:
            raise ConsentRequiredError(f"No consent record exists for athlete {athlete_id}")
        record.withdrawn_at = datetime.now(UTC)

    def require_consent(self, athlete_id: str) -> ConsentRecord:
        """Raises ConsentRequiredError if this athlete has no active
        consent on record. Call this before processing/storing any new
        data for an athlete."""
        record = self._records.get(athlete_id)
        if record is None or not record.is_active:
            raise ConsentRequiredError(
                f"Athlete {athlete_id} has no active consent on record -- "
                f"cannot process or store data until consent is recorded."
            )
        return record

    def has_consent(self, athlete_id: str) -> bool:
        record = self._records.get(athlete_id)
        return record is not None and record.is_active


@dataclass
class RetentionDecision:
    session_id: str
    should_delete_raw_video: bool
    reason: str


def decide_raw_video_retention(
    session_recorded_at: datetime,
    session_is_flagged: bool,
    retention_days: int = DEFAULT_RAW_VIDEO_RETENTION_DAYS,
    now: datetime | None = None,
) -> RetentionDecision:
    """
    Data-minimization rule (DPDP Act): raw video is deleted after
    `retention_days` UNLESS the session produced a Moderate/High risk
    flag, in which case it's kept (a clinician reviewing a flag may need
    to see the original footage). This function only decides whether
    deletion is due -- the actual file deletion is a separate, explicit
    operation (deliberately: a pure decision function is easy to test
    and audit; the delete itself should be a logged, deliberate action
    in a scheduled job, not buried in this logic).
    """
    now = now or datetime.now(UTC)
    age = now - session_recorded_at
    if session_is_flagged:
        return RetentionDecision(
            session_id="",
            should_delete_raw_video=False,
            reason="Session has a Moderate/High flag -- raw video retained for clinician review.",
        )
    if age >= timedelta(days=retention_days):
        return RetentionDecision(
            session_id="",
            should_delete_raw_video=True,
            reason=f"Session is {age.days} day(s) old, past the {retention_days}-day retention window, and unflagged.",
        )
    return RetentionDecision(
        session_id="",
        should_delete_raw_video=False,
        reason=f"Session is {age.days} day(s) old, within the {retention_days}-day retention window.",
    )
