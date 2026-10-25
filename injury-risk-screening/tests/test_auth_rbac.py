"""
Unit tests for authentication, JWT creation, password hashing, and RBAC security logic.
"""

import pytest
from src.api.auth import hash_password, verify_password, create_access_token, decode_access_token, require_role
from fastapi import HTTPException


def test_password_hashing_and_verification():
    raw_pass = "ClinicianSecret2026!"
    hashed = hash_password(raw_pass)

    assert hashed != raw_pass
    assert verify_password(raw_pass, hashed) is True
    assert verify_password("WrongPassword", hashed) is False


def test_jwt_token_encoding_and_decoding():
    payload = {"sub": "dr.smith@apex.org", "role": "doctor", "name": "Dr. Smith"}
    token = create_access_token(payload)

    assert isinstance(token, str)
    decoded = decode_access_token(token)
    assert decoded is not None
    assert decoded["sub"] == "dr.smith@apex.org"
    assert decoded["role"] == "doctor"


def test_role_based_access_control_checker():
    doctor_user = {"user_id": "u1", "role": "doctor"}
    coach_user = {"user_id": "u2", "role": "coach"}

    doctor_checker = require_role(["doctor", "physiotherapist"])

    # Doctor should pass role check
    assert doctor_checker(doctor_user) == doctor_user

    # Coach should raise HTTP 403 Forbidden
    with pytest.raises(HTTPException) as exc_info:
        doctor_checker(coach_user)
    assert exc_info.value.status_code == 403
