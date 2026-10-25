-- ============================================================================
-- DriftGuard / Injury-Risk-Screening schema
-- Target: PostgreSQL (SQLAlchemy models in src/db/models.py mirror this
-- exactly and are what the FastAPI app actually uses at runtime -- this
-- file documents the schema in plain SQL and is also what a DBA/doctor's
-- IT team would review independently of any ORM).
-- ============================================================================

CREATE TABLE IF NOT EXISTS athletes (
    athlete_id      TEXT PRIMARY KEY,          -- internal pseudonymous ID, never a real name in logs
    display_name    TEXT NOT NULL,             -- shown only inside the authenticated clinician UI
    sport           TEXT,
    date_of_birth   DATE,
    dominant_leg    TEXT CHECK (dominant_leg IN ('left', 'right', 'unknown')),
    team_or_school  TEXT,
    consent_given_at TIMESTAMPTZ,              -- DPDP Act: consent record, required before any session is stored
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    athlete_id      TEXT NOT NULL REFERENCES athletes(athlete_id),
    recorded_at     TIMESTAMPTZ NOT NULL,
    camera_setup    TEXT NOT NULL,             -- e.g. 'frontal+sagittal'
    fps             REAL NOT NULL,
    raw_video_deleted_at TIMESTAMPTZ,          -- DPDP data-minimization: raw video purged unless session flagged
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS landing_events (
    event_id        TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
    athlete_id      TEXT NOT NULL REFERENCES athletes(athlete_id),  -- denormalized for fast per-athlete queries
    event_index     INTEGER NOT NULL,          -- order within the session (0, 1, 2, ...)
    takeoff_frame   INTEGER NOT NULL,
    peak_frame      INTEGER NOT NULL,
    landing_frame   INTEGER NOT NULL,
    landing_timestamp_s REAL NOT NULL,
    fppa_left       REAL,
    fppa_right      REAL,
    knee_flexion_left REAL,
    knee_flexion_right REAL,
    trunk_lean      REAL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_landing_events_athlete ON landing_events(athlete_id, landing_timestamp_s);

CREATE TABLE IF NOT EXISTS baselines (
    athlete_id      TEXT NOT NULL REFERENCES athletes(athlete_id),
    feature_name    TEXT NOT NULL,             -- e.g. 'fppa_left'
    n_events        INTEGER NOT NULL,
    established     BOOLEAN NOT NULL,
    median_value    REAL,
    iqr_value       REAL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (athlete_id, feature_name)
);

CREATE TABLE IF NOT EXISTS risk_flags (
    flag_id         TEXT PRIMARY KEY,
    athlete_id      TEXT NOT NULL REFERENCES athletes(athlete_id),
    feature_name    TEXT NOT NULL,
    event_id        TEXT NOT NULL REFERENCES landing_events(event_id),  -- the event that triggered/confirmed this flag
    band            TEXT NOT NULL CHECK (band IN ('Low', 'Moderate', 'High')),
    deviation_iqr   REAL,
    headline        TEXT NOT NULL,
    detail          TEXT NOT NULL,
    citation        TEXT,
    reviewed_by_clinician_id TEXT,             -- NULL until a clinician opens/acknowledges this flag
    reviewed_at     TIMESTAMPTZ,
    clinician_note  TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_risk_flags_unreviewed ON risk_flags(athlete_id) WHERE reviewed_by_clinician_id IS NULL;

CREATE TABLE IF NOT EXISTS clinicians (
    clinician_id    TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    role            TEXT,                      -- e.g. 'orthopedic surgeon', 'physiotherapist'
    hospital        TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Thermal/IMU readings are stored separately because they come from
-- optional hardware add-ons (see src/sensors/ -- hardware-dependent,
-- not runnable without physical sensors). Schema included now so the
-- pipeline and API are ready the moment hardware is available.
CREATE TABLE IF NOT EXISTS thermal_readings (
    reading_id      TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
    athlete_id      TEXT NOT NULL REFERENCES athletes(athlete_id),
    captured_at     TIMESTAMPTZ NOT NULL,
    left_knee_temp_c  REAL,
    right_knee_temp_c REAL,
    bilateral_asymmetry_c REAL,                -- |left - right|; >0.5C flagged as physiologically notable
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS imu_kinematics (
    reading_id      TEXT PRIMARY KEY,
    event_id        TEXT NOT NULL REFERENCES landing_events(event_id),
    flexion_extension_deg REAL,
    internal_external_rotation_deg REAL,
    abduction_adduction_deg REAL,
    anterior_posterior_translation_mm REAL,
    medial_lateral_translation_mm REAL,
    superior_inferior_translation_mm REAL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
