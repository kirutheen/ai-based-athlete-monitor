# DriftGuard — Full Product Build Specification

Internal engineering spec. Covers every layer from raw sensor capture to
cloud deployment and clinical rollout. Each section states what gets built,
the exact interfaces/formulas/schemas involved, and the acceptance check
that closes it out. Nothing here is aspirational marketing language — where
something is unsolved or hard, it says so directly.

---

## 0. Constraints that override every design decision below

These are fixed. If a later section conflicts with one of these, the
constraint wins, not the section.

1. Every train/test split is by athlete ID. No athlete appears in both sets.
2. Models operate on a full movement event (a landing/cutting/pivot window),
   never on an isolated frame.
3. Every risk output ships with the joints/features that drove it. A bare
   score is never acceptable output.
4. UI copy says "professional assessment recommended." It never says
   "diagnosis," "injury confirmed," or "prevents injury."
5. Risk comparison uses each athlete's own median/IQR, never one fixed
   population threshold.
6. A flag requires the same risk band across 3 of the last 5 evaluated
   events for that athlete. No single-event alerts.

---

## 1. System architecture

Three layers. Each can be built, tested, and (eventually) sold somewhat
independently.

```
LAYER 1 — MEASUREMENT              LAYER 2 — INTELLIGENCE           LAYER 3 — DELIVERY
video_source.py                    feature extraction                FastAPI backend
pose_estimator.py                  personal baseline engine          Postgres + object storage
imu_fusion.py (new)                risk-band + persistence logic     Streamlit/React dashboards
thermal_capture.py (new)           explanation generator             Triage queue
                                    (later) XGBoost + SHAP            Cloud hosting, public URL
```

Repo structure extends the existing skeleton:

```
injury-risk-screening/
  src/
    capture/        video_source.py [BUILT], imu_fusion.py [NEW], thermal_capture.py [NEW]
    pose/           pose_estimator.py [BUILT]
    calibration/    camera_calibration.py [NEW]
    features/       event_segmentation.py, biomechanics.py, thermal_features.py [NEW]
    baseline/       baseline_engine.py, risk_bands.py [NEW]
    explain/        rule_explainer.py [NEW], shap_explainer.py [PHASE 2]
    models/         risk_model.py [PHASE 2 — needs labelled data first]
    api/            main.py, routers/ [NEW]
    dashboard/      doctor_view.py, coach_view.py [NEW]
    db/             schema.sql, models.py [NEW]
  scripts/          run_pose_on_video.py [BUILT], download_pose_model.py [BUILT], setup.bat [BUILT]
  tests/
  docker/
  docs/
```

---

## 2. Layer 1 — Measurement

### 2.1 Video capture (built)

`src/capture/video_source.py`. `VideoSource` class, file or camera index
input, resamples to a target FPS by dropping/duplicating source frames,
yields `Frame(image, timestamp, frame_index, source_frame_index)`. Already
implemented and unit-tested. No changes needed for Phase 1.

### 2.2 Pose estimation (built)

`src/pose/pose_estimator.py`. Wraps MediaPipe Pose Landmarker (Tasks API).
`PoseEstimator.process(frame, timestamp_ms)` returns 2D landmarks (33x4:
x, y, z, visibility, normalized image coordinates), 3D world landmarks
where available, a confidence score, and an annotated debug frame. One-Euro
and moving-average smoothing both implemented. Already built and
unit-tested against a mocked backend; needs a real model file and real
footage to validate against, not more code changes.

### 2.3 IMU sensor fusion (new — this is the hard, unbuilt part)

Purpose: resolve the 3 rotations (flexion-extension, internal-external
rotation, abduction-adduction) and 3 translations (anterior-posterior,
medial-lateral, superior-inferior) that a single camera cannot see —
specifically internal-external rotation and all three translations, which
have no visible surface-landmark signature.

Hardware: two 9-axis IMUs (accelerometer + gyroscope + magnetometer) per
limb segment being tracked — one strapped to the distal thigh, one to the
proximal shank, positioned to minimize soft-tissue movement (strapped over
bone-adjacent regions: lateral femoral condyle side for the thigh sensor,
tibial crest side for the shank sensor). Sampling rate: minimum 200 Hz
(joint rotation during a pivot happens faster than camera frame rates can
resolve; IMUs need to run well above the camera's 60 fps to be useful as
the higher-fidelity reference signal).

Functional calibration protocol (required before every session, this is
not optional): athlete performs a slow, controlled flexion-extension sweep
and a slow internal-external rotation sweep while stationary. This
establishes each IMU's local sensor axis relative to the actual anatomical
joint axis — skipping this step is the single most common cause of bad
6-DOF results in published IMU-based knee studies, because raw IMU
orientation has no inherent knowledge of anatomical axes.

Fusion algorithm: an Extended Kalman Filter (EKF) fusing gyroscope
(orientation-over-time) with accelerometer+magnetometer (drift correction
against gravity and magnetic north), run independently per IMU to get each
segment's 3D orientation, then the knee's relative rotation is computed as
the quaternion difference between the thigh-segment and shank-segment
orientations. Translation is the harder problem: pure IMU integration
drifts too fast for translation to be trustworthy alone, so translation
is estimated by fusing the IMU-derived relative orientation with the
vision-derived joint-center positions (Layer 1's camera output), using the
camera as the low-frequency drift-correction anchor for the IMU's
high-frequency signal. This is a sensor-fusion research problem, not a
weekend implementation — budget real time for it.

Output schema per timestamp:
```
{
  "timestamp_ms": int,
  "flexion_extension_deg": float,
  "internal_external_rotation_deg": float,
  "abduction_adduction_deg": float,
  "ap_translation_mm": float,
  "ml_translation_mm": float,
  "si_translation_mm": float,
  "fusion_confidence": float   # degrades if IMU/vision signals disagree beyond a set tolerance
}
```

### 2.4 Thermal imaging (new)

Hardware: one low-cost long-wave infrared sensor module (Lepton-class,
80x60 or 160x120 thermal resolution is sufficient — this does not need to
be research-grade), mounted alongside the RGB camera, roughly co-aligned
so the knee region-of-interest can be cropped consistently from both
feeds.

Capture protocol: one thermal frame captured at rest before the drill
sequence begins (baseline reading) and one immediately after each rep
(post-exertion reading). Region of interest: a fixed-size box around the
knee joint, located using the same pose landmark coordinates already
being computed in Layer 1 (project the knee landmark's image coordinates
onto the thermal sensor's frame using a one-time spatial calibration
between the two cameras).

Metric computed: bilateral temperature asymmetry —
`abs(mean_temp(left_knee_roi) - mean_temp(right_knee_roi))`, in degrees C.
Asymmetries greater than 0.5 degrees C flag as physiologically notable, per
published thermography research on musculoskeletal asymmetry. This value
is a second, independent feature entering the same personal-baseline
engine as the movement features — not diagnostic on its own, tracked the
same way (median/IQR against this athlete's own resting/post-exertion
thermal history).

### 2.5 Camera calibration

Every capture session (single or multi-camera) requires: fixed camera
height (hip height of the specific athlete, recorded), fixed distance
(3-4 m), and — for the two-camera setup — a one-time checkerboard
calibration to establish each camera's intrinsic parameters and relative
extrinsic pose, so image coordinates can be converted to real-world
metric units rather than normalized 0-1 image space. Store calibration
parameters per physical camera rig, reused across sessions until the rig
is physically moved.

---

## 3. Layer 2 — Feature extraction

### 3.1 Event segmentation (new, currently missing)

Input: continuous smoothed landmark sequence. Output: a list of
`(start_frame, end_frame)` windows, each one full movement event.

Method for Phase 1 (structured drills): detect ground contact via ankle
landmark vertical velocity crossing zero after a sustained downward
phase — i.e., the ankle y-coordinate's smoothed derivative changes sign
from strongly negative (falling) to near-zero (landed) and stays near-zero
for at least 3 consecutive frames. Window boundaries: 300 ms before the
detected contact frame to 500 ms after it — wide enough to capture the
pre-landing approach and the full absorption phase, based on the
published finding that the injury-relevant knee-abduction-moment peak
occurs within the first ~100 ms after contact.

### 3.2 Biomechanical features (per event)

Knee abduction (valgus) angle — frontal plane projection angle (FPPA), a
standard 2D video-derived proxy used in published video-based valgus
scoring: project hip, knee, and ankle landmarks onto the frontal-plane
image coordinates (x, y only, from the frontal camera), compute the angle
at the knee vertex between vector `knee to hip` and vector `knee to ankle`,
then express the deviation from 180 degrees (a perfectly straight leg) as
the valgus angle, signed so a knee position medial to the hip-ankle line
is positive (inward collapse).

```
v1 = hip_xy - knee_xy
v2 = ankle_xy - knee_xy
angle_at_knee = degrees(arccos(dot(v1, v2) / (norm(v1) * norm(v2))))
valgus_angle = 180 - angle_at_knee   # signed per medial/lateral knee position
```

Timing feature: frame index (converted to ms from event start) at which
`valgus_angle` reaches its peak within the event window — this is checked
against the "does the peak occur within the first 100 ms" signature
separately from the peak magnitude itself.

Trunk lean: angle between the shoulder-midpoint-to-hip-midpoint vector and
true vertical, at the same peak-valgus frame.

Asymmetry index: for two-legged movements, `abs(left_valgus_angle -
right_valgus_angle)` at equivalent event phase.

6-DOF features (once Layer 1.3 is built): flexion-extension range, peak
internal-external rotation, peak translations, all taken across the same
segmented event window, giving a much richer feature set than the
camera-only proxy above once available.

### 3.3 Thermal feature

Bilateral asymmetry score from Section 2.4, attached to the session
(not per-event, since thermal readings are pre/post-session, not
per-rep).

---

## 4. Layer 2 — Personal baseline engine

`src/baseline/baseline_engine.py`.

- Minimum 10 clean, high-confidence events before an athlete's baseline is
  considered established. Before that: log everything, flag nothing.
- Baseline statistic per feature: median and IQR, recomputed as a rolling
  window over the athlete's most recent N established events (N to be
  tuned; start at 20).
- Risk bands per feature, relative to the athlete's own baseline:
  - Low: within 1.5x IQR of the athlete's median
  - Moderate: 1.5-3x IQR
  - High: beyond 3x IQR
- Absolute-threshold cross-check: independent of the personal baseline,
  also compare against the published research thresholds (e.g., 4.6 degree
  knee abduction angle at initial contact) as a secondary signal — an
  athlete whose baseline itself is already risky should not be invisible
  just because they're "normal for themselves."
- Persistence rule (constraint #6): a flag only fires when the same
  athlete's risk band has been Moderate or High in at least 3 of their
  last 5 evaluated events.

---

## 5. Layer 2 — Explainability

Phase 1 (rule-based, no training data required): explanation is generated
directly from which feature(s) crossed which band, and by how much —
e.g. `"Knee abduction angle 14.2 deg vs. this athlete's baseline of 6.1
deg (2.8x IQR) — Moderate."` This is explainable by construction; no
model inference step is involved.

Phase 2 (once pilot-generated doctor labels exist): train
XGBoost/LightGBM on event-level features against doctor-reviewed labels,
attach SHAP values per prediction, and generate the same style of
explanation string but now ranked by SHAP contribution rather than raw
threshold distance. The output format to the dashboard does not change —
only the mechanism generating it does.

---

## 6. Layer 3 — Backend API

FastAPI, `src/api/main.py`. Minimum endpoint set:

```
POST   /athletes                        create athlete record
POST   /sessions                        start a capture session
POST   /sessions/{id}/events            submit a processed event (features + raw video ref)
GET    /athletes/{id}/baseline          current baseline stats
GET    /athletes/{id}/flags             flagged events for review
POST   /flags/{id}/review               doctor submits their assessment (this is the label
                                         that eventually trains the Phase 2 model)
GET    /cohort/summary                  aggregate stats for a hospital/academy's athlete group
```

Every write endpoint requires an authenticated, scoped API key per
partner organization — not open access, given this handles athlete data.

---

## 7. Layer 3 — Database schema

```sql
CREATE TABLE athletes (
    id UUID PRIMARY KEY,
    org_id UUID NOT NULL,          -- which school/academy/hospital owns this record
    display_id TEXT NOT NULL,      -- non-identifying label, not necessarily real name
    sport TEXT,
    consent_recorded_at TIMESTAMP NOT NULL
);

CREATE TABLE sessions (
    id UUID PRIMARY KEY,
    athlete_id UUID REFERENCES athletes(id),
    started_at TIMESTAMP NOT NULL,
    capture_fps INTEGER,
    camera_config TEXT             -- 'single' | 'dual_frontal_sagittal' | 'multi'
);

CREATE TABLE events (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES sessions(id),
    event_index INTEGER,
    start_ms INTEGER,
    end_ms INTEGER,
    valgus_angle_deg REAL,
    valgus_peak_timing_ms REAL,
    trunk_lean_deg REAL,
    asymmetry_index REAL,
    thermal_asymmetry_c REAL,
    -- populated once Layer 1.3 exists:
    flexion_extension_deg REAL,
    internal_external_rotation_deg REAL,
    ap_translation_mm REAL,
    ml_translation_mm REAL,
    si_translation_mm REAL,
    video_clip_ref TEXT             -- object storage path, only retained if flagged
);

CREATE TABLE flags (
    id UUID PRIMARY KEY,
    event_id UUID REFERENCES events(id),
    risk_band TEXT,                 -- 'moderate' | 'high'
    explanation TEXT,
    flagged_at TIMESTAMP,
    doctor_reviewed_at TIMESTAMP,
    doctor_assessment TEXT          -- the label for Phase 2 training
);
```

Raw video is deleted after feature extraction unless the event produced a
flag (data-minimization rule agreed earlier — this is enforced at the
database/storage layer via a scheduled cleanup job, not left as a manual
step).

---

## 8. Layer 3 — Dashboards

Doctor/physio view (build first): athlete list sorted by flag recency,
per-flag detail (explanation, video clip, baseline comparison chart),
review-submission form. Coach view (subset of the same data): athlete
list, current risk band only, no raw feature values — coaches act on the
flag, not the underlying numbers.

Both read from the same API; no separate data pipeline per view.

---

## 9. Layer 3 — Cloud deployment

Containerize the API and dashboard separately (`docker/api.Dockerfile`,
`docker/dashboard.Dockerfile`). Database: managed PostgreSQL instance, not
self-hosted, for the pilot stage — reduces ops burden a two-person team
cannot absorb. Object storage for video clips: any S3-compatible bucket
with lifecycle rules matching the data-minimization policy above.
Public URL requirement (the hospital's funding milestone): a reverse
proxy with TLS in front of the containerized API/dashboard, on any
standard cloud VM or container hosting service — this is standard
deployment, not a research problem, and should not consume disproportionate
build time relative to Layers 1 and 2.

---

## 10. Layer 3 — Triage and scaling

The mechanism that lets clinician review time scale sub-linearly with
athlete count:

1. Every event is scored automatically (Section 4/5). No human touches a
   Low-band event.
2. Moderate/High flags enter a review queue, sorted by band severity and
   flag recency.
3. A doctor's review action (Section 6, `/flags/{id}/review`) is the only
   human step in the loop, and it doubles as the labelled-data collection
   mechanism for Phase 2.
4. Distribution is B2B2C: onboard organizations (schools, academies,
   hospital partner networks), not individual athletes one at a time.

---

## 11. Data governance and consent

Enrollment requires a consent record (digital or paper, referenced by
`consent_recorded_at`) before any session data is stored, aligned to
DPDP Act basics: purpose limitation (movement-risk screening only),
data minimization (raw video deleted unless flagged, per Section 7), and
a defined retention period per organization's contract terms — build this
as an enforced database rule, not a policy document that the code ignores.

---

## 12. Validation protocol (for the 6-DOF measurement layer specifically)

Movement set: gait (walking), sit-to-stand, stair descent, and a
controlled pivot — matching the MedTech challenge protocol, and
overlapping directly with DriftGuard's own landing/cutting/deceleration
drills.

Reference system: validate Layer 1.3's IMU-fusion output against either
an optical motion-capture system or an instrumented mechanical rig, run
on healthy volunteers first. Accuracy targets to hit before this layer is
considered production-ready: rotational error within 5 degrees RMS,
translational error within 5 mm RMS, sustained across the full movement
cycle — not just at a single instant. This validation step is a
prerequisite for claiming the 6-DOF numbers mean anything clinically;
skipping it and shipping unvalidated 6-DOF output would repeat the exact
overclaiming problem this project has otherwise avoided.

---

## 13. Regulatory pathway

Once real doctors are making review decisions informed by this system's
output, it likely falls under India's Software as a Medical Device
classification with CDSCO, even framed purely as a screening aid rather
than a diagnostic tool. This does not block Phase 1 (a supervised pilot
under direct clinical oversight is a normal, permissible way to generate
the evidence a regulatory submission would eventually need), but a formal
classification and compliance assessment needs to happen before any
Phase 2 scaling beyond the initial pilot site.

---

## 14. Registry and long-term data aggregation

Once multiple partner organizations are live (Phase 2+), anonymized
event-level data (features and doctor labels, not raw video) aggregates
into a shared dataset, modeled on the structure of established national
knee-ligament registries. This is what allows the system's accuracy and
baseline calibration to improve with scale rather than staying flat per
isolated deployment — the concrete answer to "how does this get better
as it grows," not a slide claim.

---

## 15. Testing strategy

- Unit tests per module, following the existing pattern in
  `tests/test_pose_estimator.py` — smoothing and geometry functions tested
  directly; anything depending on external model/hardware (pose model,
  IMU driver, thermal camera SDK) tested against a mocked interface so
  CI does not require physical hardware.
- Integration test: full pipeline run end-to-end on a fixed reference
  video with known expected output ranges, run on every change to
  Layers 1-2.
- Validation test (Section 12): run separately, on a defined cadence
  (not per-commit — it requires physical setup with reference equipment).

---

## 16. Build order — definition of done per phase

**Phase 0 (current sprint).** Done when: single camera, live jump/landing
video in, skeleton overlay out, one real biomechanical number computed,
personal baseline comparison working on a handful of reps, a rule-based
flag with a plain-language explanation, shown on a minimal dashboard.

**Phase 1 (funded pilot).** Done when: IMU fusion validated against a
reference system at the stated accuracy targets, thermal module
integrated and producing an asymmetry score, two-camera capture live,
API and dashboard deployed to a public URL, consent/data-minimization
rules enforced in code, and a supervised pilot underway generating
doctor-reviewed labels.

**Phase 2 (scaling).** Done when: ML risk model trained on real
pilot labels with SHAP explainability live, triage queue operating with
real flag volume, B2B2C onboarding flow built, regulatory classification
assessment completed.

**Phase 3 (scaled platform).** Done when: multi-camera occlusion-robust
capture live, cross-organization registry aggregating data, baseline
calibration measurably improving with added scale.
