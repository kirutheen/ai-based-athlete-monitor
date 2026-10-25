//cmd :- cd injury-risk-screening; python -m uvicorn src.api.main:app --reload --port 8000 
//accuracy:- Set-Location injury-risk-screening>> python scripts/run_model_eval.py

# DriftGuard -- Camera-Based Knee/ACL Injury-Risk Screening

> **Not a certified medical device.** This system never outputs a diagnosis and
> never claims to prevent injury — it produces a screening signal, grounded in
> published biomechanics literature, for a clinician to review and act on.

## Project Summary

A markerless computer-vision system that watches an athlete (football/soccer,
non-contact knee/ACL-related risk) perform landing movements from ordinary RGB
video. It extracts biomechanical features (knee valgus angle, landing flexion,
trunk lean), compares them against **that athlete's own personal baseline**
(never a population norm), and produces an interpretable **Low / Moderate /
High** risk-screening signal with a plain-language, literature-cited
explanation — flagging the athlete for a clinician to review.

Design rules, non-negotiable in every layer:

- Every risk output is explained in terms of a named feature, a citation, and
  a comparison to the athlete's own history — never a bare black-box score.
- A single unusual event never produces a flag: a confirmed flag requires
  persistence (3 of the last 5 events) before it reaches a clinician.
- Train/test and baseline computation are always split by athlete ID.

## What's actually built here

This is not a slide-deck description — every module below is real, tested
code (`pytest` suite, 115 passing tests + 1 skipped for the software-only
components; see "Running the tests").

| Layer | Module | What it does |
|---|---|---|
| Capture | `src/capture/video_source.py` | Frame-rate-aware video ingestion (file or camera), resampling to a target FPS |
| Capture | `src/capture/multi_person_tracker.py` | Centroid-distance multi-person ID tracking -- scoped to well-spaced, non-crossing scenes; known ID-switch limitation on crossing paths is documented and tested, not hidden |
| Pose | `src/pose/pose_estimator.py` | MediaPipe Pose Landmarker wrapper (Tasks API) + One-Euro/moving-average smoothing |
| CLI | `scripts/run_pose_on_video.py` | Runs capture+pose on a **recorded** video, writes a per-frame CSV/Parquet + annotated debug video |
| CLI | `scripts/live_webcam_demo.py` | **Real-time webcam monitoring**: live skeleton overlay, on-the-fly event detection, live risk flags shown on screen -- see "Live webcam demo" below |
| Features | `src/features/biomechanics.py` | 2D features: FPPA (knee valgus proxy), knee flexion angle, trunk lean. **Monocular-3D features** (new): real 3D knee flexion + abduction/adduction angles computed from MediaPipe's own `pose_world_landmarks` -- live-wired into `live_webcam_demo.py`, see "Monocular-3D features" below for exactly what this does and doesn't prove |
| Features | `src/features/event_segmentation.py` | **Batch** landing-event detection from a full recorded session |
| Features | `src/features/online_event_detector.py` | **Streaming** landing-event detection, one frame at a time, bounded memory -- powers the live webcam demo |
| Baseline | `src/baseline/baseline_engine.py` | Per-athlete median/IQR baseline from a 10-event warm-up |
| Baseline | `src/baseline/risk_bands.py` | Low/Moderate/High banding + the 3-of-5 persistence rule |
| Explain | `src/explain/rule_explainer.py` | Turns a flag into a plain-English, cited explanation -- no ML, by design |
| Models | `src/models/train_risk_model.py` | ML training **pipeline** (athlete-disjoint split, XGBoost, evaluation) -- infrastructure only; see "What's still not real" below for why it can't be a validated model yet |
| Governance | `src/governance/consent.py` | Consent ledger + DPDP-style raw-video retention decision logic |
| DB | `src/db/schema.sql`, `src/db/models.py` | Full schema (SQLAlchemy models + raw SQL) -- SQLite by default, Postgres-ready |
| API | `src/api/main.py` | FastAPI backend wiring ingestion → baseline → risk → explanation → flags |
| Glue | `scripts/process_session.py` | End-to-end CLI: pose data → features → events → API, for the recorded-video demo |
| Dashboard | `src/dashboard/doctor_view.py` | Clinician Streamlit app: flag queue + per-athlete trend lines |
| Dashboard | `src/dashboard/coach_view.py` | Thin coach view: roster status only, no raw clinical numbers |
| Sensors (hardware-gated) | `src/sensors/imu_fusion.py`, `thermal_capture.py` | Interfaces + algorithm skeletons for 6-DOF IMU kinematics and infrared thermography -- honestly flagged as **not runnable without physical hardware**; see each file's docstring for exactly why |
| Validation | `src/validation/opensim_knee_model.py` | **Real OpenSim integration**: builds a genuine 6-DOF knee model (`opensim.FreeJoint`) and runs real forward kinematics -- not a stub |
| Validation | `src/validation/motion_generator.py` | Simulated parametric reference knee trajectories (gait, sit-to-stand, stair descent, pivot) |
| Validation | `src/validation/noise_model.py` | Injects configurable camera/IMU/fused noise+bias+drift onto a reference trajectory to produce a simulated "estimate" |
| Validation | `src/validation/trajectory_metrics.py` | Rotational/translational RMSE, MAE, max error, cycle-consistency -- checked against the ≤5°/≤5mm target |
| CLI | `scripts/run_opensim_validation.py` | Runs the full validation pipeline end-to-end and prints a PASS/FAIL report -- see "OpenSim validation pipeline" below |
| Validation | `src/validation/stereo_calibration.py` | **Real triangulation math** (Direct Linear Transform): two synthetic cameras, real projection + SVD-based 3D reconstruction -- proves the multi-camera math, not a real two-camera rig |
| Validation | `src/validation/depth_simulation.py` | Simulated depth-sensor readings (published noise specs) vs. monocular depth-guessing, showing how much a real depth camera would help -- not a claim about a specific real sensor |
| Demo | `scripts/thermal_sample_demo.py` | Runs the real ROI/bilateral-asymmetry logic from `thermal_capture.py` against a synthetic thermal frame, so you can see sample flagging output before buying hardware |
| Demo | `scripts/load_test_api.py` | **Real load test** against the actual running FastAPI backend -- genuine measured throughput/latency, plus an honest discussion of what's needed to scale toward many concurrent users |

## Proof-of-concept simulations for the hardware-gated / scale-gated items

Six items were previously listed as "not achieved": real camera+IMU 6-DOF
fusion, depth-sensor hardware, temperature monitoring, 100k-user scaling,
multi-camera calibration, and clinician-labelled real-world validation.
**Five of these now have an honest simulated proof-of-concept; one does not,
on purpose.**

| Item | Can it be simulated honestly? | What was built |
|---|---|---|
| Camera+IMU 6-DOF fusion | Yes, partially | The OpenSim validation pipeline's `"fused"` noise preset (`noise_model.py`) represents a camera+IMU-fused estimate's *error characteristics* -- lower noise than camera-alone, no drift (unlike IMU-alone). Real fusion math (an EKF combining both sensors, `src/sensors/imu_fusion.py`) still needs real IMU hardware to calibrate against -- see that file. |
| Depth-sensor hardware | Yes | `depth_simulation.py`: projects a known 3D point into a camera, adds published-datasheet depth-sensor noise, back-projects, and shows the accuracy gain over monocular depth-guessing. Real sensor-specific failure modes (sunlight interference, reflective surfaces) are NOT simulated -- see the module docstring. |
| Temperature monitoring | Yes | `scripts/thermal_sample_demo.py` runs the real ROI-extraction/asymmetry logic against a synthetic thermal frame with a simulated inflammation hotspot. The camera-capture step itself stays a hardware-gated stub (`thermal_capture.py`) -- only calibrated data can come from real hardware. |
| 100k-user scaling | Partially | `scripts/load_test_api.py` is a REAL load test (real server, real concurrent HTTP requests, real measured throughput) -- it proves today's single-instance bottlenecks and names the specific, standard fixes (Postgres, multiple workers, shared state). It does **not** prove a number for 100k real users; that needs real deployment hardware. |
| Multi-camera calibration | Yes | `stereo_calibration.py`: real Direct Linear Transform triangulation, tested by projecting known 3D points into two synthetic cameras and recovering them exactly, then showing how 2D pixel noise degrades 3D accuracy. Real checkerboard-based calibration of two physical cameras (`cv2.stereoCalibrate`) still needs the physical cameras in hand. |
| Clinician-labelled real-world validation | **No** | This one is refused on purpose. The entire point of this validation is that it reflects real clinicians' real judgment on real patients over time. Fabricating sample "clinician labels" wouldn't be an honest proof-of-concept -- it would be synthetic data dressed up to look like the one thing that can't be faked without being misleading. There is no code for this, and there shouldn't be. |

## Monocular-3D features (live, not simulated)

`src/pose/pose_estimator.py` has always captured MediaPipe's `pose_world_landmarks`
into `PoseResult.landmarks_3d` -- a metric-scale 3D pose estimate from a single
camera (the BlazePose GHUM model). Until now, nothing downstream used it: every
feature in `biomechanics.py` (FPPA, flexion, trunk lean) only ever read the 2D
projection. That was a real gap between "3D is a main feature" and what the
live pipeline actually computed.

That gap is now partly closed, live:

- `knee_flexion_angle_3d()` -- a true 3D angle between thigh and shank vectors
  (not a 2D sagittal-plane assumption).
- `knee_abduction_angle_3d()` -- an out-of-flexion-plane shank deviation,
  closer to a real orthopedic knee abduction/adduction angle than the 2D FPPA
  proxy, using a Grood & Suntay (1983)-style local joint frame built from the
  thigh direction and the pelvis width vector.

Both are wired live into `scripts/live_webcam_demo.py` (shown on-screen
alongside the 2D numbers) and covered by `tests/test_biomechanics_3d.py`,
which checks them against hand-computed known angles (e.g. a shank tilted by
a known out-of-plane vector recovers ≈31° abduction, computed by hand in the
test file).

**What this is not:** it is not stereo- or IMU-measured 6-DOF. MediaPipe's
monocular depth (z) estimate is known to be less accurate than its x/y
estimate, and this still only recovers 2 of the 3 rotational knee angles
(flexion and ab/adduction) -- internal/external tibial rotation cannot be
separated from ab/adduction using monocular video alone, and none of the 3
translational DOFs (AP/ML/SI) are recovered this way. That full 6-DOF target
is still the OpenSim-validated camera+IMU fusion pipeline below, which needs
real IMU hardware. This module is an honest, real upgrade to the camera-only
path, not a substitute for that pipeline.

Not yet done: persisting the 3D fields through the API/database (the
`LandingEventIn` schema and DB tables still only have the original 2D
columns) -- the live demo computes and displays them, but a schema migration
is still needed before they're stored per-event like the 2D features are.

## OpenSim validation pipeline

```bash
python scripts/run_opensim_validation.py --movement gait --noise-profile camera
python scripts/run_opensim_validation.py --movement sit_to_stand --noise-profile imu --cycles 3
```

This is a genuine implementation of the abstract's validation stage: a real
OpenSim 6-DOF knee model runs real forward kinematics on a simulated
reference trajectory, controlled noise is injected to produce a simulated
"DriftGuard estimate," and the two are compared with rotational RMSE,
translational RMSE, MAE, max error, and cycle-level consistency against the
≤5°/≤5mm target. Try `--noise-profile imu --cycles 10 --duration 3.0` to see
it correctly **fail** once IMU-style drift accumulates over time — the
pass/fail logic reacts to real input, it isn't hardcoded to pass.

**What this is not:** the reference trajectories are simulated parametric
curves shaped to stay within commonly-reported clinical ranges, not measured
from a real patient, and the noise magnitudes are illustrative defaults, not
characteristics measured from real camera/IMU hardware. A "PASS" here proves
the validation *arithmetic* is correct — it is not a real-world accuracy
claim. See `noise_model.py` and `motion_generator.py` docstrings before
citing any number from this script to a clinician.

## Live webcam demo

```bash
uvicorn src.api.main:app --reload --port 8000     # terminal 1
python scripts/live_webcam_demo.py --camera 0 --athlete-name "Athlete 1"   # terminal 2
```

Opens your webcam, draws the live skeleton + FPPA/flexion numbers on screen,
detects each landing the moment it happens (not after the fact), uploads it to
the API, and shows a flag overlay once one is confirmed. Press `q` to quit. Add
`--no-api` to run the live overlay without a backend (no persistence/flags).

This is scoped to **one athlete at a time in a structured drill** -- the same
scoping this project has had from the start (see "Known limitations" below).

## Pipeline

```
Camera (frontal + sagittal) → VideoSource → PoseEstimator (MediaPipe) → smoothing
     → event segmentation (per landing) → biomechanical features (FPPA, flexion, trunk lean)
     → personal baseline comparison → risk banding + persistence rule
     → rule-based explanation → FastAPI → clinician / coach dashboard
```

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Video / CV | OpenCV |
| Pose estimation | MediaPipe Pose Landmarker (Tasks API) |
| Numerical / Data | NumPy, pandas, SciPy |
| Backend / API | FastAPI + Uvicorn |
| UI | Streamlit |
| Database | SQLite (dev/demo) → PostgreSQL (production), via SQLAlchemy |
| Testing | pytest |

## Repo Structure

```
injury-risk-screening/
  data/               # raw + processed video, landmark, feature data (gitignored)
  models/             # MediaPipe .task model bundle goes here
  src/
    capture/          # video ingestion + multi-person tracker
    pose/             # pose estimation wrapper
    features/         # biomechanics, batch + streaming event detection
    baseline/         # personal-baseline engine + risk bands/persistence
    explain/          # rule-based explanation generator
    models/           # ML training pipeline (infrastructure, not a validated model)
    governance/       # consent ledger + retention logic
    db/               # SQLAlchemy models + schema.sql
    api/              # FastAPI app
    dashboard/        # Streamlit apps (doctor_view, coach_view)
    sensors/          # hardware-gated IMU/thermal stubs (see module docstrings)
    validation/       # real OpenSim 6-DOF model + simulated reference/noise/RMSE validation
  scripts/
    run_pose_on_video.py      # recorded-video CLI: video -> pose CSV/Parquet
    live_webcam_demo.py       # real-time webcam monitoring with live overlay + flags
    run_opensim_validation.py # OpenSim reference-vs-estimate RMSE validation pipeline
    process_session.py       # recorded-video CLI: pose data -> features -> API
    download_pose_model.py   # downloads the MediaPipe model bundle
  tests/
  docs/
  requirements.txt
```

## Getting Started

**On Windows, extract the zip with File Explorer (right-click -> Extract All),
or `Expand-Archive` in PowerShell -- `unzip` is a Linux/macOS command and
PowerShell doesn't have it:**
```powershell
Expand-Archive injury-risk-screening.zip -DestinationPath .
cd injury-risk-screening
```

```bash
# Create and activate the virtual environment
python -m venv venv
source venv/bin/activate       # Windows (PowerShell): venv\Scripts\Activate.ps1
                                # Windows (cmd.exe):     venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Download the MediaPipe pose model (needed only for Phase 1 video processing)
python scripts/download_pose_model.py
```

**Every version in `requirements.txt` has a confirmed prebuilt wheel for
Python 3.11-3.13 on both Windows and Linux** (checked directly, not assumed
-- several of the original pins didn't, which caused installs to hang or
fail outright; see the comments in `requirements.txt` for exactly which
ones and why). If `pip install -r requirements.txt` still errors with "No
matching distribution found," run `python --version` first: a much newer
or older Python than 3.11-3.13 is the most likely cause, and switching to
a Python in that range resolves it.

If PowerShell blocks `venv\Scripts\Activate.ps1` with an "execution
policy" error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy
Bypass` in that PowerShell window first, then activate again.

### Running the tests

```bash
# Software-only components (no mediapipe/cv2 required):
pytest tests/ --ignore=tests/test_pose_estimator.py

# Including the pose estimator (requires mediapipe + cv2 installed and,
# for the one real-model smoke test, the downloaded .task model file):
pytest tests/
```

### Running the live demo end-to-end

```bash
# 1. Start the backend
uvicorn src.api.main:app --reload --port 8000

# 2. Process a video into pose data (Phase 1)
python scripts/run_pose_on_video.py --input data/my_jump.mp4 --output data/my_jump_pose.parquet --model models/pose_landmarker_full.task

# 3. Extract features, segment events, and push them to the API (Phase 2)
python scripts/process_session.py --pose-data data/my_jump_pose.parquet --athlete-name "Athlete 1" --fps 30

# 4. Open the clinician dashboard
streamlit run src/dashboard/doctor_view.py

# ...and/or the coach roster view
streamlit run src/dashboard/coach_view.py
```

### Running the proof-of-concept simulations

```bash
# Multi-camera triangulation (real math against synthetic two-camera views)
pytest tests/test_stereo_calibration.py -v

# Depth-sensor simulation (published noise specs vs. monocular depth guess)
pytest tests/test_depth_simulation.py -v

# Thermal "sample" demo -- visible output, no hardware required
python scripts/thermal_sample_demo.py --hotspot-knee right --hotspot-delta-c 1.2
python scripts/thermal_sample_demo.py --hotspot-knee none          # baseline, no flag

# Real load test against the actual running backend (starts its own server)
python scripts/load_test_api.py --n-requests 200 --concurrency 20
```

## Hard Constraints

Non-negotiable design rules, enforced in code (see the modules above):

1. **Athlete-level splitting.** Baselines and any future trained model are
   computed per athlete ID, never mixed across athletes.
2. **Event-level modeling.** Risk is scored per landing event, not per frame.
3. **Explainable outputs only.** Every flag includes the feature name, the
   athlete's own baseline, the deviation, and a literature citation.
4. **Careful UI language.** Dashboard copy says "clinician review
   recommended" -- never "diagnosis," "injury confirmed," or "prevents injury."
5. **Per-athlete robust baselines.** Risk comparisons use robust per-athlete
   median/IQR statistics, never a single universal threshold.
6. **Persistence before flagging.** A flag requires 3 of the athlete's last 5
   events in the same band. No single-event alerts.

## Known limitations (honest, on purpose)

- **Multi-person tracking is scoped, not solved.** `multi_person_tracker.py`
  keeps stable IDs for a few well-spaced athletes who don't cross paths. It
  **will** swap identities when people cross or fully occlude each other --
  `tests/test_multi_person_tracker.py::test_KNOWN_LIMITATION_crossing_paths_can_swap_ids`
  enforces and documents this rather than hiding it. Not a fit for a crowded
  scrimmage; built for structured, individual/small-group drills.
- **The live webcam demo is one-athlete-at-a-time.** It does not currently
  wire in the multi-person tracker; `PoseEstimator` is configured for
  `num_poses=1`.
- **The ML training pipeline (`src/models/train_risk_model.py`) has never
  seen a real injury outcome.** Its tests use clearly-labeled synthetic data
  to prove the plumbing (athlete-disjoint split, training, evaluation,
  save/load) works -- not that any prediction is real. The system's actual
  live risk logic is the rule-based baseline/risk-band system, which needs no
  outcome labels at all. See the module's docstring for exactly why a trained
  model can't be validated without months of real follow-up data.
- **Consent/retention logic (`src/governance/consent.py`) is a ledger, not a
  legal safeguard.** It enforces "no processing without a consent record on
  file" in code, and a DPDP-style raw-video retention rule. It does not write
  your consent form's legal wording, obtain consent from anyone, or satisfy
  Institutional Ethics Committee / CDSCO approval -- those remain
  institutional steps, not code.
- **The OpenSim validation pipeline (`src/validation/`) validates the
  pipeline's math, not real-world accuracy.** The knee model and forward
  kinematics are real OpenSim calls, not mocked -- but the reference
  trajectory is a simulated parametric curve, and the noise levels are
  illustrative defaults, not characteristics measured from the actual camera
  pipeline or real IMU hardware. A "PASS" from `run_opensim_validation.py`
  means the RMSE/MAE arithmetic is correct and the illustrative noise clears
  the target -- it is not yet a real accuracy measurement. It becomes one the
  day the illustrative noise presets in `noise_model.py` are replaced with
  error measured from the real camera pipeline and real IMU hardware.
- **This still doesn't deliver true 6-DOF tibiofemoral kinematics from
  camera+IMU fusion in the live system.** The validation pipeline proves the
  *math and target-checking* work; it does not retroactively make
  `src/sensors/imu_fusion.py` a working implementation, or make the live
  webcam demo output 6-DOF instead of 2D FPPA. Those remain gated on real
  IMU hardware, exactly as before.

## What's NOT real code (and why)

`src/sensors/imu_fusion.py` and `src/sensors/thermal_capture.py` are
interfaces and algorithm skeletons, not tested working implementations. Both
require physical sensor hardware to calibrate, tune, and validate against --
there is no way to write genuinely correct code for them without the hardware
in hand. Each file's docstring explains exactly what's missing and why.

Separately, and just as structurally: **no code in this repository proves the
risk model predicts real injuries.** That requires tracking real athletes for
months and comparing who got flagged against who actually got hurt -- an
empirical result, not a software feature. Everything here is a tested,
literature-grounded *screening* tool; none of it is a *proven predictor*.

The rest of this repository -- capture, pose, features (including the live
streaming path), baseline, risk banding, explanation, database, API, consent
ledger, and dashboards -- is real, tested, and runs today on an ordinary RGB
camera with no additional hardware.
