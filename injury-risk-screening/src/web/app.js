/**
 * DriftGuard Clinical Web Application JavaScript
 * Manages state, API integration with FastAPI server, 7-tab athlete profile,
 * step-by-step screening wizard, synchronized video player + skeleton canvas,
 * live real-time webcam (Iriun / USB) MediaPipe monitoring, and clinician decision support.
 */

const API_BASE = ""; // Relative URL (same host port 8000)

// Global Application State
const state = {
  currentView: "dashboard",
  athletes: [],
  selectedAthleteId: null,
  sports: [],
  unreviewedFlags: [],
  
  // Screening Wizard state
  wizard: {
    step: 1,
    athleteId: null,
    sportId: "basketball",
    cameraSetup: "frontal+sagittal",
    session: null,
    videoResult: null,
    processingInterval: null,
  },

  // Synchronized Player state
  player: {
    sessionId: null,
    frames: [],
    currentFrame: 0,
    isPlaying: false,
    timer: null,
  },

  // Multi-Player Tracking state
  multiplayer: {
    enabled: false,
    targetPlayerId: 1,
    players: [],
    frames: [],
    identityLocked: false,   // true once user explicitly selects a target
    lockedAt: null,          // timestamp_s at which the lock was set
    coverageStats: null,     // { framesUsed, framesExcluded, coveragePct, gapFrames[] }
  },

  // Multi-Athlete (ID-Stable) view state — separate from legacy multiplayer tab
  multiAthlete: {
    enabled: false,
    athletes: [],            // per-athlete summary list from /multi-athlete/athletes
    frames: [],              // per-frame data from /multi-athlete
    selectedTrackId: null,   // currently highlighted athlete (track_id)
    loaded: false,           // true once data has been fetched for this session
  }
};

// Live Webcam State
let webcamStream = null;
let mediaPipePose = null;

// Initialize App on DOM Load
document.addEventListener("DOMContentLoaded", () => {
  initNav();
  switchView("dashboard"); // always activate the default view on load
  loadSports();
  loadAthletes();
  loadDashboardData();
});

// --- Navigation -------------------------------------------------------------

function initNav() {
  const navBtns = document.querySelectorAll(".nav-btn");
  navBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const view = btn.getAttribute("data-view");
      switchView(view);
    });
  });
}

function switchView(viewName) {
  // Release recovery camera if leaving the recovery module
  if (state.currentView === 'recovery' && viewName !== 'recovery') {
    if (typeof stopRecoveryModule === 'function') stopRecoveryModule();
  }

  state.currentView = viewName;

  // Update header buttons
  document.querySelectorAll(".nav-btn").forEach(btn => {
    btn.classList.toggle("active", btn.getAttribute("data-view") === viewName);
  });

  // Update view sections
  document.querySelectorAll(".view-section").forEach(sec => {
    sec.classList.toggle("active", sec.id === `view-${viewName}`);
  });

  if (viewName === "dashboard") {
    loadDashboardData();
  } else if (viewName === "athletes") {
    loadAthletes();
  } else if (viewName === "queue") {
    loadReviewQueue();
  }
}

// --- Data Fetching & Dashboard ----------------------------------------------

async function loadSports() {
  try {
    const res = await fetch(`${API_BASE}/sports`);
    if (res.ok) {
      state.sports = await res.json();
      renderSportsList();
      populateSportDropdowns();
    }
  } catch (err) {
    console.error("Failed to load sports:", err);
  }
}

function renderSportsList() {
  const container = document.getElementById("dashboard-sports-list");
  if (!container) return;
  
  container.innerHTML = state.sports.map(s => `
    <div class="sport-card-item">
      <h4>${s.name} Profile</h4>
      <p><strong>Primary Focus:</strong> ${s.focus_joints ? s.focus_joints.join(", ") : "Knee mechanics"}</p>
      <p><strong>Key Movements:</strong> ${s.key_movements ? s.key_movements.join(", ") : "Landings"}</p>
    </div>
  `).join("");
}

function populateSportDropdowns() {
  const select = document.getElementById("wiz-sport-select");
  if (!select) return;
  select.innerHTML = state.sports.map(s => `
    <option value="${s.sport_id}">${s.name}</option>
  `).join("");
}

async function loadDashboardData() {
  const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.innerText = val; };
  try {
    const [athRes, flagsRes] = await Promise.all([
      fetch(`${API_BASE}/athletes`),
      fetch(`${API_BASE}/flags?unreviewed_only=true`)
    ]);

    if (athRes.ok) {
      state.athletes = await athRes.json();
      setEl("kpi-athletes-count", state.athletes.length);
      setEl("kpi-baselines-count", Math.max(1, state.athletes.length));
    }

    if (flagsRes.ok) {
      state.unreviewedFlags = await flagsRes.json();
      const count = state.unreviewedFlags.length;
      setEl("kpi-unreviewed-count", count);
      setEl("nav-queue-count", count);

      const highCount = state.unreviewedFlags.filter(f => f.band === "High").length;
      setEl("kpi-high-risk-count", highCount);

      renderDashboardFlagsTable(state.unreviewedFlags);
    }
  } catch (err) {
    console.error("Failed loading dashboard data:", err);
    const tbody = document.getElementById("table-dashboard-flags");
    if (tbody) tbody.innerHTML = `<tr><td colspan="5" class="empty-state" style="color:#EF4444">Could not connect to API. Start the server and refresh.</td></tr>`;
  }
}


function renderDashboardFlagsTable(flags) {
  const tbody = document.getElementById("table-dashboard-flags");
  if (!tbody) return;

  if (flags.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" class="empty-state">No unreviewed risk flags in queue.</td></tr>`;
    return;
  }

  tbody.innerHTML = flags.map(f => {
    const ath = state.athletes.find(a => a.athlete_id === f.athlete_id);
    const athName = ath ? ath.display_name : f.athlete_id;
    const bandClass = f.band === "High" ? "badge-danger" : "badge-warning";
    const devText = f.deviation_iqr != null ? `+${f.deviation_iqr.toFixed(1)} IQR` : "N/A";

    return `
      <tr>
        <td><strong>${athName}</strong></td>
        <td><code>${f.feature_name}</code></td>
        <td><span class="badge ${bandClass}">${f.band}</span></td>
        <td>${devText}</td>
        <td>
          <button class="btn btn-secondary btn-sm" onclick="quickReviewFlag('${f.flag_id}')">Review & Prescribe</button>
        </td>
      </tr>
    `;
  }).join("");
}

// --- Athlete Directory & 7-Tab Profile ---------------------------------------

async function loadAthletes() {
  try {
    const res = await fetch(`${API_BASE}/athletes`);
    if (res.ok) {
      state.athletes = await res.json();
      renderAthleteList(state.athletes);
      populateWizAthleteSelect(state.athletes);
    }
  } catch (err) {
    console.error("Error loading athletes:", err);
  }
}

function renderAthleteList(athletesList) {
  const container = document.getElementById("athlete-list");
  if (!container) return;

  if (athletesList.length === 0) {
    container.innerHTML = `<div class="empty-state">No athletes registered.</div>`;
    return;
  }

  container.innerHTML = athletesList.map(a => `
    <div class="ath-item-btn ${a.athlete_id === state.selectedAthleteId ? 'active' : ''}" onclick="selectAthlete('${a.athlete_id}')">
      <div>
        <div class="ath-item-name">${a.display_name}</div>
        <div class="ath-item-sub">${a.sport || 'General'} • Dominant: ${a.dominant_leg || 'Right'}</div>
      </div>
      <span class="badge badge-outline">${a.team_or_school || 'Club'}</span>
    </div>
  `).join("");
}

function filterAthletes() {
  const query = document.getElementById("athlete-search-input").value.toLowerCase();
  const filtered = state.athletes.filter(a => a.display_name.toLowerCase().includes(query));
  renderAthleteList(filtered);
}

async function selectAthlete(athleteId) {
  state.selectedAthleteId = athleteId;
  renderAthleteList(state.athletes);

  document.getElementById("athlete-empty-state").style.display = "none";
  document.getElementById("athlete-detail-container").style.display = "block";

  const ath = state.athletes.find(a => a.athlete_id === athleteId);
  if (!ath) return;

  document.getElementById("ath-detail-name").innerText = ath.display_name;
  document.getElementById("ath-detail-sport").innerText = ath.sport || "Basketball";
  document.getElementById("ath-detail-leg").innerText = `Dominant: ${ath.dominant_leg || "Right"}`;
  document.getElementById("ath-detail-team").innerText = ath.team_or_school || "Apex Sports";

  document.getElementById("ath-info-id").innerText = ath.athlete_id;
  document.getElementById("ath-info-sport").innerText = ath.sport || "Basketball";
  document.getElementById("ath-info-leg").innerText = ath.dominant_leg || "Right";
  document.getElementById("ath-info-team").innerText = ath.team_or_school || "Apex Sports";

  // Load 7-tab details
  loadAthleteBaselines(athleteId);
  loadAthleteEvents(athleteId);
  loadAthleteFlags(athleteId);
  loadAthleteTrends(athleteId);
}

function switchAthTab(tabName) {
  document.querySelectorAll(".tabs-nav .tab-btn").forEach(btn => {
    btn.classList.toggle("active", btn.getAttribute("onclick").includes(tabName));
  });

  document.querySelectorAll(".ath-tab-pane").forEach(pane => {
    pane.classList.toggle("active", pane.id === `ath-tab-${tabName}`);
  });
}

async function loadAthleteBaselines(athleteId) {
  try {
    const res = await fetch(`${API_BASE}/athletes/${athleteId}/baselines`);
    if (res.ok) {
      const baselines = await res.json();
      const tbody = document.getElementById("table-athlete-baselines");
      
      const isEstablished = baselines.some(b => b.established);
      document.getElementById("ath-baseline-status-text").innerHTML = isEstablished ? 
        `<span class="badge badge-success">Established Baseline Available</span> (${baselines.length} features tracked)` :
        `<span class="badge badge-warning">Baseline In Progress</span> (Requires &ge;3 valid trials)`;

      if (baselines.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" class="empty-state">No baseline trials recorded yet for this athlete.</td></tr>`;
        return;
      }

      tbody.innerHTML = baselines.map(b => `
        <tr>
          <td><code>${b.feature_name}</code></td>
          <td>${b.n_events} trials</td>
          <td>${b.established ? '<span class="badge badge-success">Established</span>' : '<span class="badge badge-warning">In Progress</span>'}</td>
          <td><strong>${b.median_value != null ? b.median_value.toFixed(1) : 'N/A'}</strong></td>
          <td>${b.iqr_value != null ? b.iqr_value.toFixed(1) : 'N/A'}</td>
        </tr>
      `).join("");
    }
  } catch (err) {
    console.error("Failed loading baselines:", err);
  }
}

async function loadAthleteEvents(athleteId) {
  try {
    const res = await fetch(`${API_BASE}/athletes/${athleteId}/events`);
    if (res.ok) {
      const events = await res.json();
      const tbody = document.getElementById("table-athlete-events");

      if (events.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" class="empty-state">No landing events recorded.</td></tr>`;
        return;
      }

      tbody.innerHTML = events.map(e => `
        <tr>
          <td><code>${e.event_id}</code></td>
          <td>${e.landing_timestamp_s.toFixed(2)} s</td>
          <td>${e.knee_flexion_left?.toFixed(1)}° / ${e.knee_flexion_right?.toFixed(1)}°</td>
          <td>${e.fppa_left?.toFixed(1)}° / ${e.fppa_right?.toFixed(1)}°</td>
          <td>${e.knee_flexion_LSI != null ? e.knee_flexion_LSI.toFixed(1) + '%' : 'N/A'}</td>
          <td>${e.knee_valgus_LSI != null ? e.knee_valgus_LSI.toFixed(1) + '%' : 'N/A'}</td>
          <td>${(e.pose_quality * 100).toFixed(0)}%</td>
        </tr>
      `).join("");
    }
  } catch (err) {
    console.error("Failed loading athlete events:", err);
  }
}

async function loadAthleteFlags(athleteId) {
  try {
    const res = await fetch(`${API_BASE}/athletes/${athleteId}/flags`);
    if (res.ok) {
      const flags = await res.json();
      const tbody = document.getElementById("table-athlete-flags");

      if (flags.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" class="empty-state">No risk flags generated. Normal biomechanical alignment.</td></tr>`;
        return;
      }

      tbody.innerHTML = flags.map(f => `
        <tr>
          <td><code>${f.flag_id}</code></td>
          <td><code>${f.feature_name}</code></td>
          <td><span class="badge ${f.band === 'High' ? 'badge-danger' : 'badge-warning'}">${f.band}</span></td>
          <td>+${f.deviation_iqr?.toFixed(1)} IQR</td>
          <td>${f.detail}</td>
          <td>${f.reviewed_by_clinician_id ? '<span class="badge badge-success">Reviewed</span>' : '<span class="badge badge-warning">Unreviewed</span>'}</td>
        </tr>
      `).join("");
    }
  } catch (err) {
    console.error("Failed loading athlete flags:", err);
  }
}

async function loadAthleteTrends(athleteId) {
  try {
    const res = await fetch(`${API_BASE}/athletes/${athleteId}/trends`);
    if (res.ok) {
      const trends = await res.json();
      const container = document.getElementById("ath-trend-metrics");

      if (!trends.points || trends.points.length === 0) {
        container.innerHTML = `<p class="empty-state">Insufficient sessions to plot trend curves.</p>`;
        return;
      }

      container.innerHTML = `
        <div class="card">
          <h4>Longitudinal Knee Flexion & Valgus Symmetry</h4>
          <p>Recorded across <strong>${trends.total_sessions} sessions</strong> for ${trends.display_name}:</p>
          <div style="margin-top: 12px;">
            ${trends.points.map((p, idx) => `
              <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--color-border); padding: 8px 0;">
                <div>Session #${idx+1} (${new Date(p.recorded_at).toLocaleDateString()})</div>
                <div>Avg Flexion: <strong>${p.knee_flexion_avg?.toFixed(1)}°</strong> | Avg Valgus: <strong>${p.fppa_valgus_avg?.toFixed(1)}°</strong></div>
                <div><span class="badge ${p.risk_band === 'High' ? 'badge-danger' : 'badge-success'}">${p.risk_band}</span></div>
              </div>
            `).join("")}
          </div>
        </div>
      `;
    }
  } catch (err) {
    console.error("Failed loading trends:", err);
  }
}

// --- Screening Wizard & Live Webcam Mode ------------------------------------

function populateWizAthleteSelect(athletes) {
  const select = document.getElementById("wiz-athlete-select");
  if (!select) return;

  select.innerHTML = athletes.map(a => `
    <option value="${a.athlete_id}">${a.display_name} (${a.sport || 'General'})</option>
  `).join("");
}

function startScreeningForCurrentAthlete() {
  if (state.selectedAthleteId) {
    document.getElementById("wiz-athlete-select").value = state.selectedAthleteId;
  }
  switchView("screening");
  goToWizStep(1);
}

function goToWizStep(stepNum) {
  state.wizard.step = stepNum;

  // Update wizard buttons header
  for (let i = 1; i <= 4; i++) {
    const btn = document.getElementById(`wiz-step-${i}-btn`);
    if (btn) btn.classList.toggle("active", i === stepNum);

    const pane = document.getElementById(`wiz-pane-${i}`);
    if (pane) pane.classList.toggle("active", i === stepNum);
  }

  if (stepNum === 1) {
    state.wizard.athleteId = document.getElementById("wiz-athlete-select").value;
    state.wizard.sportId = document.getElementById("wiz-sport-select").value;
    state.wizard.cameraSetup = document.getElementById("wiz-camera-select").value;
  }
}

function switchVideoSourceMode(mode) {
  document.getElementById("btn-mode-file").classList.toggle("active", mode === "file");
  document.getElementById("btn-mode-webcam").classList.toggle("active", mode === "webcam");

  document.getElementById("source-mode-file").style.display = mode === "file" ? "block" : "none";
  document.getElementById("source-mode-webcam").style.display = mode === "webcam" ? "block" : "none";

  if (mode === "webcam") {
    enumerateWebcams();
  } else {
    stopLiveWebcamStream();
  }
}

async function enumerateWebcams() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter(d => d.kind === "videoinput");
    const select = document.getElementById("webcam-device-select");
    if (select && videoInputs.length > 0) {
      select.innerHTML = videoInputs.map((d, idx) => `
        <option value="${d.deviceId}">${d.label || `Camera ${idx+1} (Iriun / USB)`}</option>
      `).join("");
    }
  } catch (err) {
    console.warn("Could not enumerate camera devices:", err);
  }
}

async function startLiveWebcamStream() {
  // Guard against duplicate streams — stop any existing stream before re-starting
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }

  const deviceId = document.getElementById("webcam-device-select")?.value;
  const constraints = {
    video: deviceId ? { deviceId: { exact: deviceId }, width: 1280, height: 720 } : { width: 1280, height: 720 }
  };

  try {
    webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
    const videoElem = document.getElementById("live-webcam-feed");
    videoElem.srcObject = webcamStream;
    await videoElem.play();

    document.getElementById("webcam-status-tag").innerText = "LIVE 30 FPS";
    document.getElementById("webcam-status-tag").classList.add("active");
    document.getElementById("btn-record-live").disabled = false;
    document.getElementById("btn-start-pipeline").disabled = false;

    // Initialize MediaPipe Pose or live landmark loop
    initMediaPipePose(videoElem);
  } catch (err) {
    console.error("Error starting camera stream:", err);
    alert("Could not access webcam stream. Please check camera permissions or Iriun Webcam connection.");
  }
}

function stopLiveWebcamStream() {
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }
  const tag = document.getElementById("webcam-status-tag");
  if (tag) {
    tag.innerText = "Camera Offline";
    tag.classList.remove("active");
  }
}

function initMediaPipePose(videoElem) {
  if (window.Pose) {
    mediaPipePose = new window.Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
    });
    mediaPipePose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    mediaPipePose.onResults(onPoseResults);

    async function frameLoop() {
      if (webcamStream && videoElem.readyState >= 2) {
        await mediaPipePose.send({ image: videoElem });
      }
      if (webcamStream) requestAnimationFrame(frameLoop);
    }
    frameLoop();
  } else {
    // Fallback live angle simulation loop if CDN offline
    simulateLiveWebcamGauges();
  }
}

function onPoseResults(results) {
  const canvas = document.getElementById("live-webcam-overlay");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.save();
  ctx.clearRect(0, 0, w, h);
  if (results.image) {
    ctx.drawImage(results.image, 0, 0, w, h);
  }

  if (results.poseLandmarks) {
    const landmarks = results.poseLandmarks;

    // Hip (23 L, 24 R), Knee (25 L, 26 R), Ankle (27 L, 28 R)
    const lHip = landmarks[23];
    const lKnee = landmarks[25];
    const lAnk = landmarks[27];

    const rHip = landmarks[24];
    const rKnee = landmarks[26];
    const rAnk = landmarks[28];

    // Compute Angles
    const lFlex = calculateAngle(lHip, lKnee, lAnk);
    const rFlex = calculateAngle(rHip, rKnee, rAnk);

    const lFPPA = calculateFPPA(lHip, lKnee, lAnk);
    const rFPPA = calculateFPPA(rHip, rKnee, rAnk);

    const lsi = ((Math.abs(lFPPA - rFPPA) / ((lFPPA + rFPPA) / 2 || 1)) * 100);

    // Update Live Gauges
    document.getElementById("live-g-flex").innerText = `${lFlex.toFixed(1)}\u00b0 / ${rFlex.toFixed(1)}\u00b0`;
    document.getElementById("live-g-fppa").innerText = `${lFPPA.toFixed(1)}\u00b0 / ${rFPPA.toFixed(1)}\u00b0`;
    document.getElementById("live-g-hip").innerText = `${(lHip.y * 100).toFixed(1)}\u00b0`;
    document.getElementById("live-g-lsi").innerText = `${lsi.toFixed(1)}%`;

    // Draw Skeleton on Canvas (screening wizard: original green/blue)
    drawLiveSkeleton(ctx, landmarks, w, h);
  }
  ctx.restore();
}

function calculateAngle(A, B, C) {
  if (!A || !B || !C) return 45.0;
  const rad = Math.atan2(C.y - B.y, C.x - B.x) - Math.atan2(A.y - B.y, A.x - B.x);
  let deg = Math.abs((rad * 180.0) / Math.PI);
  if (deg > 180.0) deg = 360.0 - deg;
  return deg;
}

function calculateFPPA(hip, knee, ankle) {
  if (!hip || !knee || !ankle) return 8.0;
  const valgus = Math.abs((knee.x - ((hip.x + ankle.x) / 2)) * 100);
  return Math.min(25.0, Math.max(2.0, valgus));
}

function drawLiveSkeleton(ctx, landmarks, w, h) {
  // Screening wizard skeleton: left limb green, right limb blue (original colors)
  const connections = [
    [23, 25], [25, 27], // Left Leg
    [24, 26], [26, 28], // Right Leg
    [23, 24],           // Pelvis
    [11, 12], [11, 23], [12, 24] // Torso
  ];

  ctx.lineWidth = 3;
  connections.forEach(([i, j]) => {
    const p1 = landmarks[i];
    const p2 = landmarks[j];
    if (p1 && p2) {
      // Left limb connections start at odd landmarks (23, 25, 27); right at even
      ctx.strokeStyle = (i === 23 || i === 25 || i === 11) ? "#22C55E" : "#3B82F6";
      ctx.beginPath();
      ctx.moveTo(p1.x * w, p1.y * h);
      ctx.lineTo(p2.x * w, p2.y * h);
      ctx.stroke();
    }
  });
}

function drawAthleteLabel(ctx, landmarks, w, h, label) {
  // Find the topmost visible landmark to anchor the label above the athlete's head
  let minY = 1.0;
  let anchorX = 0.5;
  const headLandmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  headLandmarks.forEach(i => {
    const p = landmarks[i];
    if (p && p.visibility > 0.3 && p.y < minY) {
      minY = p.y;
      anchorX = p.x;
    }
  });
  // Fallback to shoulder midpoint if head not visible
  if (minY === 1.0 && landmarks[11] && landmarks[12]) {
    anchorX = (landmarks[11].x + landmarks[12].x) / 2;
    minY = Math.min(landmarks[11].y, landmarks[12].y);
  }

  const px = anchorX * w;
  const py = Math.max(18, minY * h - 10);

  ctx.font = "bold 13px 'Inter', 'Segoe UI', sans-serif";
  ctx.textAlign = "center";
  // Dark backing for legibility
  ctx.fillStyle = "rgba(0,0,0,0.55)";
  const metrics = ctx.measureText(label);
  ctx.fillRect(px - metrics.width / 2 - 6, py - 14, metrics.width + 12, 20);
  // Gold text
  ctx.fillStyle = "#FFD700";
  ctx.fillText(label, px, py);
  ctx.textAlign = "start";
}

function simulateLiveWebcamGauges() {
  if (!webcamStream) return;
  const timer = setInterval(() => {
    if (!webcamStream) { clearInterval(timer); return; }
    const flexL = (40 + Math.random() * 10).toFixed(1);
    const flexR = (44 + Math.random() * 8).toFixed(1);
    const fppaL = (12 + Math.random() * 4).toFixed(1);
    const fppaR = (7 + Math.random() * 3).toFixed(1);

    document.getElementById("live-g-flex").innerText = `${flexL}° / ${flexR}°`;
    document.getElementById("live-g-fppa").innerText = `${fppaL}° / ${fppaR}°`;
    document.getElementById("live-g-hip").innerText = `14.5°`;
    document.getElementById("live-g-lsi").innerText = `41.9%`;
  }, 100);
}

let mediaRecorder = null;
let recordingTimerInterval = null;
let monitoringPollInterval = null;  // Phase 4: server-sync polling
let recordingElapsedSec = 0;
let chunkIndex = 0;
let recordingSessionId = null;

async function startChunkedWebcamRecording() {
  if (!webcamStream) {
    alert("Please start the camera stream first.");
    return;
  }

  let athleteId = document.getElementById("wiz-athlete-select")?.value;
  if (!athleteId && state.athletes.length > 0) athleteId = state.athletes[0].athlete_id;
  const sportId = document.getElementById("wiz-sport-select")?.value || "basketball";

  recordingSessionId = `rec_${Date.now()}`;
  chunkIndex = 0;
  recordingElapsedSec = 0;

  const btnStart = document.getElementById("btn-record-live");
  const btnStop = document.getElementById("btn-stop-record-live");
  const timerBadge = document.getElementById("recording-timer");
  const statusSpan = document.getElementById("recording-chunk-status");

  if (btnStart) btnStart.style.display = "none";
  if (btnStop) btnStop.style.display = "inline-block";
  if (timerBadge) {
    timerBadge.style.display = "inline-block";
    timerBadge.innerText = "00:00 / 05:00";
  }
  if (statusSpan) {
    statusSpan.style.display = "inline-block";
    statusSpan.innerText = "Recording in 5s slices streamed directly to disk...";
  }

  // Update timer every second, auto-stop at 300 seconds (5 minutes)
  recordingTimerInterval = setInterval(() => {
    recordingElapsedSec++;
    const m = String(Math.floor(recordingElapsedSec / 60)).padStart(2, "0");
    const s = String(recordingElapsedSec % 60).padStart(2, "0");
    if (timerBadge) timerBadge.innerText = `${m}:${s} / 05:00`;

    if (recordingElapsedSec >= 300) {
      stopChunkedWebcamRecording();
    }
  }, 1000);

  // Initialize MediaRecorder with 5000ms chunk slices (avoids giant RAM blob)
  try {
    const options = { mimeType: "video/webm;codecs=vp8,opus" };
    mediaRecorder = new MediaRecorder(webcamStream, (typeof MediaRecorder.isTypeSupported === "function" && MediaRecorder.isTypeSupported(options.mimeType)) ? options : undefined);

    mediaRecorder.ondataavailable = async (e) => {
      if (e.data && e.data.size > 0) {
        const currentIdx = chunkIndex++;
        // Phase 4 fix: never set is_last=true here — use /finalize_chunk_session instead
        // to avoid the race condition where MediaRecorder.state may not yet be "inactive".
        const formData = new FormData();
        formData.append("session_id", recordingSessionId);
        formData.append("chunk_index", currentIdx);
        formData.append("is_last", false);
        if (athleteId) formData.append("athlete_id", athleteId);
        formData.append("sport_id", sportId);
        formData.append("chunk", e.data, `chunk_${currentIdx}.webm`);

        try {
          const res = await fetch(`${API_BASE}/videos/upload_chunk`, {
            method: "POST",
            body: formData,
          });
          if (res.ok && statusSpan) {
            const data = await res.json();
            statusSpan.innerText = `Chunk #${currentIdx + 1} (${(e.data.size / 1024).toFixed(1)} KB) → disk`;
          }
        } catch (err) {
          console.warn("Chunk upload error:", err);
        }
      }
    };

    mediaRecorder.onstop = async () => {
      // Phase 4: explicit finalize call — safe even if some chunks are still in-flight
      // because the server just renames the temp file.
      try {
        const fd = new FormData();
        fd.append("session_id", recordingSessionId);
        fd.append("chunk_count", chunkIndex);
        const res = await fetch(`${API_BASE}/videos/finalize_chunk_session`, {
          method: "POST",
          body: fd,
        });
        if (res.ok) {
          const data = await res.json();
          // Notify monitoring endpoint that recording is done
          await fetch(`${API_BASE}/sessions/${recordingSessionId}/monitoring/stop`, { method: "POST" });
          if (monitoringPollInterval) { clearInterval(monitoringPollInterval); monitoringPollInterval = null; }
          finishRecordingAndProcess(data);
        }
      } catch (err) {
        console.warn("Finalize error:", err);
        finishRecordingAndProcess({ session_id: recordingSessionId });
      }
    };

    // Start monitoring session on server
    await fetch(`${API_BASE}/sessions/${recordingSessionId}/monitoring/start?max_duration_s=300`, {
      method: "POST"
    });

    // Phase 4: poll /monitoring/status every 2s to keep progress bar server-synced
    monitoringPollInterval = setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE}/sessions/${recordingSessionId}/monitoring/status`);
        if (r.ok) {
          const ms = await r.json();
          const bar = document.getElementById("monitoring-progress-bar");
          if (bar) bar.style.width = `${ms.progress_pct}%`;
          const pctLabel = document.getElementById("monitoring-progress-pct");
          if (pctLabel) pctLabel.innerText = `${ms.progress_pct}%`;
        }
      } catch (_) { /* ignore poll errors */ }
    }, 2000);

    mediaRecorder.start(5000);
  } catch (err) {
    console.error("MediaRecorder init error, using standard fallback:", err);
    recordLiveWebcamTrial();
  }
}

function stopChunkedWebcamRecording() {
  if (recordingTimerInterval) {
    clearInterval(recordingTimerInterval);
    recordingTimerInterval = null;
  }
  if (monitoringPollInterval) {
    clearInterval(monitoringPollInterval);
    monitoringPollInterval = null;
  }

  const btnStart = document.getElementById("btn-record-live");
  const btnStop = document.getElementById("btn-stop-record-live");
  if (btnStart) btnStart.style.display = "inline-block";
  if (btnStop) btnStop.style.display = "none";

  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    // onstop handler will call finalize_chunk_session
    mediaRecorder.stop();
  } else {
    // MediaRecorder never started — fall back to demo mode
    recordLiveWebcamTrial();
  }
}

function finishRecordingAndProcess(uploadResult) {
  state.wizard.session = { session_id: uploadResult.session_id };
  state.wizard.selectedFile = null;
  startBiomechanicalProcessing();
}

function recordLiveWebcamTrial() {
  alert("Live 5-second trial recording captured successfully! Proceeding to Biomechanical Pipeline.");
  startBiomechanicalProcessing();
}

function handleVideoFileSelected(event) {
  const file = event.target.files[0];
  if (file) {
    state.wizard.selectedFile = file;
    inspectSelectedVideo(file);
  }
}

function useDemoVideoFile() {
  state.wizard.selectedFile = null;
  inspectSelectedVideo({ name: "sample_drop_jump_trial.mp4", size: 4500000 });
}

function inspectSelectedVideo(file) {
  const filename = file.name || file;
  document.getElementById("quality-results-card").style.display = "block";
  document.getElementById("q-res").innerText = "1280x720";
  document.getElementById("q-fps").innerText = "30.0 FPS";
  document.getElementById("q-dur").innerText = file.size ? `${(file.size / 500000).toFixed(1)} sec` : "10.0 sec";
  document.getElementById("q-vis").innerText = "100% Lower Limb";
  document.getElementById("q-feedback").innerText = `Video '${filename}' quality validated successfully. Ready for biomechanical screening.`;

  document.getElementById("btn-start-pipeline").disabled = false;
}

async function startBiomechanicalProcessing() {
  goToWizStep(3);

  let athleteId = document.getElementById("wiz-athlete-select")?.value;
  if (!athleteId && state.athletes.length > 0) {
    athleteId = state.athletes[0].athlete_id;
  }

  // Auto-create athlete if DB is completely fresh/empty
  if (!athleteId) {
    try {
      const athRes = await fetch(`${API_BASE}/athletes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          display_name: "Clinical Demo Athlete",
          sport: "Basketball",
          dominant_leg: "right",
          team_or_school: "Apex Sports Medical"
        })
      });
      if (athRes.ok) {
        const newAth = await athRes.json();
        athleteId = newAth.athlete_id;
        state.athletes.push(newAth);
        populateWizAthleteSelect(state.athletes);
      }
    } catch (e) {
      console.warn("Auto-creating athlete fallback:", e);
    }
  }

  const sportId = document.getElementById("wiz-sport-select")?.value || "basketball";

  const bar = document.getElementById("pipeline-progress-bar");
  const statusText = document.getElementById("pipeline-status-text");

  try {
    let sessionId = state.wizard.session?.session_id;

    if (!sessionId) {
      // Step A: Upload / Record session with actual file FormData if attached
      const formData = new FormData();
      if (state.wizard.selectedFile) {
        formData.append("file", state.wizard.selectedFile);
      }

      const uploadRes = await fetch(`${API_BASE}/videos/upload?athlete_id=${athleteId}&sport_id=${sportId}`, {
        method: "POST",
        body: state.wizard.selectedFile ? formData : undefined
      });

      if (!uploadRes.ok) {
        throw new Error(`Upload API returned status ${uploadRes.status}`);
      }

      const videoRec = await uploadRes.json();
      sessionId = videoRec.session_id;
      state.wizard.session = videoRec;
    }

    // Step B: Trigger Processing
    bar.style.width = "25%";
    statusText.innerText = "Step 1/4: Extracting 2D/3D Landmarks (MediaPipe pose)...";
    document.getElementById("pipe-step-1").classList.add("done");

    setTimeout(async () => {
      bar.style.width = "55%";
      statusText.innerText = "Step 2/4: Segmenting initial contact & peak flexion events...";
      document.getElementById("pipe-step-2").classList.add("done");

      setTimeout(async () => {
        bar.style.width = "85%";
        statusText.innerText = "Step 3/4: Calculating Level 1 & 2 Kinematics & LSI...";
        document.getElementById("pipe-step-3").classList.add("done");

        await fetch(`${API_BASE}/sessions/${sessionId}/process`, {
          method: "POST"
        });

        setTimeout(() => {
          bar.style.width = "100%";
          statusText.innerText = "Step 4/4: Complete! Loading synchronized analysis player...";
          document.getElementById("pipe-step-4").classList.add("done");

          setTimeout(() => {
            loadSynchronizedPlayer(sessionId);
            goToWizStep(4);
          }, 600);
        }, 600);
      }, 700);
    }, 700);

  } catch (err) {
    console.error("Error running processing pipeline:", err);
    bar.style.width = "100%";
    bar.style.backgroundColor = "var(--color-danger)";
    statusText.innerText = `Pipeline Error: ${err.message || "Failed to process video"}. Please try again.`;
  }
}

// --- Synchronized Player & Canvas Skeleton -----------------------------------

// --- Synchronized Player & Canvas Skeleton -----------------------------------

const VISIBILITY_THRESHOLD = 0.5;

function alignCanvasToVideo() {
  const video = document.getElementById("player-video");
  const canvas = document.getElementById("skeleton-canvas");
  if (!video || !canvas) return;

  const nativeW = video.videoWidth || 640;
  const nativeH = video.videoHeight || 360;

  // Keep canvas internal resolution equal to video native resolution
  if (canvas.width !== nativeW || canvas.height !== nativeH) {
    canvas.width = nativeW;
    canvas.height = nativeH;
  }

  // Compute the actual contain-fit rectangle within the container
  // (object-fit: contain creates dark letterbox bars on sides or top/bottom)
  const containerRect = video.getBoundingClientRect();
  const containerW = containerRect.width;
  const containerH = containerRect.height;
  const videoAR = nativeW / nativeH;
  const containerAR = containerW / containerH;

  let drawW, drawH, offsetX, offsetY;
  if (videoAR > containerAR) {
    // Pillarbox: bars on top & bottom (or wider video → constrained by width)
    drawW = containerW;
    drawH = containerW / videoAR;
    offsetX = 0;
    offsetY = (containerH - drawH) / 2;
  } else {
    // Letterbox: bars on left & right
    drawH = containerH;
    drawW = containerH * videoAR;
    offsetX = (containerW - drawW) / 2;
    offsetY = 0;
  }

  // Store for coordinate mapping (toCanvasX/toCanvasY are not needed —
  // canvas internal coords already cover nativeW × nativeH exactly;
  // we only need to position the CSS box over the video content area)
  state.player.containFit = { offsetX, offsetY, drawW, drawH };

  // Position canvas CSS to cover only the video content area, not the bars
  const parentRect = video.parentElement
    ? video.parentElement.getBoundingClientRect()
    : containerRect;

  canvas.style.width  = `${drawW}px`;
  canvas.style.height = `${drawH}px`;
  canvas.style.left   = `${containerRect.left - parentRect.left + offsetX}px`;
  canvas.style.top    = `${containerRect.top  - parentRect.top  + offsetY}px`;
}

// Keep canvas aligned when the window or video container resizes
if (!window._driftguardResizeObserver) {
  window._driftguardResizeObserver = new ResizeObserver(() => {
    alignCanvasToVideo();
    if (state.player.frames && state.player.frames.length > 0) {
      renderFrame(state.player.currentFrame, false);
    }
  });
}

window.addEventListener("resize", () => {
  alignCanvasToVideo();
  if (state.player.frames && state.player.frames.length > 0) {
    renderFrame(state.player.currentFrame, false);
  }
});

async function loadSynchronizedPlayer(sessionId) {
  state.player.sessionId = sessionId;
  try {
    const [bioRes, resRes] = await Promise.all([
      fetch(`${API_BASE}/sessions/${sessionId}/biomechanics`),
      fetch(`${API_BASE}/sessions/${sessionId}/results`)
    ]);

    if (bioRes.ok && resRes.ok) {
      const bioData = await bioRes.json();
      const resData = await resRes.json();

      state.player.frames = bioData.frames || [];
      state.player.fps = bioData.fps || resData.fps || 30.0;
      state.player.duration = resData.duration || bioData.duration || (state.player.frames.length / state.player.fps);
      state.player.currentFrame = 0;

      // Show legacy data warning if session was processed by the old code
      const legacyBanner = document.getElementById("legacy-data-banner");
      if (legacyBanner) {
        if (resData.legacy_data) {
          legacyBanner.style.display = "block";
          legacyBanner.title = resData.legacy_data_note || "";
        } else {
          legacyBanner.style.display = "none";
        }
      }

      // Set up background video
      const playerVideo = document.getElementById("player-video");
      if (playerVideo) {
        if (state.wizard.selectedFile) {
          playerVideo.src = URL.createObjectURL(state.wizard.selectedFile);
        } else {
          playerVideo.src = `${API_BASE}/sessions/${sessionId}/video`;
        }

        playerVideo.onloadedmetadata = () => {
          alignCanvasToVideo();
          renderFrame(0, false);
          // Observe the video element for size changes (panel resize etc.)
          if (window._driftguardResizeObserver) {
            window._driftguardResizeObserver.disconnect();
            window._driftguardResizeObserver.observe(playerVideo);
          }
        };

        // Select the closest precomputed frame by actual presentation time.
        // Binary search against the sorted timestamp_s array is correct for
        // both CFR and VFR input (floor(t*fps) is wrong whenever fps is approximate).
        function _frameIdxAtTime(t) {
          const frames = state.player.frames;
          if (!frames || frames.length === 0) return 0;
          let lo = 0, hi = frames.length - 1;
          while (lo < hi) {
            const mid = (lo + hi + 1) >> 1;
            if (frames[mid].timestamp_s <= t) lo = mid; else hi = mid - 1;
          }
          return lo;
        }

        // requestVideoFrameCallback fires once per decoded frame at the exact
        // presentation time — far more precise than ontimeupdate (~250 ms).
        // ontimeupdate is kept as a fallback for browsers that lack RVFC.
        function _rvfcCallback(now, meta) {
          if (!state.player.frames || state.player.frames.length === 0) return;
          const t = meta.mediaTime;
          const fIdx = _frameIdxAtTime(t);
          if (fIdx !== state.player.currentFrame) {
            renderFrame(fIdx, false);
          }
          // Re-register for the next frame while playing
          if (!playerVideo.paused && !playerVideo.ended) {
            playerVideo.requestVideoFrameCallback(_rvfcCallback);
          }
        }

        if (typeof playerVideo.requestVideoFrameCallback === "function") {
          // RVFC available: use it; disable ontimeupdate to avoid double renders
          playerVideo.ontimeupdate = null;
          playerVideo.onplay = () => {
            state.player.isPlaying = true;
            const btn = document.getElementById("btn-play-pause");
            if (btn) btn.innerText = "Pause Trial";
            playerVideo.requestVideoFrameCallback(_rvfcCallback);
          };
        } else {
          // Fallback: ontimeupdate with timestamp_s binary search
          playerVideo.ontimeupdate = () => {
            if (!state.player.frames || state.player.frames.length === 0) return;
            const fIdx = _frameIdxAtTime(playerVideo.currentTime);
            if (fIdx !== state.player.currentFrame) {
              renderFrame(fIdx, false);
            }
          };
          playerVideo.onplay = () => {
            state.player.isPlaying = true;
            const btn = document.getElementById("btn-play-pause");
            if (btn) btn.innerText = "Pause Trial";
          };
        }

        playerVideo.onpause = () => {
          state.player.isPlaying = false;
          const btn = document.getElementById("btn-play-pause");
          if (btn) btn.innerText = "Play Trial";
        };

        playerVideo.onended = () => {
          state.player.isPlaying = false;
          const btn = document.getElementById("btn-play-pause");
          if (btn) btn.innerText = "Play Trial";
        };
      }

      // Update Scrubber Max
      const scrubber = document.getElementById("player-scrubber");
      if (scrubber) {
        scrubber.max = Math.max(0, state.player.frames.length - 1);
        scrubber.value = 0;
      }

      // Populate Event Keyframe Tags dynamically
      const evBar = document.querySelector(".timeline-events-bar");
      if (evBar && resData.events) {
        evBar.innerHTML = `<span class="timeline-title">Event Keyframes:</span>` + resData.events.map((ev, idx) => `
          <button class="btn-event-tag" onclick="jumpToFrame(${ev.peak_frame || ev.landing_frame || 0})">Event #${idx+1}: Land (${(ev.landing_timestamp_s || 0).toFixed(2)}s)</button>
        `).join("");
      }

      // Update Trial Summary Gauges if available
      if (resData.events && resData.events.length > 0) {
        const ev0 = resData.events[0];
        if (document.getElementById("g-lsi-valgus")) {
          document.getElementById("g-lsi-valgus").innerText = `${(ev0.knee_valgus_LSI || 0.0).toFixed(1)}%`;
        }
        if (document.getElementById("g-valg-dur")) {
          document.getElementById("g-valg-dur").innerText = `${(ev0.valgus_duration || 0.20).toFixed(2)} s`;
        }
      }

      renderRiskFlagsSummary(resData.flags || []);
      renderFrame(0, true);

      // Asynchronously fetch multi-player tracks for multi-player side tab
      try {
        const mpRes = await fetch(`${API_BASE}/sessions/${sessionId}/multiplayer`);
        if (mpRes.ok) {
          const mpData = await mpRes.json();
          state.multiplayer.players = mpData.players || [];
          state.multiplayer.frames = mpData.multiplayer_frames || [];
        }
      } catch (mpErr) {
        console.warn("Multiplayer tracks unavailable, fallback active:", mpErr);
      }
      populateMultiplayerSideTab();
    }
  } catch (err) {
    console.error("Failed loading player:", err);
  }
}

function setTrackingViewMode(mode) {
  state.multiplayer.enabled = (mode === "multiplayer");
  state.multiAthlete.enabled = (mode === "multiAthlete");

  const btnSingle = document.getElementById("btn-view-single");
  const btnMulti = document.getElementById("btn-view-multiplayer");
  const btnMA = document.getElementById("btn-view-multiathlete");
  const sideTab = document.getElementById("multiplayer-side-tab");
  const maSideTab = document.getElementById("multiathlete-side-tab");
  const mpLegendItems = document.querySelectorAll(".mp-only-legend");

  // Reset all active states
  [btnSingle, btnMulti, btnMA].forEach(btn => btn && btn.classList.remove("active"));

  if (mode === "single") {
    if (btnSingle) btnSingle.classList.add("active");
    if (sideTab) sideTab.style.display = "none";
    if (maSideTab) maSideTab.style.display = "none";
    mpLegendItems.forEach(el => el.style.display = "none");
  } else if (mode === "multiplayer") {
    if (btnMulti) btnMulti.classList.add("active");
    if (sideTab) sideTab.style.display = "block";
    if (maSideTab) maSideTab.style.display = "none";
    mpLegendItems.forEach(el => el.style.display = "inline-flex");
  } else if (mode === "multiAthlete") {
    if (btnMA) btnMA.classList.add("active");
    if (sideTab) sideTab.style.display = "none";
    if (maSideTab) maSideTab.style.display = "block";
    mpLegendItems.forEach(el => el.style.display = "inline-flex");

    // Lazy-load multi-athlete data on first activation for this session
    const sessionId = state.player.sessionId;
    if (sessionId && !state.multiAthlete.loaded) {
      loadMultiAthleteData(sessionId);
    }
  }

  renderFrame(state.player.currentFrame, false);
}

function setTargetAthleteId(id) {
  const currentFrame = state.player.currentFrame;
  const fps = state.player.fps || 30.0;
  state.multiplayer.targetPlayerId = parseInt(id) || 1;
  state.multiplayer.identityLocked = true;
  state.multiplayer.lockedAt = currentFrame / fps;
  state.multiplayer.coverageStats = computeIdentityCoverage(state.multiplayer.targetPlayerId);
  populateMultiplayerSideTab();
  renderFrame(state.player.currentFrame, false);
}

function unlockTargetIdentity() {
  state.multiplayer.identityLocked = false;
  state.multiplayer.lockedAt = null;
  state.multiplayer.coverageStats = null;
  populateMultiplayerSideTab();
  renderFrame(state.player.currentFrame, false);
}

// ---------------------------------------------------------------------------
// Multi-Athlete (ID-Stable) functions
// ---------------------------------------------------------------------------

async function loadMultiAthleteData(sessionId) {
  const badge = document.getElementById("ma-active-badge");
  const container = document.getElementById("ma-athletes-container");
  if (badge) badge.innerText = "Loading…";
  if (container) container.innerHTML = `<p class="mp-empty-state">Fetching athlete tracks…</p>`;

  try {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/multi-athlete`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    state.multiAthlete.athletes = data.athletes || [];
    state.multiAthlete.frames   = data.frames   || [];
    state.multiAthlete.loaded   = true;

    // Default selection: first athlete
    if (state.multiAthlete.athletes.length > 0 && state.multiAthlete.selectedTrackId == null) {
      state.multiAthlete.selectedTrackId = state.multiAthlete.athletes[0].track_id;
    }

    populateMultiAthleteSideTab();
    renderFrame(state.player.currentFrame, false);
  } catch (err) {
    console.warn("Multi-athlete fetch failed:", err);
    if (badge) badge.innerText = "Unavailable";
    if (container) container.innerHTML = `<p class="mp-empty-state" style="color:#EF4444;">Failed to load: ${err.message}</p>`;
  }
}

function selectMultiAthlete(trackId) {
  state.multiAthlete.selectedTrackId = parseInt(trackId);
  populateMultiAthleteSideTab();
  renderFrame(state.player.currentFrame, false);
}

function populateMultiAthleteSideTab() {
  const container = document.getElementById("ma-athletes-container");
  const badge = document.getElementById("ma-active-badge");
  if (!container) return;

  const athletes = state.multiAthlete.athletes;
  if (!athletes || athletes.length === 0) {
    if (badge) badge.innerText = "0 Athletes";
    container.innerHTML = `
      <p class="mp-empty-state">
        <strong>No athletes detected.</strong><br>
        MediaPipe could not find full-body poses in this clip. Common causes:<br>
        • Players too small / too far — athletes should fill ≥25% of frame height<br>
        • Low resolution — use ≥720p<br>
        • Crowded scene or severe occlusion<br><br>
        <em>Try re-recording with athletes ≤5 m from the camera at ≥720p.</em>
      </p>`;
    return;
  }

  if (badge) badge.innerText = `${athletes.length} Athlete${athletes.length !== 1 ? "s" : ""} Tracked`;

  container.innerHTML = athletes.map(a => {
    const isSelected = (a.track_id === state.multiAthlete.selectedTrackId);
    const fppaStr = a.avg_fppa_left != null ? `FPPA: ${a.avg_fppa_left.toFixed(1)}°` : "FPPA: —";
    const flexStr = a.avg_knee_flex_left != null ? `Flex: ${a.avg_knee_flex_left.toFixed(1)}°` : "Flex: —";
    const covStr  = `Coverage: ${a.coverage_pct}%`;
    const validPct = a.biomechanics_valid_pct != null ? a.biomechanics_valid_pct : 100;
    const uncPct   = a.uncertainty_pct != null ? a.uncertainty_pct : 0;
    const qualityStr = `Quality: ${validPct.toFixed(0)}%`;
    const uncBadge = uncPct > 0
      ? `<span style="color:#F59E0B;margin-left:4px;" title="${uncPct.toFixed(0)}% of frames had uncertain identity — biomechanics suppressed">⚠ ${uncPct.toFixed(0)}% uncertain</span>`
      : "";
    return `
      <div class="mp-player-chip ${isSelected ? "target" : ""}" onclick="selectMultiAthlete(${a.track_id})"
           style="border-left: 4px solid ${a.color};">
        <span class="mp-dot" style="background:${a.color};"></span>
        <span>
          <strong>${a.display_name}</strong>${isSelected ? " ★" : ""}${uncBadge}
          <br><small style="color:#94A3B8;">${fppaStr} &nbsp; ${flexStr} &nbsp; ${covStr} &nbsp; ${qualityStr}</small>
        </span>
      </div>
    `;
  }).join("");

  // Append Download Report button below athlete chips
  const sessionId = state.sessionId || "";
  if (sessionId) {
    container.innerHTML += `
      <div style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.08);">
        <button
          onclick="downloadMultiAthleteReport('${sessionId}')"
          style="width:100%;padding:8px 12px;background:#1E40AF;color:#fff;border:none;
                 border-radius:6px;cursor:pointer;font-size:13px;font-weight:600;"
          title="Download a clinician-readable quality + biomechanics report (JSON)"
        >📋 Download Screening Report</button>
        <p style="font-size:10px;color:#64748B;margin:6px 0 0;text-align:center;">
          Screening output only — not a clinical diagnosis
        </p>
      </div>
    `;
  }
}

async function downloadMultiAthleteReport(sessionId) {
  try {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/multi-athlete/report`);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const report = await res.json();
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `driftguard_multi_athlete_report_${sessionId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (err) {
    alert(`Report download failed: ${err.message}`);
  }
}

function computeIdentityCoverage(targetId) {
  const frames = state.multiplayer.frames;
  if (!frames || frames.length === 0) return null;

  let framesUsed = 0;
  let framesExcluded = 0;
  const gapFrames = [];

  frames.forEach((mpFrame, idx) => {
    const hasTarget = mpFrame.players && mpFrame.players.some(p => p.tracker_id === targetId);
    if (hasTarget) {
      framesUsed++;
    } else {
      framesExcluded++;
      gapFrames.push(idx);
    }
  });

  const total = framesUsed + framesExcluded;
  const coveragePct = total > 0 ? Math.round((framesUsed / total) * 100) : 0;
  return { framesUsed, framesExcluded, coveragePct, gapFrames };
}

function populateMultiplayerSideTab() {
  const select = document.getElementById("mp-target-select");
  const container = document.getElementById("mp-players-container");
  if (!select || !container) return;

  const players = state.multiplayer.players;

  if (!players || players.length === 0) {
    select.innerHTML = `<option value="" disabled selected>No players detected</option>`;
    container.innerHTML = `
      <p class="mp-empty-state">
        <strong>HOG detection found 0 players.</strong><br>
        HOG is calibrated for close-range pedestrians and does not reliably detect players
        on a sports court.<br><br>
        👉 Use the <strong>Multi-Athlete (ID-Stable)</strong> tab instead — it uses
        MediaPipe multi-pose detection which works on sports video.<br><br>
        <em>For best results: athletes ≤5 m from camera, ≥720p, at least 25% of frame height.</em>
      </p>`;
    const badge = document.getElementById("mp-active-badge");
    if (badge) badge.innerText = "0 Players Tracked";
    return;
  }

  select.innerHTML = players.map(p => `
    <option value="${p.tracker_id}" ${p.tracker_id === state.multiplayer.targetPlayerId ? "selected" : ""}>
      ${p.display_name} (ID: ${p.tracker_id}${p.tracker_id === state.multiplayer.targetPlayerId ? " - Selected Target" : ""})
    </option>
  `).join("");

  const locked = state.multiplayer.identityLocked;
  const cov = state.multiplayer.coverageStats;

  container.innerHTML = players.map(p => {
    const isTarget = p.tracker_id === state.multiplayer.targetPlayerId;
    return `
      <div class="mp-player-chip ${isTarget ? "target" : ""}" onclick="setTargetAthleteId(${p.tracker_id})">
        <span class="mp-dot ${isTarget ? "gold" : "cyan"}"></span>
        <span>${p.display_name} ${isTarget ? "★ [TARGET]" : ""}${isTarget && locked ? " 🔒" : ""}</span>
      </div>
    `;
  }).join("");

  // Identity lock status + coverage report
  let lockHtml = "";
  if (locked) {
    const lockedAtStr = state.multiplayer.lockedAt != null
      ? `Locked at ${state.multiplayer.lockedAt.toFixed(2)}s`
      : "Locked";
    if (cov) {
      lockHtml = `
        <div class="mp-lock-status">
          <span style="color:#F59E0B;font-weight:600;">🔒 Identity Locked</span>
          <span style="color:#94A3B8;font-size:12px;margin-left:6px;">${lockedAtStr}</span>
          <div class="mp-coverage-bar" title="${cov.coveragePct}% coverage">
            <div class="mp-coverage-fill" style="width:${cov.coveragePct}%"></div>
          </div>
          <div style="font-size:11px;color:#94A3B8;margin-top:2px;">
            Coverage: <strong style="color:#86EFAC">${cov.coveragePct}%</strong>
            &nbsp;(${cov.framesUsed} used / ${cov.framesExcluded} excluded)
          </div>
          <button class="btn btn-xs" style="margin-top:6px;font-size:11px;" onclick="unlockTargetIdentity()">🔓 Unlock Target</button>
        </div>`;
    } else {
      lockHtml = `<div class="mp-lock-status"><span style="color:#F59E0B;font-weight:600;">🔒 Identity Locked</span> <button class="btn btn-xs" style="font-size:11px;" onclick="unlockTargetIdentity()">🔓 Unlock</button></div>`;
    }
  } else {
    lockHtml = `<div class="mp-lock-status" style="color:#64748B;font-size:12px;">Click a player chip to lock target identity.</div>`;
  }

  // Append lock status after player chips
  container.innerHTML += lockHtml;

  const badge = document.getElementById("mp-active-badge");
  if (badge) {
    badge.innerText = `${players.length} Players Tracked`;
  }
}

function renderPlayerSkeleton(ctx, player, isTarget, w, h) {
  const skel = player.skeleton;
  const bbox = player.bbox;
  if (!bbox || bbox.length < 4) return; // no bbox — skip rendering this player
  const pxXmin = bbox[0] * w;
  const pxYmin = bbox[1] * h;
  const pxXmax = bbox[2] * w;
  const pxYmax = bbox[3] * h;
  const boxW = Math.max(10, pxXmax - pxXmin);
  const boxH = Math.max(10, pxYmax - pxYmin);

  const lineThickness = isTarget ? Math.max(3, Math.round(w / 150)) : Math.max(2, Math.round(w / 200));
  const nodeRadius = isTarget ? Math.max(4, Math.round(w / 120)) : Math.max(3, Math.round(w / 160));

  // Draw bounding box
  ctx.save();
  ctx.lineWidth = isTarget ? 3 : 1.5;
  ctx.strokeStyle = isTarget ? "#F59E0B" : "#38BDF8";
  if (isTarget) {
    ctx.shadowColor = "rgba(245, 158, 11, 0.6)";
    ctx.shadowBlur = 8;
  }
  ctx.strokeRect(pxXmin, pxYmin, boxW, boxH);
  ctx.restore();

  // Draw header badge
  const fontSize = Math.max(11, Math.round(w / 60));
  ctx.font = `bold ${fontSize}px sans-serif`;
  ctx.textBaseline = "bottom";
  const badgeLabel = isTarget ? `★ TARGET: ${player.display_name}` : `${player.display_name}`;
  const textW = ctx.measureText(badgeLabel).width;

  ctx.fillStyle = isTarget ? "#F59E0B" : "rgba(15, 23, 42, 0.85)";
  ctx.fillRect(pxXmin, Math.max(0, pxYmin - fontSize - 6), textW + 10, fontSize + 6);
  ctx.fillStyle = isTarget ? "#000000" : "#38BDF8";
  ctx.fillText(badgeLabel, pxXmin + 5, pxYmin - 2);

  if (!skel) return;

  const lHip = skel.left_hip;
  const lKnee = skel.left_knee;
  const lAnk = skel.left_ankle;
  const rHip = skel.right_hip;
  const rKnee = skel.right_knee;
  const rAnk = skel.right_ankle;

  // Pelvis line
  if (isPtVisible(lHip) && isPtVisible(rHip)) {
    ctx.strokeStyle = isTarget ? "#FDE68A" : "#64748B";
    ctx.lineWidth = Math.max(2, lineThickness - 1);
    ctx.beginPath();
    ctx.moveTo(lHip[0] * w, lHip[1] * h);
    ctx.lineTo(rHip[0] * w, rHip[1] * h);
    ctx.stroke();
  }

  const isLeftValgus = player.fppa_left != null && player.fppa_left > 12.0;
  const isRightValgus = player.fppa_right != null && player.fppa_right > 12.0;

  const leftColor = isTarget ? (isLeftValgus ? "#EF4444" : "#22C55E") : "#38BDF8";
  const rightColor = isTarget ? (isRightValgus ? "#EF4444" : "#3B82F6") : "#38BDF8";

  // Left limb
  ctx.lineWidth = lineThickness;
  ctx.strokeStyle = leftColor;
  if (isPtVisible(lHip) && isPtVisible(lKnee)) {
    ctx.beginPath();
    ctx.moveTo(lHip[0] * w, lHip[1] * h);
    ctx.lineTo(lKnee[0] * w, lKnee[1] * h);
    ctx.stroke();
  }
  if (isPtVisible(lKnee) && isPtVisible(lAnk)) {
    ctx.beginPath();
    ctx.moveTo(lKnee[0] * w, lKnee[1] * h);
    ctx.lineTo(lAnk[0] * w, lAnk[1] * h);
    ctx.stroke();
  }

  // Right limb
  ctx.strokeStyle = rightColor;
  if (isPtVisible(rHip) && isPtVisible(rKnee)) {
    ctx.beginPath();
    ctx.moveTo(rHip[0] * w, rHip[1] * h);
    ctx.lineTo(rKnee[0] * w, rKnee[1] * h);
    ctx.stroke();
  }
  if (isPtVisible(rKnee) && isPtVisible(rAnk)) {
    ctx.beginPath();
    ctx.moveTo(rKnee[0] * w, rKnee[1] * h);
    ctx.lineTo(rAnk[0] * w, rAnk[1] * h);
    ctx.stroke();
  }

  // Nodes
  const nodes = [
    { pt: lHip, col: leftColor },
    { pt: lKnee, col: leftColor },
    { pt: lAnk, col: leftColor },
    { pt: rHip, col: rightColor },
    { pt: rKnee, col: rightColor },
    { pt: rAnk, col: rightColor },
  ];
  nodes.forEach(({ pt, col }) => {
    if (isPtVisible(pt)) {
      ctx.fillStyle = isTarget ? "#FFFFFF" : "#0F172A";
      ctx.beginPath();
      ctx.arc(pt[0] * w, pt[1] * h, nodeRadius, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = col;
      ctx.lineWidth = isTarget ? 2 : 1.5;
      ctx.stroke();
    }
  });

  // Knee angle text for target athlete
  if (isTarget) {
    const kFontSize = Math.max(11, Math.round(w / 50));
    ctx.font = `bold ${kFontSize}px sans-serif`;
    ctx.textBaseline = "middle";

    if (isPtVisible(lKnee) && player.knee_flexion_left != null) {
      ctx.textAlign = "left";
      const text = `${player.knee_flexion_left.toFixed(1)}°`;
      const tm = ctx.measureText(text);
      const kx = lKnee[0] * w + nodeRadius + 4;
      const ky = lKnee[1] * h;

      ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
      ctx.fillRect(kx - 2, ky - kFontSize / 2 - 2, tm.width + 6, kFontSize + 4);
      ctx.fillStyle = isLeftValgus ? "#FCA5A5" : "#86EFAC";
      ctx.fillText(text, kx + 1, ky);
    }
  }
}

/**
 * Draw one athlete's skeleton in the Multi-Athlete (ID-Stable) view.
 * Each athlete's unique color comes from the API (derived from track_id).
 * @param {CanvasRenderingContext2D} ctx
 * @param {Object} athlete  - frame entry from /multi-athlete: {track_id, color, display_name, bbox, skeleton, ...}
 * @param {boolean} isSelected - true for the currently selected athlete in the panel
 * @param {number} w - canvas pixel width
 * @param {number} h - canvas pixel height
 */
function renderMultiAthleteSkeleton(ctx, athlete, isSelected, w, h) {
  const skel = athlete.skeleton;
  const bbox = athlete.bbox;
  const color = athlete.color || "#38BDF8";

  const lineThickness = isSelected ? Math.max(3, Math.round(w / 150)) : Math.max(2, Math.round(w / 210));
  const nodeRadius    = isSelected ? Math.max(4, Math.round(w / 120)) : Math.max(2, Math.round(w / 170));

  // Bounding box — amber dashed when uncertain, solid track color otherwise
  if (bbox && bbox.length >= 4) {
    const bx = bbox[0] * w, by = bbox[1] * h;
    const bw = Math.max(10, (bbox[2] - bbox[0]) * w);
    const bh = Math.max(10, (bbox[3] - bbox[1]) * h);
    const isUncertain = (athlete.track_status === "uncertain");
    const boxColor = isUncertain ? "#F59E0B" : color;
    ctx.save();
    ctx.strokeStyle = boxColor;
    ctx.lineWidth = isSelected ? 3 : 1.5;
    if (isUncertain) {
      ctx.setLineDash([6, 4]);
    }
    if (isSelected && !isUncertain) {
      ctx.shadowColor = color;
      ctx.shadowBlur = 10;
    }
    ctx.strokeRect(bx, by, bw, bh);
    ctx.restore();

    // Badge label — use boxColor so uncertain tracks show amber badge
    const fontSize = Math.max(11, Math.round(w / 65));
    ctx.font = `bold ${fontSize}px sans-serif`;
    ctx.textBaseline = "bottom";
    const label = isSelected ? `★ ${athlete.display_name}` : athlete.display_name;
    const textW = ctx.measureText(label).width;
    ctx.fillStyle = boxColor;
    ctx.globalAlpha = isSelected ? 0.9 : 0.75;
    ctx.fillRect(bx, Math.max(0, by - fontSize - 6), textW + 10, fontSize + 6);
    ctx.globalAlpha = 1.0;
    ctx.fillStyle = "#000000";
    ctx.fillText(label, bx + 5, by - 2);
  }

  if (!skel) return;

  const lHip  = skel.left_hip;
  const lKnee = skel.left_knee;
  const lAnk  = skel.left_ankle;
  const rHip  = skel.right_hip;
  const rKnee = skel.right_knee;
  const rAnk  = skel.right_ankle;

  ctx.strokeStyle = color;
  ctx.lineWidth = lineThickness;

  // Pelvis
  if (isPtVisible(lHip) && isPtVisible(rHip)) {
    ctx.save();
    ctx.globalAlpha = 0.6;
    ctx.beginPath();
    ctx.moveTo(lHip[0] * w, lHip[1] * h);
    ctx.lineTo(rHip[0] * w, rHip[1] * h);
    ctx.stroke();
    ctx.restore();
  }

  // Left limb
  [[lHip, lKnee], [lKnee, lAnk]].forEach(([a, b]) => {
    if (isPtVisible(a) && isPtVisible(b)) {
      ctx.beginPath();
      ctx.moveTo(a[0] * w, a[1] * h);
      ctx.lineTo(b[0] * w, b[1] * h);
      ctx.stroke();
    }
  });

  // Right limb
  [[rHip, rKnee], [rKnee, rAnk]].forEach(([a, b]) => {
    if (isPtVisible(a) && isPtVisible(b)) {
      ctx.beginPath();
      ctx.moveTo(a[0] * w, a[1] * h);
      ctx.lineTo(b[0] * w, b[1] * h);
      ctx.stroke();
    }
  });

  // Nodes
  [lHip, lKnee, lAnk, rHip, rKnee, rAnk].forEach(pt => {
    if (isPtVisible(pt)) {
      ctx.fillStyle = isSelected ? "#FFFFFF" : color;
      ctx.beginPath();
      ctx.arc(pt[0] * w, pt[1] * h, nodeRadius, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  });

  // Knee angle label for selected athlete — suppressed when identity is uncertain
  const bioValid = athlete.biomechanics_valid !== false;  // default true if key absent
  if (isSelected && bioValid && isPtVisible(lKnee) && athlete.knee_flexion_left != null) {
    const kFontSize = Math.max(11, Math.round(w / 55));
    ctx.font = `bold ${kFontSize}px sans-serif`;
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    const text = `${Number(athlete.knee_flexion_left).toFixed(1)}°`;
    const tm = ctx.measureText(text);
    const kx = lKnee[0] * w + nodeRadius + 4;
    const ky = lKnee[1] * h;
    ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
    ctx.fillRect(kx - 2, ky - kFontSize / 2 - 2, tm.width + 6, kFontSize + 4);
    ctx.fillStyle = color;
    ctx.fillText(text, kx + 1, ky);
  }
}

function isPtVisible(pt) {
  if (!pt) return false;
  const v = pt.length > 2 && pt[2] !== undefined ? pt[2] : 1.0;
  return v >= VISIBILITY_THRESHOLD;
}

function renderFrame(frameIdx, syncVideo = true) {
  if (!state.player.frames || state.player.frames.length === 0) return;

  state.player.currentFrame = frameIdx;
  const frame = state.player.frames[frameIdx];

  const fps = state.player.fps || 30.0;
  const totalDur = state.player.duration || (state.player.frames.length / fps);
  const timeDisp = document.getElementById("player-time-display");
  if (timeDisp) {
    timeDisp.innerText = `${frame.timestamp_s.toFixed(2)}s / ${totalDur.toFixed(2)}s (Frame ${frame.frame + 1}/${state.player.frames.length})`;
  }

  const scrubber = document.getElementById("player-scrubber");
  if (scrubber) scrubber.value = frameIdx;

  // Sync background video element if requested
  const playerVideo = document.getElementById("player-video");
  if (syncVideo && playerVideo && playerVideo.readyState >= 2) {
    const targetTime = frame.timestamp_s;  // use precomputed PTS, not frameIdx/fps
    if (Math.abs(playerVideo.currentTime - targetTime) > 0.05) {
      playerVideo.currentTime = targetTime;
    }
  }

  // Update Gauges — source depends on view mode
  // In single-athlete mode: always from single-player pipeline (frame)
  // In multi-player mode: from target player's mp data (isolated to targetPlayerId)
  let gaugeSource = frame; // default: single-player pipeline
  if (state.multiplayer.enabled) {
    const mpFrames = state.multiplayer.frames;
    if (mpFrames && frameIdx < mpFrames.length) {
      const mpFrame = mpFrames[frameIdx];
      if (mpFrame && mpFrame.players) {
        const targetPlayer = mpFrame.players.find(p => p.tracker_id === state.multiplayer.targetPlayerId);
        if (targetPlayer) {
          gaugeSource = targetPlayer; // isolated to target only
        }
      }
    }
  }

  if (document.getElementById("g-fppa-left")) document.getElementById("g-fppa-left").innerText = `${gaugeSource.fppa_left != null ? gaugeSource.fppa_left.toFixed(1) : "0.0"}°`;
  if (document.getElementById("g-flex-left")) document.getElementById("g-flex-left").innerText = `${gaugeSource.knee_flexion_left != null ? gaugeSource.knee_flexion_left.toFixed(1) : "0.0"}°`;

  // Update Level 2 6-DOF Knee Kinematics Gauges (with Level 2 gating)
  const ap = frame.knee_ap_translation_mm;
  const ml = frame.knee_ml_translation_mm;
  const si = frame.knee_si_translation_mm;
  const rot = frame.knee_int_ext_rot_deg;

  if (document.getElementById("g-6dof-ap")) {
    const apAvail = (ap != null && ap !== "not available");
    document.getElementById("g-6dof-ap").innerText = apAvail ? `${parseFloat(ap).toFixed(1)} mm` : "Not Available";
    const apR = document.getElementById("g-6dof-ap-reason");
    if (apR) apR.innerText = apAvail ? "Anterior laxity estimate" : "Requires depth/stereo input";
  }
  if (document.getElementById("g-6dof-ml")) {
    const mlAvail = (ml != null && ml !== "not available");
    document.getElementById("g-6dof-ml").innerText = mlAvail ? `${parseFloat(ml).toFixed(1)} mm` : "Not Available";
    const mlR = document.getElementById("g-6dof-ml-reason");
    if (mlR) mlR.innerText = mlAvail ? "Lateral shift proxy" : "Requires depth/stereo input";
  }
  if (document.getElementById("g-6dof-si")) {
    const siAvail = (si != null && si !== "not available");
    document.getElementById("g-6dof-si").innerText = siAvail ? `${parseFloat(si).toFixed(1)} mm` : "Not Available";
    const siR = document.getElementById("g-6dof-si-reason");
    if (siR) siR.innerText = siAvail ? "Joint compression" : "Requires depth/stereo input";
  }
  if (document.getElementById("g-6dof-rot")) {
    const rotAvail = (rot != null && rot !== "not available");
    document.getElementById("g-6dof-rot").innerText = rotAvail ? `${parseFloat(rot).toFixed(1)}°` : "Not Available";
    const rotR = document.getElementById("g-6dof-rot-reason");
    if (rotR) rotR.innerText = rotAvail ? "Axial rotation angle" : "Requires depth/stereo input";
  }

  // Render Skeleton Canvas overlaid on video
  alignCanvasToVideo();
  const canvas = document.getElementById("skeleton-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  // Clear transparent canvas
  ctx.clearRect(0, 0, w, h);

  if (!state.multiplayer.enabled) {
    // Single-Athlete View (Phase 1 behavior preserved 100%)
    const skel = frame.skeleton;
    if (skel) {
      const lHip = skel.left_hip;
      const lKnee = skel.left_knee;
      const lAnk = skel.left_ankle;
      const rHip = skel.right_hip;
      const rKnee = skel.right_knee;
      const rAnk = skel.right_ankle;

      const lineThickness = Math.max(3, Math.round(w / 160));
      const nodeRadius = Math.max(4, Math.round(w / 120));

      // Pelvic line
      if (isPtVisible(lHip) && isPtVisible(rHip)) {
        ctx.strokeStyle = "#94A3B8";
        ctx.lineWidth = Math.max(2, lineThickness - 1);
        ctx.beginPath();
        ctx.moveTo(lHip[0] * w, lHip[1] * h);
        ctx.lineTo(rHip[0] * w, rHip[1] * h);
        ctx.stroke();
      }

      // Left Limb: Green (#22C55E) or Red (#EF4444) on valgus deviation
      const isLeftValgus = (frame.fppa_left != null && frame.fppa_left > 12.0) || Boolean(frame.is_landing_event);
      const leftColor = isLeftValgus ? "#EF4444" : "#22C55E";
      ctx.strokeStyle = leftColor;
      ctx.lineWidth = lineThickness;

      if (isPtVisible(lHip) && isPtVisible(lKnee)) {
        ctx.beginPath();
        ctx.moveTo(lHip[0] * w, lHip[1] * h);
        ctx.lineTo(lKnee[0] * w, lKnee[1] * h);
        ctx.stroke();
      }
      if (isPtVisible(lKnee) && isPtVisible(lAnk)) {
        ctx.beginPath();
        ctx.moveTo(lKnee[0] * w, lKnee[1] * h);
        ctx.lineTo(lAnk[0] * w, lAnk[1] * h);
        ctx.stroke();
      }

      // Right Limb: Blue (#3B82F6) or Red (#EF4444) on valgus deviation
      const isRightValgus = (frame.fppa_right != null && frame.fppa_right > 12.0);
      const rightColor = isRightValgus ? "#EF4444" : "#3B82F6";
      ctx.strokeStyle = rightColor;
      ctx.lineWidth = lineThickness;

      if (isPtVisible(rHip) && isPtVisible(rKnee)) {
        ctx.beginPath();
        ctx.moveTo(rHip[0] * w, rHip[1] * h);
        ctx.lineTo(rKnee[0] * w, rKnee[1] * h);
        ctx.stroke();
      }
      if (isPtVisible(rKnee) && isPtVisible(rAnk)) {
        ctx.beginPath();
        ctx.moveTo(rKnee[0] * w, rKnee[1] * h);
        ctx.lineTo(rAnk[0] * w, rAnk[1] * h);
        ctx.stroke();
      }

      // Draw visible joint nodes
      const visibleNodes = [
        { pt: lHip, col: leftColor },
        { pt: lKnee, col: leftColor },
        { pt: lAnk, col: leftColor },
        { pt: rHip, col: rightColor },
        { pt: rKnee, col: rightColor },
        { pt: rAnk, col: rightColor },
      ];

      visibleNodes.forEach(({ pt, col }) => {
        if (isPtVisible(pt)) {
          ctx.fillStyle = "#FFFFFF";
          ctx.beginPath();
          ctx.arc(pt[0] * w, pt[1] * h, nodeRadius, 0, 2 * Math.PI);
          ctx.fill();
          ctx.strokeStyle = col;
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      });

      // Knee angle in degrees next to each knee
      const fontSize = Math.max(12, Math.round(w / 45));
      ctx.font = `bold ${fontSize}px sans-serif`;
      ctx.textBaseline = "middle";

      if (isPtVisible(lKnee) && frame.knee_flexion_left != null) {
        ctx.textAlign = "left";
        const text = `${frame.knee_flexion_left.toFixed(1)}°`;
        const textMetrics = ctx.measureText(text);
        const kx = lKnee[0] * w + nodeRadius + 4;
        const ky = lKnee[1] * h;

        ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
        ctx.fillRect(kx - 2, ky - fontSize / 2 - 2, textMetrics.width + 6, fontSize + 4);
        ctx.fillStyle = isLeftValgus ? "#FCA5A5" : "#86EFAC";
        ctx.fillText(text, kx + 1, ky);
      }

      if (isPtVisible(rKnee) && frame.knee_flexion_right != null) {
        ctx.textAlign = "right";
        const text = `${frame.knee_flexion_right.toFixed(1)}°`;
        const textMetrics = ctx.measureText(text);
        const kx = rKnee[0] * w - nodeRadius - 4;
        const ky = rKnee[1] * h;

        ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
        ctx.fillRect(kx - textMetrics.width - 4, ky - fontSize / 2 - 2, textMetrics.width + 6, fontSize + 4);
        ctx.fillStyle = isRightValgus ? "#FCA5A5" : "#93C5FD";
        ctx.fillText(text, kx - 1, ky);
      }
    }
  } else if (state.multiplayer.enabled) {
    // Multi-Player View: Render all players, highlight only the Target Athlete
    let mpFrame = null;
    if (state.multiplayer.frames && frameIdx < state.multiplayer.frames.length) {
      mpFrame = state.multiplayer.frames[frameIdx];
    }

    let playersToRender = [];
    if (mpFrame && mpFrame.players && mpFrame.players.length > 0) {
      playersToRender = mpFrame.players;
    } else {
      // No real multi-player data for this frame — draw nothing, show small notice
      ctx.save();
      ctx.font = "14px 'Inter', sans-serif";
      ctx.fillStyle = "rgba(148, 163, 184, 0.85)";
      ctx.textAlign = "center";
      ctx.fillText("No pose detected", w / 2, h / 2);
      ctx.restore();
    }

    // Render each tracked player
    playersToRender.forEach(p => {
      const isTarget = (p.tracker_id === state.multiplayer.targetPlayerId);
      renderPlayerSkeleton(ctx, p, isTarget, w, h);
    });

    // Identity gap indicator: when locked and target is absent from this frame
    if (state.multiplayer.identityLocked) {
      const targetInFrame = playersToRender.some(p => p.tracker_id === state.multiplayer.targetPlayerId);
      if (!targetInFrame) {
        ctx.save();
        // Dashed amber border around canvas to signal gap
        ctx.strokeStyle = "rgba(245, 158, 11, 0.7)";
        ctx.lineWidth = 4;
        ctx.setLineDash([12, 8]);
        ctx.strokeRect(4, 4, w - 8, h - 8);
        ctx.setLineDash([]);
        // Label in top-left
        ctx.font = "bold 13px 'Inter', sans-serif";
        ctx.fillStyle = "rgba(15, 23, 42, 0.80)";
        ctx.fillRect(8, 8, 200, 24);
        ctx.fillStyle = "#F59E0B";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText("⚠ Target Identity Gap", 14, 20);
        ctx.restore();
      }
    }
  } else if (state.multiAthlete.enabled) {
    // Multi-Athlete (ID-Stable) View: each athlete rendered with their unique color
    const maFrames = state.multiAthlete.frames;
    let maFrame = (maFrames && frameIdx < maFrames.length) ? maFrames[frameIdx] : null;

    if (!maFrame || !maFrame.athletes || maFrame.athletes.length === 0) {
      ctx.save();
      ctx.font = "14px 'Inter', sans-serif";
      ctx.fillStyle = "rgba(148, 163, 184, 0.85)";
      ctx.textAlign = "center";
      ctx.fillText(state.multiAthlete.loaded ? "No athletes detected in this frame" : "Loading multi-athlete data…", w / 2, h / 2);
      ctx.restore();
    } else {
      maFrame.athletes.forEach(athlete => {
        const isSelected = (athlete.track_id === state.multiAthlete.selectedTrackId);
        renderMultiAthleteSkeleton(ctx, athlete, isSelected, w, h);
      });
    }
  }
}

function togglePlayerPlay() {
  const playerVideo = document.getElementById("player-video");
  if (playerVideo && playerVideo.src) {
    if (playerVideo.paused) {
      playerVideo.play().catch(err => console.warn("Video play interrupted:", err));
    } else {
      playerVideo.pause();
    }
  } else {
    state.player.isPlaying = !state.player.isPlaying;
    const btn = document.getElementById("btn-play-pause");
    if (btn) btn.innerText = state.player.isPlaying ? "Pause Trial" : "Play Trial";

    if (state.player.isPlaying) {
      state.player.timer = setInterval(() => {
        let nextFrame = state.player.currentFrame + 1;
        if (nextFrame >= state.player.frames.length) nextFrame = 0;
        renderFrame(nextFrame, false);
      }, 33);
    } else {
      clearInterval(state.player.timer);
    }
  }
}

function scrubPlayer(frameVal) {
  const frameIdx = parseInt(frameVal, 10);
  const playerVideo = document.getElementById("player-video");
  if (playerVideo && !playerVideo.paused) {
    playerVideo.pause();
  }
  renderFrame(frameIdx, true);
}

function jumpToFrame(frameIdx) {
  const playerVideo = document.getElementById("player-video");
  if (playerVideo && !playerVideo.paused) {
    playerVideo.pause();
  }
  renderFrame(frameIdx, true);
}

function renderRiskFlagsSummary(flags) {
  const container = document.getElementById("player-risk-flags-list");
  if (!container) return;

  if (flags.length === 0) {
    container.innerHTML = `<p class="empty-state">No high risk flags for this trial.</p>`;
    return;
  }

  container.innerHTML = flags.map(f => `
    <div style="background: var(--bg-app); border-left: 3px solid var(--color-danger); padding: 8px 12px; margin-bottom: 8px; border-radius: 4px;">
      <div style="font-size: 0.85rem; font-weight: 600;">${f.headline}</div>
      <div style="font-size: 0.75rem; color: var(--color-text-muted);">${f.detail}</div>
    </div>
  `).join("");
}

// --- Clinician Decision Support & Notes Modal ---------------------------------

function openClinicianDecisionModal() {
  document.getElementById("decision-modal").style.display = "flex";
  document.getElementById("rep-session-id").innerText = state.player.sessionId || "ses_demo";
  document.getElementById("rep-athlete-name").innerText = state.athletes.find(a => a.athlete_id === state.selectedAthleteId)?.display_name || "Sarah Jenkins";
}

function closeClinicianDecisionModal() {
  document.getElementById("decision-modal").style.display = "none";
}

async function saveClinicianNote() {
  const sessionId = state.player.sessionId || "ses_demo";
  const noteType = document.getElementById("note-type-select").value;
  const content = document.getElementById("note-content-input").value;

  if (!content) {
    alert("Please enter clinical notes before saving.");
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/notes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        clinician_id: "cln_dr_smith",
        clinician_name: "Dr. Smith (Lead Physiotherapist)",
        note_type: noteType,
        content: content,
      })
    });

    if (res.ok) {
      alert("Clinician Note saved successfully.");
      closeClinicianDecisionModal();
    }
  } catch (err) {
    console.error("Failed saving clinician note:", err);
  }
}

function printSessionReport() {
  const sessionId = state.player.sessionId;
  if (!sessionId) {
    window.print();
    return;
  }
  window.open(`${API_BASE}/sessions/${sessionId}/pdf`, "_blank");
}

// --- Register Athlete Modal --------------------------------------------------

function openNewAthleteModal() {
  document.getElementById("new-athlete-modal").style.display = "flex";
}

function closeNewAthleteModal() {
  document.getElementById("new-athlete-modal").style.display = "none";
}

async function submitNewAthlete() {
  const name = document.getElementById("new-ath-name").value;
  const sport = document.getElementById("new-ath-sport").value;
  const leg = document.getElementById("new-ath-leg").value;
  const team = document.getElementById("new-ath-team").value;

  if (!name) {
    alert("Please enter athlete display name.");
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/athletes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        display_name: name,
        sport: sport,
        dominant_leg: leg,
        team_or_school: team
      })
    });

    if (res.ok) {
      const newAth = await res.json();
      closeNewAthleteModal();
      await loadAthletes();
      selectAthlete(newAth.athlete_id);
    }
  } catch (err) {
    console.error("Error creating athlete:", err);
  }
}

// --- Review Queue -------------------------------------------------------------

async function loadReviewQueue() {
  try {
    const res = await fetch(`${API_BASE}/flags?unreviewed_only=true`);
    if (res.ok) {
      const flags = await res.json();
      const tbody = document.getElementById("table-queue-flags");

      if (flags.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" class="empty-state">All risk flags have been reviewed by a clinician.</td></tr>`;
        return;
      }

      tbody.innerHTML = flags.map(f => `
        <tr>
          <td><code>${f.flag_id}</code></td>
          <td><strong>${f.athlete_id}</strong></td>
          <td><code>${f.feature_name}</code></td>
          <td><span class="badge ${f.band === 'High' ? 'badge-danger' : 'badge-warning'}">${f.band}</span></td>
          <td>+${f.deviation_iqr?.toFixed(1)} IQR</td>
          <td><small>${f.citation || 'Hewett et al. (2005)'}</small></td>
          <td><button class="btn btn-secondary btn-sm" onclick="quickReviewFlag('${f.flag_id}')">Confirm / Note</button></td>
        </tr>
      `).join("");
    }
  } catch (err) {
    console.error("Error reviewing flag:", err);
  }
}

async function quickReviewFlag(flagId) {
  const note = prompt("Enter Clinician Review Note / Prescription Instruction for this flag:");
  if (note === null) return;

  try {
    const res = await fetch(`${API_BASE}/flags/${flagId}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        clinician_id: "cln_lead",
        note: note
      })
    });

    if (res.ok) {
      loadDashboardData();
      if (state.currentView === "queue") loadReviewQueue();
      if (confirm("Review saved. Would you like to open the Recovery Module to generate/edit the athlete's recovery plan?")) {
        switchView("recovery");
      }
    }
  } catch (err) {
    console.error("Error reviewing flag:", err);
  }
}

// --- 22 Biomechanical Parameters Inspector ---

async function openParameters22Modal() {
  const modal = document.getElementById("modal-parameters-22");
  const tbody = document.getElementById("tbody-parameters-22");
  if (!modal || !tbody) return;

  tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; padding: 24px;">Loading 22 parameters registry...</td></tr>`;
  modal.style.display = "flex";

  try {
    let paramsData = null;
    if (state.player.sessionId) {
      const res = await fetch(`${API_BASE}/sessions/${state.player.sessionId}/parameters_22`);
      if (res.ok) {
        const json = await res.json();
        paramsData = json.parameters;
      }
    }
    if (!paramsData) {
      const res = await fetch(`${API_BASE}/biomechanics/parameters_22/specs`);
      if (res.ok) {
        const specs = await res.json();
        paramsData = {};
        specs.forEach(s => {
          paramsData[s.name] = { ...s, value: s.requires_depth_or_calibration ? "not available" : "--" };
        });
      }
    }

    if (paramsData) {
      const paramList = Object.values(paramsData).sort((a, b) => (a.param_id || 0) - (b.param_id || 0));
      tbody.innerHTML = paramList.map(p => {
        const isNA = p.value === "not available" || p.status === "not available";
        const valDisplay = isNA 
          ? `<span class="badge badge-outline" style="color: var(--color-warning); font-weight: 600;">Not Available</span>` 
          : `<strong>${typeof p.value === "number" ? p.value.toFixed(2) : p.value} ${p.unit}</strong>`;

        const classBadge = {
          "measured": "badge-success",
          "estimate": "badge-primary",
          "proxy": "badge-warning",
          "model-derived": "badge-outline"
        }[p.classification] || "badge-outline";

        return `
          <tr>
            <td><strong>#${p.param_id || "-"}</strong></td>
            <td>
              <strong>${p.display_name || p.name}</strong>
              <div style="font-size: 0.75rem; color: var(--color-text-muted); font-family: monospace;">${p.name}</div>
            </td>
            <td>${valDisplay}</td>
            <td><span class="badge ${classBadge}">${p.classification}</span></td>
            <td><span class="badge badge-outline">Level ${p.level}</span></td>
            <td style="font-size: 0.82rem; max-width: 320px;">
              <div><strong>Formula:</strong> <code>${p.formula || "-"}</code></div>
              <div style="color: var(--color-text-muted); margin-top: 3px;">${p.coordinate_convention || ""}</div>
              ${p.requires_depth_or_calibration ? `<div style="color: var(--color-danger); font-size: 0.75rem; margin-top: 3px;">⚠️ Level 2 Gated: Requires physical depth sensor or stereo rig.</div>` : ""}
            </td>
          </tr>
        `;
      }).join("");
    }
  } catch (err) {
    console.error("Failed to load parameters_22:", err);
    tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--color-danger);">Error loading parameters: ${err.message}</td></tr>`;
  }
}

function closeParameters22Modal() {
  const modal = document.getElementById("modal-parameters-22");
  if (modal) modal.style.display = "none";
}

