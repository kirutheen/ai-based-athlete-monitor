/**
 * Athlete Recovery Module (v3) Frontend Engine
 * Connects to FastAPI /api/reports/analyze, /api/plans, /api/days, /api/exercises/media
 *
 * CLINICAL NOTICE: Camera observations in this module are for reference and
 * exercise guidance only. They do not constitute a clinical diagnosis, medical
 * clearance, or modification of any treatment plan. All recovery plans remain
 * subject to clinician review and approval.
 */

let currentRecoveryState = {
  athlete_id: 'ath_demo1',
  report_id: null,
  extracted: null,
  active_plan: null,
  active_plan_id: null,
  selected_day: 1,
  current_day_log: null,
  exercise_stages: {} // { ex_name: 'teach' | 'demo' | 'track' }
};

// ---------------------------------------------------------------------------
// Shared Recovery Camera Manager
// One stream at a time; Start/Stop/Retry; releases on stop or module leave.
// Used by the top-level recovery panel AND per-exercise Stage 3 camera.
// ---------------------------------------------------------------------------
const RecoveryCameraManager = {
  _stream: null,
  _animFrame: null,
  _poseEngine: null,
  _activeCanvasId: null,
  _activeVideoId: null,
  _activeLabelId: null,
  _activeStatusId: null,

  /**
   * Start the camera and draw onto the given video/canvas pair.
   * @param {string} videoId   - id of <video> element
   * @param {string} canvasId  - id of <canvas> element
   * @param {string} labelId   - id of label span element (shows "Athlete 01")
   * @param {string} statusId  - id of status badge
   */
  async start(videoId, canvasId, labelId, statusId) {
    // Release any existing stream first
    this.stop();

    this._activeVideoId = videoId;
    this._activeCanvasId = canvasId;
    this._activeLabelId = labelId;
    this._activeStatusId = statusId;

    const statusEl = document.getElementById(statusId);
    if (statusEl) statusEl.innerText = 'Requesting camera…';

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      if (statusEl) statusEl.innerText = '⚠️ Webcam API not supported in this browser.';
      return;
    }

    try {
      this._stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      const video = document.getElementById(videoId);
      const canvas = document.getElementById(canvasId);
      if (!video || !canvas) { this.stop(); return; }

      video.srcObject = this._stream;
      video.style.display = 'none'; // video feed drawn onto canvas
      await video.play();

      canvas.width = 640;
      canvas.height = 480;

      if (statusEl) statusEl.innerText = '🟢 Camera active — Athlete 01 tracked';

      // Try to init MediaPipe Pose for gold skeleton overlay
      if (window.Pose) {
        this._poseEngine = new window.Pose({
          locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${f}`
        });
        this._poseEngine.setOptions({ modelComplexity: 0, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
        this._poseEngine.onResults((results) => {
          this._drawFrame(results, canvas);
        });
        // Frame loop
        const loop = async () => {
          if (!this._stream) return;
          const vid = document.getElementById(this._activeVideoId);
          if (vid && !vid.paused) {
            try { await this._poseEngine.send({ image: vid }); } catch (e) { /* ignore */ }
          }
          this._animFrame = requestAnimationFrame(loop);
        };
        this._animFrame = requestAnimationFrame(loop);
      } else {
        // Fallback: draw raw video without skeleton
        const fallbackLoop = () => {
          if (!this._stream) return;
          const vid = document.getElementById(this._activeVideoId);
          const cvs = document.getElementById(this._activeCanvasId);
          if (vid && cvs) {
            const ctx = cvs.getContext('2d');
            ctx.drawImage(vid, 0, 0, cvs.width, cvs.height);
            this._drawLabel(ctx, cvs.width, cvs.height, 'Athlete 01 (No pose engine)');
          }
          this._animFrame = requestAnimationFrame(fallbackLoop);
        };
        this._animFrame = requestAnimationFrame(fallbackLoop);
      }
    } catch (err) {
      if (statusEl) statusEl.innerText = '⚠️ Camera permission denied. Use manual self-log below.';
      this._stream = null;
    }
  },

  stop() {
    if (this._animFrame) {
      cancelAnimationFrame(this._animFrame);
      this._animFrame = null;
    }
    if (this._poseEngine) {
      try { this._poseEngine.close(); } catch (e) { /* ignore */ }
      this._poseEngine = null;
    }
    if (this._stream) {
      this._stream.getTracks().forEach(t => t.stop());
      this._stream = null;
    }
    // Clear canvas
    const canvas = this._activeCanvasId ? document.getElementById(this._activeCanvasId) : null;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    const statusEl = this._activeStatusId ? document.getElementById(this._activeStatusId) : null;
    if (statusEl) statusEl.innerText = 'Camera stopped.';
  },

  /** Draw pose frame with gold skeleton + Athlete 01 label. */
  _drawFrame(results, canvas) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (results.image) ctx.drawImage(results.image, 0, 0, w, h);

    if (results.poseLandmarks && results.poseLandmarks.length > 0) {
      this._drawGoldSkeleton(ctx, results.poseLandmarks, w, h);
      this._drawLabel(ctx, w, h, 'Athlete 01');
    }
    // No skeleton on detection loss — canvas cleared above, no stale lines.
  },

  /** Gold skeleton for recovery module (separate from screening wizard green/blue). */
  _drawGoldSkeleton(ctx, landmarks, w, h) {
    const GOLD = '#FFD700';
    const connections = [
      [23, 25], [25, 27], // Left Leg
      [24, 26], [26, 28], // Right Leg
      [23, 24],           // Pelvis
      [11, 12], [11, 23], [12, 24] // Torso
    ];
    ctx.lineWidth = 3;
    ctx.strokeStyle = GOLD;
    connections.forEach(([i, j]) => {
      const p1 = landmarks[i];
      const p2 = landmarks[j];
      if (p1 && p2 && p1.visibility > 0.4 && p2.visibility > 0.4) {
        ctx.beginPath();
        ctx.moveTo(p1.x * w, p1.y * h);
        ctx.lineTo(p2.x * w, p2.y * h);
        ctx.stroke();
      }
    });
    // Joint dots
    ctx.fillStyle = GOLD;
    [11, 12, 23, 24, 25, 26, 27, 28].forEach(i => {
      const p = landmarks[i];
      if (p && p.visibility > 0.4) {
        ctx.beginPath();
        ctx.arc(p.x * w, p.y * h, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  },

  /** Draw 'Athlete 01' label anchored above the topmost visible landmark. */
  _drawLabel(ctx, w, h, label) {
    // Use top-left corner as fallback anchor
    const px = w * 0.5;
    const py = 22;
    ctx.font = "bold 13px 'Inter', 'Segoe UI', sans-serif";
    ctx.textAlign = 'center';
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    const m = ctx.measureText(label);
    ctx.fillRect(px - m.width / 2 - 6, py - 14, m.width + 12, 20);
    ctx.fillStyle = '#FFD700';
    ctx.fillText(label, px, py);
    ctx.textAlign = 'start';
  }
};

/**
 * Call when navigating away from the Recovery module.
 * Releases camera stream if active.
 */
function stopRecoveryModule() {
  RecoveryCameraManager.stop();
}

document.addEventListener('DOMContentLoaded', () => {
  initRecoveryModule();
});

function initRecoveryModule() {
  const btnAnalyze = document.getElementById('btn-analyze-report');
  if (btnAnalyze) {
    btnAnalyze.addEventListener('click', handleAnalyzeReport);
  }

  const btnConfirm = document.getElementById('btn-confirm-plan');
  if (btnConfirm) {
    btnConfirm.addEventListener('click', handleConfirmPlan);
  }

  // Wire top-level recovery camera panel buttons
  const btnRecStart = document.getElementById('btn-rec-cam-start');
  const btnRecStop  = document.getElementById('btn-rec-cam-stop');
  const btnRecRetry = document.getElementById('btn-rec-cam-retry');
  if (btnRecStart) btnRecStart.addEventListener('click', () => {
    RecoveryCameraManager.start(
      'rec-live-video', 'rec-live-canvas', 'rec-cam-label', 'rec-cam-status'
    );
  });
  if (btnRecStop) btnRecStop.addEventListener('click', () => RecoveryCameraManager.stop());
  if (btnRecRetry) btnRecRetry.addEventListener('click', () => {
    RecoveryCameraManager.stop();
    RecoveryCameraManager.start(
      'rec-live-video', 'rec-live-canvas', 'rec-cam-label', 'rec-cam-status'
    );
  });

  const txtInput = document.getElementById('recovery-report-input');
  if (txtInput && !txtInput.value.trim()) {
    txtInput.value = `CLINICAL CONSULTATION REPORT
Patient: Sarah Jenkins (Athlete ID: ath_demo1)
Date: October 6, 2026
Diagnosis: Right Knee Grade II ACL Sprain with mild patellar tendinopathy.
Phase: Acute Protection & Restoration Phase
Recommended Duration: 30 days recovery protocol.

Plan Directives:
1. Hydration: Maintain minimum 8 glasses (2.0L) of water daily.
2. Medications:
   - Ibuprofen 400mg twice daily with meals for 7 days.
   - Acetaminophen 500mg as needed for pain (max 10 days).
3. Nutrition:
   - High protein synthesis (1.6g-2.0g/kg) daily for full 30 days.
   - Collagen (15g) + Vitamin C (500mg) pre-rehab for 21 days.
   - Omega-3 Fatty Acids (2000mg EPA/DHA) for 30 days.
4. Physical Therapy Exercises:
   - Quad Setting (3x10, 5s hold) twice daily for 14 days.
   - Straight Leg Raise (3x12) once daily for 30 days.
   - Seated Heel Slides (2x15) twice daily for 21 days.
   - Terminal Knee Extension (TKE) with band (3x15) once daily for 30 days.`;
  }
}

async function handleAnalyzeReport() {
  const txtInput = document.getElementById('recovery-report-input');
  const athleteSelect = document.getElementById('recovery-athlete-select');
  
  const report_text = txtInput ? txtInput.value.trim() : '';
  const athlete_id = athleteSelect ? athleteSelect.value : 'ath_demo1';

  if (!report_text) {
    alert('Please enter or paste a medical report text.');
    return;
  }

  currentRecoveryState.athlete_id = athlete_id;

  try {
    const res = await fetch('/api/reports/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ athlete_id, report_text })
    });
    
    if (!res.ok) throw new Error('Report analysis request failed.');
    
    const data = await res.json();
    currentRecoveryState.report_id = data.report_id;
    currentRecoveryState.extracted = data.extracted;

    // Display confirmation & editing workspace
    renderConfirmationWorkspace(data.extracted);
  } catch (err) {
    alert('Analysis Error: ' + err.message);
  }
}

function renderConfirmationWorkspace(ext) {
  const stepIntake = document.getElementById('rec-step-intake');
  const stepConfirm = document.getElementById('rec-step-confirm');
  const stepDaily = document.getElementById('rec-step-daily');

  if (stepIntake) stepIntake.style.display = 'none';
  if (stepDaily) stepDaily.style.display = 'none';
  if (stepConfirm) stepConfirm.style.display = 'block';

  // Fill workspace fields
  document.getElementById('edit-diagnosis').value = ext.diagnosis || '';
  document.getElementById('edit-body-part').value = ext.body_part || 'Knee';
  document.getElementById('edit-phase').value = ext.phase || 'Acute Recovery';
  document.getElementById('edit-duration').value = ext.recovery_duration_days || 30;
  document.getElementById('edit-hydration').value = ext.hydration_target_glasses || 8;

  // Render editable medications table
  renderEditMedicationsTable(ext.medications || []);
  // Render editable nutrition items
  renderEditNutritionTable(ext.nutrition_items || []);
  // Render editable exercises
  renderEditExercisesTable(ext.exercises || []);
}

function renderEditMedicationsTable(meds) {
  const container = document.getElementById('edit-meds-container');
  if (!container) return;
  
  let html = `<table class="data-table">
    <thead>
      <tr>
        <th>Medication</th>
        <th>Dosage</th>
        <th>Frequency</th>
        <th>Duration (Days)</th>
      </tr>
    </thead>
    <tbody>`;
  
  meds.forEach((m, idx) => {
    html += `<tr>
      <td><input type="text" class="form-control med-name" value="${m.name}" data-idx="${idx}"></td>
      <td><input type="text" class="form-control med-dose" value="${m.dosage}" data-idx="${idx}"></td>
      <td><input type="text" class="form-control med-freq" value="${m.frequency}" data-idx="${idx}"></td>
      <td><input type="number" class="form-control med-dur" value="${m.duration_days}" data-idx="${idx}"></td>
    </tr>`;
  });
  html += `</tbody></table>`;
  container.innerHTML = html;
}

function renderEditNutritionTable(items) {
  const container = document.getElementById('edit-nutrition-container');
  if (!container) return;

  let html = '';
  items.forEach((item, idx) => {
    const foodListStr = (item.example_foods || []).join('\n');
    html += `<div class="rec-item-card">
      <div class="form-grid">
        <div class="form-group">
          <label>Nutritional Target</label>
          <input type="text" class="form-control nutr-title" value="${item.title}" data-idx="${idx}">
        </div>
        <div class="form-group">
          <label>Target Amount / Frequency</label>
          <input type="text" class="form-control nutr-target" value="${item.target}" data-idx="${idx}">
        </div>
        <div class="form-group">
          <label>Active Duration (Days)</label>
          <input type="number" class="form-control nutr-dur" value="${item.duration_days}" data-idx="${idx}">
        </div>
      </div>
      <div class="form-group" style="margin-top:8px;">
        <label>3-5 Everyday Food Examples (one per line)</label>
        <textarea class="form-control nutr-foods" rows="3" data-idx="${idx}">${foodListStr}</textarea>
      </div>
    </div>`;
  });
  container.innerHTML = html;
}

function renderEditExercisesTable(exList) {
  const container = document.getElementById('edit-exercises-container');
  if (!container) return;

  let html = '';
  exList.forEach((ex, idx) => {
    html += `<div class="rec-item-card">
      <div class="form-grid">
        <div class="form-group">
          <label>Exercise Name</label>
          <input type="text" class="form-control ex-name" value="${ex.name}" data-idx="${idx}">
        </div>
        <div class="form-group">
          <label>Target Reps / Sets</label>
          <input type="text" class="form-control ex-reps" value="${ex.target_reps}" data-idx="${idx}">
        </div>
        <div class="form-group">
          <label>Frequency</label>
          <input type="text" class="form-control ex-freq" value="${ex.frequency}" data-idx="${idx}">
        </div>
        <div class="form-group">
          <label>Duration (Days)</label>
          <input type="number" class="form-control ex-dur" value="${ex.duration_days}" data-idx="${idx}">
        </div>
      </div>
      <div class="form-group" style="margin-top:8px;">
        <label>Teach Copy (Step-by-step guidance)</label>
        <textarea class="form-control ex-teach" rows="2" data-idx="${idx}">${ex.teach_copy}</textarea>
      </div>
      <div class="form-grid" style="margin-top:8px;">
        <div class="form-group">
          <label>Caution (What to avoid)</label>
          <input type="text" class="form-control ex-caution" value="${ex.caution || ''}" data-idx="${idx}">
        </div>
        <div class="form-group">
          <label>Form Cue</label>
          <input type="text" class="form-control ex-cue" value="${ex.form_cue || ''}" data-idx="${idx}">
        </div>
      </div>
    </div>`;
  });
  container.innerHTML = html;
}

async function handleConfirmPlan() {
  const athlete_id = currentRecoveryState.athlete_id;
  const report_id = currentRecoveryState.report_id || 'rep_manual';

  const diagnosis = document.getElementById('edit-diagnosis').value.trim();
  const body_part = document.getElementById('edit-body-part').value.trim();
  const phase = document.getElementById('edit-phase').value.trim();
  const recovery_duration_days = parseInt(document.getElementById('edit-duration').value, 10) || 30;
  const hydration_target_glasses = parseInt(document.getElementById('edit-hydration').value, 10) || 8;

  // Read medications from inputs
  const meds = [];
  document.querySelectorAll('.med-name').forEach((el, idx) => {
    const doseEl = document.querySelector(`.med-dose[data-idx="${idx}"]`);
    const freqEl = document.querySelector(`.med-freq[data-idx="${idx}"]`);
    const durEl = document.querySelector(`.med-dur[data-idx="${idx}"]`);
    if (el.value.trim()) {
      meds.push({
        name: el.value.trim(),
        dosage: doseEl ? doseEl.value.trim() : '',
        frequency: freqEl ? freqEl.value.trim() : '',
        duration_days: durEl ? parseInt(durEl.value, 10) : recovery_duration_days
      });
    }
  });

  // Read nutrition items
  const nutrItems = [];
  document.querySelectorAll('.nutr-title').forEach((el, idx) => {
    const targetEl = document.querySelector(`.nutr-target[data-idx="${idx}"]`);
    const durEl = document.querySelector(`.nutr-dur[data-idx="${idx}"]`);
    const foodsEl = document.querySelector(`.nutr-foods[data-idx="${idx}"]`);
    
    const foodsList = foodsEl ? foodsEl.value.split('\n').map(s => s.trim()).filter(Boolean) : [];
    if (el.value.trim()) {
      nutrItems.push({
        title: el.value.trim(),
        target: targetEl ? targetEl.value.trim() : '',
        duration_days: durEl ? parseInt(durEl.value, 10) : recovery_duration_days,
        example_foods: foodsList
      });
    }
  });

  // Read exercises
  const exercises = [];
  document.querySelectorAll('.ex-name').forEach((el, idx) => {
    const repsEl = document.querySelector(`.ex-reps[data-idx="${idx}"]`);
    const freqEl = document.querySelector(`.ex-freq[data-idx="${idx}"]`);
    const durEl = document.querySelector(`.ex-dur[data-idx="${idx}"]`);
    const teachEl = document.querySelector(`.ex-teach[data-idx="${idx}"]`);
    const cautionEl = document.querySelector(`.ex-caution[data-idx="${idx}"]`);
    const cueEl = document.querySelector(`.ex-cue[data-idx="${idx}"]`);

    if (el.value.trim()) {
      exercises.push({
        name: el.value.trim(),
        target_reps: repsEl ? repsEl.value.trim() : '',
        frequency: freqEl ? freqEl.value.trim() : '',
        duration_days: durEl ? parseInt(durEl.value, 10) : recovery_duration_days,
        teach_copy: teachEl ? teachEl.value.trim() : '',
        caution: cautionEl ? cautionEl.value.trim() : '',
        form_cue: cueEl ? cueEl.value.trim() : ''
      });
    }
  });

  const payload = {
    athlete_id,
    report_id,
    diagnosis,
    body_part,
    phase,
    recovery_duration_days,
    recovery_duration_source: "confirmed",
    hydration_target_glasses,
    medications: meds,
    nutrition_items: nutrItems,
    exercises: exercises
  };

  try {
    const res = await fetch('/api/plans', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error('Failed to create recovery plan.');

    const data = await res.json();
    currentRecoveryState.active_plan_id = data.plan_id;
    currentRecoveryState.selected_day = 1;

    // Load and render Day 1 View
    loadPlanDay(data.plan_id, 1);
  } catch (err) {
    alert('Plan Creation Error: ' + err.message);
  }
}

async function loadPlanDay(plan_id, day_number) {
  try {
    // 1. Get full plan info first if not loaded
    if (!currentRecoveryState.active_plan || currentRecoveryState.active_plan.id !== plan_id) {
      const planRes = await fetch(`/api/plans/${plan_id}`);
      if (!planRes.ok) throw new Error('Could not fetch recovery plan.');
      currentRecoveryState.active_plan = await planRes.json();
    }

    // 2. Fetch specific day log
    const dayRes = await fetch(`/api/plans/${plan_id}/days/${day_number}`);
    if (!dayRes.ok) throw new Error(`Could not fetch Day ${day_number} log.`);
    const dayData = await dayRes.json();

    currentRecoveryState.selected_day = day_number;
    currentRecoveryState.current_day_log = dayData;

    renderDailyPlanView(currentRecoveryState.active_plan, dayData);
  } catch (err) {
    alert('Day Load Error: ' + err.message);
  }
}

function renderDailyPlanView(plan, dayLog) {
  const stepIntake = document.getElementById('rec-step-intake');
  const stepConfirm = document.getElementById('rec-step-confirm');
  const stepDaily = document.getElementById('rec-step-daily');

  if (stepIntake) stepIntake.style.display = 'none';
  if (stepConfirm) stepConfirm.style.display = 'none';
  if (stepDaily) stepDaily.style.display = 'block';

  // 1. Header Banner & Microcopy Progress
  document.getElementById('daily-banner-title').innerText = `${plan.diagnosis} — Day ${dayLog.day_number} of ${plan.recovery_duration_days}`;
  document.getElementById('daily-banner-phase').innerText = plan.phase;

  // Calculate microcopy encouraging progress
  const cats = dayLog.category_completion || {};
  const completedCats = Object.values(cats).filter(Boolean).length;
  const pct = Math.round((completedCats / 4) * 100);

  const barEl = document.getElementById('daily-progress-bar-fill');
  if (barEl) barEl.style.width = `${pct}%`;

  let encourTxt = "Nice — let's get today's recovery routine started!";
  if (pct === 100) {
    encourTxt = "Outstanding — all recovery targets completed for today!";
  } else if (cats.hydration && cats.medication) {
    encourTxt = "Great job — hydration & meds are sorted for today!";
  } else if (completedCats >= 2) {
    encourTxt = `You're ${pct}% there — steady progress builds resilience.`;
  }
  document.getElementById('daily-progress-text').innerText = encourTxt;

  // 2. Render Horizontal Day Strip Node
  renderDayStrip(plan.id, plan.recovery_duration_days, dayLog.day_number, plan.days || []);

  // 3. Hydration Card
  renderHydrationCard(dayLog, plan.hydration_target_glasses);

  // 4. Active Medications Checklist
  renderMedicationChecklist(dayLog.active_medications, dayLog.meds_taken, dayLog.id);

  // 5. Active Nutrition Checklist
  renderNutritionChecklist(dayLog.active_nutrition_items, dayLog.nutrition_logged, dayLog.id);

  // 6. Active Exercises Workspace (3-Stage Flow)
  renderExercisesWorkspace(dayLog.active_exercises, dayLog.exercise_reps, dayLog.exercise_self_logged, dayLog.id);
}

function renderDayStrip(plan_id, total_days, active_day, days_summary) {
  const container = document.getElementById('horizontal-day-strip');
  if (!container) return;

  const statusMap = {};
  days_summary.forEach(d => {
    statusMap[d.day_number] = d.status;
  });

  let html = '';
  for (let i = 1; i <= total_days; i++) {
    const isActive = i === active_day;
    const isFuture = i > active_day;
    const isCompleted = statusMap[i] === 'complete';

    let classes = 'day-strip-node';
    if (isActive) classes += ' active';
    if (isFuture) classes += ' future';
    if (isCompleted) classes += ' completed-day';

    html += `<div class="${classes}" onclick="loadPlanDay('${plan_id}', ${i})">
      <div class="day-node-num">Day ${i}</div>
      <div class="day-node-sub">${isCompleted ? '✓ Done' : (isFuture ? 'Preview' : 'Active')}</div>
    </div>`;
  }
  container.innerHTML = html;
}

function renderHydrationCard(dayLog, targetGlasses) {
  const container = document.getElementById('hydration-card-body');
  if (!container) return;

  const current = dayLog.hydration_count || 0;
  const isDone = current >= targetGlasses;

  let html = `<div class="rec-item-card ${isDone ? 'completed' : ''}">
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <div>
        <strong>Hydration Target: ${targetGlasses} Glasses (approx 2.0 Liters)</strong>
        <p style="margin:4px 0 0 0; color:var(--rec-text-muted); font-size:0.88rem;">
          ${isDone ? '\u2713 Hydration target met for today!' : `${targetGlasses - current} more glasses to go today.`}
        </p>
      </div>
      <div class="glass-counter">
        <button class="glass-btn" onclick="updateHydrationCount(${dayLog.hydration_count - 1})">-</button>
        <span style="font-size:1.4rem; font-weight:700;">${current} / ${targetGlasses}</span>
        <button class="glass-btn" onclick="updateHydrationCount(${dayLog.hydration_count + 1})">+</button>
      </div>
    </div>
  </div>`;
  container.innerHTML = html;
}

async function updateHydrationCount(newVal) {
  if (newVal < 0) newVal = 0;
  const log_id = currentRecoveryState.current_day_log.id;
  await sendDayUpdate(log_id, { hydration_count: newVal });
}

function renderMedicationChecklist(activeMeds, takenMeds, log_id) {
  const container = document.getElementById('medication-checklist-body');
  if (!container) return;

  if (!activeMeds || activeMeds.length === 0) {
    container.innerHTML = `<p style="color:var(--rec-text-muted); font-style:italic;">No active medications scheduled for Day ${currentRecoveryState.selected_day}.</p>`;
    return;
  }

  const takenSet = new Set(takenMeds || []);
  let html = '';

  activeMeds.forEach(m => {
    const isChecked = takenSet.has(m.name);
    html += `<div class="rec-item-card ${isChecked ? 'completed' : ''}">
      <label style="display:flex; align-items:center; gap:12px; cursor:pointer;">
        <input type="checkbox" class="rec-checkbox" ${isChecked ? 'checked' : ''} onchange="toggleMedication('${m.name}', this.checked)">
        <div>
          <strong style="font-size:0.98rem;">${m.name}</strong> — ${m.dosage} (${m.frequency})
          <div style="font-size:0.82rem; color:var(--rec-text-muted);">Active through Day ${m.duration_days}</div>
        </div>
      </label>
    </div>`;
  });
  container.innerHTML = html;
}

async function toggleMedication(medName, isChecked) {
  const currentTaken = new Set(currentRecoveryState.current_day_log.meds_taken || []);
  if (isChecked) {
    currentTaken.add(medName);
  } else {
    currentTaken.delete(medName);
  }
  const log_id = currentRecoveryState.current_day_log.id;
  await sendDayUpdate(log_id, { meds_taken: Array.from(currentTaken) });
}

function renderNutritionChecklist(activeNutr, loggedNutr, log_id) {
  const container = document.getElementById('nutrition-checklist-body');
  if (!container) return;

  if (!activeNutr || activeNutr.length === 0) {
    container.innerHTML = `<p style="color:var(--rec-text-muted); font-style:italic;">No active nutrition items scheduled for Day ${currentRecoveryState.selected_day}.</p>`;
    return;
  }

  const loggedSet = new Set(loggedNutr || []);
  let html = '';

  activeNutr.forEach((item, idx) => {
    const isChecked = loggedSet.has(item.title);
    const foodsList = item.example_foods || [];

    html += `<div class="rec-item-card ${isChecked ? 'completed' : ''}">
      <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <label style="display:flex; align-items:center; gap:12px; cursor:pointer; flex:1;">
          <input type="checkbox" class="rec-checkbox" ${isChecked ? 'checked' : ''} onchange="toggleNutrition('${item.title}', this.checked)">
          <div>
            <strong style="font-size:0.98rem;">${item.title}</strong>
            <div style="font-size:0.88rem; color:var(--rec-text-dark); margin-top:2px;">${item.target}</div>
          </div>
        </label>
        ${foodsList.length > 0 ? `<button class="food-toggle-btn" onclick="toggleFoodExamplesDrawer('food-drawer-${idx}')">
          🥗 3-5 Everyday Food Examples ▼
        </button>` : ''}
      </div>

      ${foodsList.length > 0 ? `<div id="food-drawer-${idx}" class="food-examples-drawer" style="display:none;">
        <strong style="font-size:0.85rem; color:var(--rec-sage);">Recommended Everyday Food Sources:</strong>
        <ul class="food-examples-list">
          ${foodsList.map(f => `<li>${f}</li>`).join('')}
        </ul>
      </div>` : ''}
    </div>`;
  });
  container.innerHTML = html;
}

function toggleFoodExamplesDrawer(drawerId) {
  const el = document.getElementById(drawerId);
  if (el) {
    el.style.display = el.style.display === 'none' ? 'block' : 'none';
  }
}

async function toggleNutrition(title, isChecked) {
  const currentLogged = new Set(currentRecoveryState.current_day_log.nutrition_logged || []);
  if (isChecked) {
    currentLogged.add(title);
  } else {
    currentLogged.delete(title);
  }
  const log_id = currentRecoveryState.current_day_log.id;
  await sendDayUpdate(log_id, { nutrition_logged: Array.from(currentLogged) });
}

function renderExercisesWorkspace(activeEx, loggedReps, selfLogged, log_id) {
  const container = document.getElementById('exercise-workspace-body');
  if (!container) return;

  if (!activeEx || activeEx.length === 0) {
    container.innerHTML = `<p style="color:var(--rec-text-muted); font-style:italic;">No active exercises scheduled for Day ${currentRecoveryState.selected_day}.</p>`;
    return;
  }

  const selfLoggedSet = new Set(selfLogged || []);
  const repsMap = {};
  (loggedReps || []).forEach(r => {
    if (r && r.name) repsMap[r.name] = r.reps;
  });

  let html = '';
  activeEx.forEach((ex, idx) => {
    const isSelfLogged = selfLoggedSet.has(ex.name);
    const currentReps = repsMap[ex.name] || 0;
    const stage = currentRecoveryState.exercise_stages[ex.name] || 'teach';

    html += `<div class="exercise-flow-card" id="ex-card-${idx}">
      <!-- 3-Stage Nav Header -->
      <div class="ex-stage-nav">
        <div class="ex-stage-tab ${stage === 'teach' ? 'active' : ''}" onclick="switchExStage('${ex.name}', 'teach', ${idx})">
          1. Teach (Guidance)
        </div>
        <div class="ex-stage-tab ${stage === 'demo' ? 'active' : ''}" onclick="switchExStage('${ex.name}', 'demo', ${idx})">
          2. Demo (Visual)
        </div>
        <div class="ex-stage-tab ${stage === 'track' ? 'active' : ''}" onclick="switchExStage('${ex.name}', 'track', ${idx})">
          3. Track (Camera / Log)
        </div>
      </div>

      <div class="ex-stage-content">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
          <div>
            <h4 style="margin:0; font-size:1.1rem; color:var(--rec-text-dark);">${ex.name}</h4>
            <span style="font-size:0.85rem; color:var(--rec-text-muted);">${ex.target_reps} — ${ex.frequency}</span>
          </div>
          ${isSelfLogged ? '<span class="rec-phase-badge">✓ Manually Logged</span>' : ''}
        </div>

        <!-- Stage 1: Teach -->
        <div id="ex-stage-teach-${idx}" style="display: ${stage === 'teach' ? 'block' : 'none'};">
          <p style="margin:0 0 12px 0; font-size:0.95rem; line-height:1.5;">${ex.teach_copy}</p>
          ${ex.caution ? `<div class="ex-caution-box">⚠️ <strong>Caution:</strong> ${ex.caution}</div>` : ''}
          ${ex.form_cue ? `<div class="ex-cue-box">💡 <strong>Form Cue:</strong> ${ex.form_cue}</div>` : ''}
          <div style="margin-top:16px; text-align:right;">
            <button class="btn-rec-outline" onclick="switchExStage('${ex.name}', 'demo', ${idx})">Next: View Visual Demo →</button>
          </div>
        </div>

        <!-- Stage 2: Instructions (text only — no demo video/image/thumbnail) -->

        <div id="ex-stage-demo-${idx}" style="display: ${stage === 'demo' ? 'block' : 'none'};">
          <div class="ex-media-container" id="ex-media-box-${idx}">
            <p style="color:var(--rec-text-muted);">Loading demonstration graphic...</p>
          </div>
          <div style="margin-top:16px; display:flex; justify-content:space-between;">
            <button class="btn-rec-outline" onclick="switchExStage('${ex.name}', 'teach', ${idx})">← Back to Guidance</button>
            <button class="btn-rec-primary" onclick="switchExStage('${ex.name}', 'track', ${idx})">Proceed to Track & Log →</button>
          </div>
        </div>

        <!-- Stage 3: Track (camera starts only on explicit click) -->
        <div id="ex-stage-track-${idx}" style="display: ${stage === 'track' ? 'block' : 'none'};">
          <div class="tracking-canvas-box" id="ex-camera-box-${idx}" style="position:relative;">
            <video id="ex-video-${idx}" playsinline muted style="display:none;"></video>
            <canvas id="ex-canvas-${idx}" style="width:100%;border-radius:8px;"></canvas>
            <div id="ex-cam-overlay-${idx}" style="position:absolute;bottom:12px;left:12px;background:rgba(0,0,0,0.6);color:#fff;padding:6px 12px;border-radius:6px;font-size:0.85rem;">
              Click Start Camera to begin
            </div>
          </div>
          <!-- Start/Stop/Retry buttons — built by buildExerciseCameraPanel() after render -->

          <!-- Clinical Notice -->
          <div style="margin:10px 0 6px;padding:8px 12px;background:rgba(251,191,36,0.1);border-left:3px solid #F59E0B;border-radius:4px;font-size:0.8rem;color:#92400E;">
            ⚠️ Camera observations are for exercise guidance only. They do not constitute a clinical diagnosis, clearance, or treatment change. All plans remain subject to clinician review.
          </div>

          <!-- Camera Reserve / Manual Fallback Toggle -->
          <div class="camera-fallback-box">
            <p style="margin:0 0 10px 0; font-size:0.9rem; color:var(--rec-text-dark);">
              Camera unavailable or practicing offline? No worries — count your reps manually.
            </p>
            <button class="btn-rec-outline" onclick="toggleManualSelfLog('${ex.name}', ${!isSelfLogged})">
              ${isSelfLogged ? '\u2713 Marked Completed Manually (Click to undo)' : '\u270b I completed this — count me in manually'}
            </button>
          </div>
        </div>

      </div>
    </div>`;
  });

  container.innerHTML = html;

  // Post-render: populate Stage 2 text instructions and Stage 3 camera controls
  activeEx.forEach((ex, idx) => {
    showExerciseTextInstructions(ex.name, idx);
    buildExerciseCameraPanel(idx);
  });
}

function switchExStage(exName, stage, idx) {
  // If leaving 'track' stage, release camera to avoid leaks
  const prevStage = currentRecoveryState.exercise_stages[exName];
  if (prevStage === 'track' && stage !== 'track') {
    RecoveryCameraManager.stop();
  }

  currentRecoveryState.exercise_stages[exName] = stage;

  const teachEl = document.getElementById(`ex-stage-teach-${idx}`);
  const demoEl  = document.getElementById(`ex-stage-demo-${idx}`);
  const trackEl = document.getElementById(`ex-stage-track-${idx}`);

  if (teachEl) teachEl.style.display = stage === 'teach' ? 'block' : 'none';
  if (demoEl)  demoEl.style.display  = stage === 'demo'  ? 'block' : 'none';
  if (trackEl) trackEl.style.display = stage === 'track' ? 'block' : 'none';

  // Highlight tab
  const card = document.getElementById(`ex-card-${idx}`);
  if (card) {
    const tabs = card.querySelectorAll('.ex-stage-tab');
    tabs.forEach((tab, tIdx) => {
      tab.classList.toggle('active',
        (tIdx === 0 && stage === 'teach') ||
        (tIdx === 1 && stage === 'demo')  ||
        (tIdx === 2 && stage === 'track')
      );
    });
  }

  // Stage 3: do NOT auto-request camera — user clicks Start Camera button
}

// fetchAndDisplayMedia REMOVED: Demo videos, images, and thumbnails are no longer
// embedded in the Exercise Stage 2 panel. Stage 2 shows text instructions only.
// See showExerciseTextInstructions() defined above.

// startCameraTrackingForExercise REMOVED: Camera permission is no longer requested
// automatically on entering Stage 3. User must click "Start Camera" explicitly.
// The shared RecoveryCameraManager handles the stream; see buildExerciseCameraPanel().

async function toggleManualSelfLog(exName, shouldLog) {
  const currentSelf = new Set(currentRecoveryState.current_day_log.exercise_self_logged || []);
  if (shouldLog) {
    currentSelf.add(exName);
  } else {
    currentSelf.delete(exName);
  }
  const log_id = currentRecoveryState.current_day_log.id;
  await sendDayUpdate(log_id, { exercise_self_logged: Array.from(currentSelf) });
}

async function sendDayUpdate(log_id, updateData) {
  try {
    const res = await fetch(`/api/days/${log_id}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updateData)
    });

    if (!res.ok) throw new Error('Failed to update day log.');

    const updatedLog = await res.json();
    currentRecoveryState.current_day_log = updatedLog;

    // Refresh view
    loadPlanDay(updatedLog.plan_id, updatedLog.day_number);
  } catch (err) {
    alert('Update Error: ' + err.message);
  }
}
