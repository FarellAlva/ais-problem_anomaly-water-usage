/* app.js — Interaktivitas Dashboard WaterAIS & Canvas Rendering High-DPI */

const STATE = {
  initData: null,
  benchData: null,
  simData: null,
  simCursor: 23,
  simTimer: null,
  autoMode: true,
  valveOpen: true,
  lastTestedPoint: null, // { hour, mean_flow, is_anomaly, label }
};

async function apiGet(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

async function apiPost(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

function setupHiDPICanvas(canvas, cssHeight) {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || canvas.parentElement.clientWidth || 720;
  const height = cssHeight || canvas.height || 280;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, w: width, h: height };
}

function updateValveUI() {
  const modeBtn = document.getElementById("btn-mode-toggle");
  const pill = document.getElementById("valve-pill");
  modeBtn.textContent = STATE.autoMode
    ? "Mode: AUTO (Proteksi Aktif)"
    : "Mode: MANUAL (Bypass)";

  if (STATE.valveOpen) {
    pill.className = "pill open";
    pill.textContent = "Solenoid Valve: OPEN (Mengalir)";
  } else {
    pill.className = "pill closed";
    pill.textContent = "Solenoid Valve: SHUT-OFF (Tertutup Otomatis)";
  }
}

function updateAISKPIs(aisPayload) {
  const tr = aisPayload.trace || {};
  document.getElementById("kpi-detectors").textContent = `${tr.detectors_created || 0} Sel`;
  document.getElementById("kpi-det-sub").textContent =
    `Thymus censorship ${(tr.train_time_sec || 0).toFixed(3)}s | Gugur Self: ${tr.censored_by_self || 0}`;
}

function renderActiveHourCard(hData) {
  if (!hData) return;
  const badge = document.getElementById("cur-hour-badge");
  const box = document.getElementById("cur-hour-summary");
  const isAnom = hData.ais.is_anomaly;

  badge.className = `badge ${isAnom ? "badge-anomaly" : "badge-normal"}`;
  badge.textContent = `Jam ${String(hData.hour).padStart(2, "0")}:00 — ${hData.ais.status_label}`;

  if (STATE.autoMode && isAnom) {
    STATE.valveOpen = false;
  } else if (!STATE.autoMode) {
    STATE.valveOpen = true;
  }
  updateValveUI();

  box.innerHTML = `
    <div style="margin-bottom:6px;"><strong>Waktu:</strong> ${hData.label} ${hData.injected ? '<span class="badge badge-warn">Injeksi Anomali</span>' : ''}</div>
    <div style="margin-bottom:6px;"><strong>4×15m:</strong> [${hData.slot_flows.map(v => v.toFixed(2)).join(", ")}] L/min</div>
    <div style="margin-bottom:6px;"><strong>Rata-rata 1-Jam:</strong> ${hData.mean_flow_lpm.toFixed(3)} L/min (${hData.total_liters.toFixed(1)} L/jam) | <strong>Aktif:</strong> ${hData.active_slots}/4 slot</div>
    <div style="margin-bottom:6px;"><strong>AIS V-Detector:</strong> <span class="badge ${isAnom ? "badge-anomaly" : "badge-normal"}">${hData.ais.status_label}</span> (${hData.ais.confidence_pct.toFixed(1)}% yakin, ${hData.ais.triggered_detectors} detektor)</div>
    <div style="margin-bottom:6px;"><strong>Diagnosis:</strong> ${hData.ais.anomaly_category}</div>
    <div style="margin-bottom:6px;"><strong>Z-Score 15m / 1h:</strong> ${hData.zscore_15m.is_anomaly ? "ANOMALI" : "NORMAL"} / ${hData.zscore_1h.is_anomaly ? "ANOMALI" : "NORMAL"}</div>
    <div style="color: var(--muted); font-size:0.78rem;">${hData.ais.reason}</div>
  `;
}

function drawSimulationChart() {
  const canvas = document.getElementById("canvas-sim-24h");
  if (!canvas || !STATE.simData) return;
  const { ctx, w, h } = setupHiDPICanvas(canvas, 290);
  const hours = STATE.simData.hours;
  const pad = { l: 48, r: 16, t: 24, b: 36 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;

  const maxSlot = Math.max(2.8, ...hours.flatMap(x => x.slot_flows));
  const yMax = Math.ceil(maxSlot * 1.12 * 2) / 2;

  // Grid Y
  ctx.strokeStyle = "#e2e8f0";
  ctx.fillStyle = "#64748b";
  ctx.font = "11px sans-serif";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const val = (yMax * i) / 4;
    const y = pad.t + plotH - (val / yMax) * plotH;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
    ctx.fillText(val.toFixed(1), 6, y + 4);
  }

  const hourW = plotW / 24;
  // Highlight anomaly hours & active cursor
  hours.forEach((hd, idx) => {
    const x0 = pad.l + idx * hourW;
    if (idx <= STATE.simCursor && hd.ais.is_anomaly) {
      ctx.fillStyle = "rgba(239, 68, 68, 0.14)";
      ctx.fillRect(x0, pad.t, hourW, plotH);
    }
    if (idx === STATE.simCursor) {
      ctx.strokeStyle = "#0284c7";
      ctx.lineWidth = 1.6;
      ctx.strokeRect(x0 + 1, pad.t, hourW - 2, plotH);
    }
  });

  // 15-min flow bars up to cursor
  const subW = Math.max(2, hourW / 4 - 1.5);
  hours.forEach((hd, idx) => {
    if (idx > STATE.simCursor) return;
    hd.slot_flows.forEach((v, sIdx) => {
      const bx = pad.l + idx * hourW + sIdx * (hourW / 4) + 1;
      const bh = (v / yMax) * plotH;
      const by = pad.t + plotH - bh;
      ctx.fillStyle = hd.ais.is_anomaly ? "#ef4444" : "#38bdf8";
      ctx.fillRect(bx, by, subW, bh);
    });
  });

  // 1-hour mean line
  ctx.strokeStyle = "#0f172a";
  ctx.lineWidth = 2;
  ctx.beginPath();
  hours.forEach((hd, idx) => {
    if (idx > STATE.simCursor) return;
    const cx = pad.l + idx * hourW + hourW / 2;
    const cy = pad.t + plotH - (hd.mean_flow_lpm / yMax) * plotH;
    if (idx === 0) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  });
  ctx.stroke();

  // Points on mean line
  hours.forEach((hd, idx) => {
    if (idx > STATE.simCursor) return;
    const cx = pad.l + idx * hourW + hourW / 2;
    const cy = pad.t + plotH - (hd.mean_flow_lpm / yMax) * plotH;
    ctx.beginPath();
    ctx.arc(cx, cy, hd.ais.is_anomaly ? 4.5 : 3, 0, Math.PI * 2);
    ctx.fillStyle = hd.ais.is_anomaly ? "#dc2626" : "#0284c7";
    ctx.fill();
  });

  // X labels
  ctx.fillStyle = "#475569";
  ctx.font = "11px sans-serif";
  for (let hr = 0; hr < 24; hr += 2) {
    const cx = pad.l + hr * hourW + 2;
    ctx.fillText(`${String(hr).padStart(2, "0")}:00`, cx, h - 12);
  }
  ctx.fillText("Laju (L/min)", 6, 14);
}

function renderSimulationTable() {
  const tbody = document.getElementById("sim-log-body");
  if (!STATE.simData) return;
  tbody.innerHTML = STATE.simData.hours
    .slice(0, STATE.simCursor + 1)
    .map((hd, idx) => {
      const aBadge = hd.ais.is_anomaly
        ? `<span class="badge badge-anomaly">ANOMALI</span>`
        : `<span class="badge badge-normal">NORMAL</span>`;
      const z15Badge = hd.zscore_15m.is_anomaly
        ? `<span class="badge badge-warn">ANOM</span>`
        : `Normal`;
      const z1hBadge = hd.zscore_1h.is_anomaly
        ? `<span class="badge badge-warn">ANOM</span>`
        : `Normal`;
      return `
        <tr class="${hd.ais.is_anomaly ? "row-anom" : ""} ${idx === STATE.simCursor ? "row-highlight" : ""}">
          <td><strong>${String(hd.hour).padStart(2, "0")}:00</strong></td>
          <td>[${hd.slot_flows.map(v => v.toFixed(2)).join(", ")}]</td>
          <td>${hd.mean_flow_lpm.toFixed(3)} / ${hd.total_liters.toFixed(1)}L</td>
          <td>${hd.active_slots}/4</td>
          <td>${aBadge}</td>
          <td>${z15Badge}</td>
          <td>${z1hBadge}</td>
          <td>${hd.ais.anomaly_category} (${hd.ais.valve_action})</td>
        </tr>
      `;
    })
    .join("");
}

async function runSimulation() {
  if (STATE.simTimer) {
    clearInterval(STATE.simTimer);
    STATE.simTimer = null;
    document.getElementById("btn-sim-play").textContent = "Play Jam-per-Jam";
  }
  const day_number = parseInt(document.getElementById("sim-day").value, 10);
  const inject_preset = document.getElementById("sim-preset").value;
  const inject_hour = parseInt(document.getElementById("sim-hour").value, 10) || 0;
  STATE.valveOpen = true;
  STATE.simData = await apiPost("/api/simulate_day", {
    day_number,
    inject_preset,
    inject_hour,
    auto_mode: STATE.autoMode,
  });
  STATE.simCursor = 23;
  // Cek apakah ada anomali sepanjang hari
  const anyAnom = STATE.simData.hours.some(h => h.ais.is_anomaly);
  if (STATE.autoMode && anyAnom) {
    STATE.valveOpen = false;
  }
  const focusIdx = inject_preset !== "none" ? inject_hour : 23;
  renderActiveHourCard(STATE.simData.hours[focusIdx]);
  drawSimulationChart();
  renderSimulationTable();
}

function togglePlayback() {
  const btn = document.getElementById("btn-sim-play");
  if (STATE.simTimer) {
    clearInterval(STATE.simTimer);
    STATE.simTimer = null;
    btn.textContent = "Play Jam-per-Jam";
    return;
  }
  if (!STATE.simData) return;
  if (STATE.simCursor >= 23) {
    STATE.simCursor = 0;
    STATE.valveOpen = true;
  }
  btn.textContent = "Pause";
  STATE.simTimer = setInterval(() => {
    if (STATE.simCursor < 23) {
      STATE.simCursor += 1;
      renderActiveHourCard(STATE.simData.hours[STATE.simCursor]);
      drawSimulationChart();
      renderSimulationTable();
    } else {
      clearInterval(STATE.simTimer);
      STATE.simTimer = null;
      btn.textContent = "Play Jam-per-Jam";
    }
  }, 450);
}

async function evaluateLabHour() {
  const day_name = document.getElementById("lab-day").value;
  const hour = parseInt(document.getElementById("lab-hour").value, 10) || 0;
  const slot_flows = [
    parseFloat(document.getElementById("lab-q1").value) || 0,
    parseFloat(document.getElementById("lab-q2").value) || 0,
    parseFloat(document.getElementById("lab-q3").value) || 0,
    parseFloat(document.getElementById("lab-q4").value) || 0,
  ];
  const res = await apiPost("/api/evaluate_hour", {
    slot_flows,
    hour,
    day_name,
    auto_mode: STATE.autoMode,
  });
  const ag = res.antigen;
  STATE.lastTestedPoint = {
    hour: ag.hour,
    mean_flow: ag.mean_flow_lpm,
    is_anomaly: res.ais_vdetector.is_anomaly,
  };

  document.getElementById("lab-features-bar").innerHTML =
    `<strong>Vektor Agregator 1-Jam:</strong> Rata-rata = <b>${ag.mean_flow_lpm.toFixed(3)} L/min</b> (${ag.total_liters.toFixed(1)} L/jam) | ` +
    `Puncak 15m = <b>${ag.max_flow_lpm.toFixed(2)} L/min</b> | Minimum = <b>${ag.min_flow_lpm.toFixed(2)} L/min</b> | ` +
    `Slot Aktif = <b>${ag.active_slots}/4 (${(ag.active_ratio * 100).toFixed(0)}%)</b>`;

  const cards = [
    {
      title: "1. AIS V-Detector (Context 7D — Utama)",
      anom: res.ais_vdetector.is_anomaly,
      lines: [
        `Status: <b>${res.ais_vdetector.status_label}</b> (${res.ais_vdetector.confidence_pct}%)`,
        `Diagnosis: <b>${res.ais_vdetector.anomaly_category}</b>`,
        `Detektor Aktif: ${res.ais_vdetector.triggered_detectors} (margin ${res.ais_vdetector.max_detector_margin >= 0 ? "+" : ""}${res.ais_vdetector.max_detector_margin.toFixed(3)})`,
        `Jarak ke Self: ${res.ais_vdetector.min_dist_to_self.toFixed(3)} (terdekat: ${res.ais_vdetector.nearest_self_label})`,
      ],
    },
    {
      title: "2. AIS Per-Slot (168 × 2 Sampel)",
      anom: res.ais_perslot.is_anomaly,
      lines: [
        `Status: <b>${res.ais_perslot.status_label}</b>`,
        `Detektor Slot Aktif: ${res.ais_perslot.triggered_detectors}`,
        `Jarak ke 2 Sampel Slot: ${res.ais_perslot.min_dist_to_self.toFixed(3)}`,
        `Catatan: Hanya membandingkan 2 sampel historis slot ini.`,
      ],
    },
    {
      title: "3. Z-Score 15-Menit (Jurnal JUTISI)",
      anom: res.zscore_15m.is_anomaly,
      lines: [
        `Status: <b>${res.zscore_15m.is_anomaly ? "ANOMALI" : "NORMAL"}</b>`,
        `Observasi maks vs UCL: ${res.zscore_15m.observed_value.toFixed(2)} vs ${res.zscore_15m.ucl_final.toFixed(2)} L/min`,
        `Slot Pemicu: ${res.zscore_15m.triggered_slots.length ? res.zscore_15m.triggered_slots.join(", ") : "Tidak ada"}`,
        `Rentan alarm palsu jika jadwal mandi geser 15 menit.`,
      ],
    },
    {
      title: "4. Z-Score 1-Jam (Slot N=2)",
      anom: res.zscore_1h.is_anomaly,
      lines: [
        `Status: <b>${res.zscore_1h.is_anomaly ? "ANOMALI" : "NORMAL"}</b>`,
        `Rata-rata 1-Jam vs UCL: ${res.zscore_1h.observed_value.toFixed(3)} vs ${res.zscore_1h.ucl_final.toFixed(3)} L/min`,
        `Rasio terhadap UCL: ${res.zscore_1h.score.toFixed(2)}x`,
        `Rentan lolos pada kebocoran sedang di jam aktif.`,
      ],
    },
  ];

  document.getElementById("lab-verdicts").innerHTML = cards
    .map(
      c => `
      <div class="verdict-card ${c.anom ? "anomaly" : "normal"}">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
          <strong style="font-size:0.86rem;">${c.title}</strong>
          <span class="badge ${c.anom ? "badge-anomaly" : "badge-normal"}">${c.anom ? "ANOMALI" : "NORMAL"}</span>
        </div>
        ${c.lines.map(l => `<div style="font-size:0.8rem; margin-top:2px;">${l}</div>`).join("")}
      </div>`
    )
    .join("");

  const two = res.two_samples_slot || [];
  document.getElementById("lab-slot-history").innerHTML = `
    <strong>Referensi 2 Sampel Historis pada (${day_name}, ${String(hour).padStart(2, "0")}:00):</strong><br/>
    ${two.map(s => `Minggu-${s.week} (${s.date_str}): [${s.slot_flows.map(v => v.toFixed(2)).join(", ")}] → rata-rata ${s.mean_flow_lpm.toFixed(3)} L/min (${s.active_slots}/4 aktif)`).join("<br/>")}
    <br/><em>Sementara V-Detector AIS memperhitungkan konteks sirkadian ±1 jam dari seluruh 14 hari (336 Self-Cells) agar keputusan kebal terhadap pergeseran jam wajar dan peka terhadap aliran kontinyu 4/4 slot.</em>
  `;
  drawImmunologicalSpace();
}

function drawHeatmap() {
  const canvas = document.getElementById("canvas-heatmap");
  if (!canvas || !STATE.initData) return;
  const { ctx, w, h } = setupHiDPICanvas(canvas, 330);
  const hm = STATE.initData.heatmap;
  const days = STATE.initData.days;
  const pad = { l: 120, r: 20, t: 24, b: 28 };
  const cellW = (w - pad.l - pad.r) / 24;
  const cellH = (h - pad.t - pad.b) / 14;
  const maxVal = 1.3;

  hm.forEach((row, d) => {
    ctx.fillStyle = "#334155";
    ctx.font = "11px sans-serif";
    ctx.fillText(
      `${days[d].day_name.slice(0, 3)} ${days[d].date_str.slice(0, 6)} (M${days[d].week_number})`,
      6,
      pad.t + d * cellH + cellH * 0.7
    );
    row.forEach((cell, hr) => {
      const t = Math.min(1.0, cell.mean_flow_lpm / maxVal);
      const r = Math.round(240 - t * 225);
      const g = Math.round(249 - t * 130);
      const b = Math.round(255 - t * 55);
      ctx.fillStyle = cell.mean_flow_lpm === 0 ? "#f1f5f9" : `rgb(${r},${g},${b})`;
      ctx.fillRect(pad.l + hr * cellW, pad.t + d * cellH, cellW - 1, cellH - 1);
    });
  });

  // Pemisah Minggu 1 vs Minggu 2
  ctx.strokeStyle = "#dc2626";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t + 7 * cellH);
  ctx.lineTo(w - pad.r, pad.t + 7 * cellH);
  ctx.stroke();

  ctx.fillStyle = "#475569";
  ctx.font = "11px sans-serif";
  for (let hr = 0; hr < 24; hr += 2) {
    ctx.fillText(String(hr).padStart(2, "0"), pad.l + hr * cellW + 4, h - 8);
  }

  canvas.onclick = ev => {
    const rect = canvas.getBoundingClientRect();
    const cx = ev.clientX - rect.left;
    const cy = ev.clientY - rect.top;
    const hr = Math.floor((cx - pad.l) / cellW);
    const d = Math.floor((cy - pad.t) / cellH);
    if (d >= 0 && d < 14 && hr >= 0 && hr < 24) {
      document.getElementById("slot-day").value = days[d].day_name;
      document.getElementById("slot-hour").value = hr;
      inspectWeeklySlot();
    }
  };
}

function drawCircadian() {
  const canvas = document.getElementById("canvas-circadian");
  if (!canvas || !STATE.initData) return;
  const { ctx, w, h } = setupHiDPICanvas(canvas, 260);
  const circ = STATE.initData.circadian;
  const pad = { l: 42, r: 16, t: 20, b: 30 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;
  const yMax = 1.4;

  ctx.strokeStyle = "#e2e8f0";
  ctx.fillStyle = "#64748b";
  ctx.font = "10.5px sans-serif";
  for (let i = 0; i <= 4; i++) {
    const val = (yMax * i) / 4;
    const y = pad.t + plotH - (val / yMax) * plotH;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
    ctx.fillText(val.toFixed(2), 4, y + 3);
  }

  // Area rentang maks historis
  ctx.fillStyle = "rgba(56, 189, 248, 0.22)";
  ctx.beginPath();
  circ.forEach((c, i) => {
    const x = pad.l + (i / 23) * plotW;
    const y = pad.t + plotH - (c.max_hourly_mean / yMax) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo(pad.l + plotW, pad.t + plotH);
  ctx.lineTo(pad.l, pad.t + plotH);
  ctx.closePath();
  ctx.fill();

  // Garis rata-rata 14 hari
  ctx.strokeStyle = "#0284c7";
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  circ.forEach((c, i) => {
    const x = pad.l + (i / 23) * plotW;
    const y = pad.t + plotH - (c.mean_flow / yMax) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "#475569";
  for (let hr = 0; hr < 24; hr += 3) {
    const x = pad.l + (hr / 23) * plotW;
    ctx.fillText(`${String(hr).padStart(2, "0")}:00`, x - 10, h - 10);
  }
}

async function inspectWeeklySlot() {
  const day = document.getElementById("slot-day").value;
  const hour = parseInt(document.getElementById("slot-hour").value, 10) || 0;
  const data = await apiGet(`/api/weekly_slot?day=${encodeURIComponent(day)}&hour=${hour}`);
  const box = document.getElementById("slot-inspector-detail");
  const s1 = data.samples[0];
  const s2 = data.samples[1];
  const z = data.zscore_1h_stat;

  box.innerHTML = `
    <div style="margin-bottom:8px;">
      <strong>Slot Mingguan:</strong> ${data.day_name} pukul ${String(data.hour).padStart(2, "0")}:00–${String((data.hour + 1) % 24).padStart(2, "0")}:00
    </div>
    ${s1 ? `<div>• <b>Minggu-1 (${s1.date_str}):</b> [${s1.slot_flows.map(v => v.toFixed(2)).join(", ")}] → rata-rata <b>${s1.mean_flow_lpm.toFixed(3)} L/min</b></div>` : ""}
    ${s2 ? `<div>• <b>Minggu-2 (${s2.date_str}):</b> [${s2.slot_flows.map(v => v.toFixed(2)).join(", ")}] → rata-rata <b>${s2.mean_flow_lpm.toFixed(3)} L/min</b></div>` : ""}
    <div class="callout">
      <strong>Perhitungan Z-Score 1-Jam (N=2):</strong> μ = ${z.mean.toFixed(3)}, s = ${z.std.toFixed(3)} → UCL = <b>${z.ucl_final.toFixed(3)} L/min</b>.<br/>
      <strong>UCL 15-Menit Z-Score:</strong> [${data.zscore_15m_ucls.map(u => u.toFixed(2)).join(", ")}] L/min.<br/>
      <strong>Mekanisme AIS:</strong> Alih-alih terkurung pada 2 angka di atas, V-Detector didukung <b>${data.ais_circadian_support_count}</b> sampel Self tetangga sirkadian (±1 jam lintas 14 hari, maks wajar ${data.ais_circadian_mean_max.toFixed(3)} L/min) dan memeriksa pola 4/4 slot aktif kontinyu.
    </div>
  `;
}

function drawTrainingTrace() {
  const canvas = document.getElementById("canvas-trace");
  if (!canvas || !STATE.initData) return;
  const { ctx, w, h } = setupHiDPICanvas(canvas, 290);
  const steps = (STATE.initData.ais.trace && STATE.initData.ais.trace.history_steps) || [];
  if (!steps.length) return;

  const pad = { l: 50, r: 16, t: 22, b: 32 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;
  const maxAtt = steps[steps.length - 1].attempt || 1;
  const maxY = Math.max(100, ...steps.map(s => Math.max(s.detectors, s.censored, s.redundant)));

  ctx.strokeStyle = "#e2e8f0";
  ctx.fillStyle = "#64748b";
  ctx.font = "10.5px sans-serif";
  for (let i = 0; i <= 4; i++) {
    const val = (maxY * i) / 4;
    const y = pad.t + plotH - (val / maxY) * plotH;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
    ctx.fillText(Math.round(val), 6, y + 3);
  }

  const series = [
    { key: "detectors", color: "#0284c7", label: "Mature V-Detectors" },
    { key: "censored", color: "#dc2626", label: "Censored Self" },
    { key: "redundant", color: "#94a3b8", label: "Redundant" },
  ];
  series.forEach(ser => {
    ctx.strokeStyle = ser.color;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    steps.forEach((s, i) => {
      const x = pad.l + (s.attempt / maxAtt) * plotW;
      const y = pad.t + plotH - (s[ser.key] / maxY) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  // Legend
  let lx = pad.l + 8;
  series.forEach(ser => {
    ctx.fillStyle = ser.color;
    ctx.fillRect(lx, 8, 12, 8);
    ctx.fillStyle = "#334155";
    ctx.fillText(ser.label, lx + 16, 16);
    lx += 130;
  });
  ctx.fillStyle = "#475569";
  ctx.fillText("Kandidat Sel Imun Dievaluasi", pad.l + plotW / 2 - 70, h - 8);
}

function drawImmunologicalSpace() {
  const canvas = document.getElementById("canvas-space");
  if (!canvas || !STATE.initData) return;
  const { ctx, w, h } = setupHiDPICanvas(canvas, 290);
  const selfPts = STATE.initData.self_projection || [];
  const detPts = STATE.initData.ais.detector_projection || [];
  const pad = { l: 44, r: 16, t: 22, b: 32 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;
  const yMax = 2.6;

  ctx.strokeStyle = "#e2e8f0";
  ctx.fillStyle = "#64748b";
  ctx.font = "10.5px sans-serif";
  for (let i = 0; i <= 4; i++) {
    const val = (yMax * i) / 4;
    const y = pad.t + plotH - (val / yMax) * plotH;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
    ctx.fillText(val.toFixed(1), 6, y + 3);
  }

  // V-Detectors (orange circles)
  detPts.forEach(d => {
    if (d.mean_flow > yMax) return;
    const x = pad.l + (d.hour / 24) * plotW;
    const y = pad.t + plotH - (d.mean_flow / yMax) * plotH;
    ctx.beginPath();
    ctx.arc(x, y, 3.2, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(249, 115, 22, 0.35)";
    ctx.fill();
  });

  // Self Antigens (green)
  selfPts.forEach(s => {
    const x = pad.l + (s.hour / 24) * plotW;
    const y = pad.t + plotH - (s.mean_flow / yMax) * plotH;
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(16, 185, 129, 0.78)";
    ctx.fill();
  });

  // Last tested point from Lab
  if (STATE.lastTestedPoint) {
    const pt = STATE.lastTestedPoint;
    const x = pad.l + (pt.hour / 24) * plotW;
    const y = pad.t + plotH - (Math.min(yMax, pt.mean_flow) / yMax) * plotH;
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fillStyle = pt.is_anomaly ? "#dc2626" : "#2563eb";
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  ctx.fillStyle = "#475569";
  for (let hr = 0; hr <= 24; hr += 3) {
    ctx.fillText(`${String(hr).padStart(2, "0")}`, pad.l + (hr / 24) * plotW - 6, h - 10);
  }
  ctx.fillStyle = "#065f46";
  ctx.fillText("● Self (336 Jam)", pad.l + 8, 15);
  ctx.fillStyle = "#c2410c";
  ctx.fillText("● V-Detectors (Non-Self)", pad.l + 120, 15);

  const tr = STATE.initData.ais.trace || {};
  document.getElementById("ais-trace-summary").innerHTML = `
    <strong>Statistik Pelatihan V-Detector Aktif:</strong>
    Self-Cells = <b>${tr.n_self_cells}</b> |
    r_self = <b>${tr.r_self}</b> |
    Kandidat Dievaluasi = <b>${tr.total_attempts}</b> |
    Gugur Self (Censored) = <b>${tr.censored_by_self}</b> |
    Redundant = <b>${tr.discarded_redundant}</b> |
    Detektor Dewasa = <b>${tr.detectors_created}</b> |
    Radius Detektor (min / median / maks) = <b>${tr.radius_min} / ${tr.radius_median} / ${tr.radius_max}</b>.
  `;
}

function renderBenchmarkTab() {
  if (!STATE.benchData || !STATE.benchData.benchmark_summary) {
    document.getElementById("bench-metrics-table").innerHTML =
      "Hasil benchmark belum ditemukan di folder results/. Jalankan <code>python run_experiments.py</code>.";
    return;
  }
  const bs = STATE.benchData.benchmark_summary;
  const models = bs.benchmark.models;
  const holdout = bs.holdout_w1_w2;
  const lodo = STATE.benchData.lodo_cv || {};

  const vdetF1 = models.WaterAIS_VDetector.f1_score.mean * 100;
  const z15F1 = models.ZScore_15Min_JUTISI.f1_score * 100;
  document.getElementById("kpi-holdout").textContent = `FPR ${(holdout.fpr * 100).toFixed(2)}%`;
  document.getElementById("kpi-holdout-sub").textContent =
    `${holdout.false_positives}/${holdout.test_hours} alarm palsu pada Minggu-2 (dilatih hanya Minggu-1)`;
  document.getElementById("kpi-f1").textContent = `${vdetF1.toFixed(1)}% vs ${z15F1.toFixed(1)}%`;
  document.getElementById("kpi-f1-sub").textContent = "F1-Score WaterAIS V-Detector vs Z-Score JUTISI";

  const keys = [
    ["ZScore_15Min_JUTISI", "Z-Score 15m (JUTISI N=2)"],
    ["ZScore_1Hour_N2", "Z-Score 1h (Slot N=2)"],
    ["PerSlot_AIS_N2", "Per-Slot AIS (Slot N=2)"],
    ["WaterAIS_VDetector", "WaterAIS V-Detector (7D)"],
  ];
  const valOf = (obj, m) => (typeof obj[m] === "object" ? obj[m].mean * 100 : obj[m] * 100);

  document.getElementById("bench-metrics-table").innerHTML = `
    <div style="margin-bottom:10px;">
      <strong>Holdout Minggu-1 → Minggu-2 (168 jam):</strong>
      WaterAIS = <b>${holdout.false_positives}/${holdout.test_hours}</b> (${(holdout.fpr * 100).toFixed(2)}% FPR)
      ${holdout.perslot_false_positives !== undefined ? ` | Per-Slot AIS = <b>${holdout.perslot_false_positives}/${holdout.test_hours}</b> (${(holdout.perslot_fpr * 100).toFixed(2)}%)` : ""}
      ${holdout.zscore_1h_false_positives !== undefined ? ` | Z-Score 1h = <b>${holdout.zscore_1h_false_positives}/${holdout.test_hours}</b> (${(holdout.zscore_1h_fpr * 100).toFixed(2)}%)` : ""}<br/>
      <strong>14-Fold Leave-One-Day-Out (336 Jam Asli):</strong>
      WaterAIS = <b>${lodo.total_fp ?? "-"}/${lodo.total_hours ?? "-"}</b> (${lodo.overall_fpr !== undefined ? (lodo.overall_fpr * 100).toFixed(2) : "-"}% FPR)
      ${lodo.perslot_total_fp !== undefined ? ` | Per-Slot AIS = <b>${lodo.perslot_total_fp}/${lodo.total_hours}</b> (${(lodo.perslot_overall_fpr * 100).toFixed(2)}%)` : ""}
      ${lodo.zscore_1h_total_fp !== undefined ? ` | Z-Score 1h = <b>${lodo.zscore_1h_total_fp}/${lodo.total_hours}</b> (${(lodo.zscore_1h_overall_fpr * 100).toFixed(2)}%)` : ""}
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Metode</th><th>Acc</th><th>Prec</th><th>Recall</th><th>F1</th><th>FPR</th></tr></thead>
        <tbody>
          ${keys.map(([k, lbl]) => {
            const m = models[k];
            return `<tr>
              <td><strong>${lbl}</strong></td>
              <td>${valOf(m, "accuracy").toFixed(1)}%</td>
              <td>${valOf(m, "precision").toFixed(1)}%</td>
              <td>${valOf(m, "recall").toFixed(1)}%</td>
              <td><b>${valOf(m, "f1_score").toFixed(1)}%</b></td>
              <td>${valOf(m, "fpr").toFixed(1)}%</td>
            </tr>`;
          }).join("")}
        </tbody>
      </table>
    </div>
  `;

  // Draw grouped bar chart
  const canvas = document.getElementById("canvas-bench");
  const { ctx, w, h } = setupHiDPICanvas(canvas, 280);
  const pad = { l: 42, r: 16, t: 28, b: 36 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;

  ctx.strokeStyle = "#e2e8f0";
  ctx.fillStyle = "#64748b";
  ctx.font = "10.5px sans-serif";
  for (let i = 0; i <= 5; i++) {
    const v = i * 20;
    const y = pad.t + plotH - (v / 100) * plotH;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
    ctx.fillText(`${v}%`, 4, y + 3);
  }

  const metrics = [
    { key: "accuracy", color: "#38bdf8", label: "Accuracy" },
    { key: "recall", color: "#34d399", label: "Recall" },
    { key: "f1_score", color: "#0284c7", label: "F1-Score" },
  ];
  const grpW = plotW / keys.length;
  const barW = grpW / 4.2;

  keys.forEach(([k, lbl], gi) => {
    const gx = pad.l + gi * grpW + grpW * 0.14;
    metrics.forEach((mt, mi) => {
      const val = valOf(models[k], mt.key);
      const bh = (val / 100) * plotH;
      const bx = gx + mi * (barW + 3);
      const by = pad.t + plotH - bh;
      ctx.fillStyle = mt.color;
      ctx.fillRect(bx, by, barW, bh);
      ctx.fillStyle = "#0f172a";
      ctx.font = "9.5px sans-serif";
      ctx.fillText(`${val.toFixed(0)}`, bx + 1, by - 3);
    });
    ctx.fillStyle = "#334155";
    ctx.font = "10px sans-serif";
    ctx.fillText(lbl.split(" (")[0], pad.l + gi * grpW + 8, h - 10);
  });

  let lx = pad.l;
  metrics.forEach(mt => {
    ctx.fillStyle = mt.color;
    ctx.fillRect(lx, 8, 12, 8);
    ctx.fillStyle = "#334155";
    ctx.fillText(mt.label, lx + 16, 16);
    lx += 95;
  });

  // Scenario matrix
  const scRows = bs.benchmark.scenario_comparison || [];
  document.getElementById("bench-scenario-body").innerHTML = scRows
    .map(r => {
      const badge = (pred, expected) => {
        const ok = pred === expected;
        const txt = pred ? "ANOM" : "NORM";
        return `<span class="badge ${ok ? "badge-normal" : "badge-anomaly"}">${txt}${ok ? "" : " (X)"}</span>`;
      };
      return `
        <tr>
          <td><strong>${r.scenario_id}</strong></td>
          <td>${r.name}</td>
          <td>${r.day_name.slice(0, 3)} ${String(r.hour).padStart(2, "0")}:00</td>
          <td>[${r.slot_flows.map(v => v.toFixed(2)).join(", ")}]</td>
          <td><span class="badge ${r.expected_anomaly ? "badge-anomaly" : "badge-normal"}">${r.expected_anomaly ? "ANOMALI" : "NORMAL"}</span></td>
          <td>${badge(r.zscore_15m, r.expected_anomaly)}</td>
          <td>${badge(r.zscore_1h, r.expected_anomaly)}</td>
          <td>${badge(r.perslot_ais, r.expected_anomaly)}</td>
          <td>${badge(r.vdetector_ais, r.expected_anomaly)}</td>
        </tr>
      `;
    })
    .join("");
}

function wireEvents() {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
      // Re-draw canvases when tab becomes visible
      if (btn.dataset.tab === "tab-sim") drawSimulationChart();
      if (btn.dataset.tab === "tab-dataset") {
        drawHeatmap();
        drawCircadian();
      }
      if (btn.dataset.tab === "tab-ais") {
        drawTrainingTrace();
        drawImmunologicalSpace();
      }
      if (btn.dataset.tab === "tab-bench") renderBenchmarkTab();
    });
  });

  document.getElementById("btn-mode-toggle").addEventListener("click", () => {
    STATE.autoMode = !STATE.autoMode;
    if (!STATE.autoMode) STATE.valveOpen = true;
    updateValveUI();
    runSimulation();
  });

  document.getElementById("btn-valve-reset").addEventListener("click", () => {
    STATE.valveOpen = true;
    updateValveUI();
  });

  const sR = document.getElementById("slider-rself");
  const sM = document.getElementById("slider-maxdet");
  sR.addEventListener("input", () => (document.getElementById("val-rself").textContent = Number(sR.value).toFixed(2)));
  sM.addEventListener("input", () => (document.getElementById("val-maxdet").textContent = sM.value));

  document.getElementById("btn-retrain").addEventListener("click", async () => {
    const status = document.getElementById("train-status");
    status.textContent = "Melatih ulang...";
    const aisPayload = await apiPost("/api/train", {
      r_self: parseFloat(sR.value),
      max_detectors: parseInt(sM.value, 10),
      seed: 42,
    });
    STATE.initData.ais = aisPayload;
    updateAISKPIs(aisPayload);
    status.textContent = `Selesai (${aisPayload.trace.train_time_sec}s)`;
    drawTrainingTrace();
    drawImmunologicalSpace();
    await runSimulation();
    await evaluateLabHour();
  });

  document.getElementById("btn-sim-run").addEventListener("click", runSimulation);
  document.getElementById("btn-sim-play").addEventListener("click", togglePlayback);
  document.getElementById("btn-sim-step").addEventListener("click", () => {
    if (!STATE.simData) return;
    STATE.simCursor = (STATE.simCursor + 1) % 24;
    if (STATE.simCursor === 0) STATE.valveOpen = true;
    renderActiveHourCard(STATE.simData.hours[STATE.simCursor]);
    drawSimulationChart();
    renderSimulationTable();
  });
  document.getElementById("btn-sim-reset").addEventListener("click", () => {
    if (!STATE.simData) return;
    STATE.simCursor = 0;
    STATE.valveOpen = true;
    renderActiveHourCard(STATE.simData.hours[0]);
    drawSimulationChart();
    renderSimulationTable();
  });

  document.getElementById("btn-lab-eval").addEventListener("click", evaluateLabHour);
  document.querySelectorAll(".chip[data-preset]").forEach(ch => {
    ch.addEventListener("click", () => {
      const p = ch.dataset.preset;
      const setVals = (day, hr, q1, q2, q3, q4) => {
        document.getElementById("lab-day").value = day;
        document.getElementById("lab-hour").value = hr;
        document.getElementById("lab-q1").value = q1;
        document.getElementById("lab-q2").value = q2;
        document.getElementById("lab-q3").value = q3;
        document.getElementById("lab-q4").value = q4;
      };
      if (p === "slow_leak") setVals("Monday", 10, 0.35, 0.35, 0.35, 0.35);
      if (p === "night_leak") setVals("Thursday", 2, 0.45, 0.45, 0.45, 0.45);
      if (p === "open_tap") setVals("Tuesday", 3, 1.20, 1.20, 0.00, 0.00);
      if (p === "shift_normal") setVals("Sunday", 7, 0.00, 2.35, 0.00, 0.00);
      if (p === "night_drip") setVals("Tuesday", 2, 0.00, 0.18, 0.00, 0.00);
      if (p === "burst") setVals("Tuesday", 8, 5.20, 4.80, 0.00, 0.00);
      evaluateLabHour();
    });
  });

  document.getElementById("slot-day").addEventListener("change", inspectWeeklySlot);
  document.getElementById("slot-hour").addEventListener("change", inspectWeeklySlot);
  window.addEventListener("resize", () => {
    drawSimulationChart();
    drawHeatmap();
    drawCircadian();
    drawTrainingTrace();
    drawImmunologicalSpace();
    renderBenchmarkTab();
  });
}

async function bootstrap() {
  wireEvents();
  updateValveUI();
  STATE.initData = await apiGet("/api/init");
  updateAISKPIs(STATE.initData.ais);

  const daySel = document.getElementById("sim-day");
  daySel.innerHTML = STATE.initData.days
    .map(
      d =>
        `<option value="${d.day_number}" ${d.day_number === 13 ? "selected" : ""}>Hari ${d.day_number + 1}: ${d.day_name}, ${d.date_str} (M${d.week_number} • ${d.total_liters} L)</option>`
    )
    .join("");

  try {
    STATE.benchData = await apiGet("/api/benchmark");
    renderBenchmarkTab();
  } catch (e) {
    console.warn("Benchmark belum tersedia:", e);
  }

  await runSimulation();
  await evaluateLabHour();
  await inspectWeeklySlot();
}

bootstrap();

