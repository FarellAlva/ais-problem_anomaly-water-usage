/* app.js — Interaktivitas Dashboard WaterAIS (Pure AIS + Chart Overlay) & Canvas Rendering High-DPI */

const STATE = {
  initData: null,
  benchData: null,
  simData: null,
  simCursor: 23,
  simTimer: null,
  autoMode: true,
  valveOpen: true,
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

  const marginTxt = `${hData.ais.max_detector_margin >= 0 ? "+" : ""}${hData.ais.max_detector_margin.toFixed(3)}`;
  box.innerHTML = `
    <div style="margin-bottom:6px;"><strong>Waktu Uji:</strong> ${hData.label} ${hData.injected ? '<span class="badge badge-warn">Injeksi Anomali</span>' : ""}</div>
    <div style="margin-bottom:6px;"><strong>4×15m:</strong> [${hData.slot_flows.map(v => v.toFixed(2)).join(", ")}] L/min</div>
    <div style="margin-bottom:6px;"><strong>Rata-rata Uji:</strong> ${hData.mean_flow_lpm.toFixed(3)} L/min (Mean M1+M2: ${(hData.combined_mean || 0).toFixed(3)}) | <strong>Aktif:</strong> ${hData.active_slots}/4</div>
    <div style="margin-bottom:6px;"><strong>Proyeksi Batas AIS:</strong> Intermiten 2/4 &le; <b>${(hData.ais_upper || 0).toFixed(2)}</b> | Kontinyu 4/4 &ge; <b>${(hData.ais_leak_limit || 0).toFixed(2)}</b> L/min</div>
    <div style="margin-bottom:6px;"><strong>Keputusan AIS 7D:</strong> <span class="badge ${isAnom ? "badge-anomaly" : "badge-normal"}">${hData.ais.status_label}</span> (${hData.ais.confidence_pct.toFixed(1)}% yakin, ${hData.ais.triggered_detectors} detektor)</div>
    <div style="margin-bottom:6px;"><strong>Jarak ke Self:</strong> ${hData.ais.min_dist_to_self.toFixed(3)} (margin V-Det: ${marginTxt})</div>
    <div style="margin-bottom:6px;"><strong>Diagnosis Imunologis:</strong> ${hData.ais.anomaly_category}</div>
    <div style="color: var(--muted); font-size:0.78rem;">${hData.ais.reason}</div>
  `;
}

function drawSimulationChart() {
  const canvas = document.getElementById("canvas-sim-24h");
  if (!canvas || !STATE.simData) return;
  const { ctx, w, h } = setupHiDPICanvas(canvas, 320);
  const hours = STATE.simData.hours || [];
  const profile = STATE.simData.threshold_profile || [];
  if (!hours.length) return;

  const showZ = document.getElementById("chk-show-zscore")?.checked ?? true;
  const showAIS = document.getElementById("chk-show-ais")?.checked ?? true;
  const showMean = document.getElementById("chk-show-mean")?.checked ?? true;

  const pad = { l: 48, r: 18, t: 28, b: 36 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;

  const maxData = Math.max(
    1.6,
    ...hours.map(x => x.mean_flow_lpm),
    ...(showZ ? profile.map(p => p.zscore_ucl) : [0]),
    ...(showAIS ? profile.map(p => p.ais_upper) : [0])
  );
  const yMax = Math.ceil(maxData * 1.15 * 2) / 2;
  const hourW = plotW / 24;
  const xAt = idx => pad.l + (idx + 0.5) * hourW;
  const yAt = val => pad.t + plotH - (Math.min(yMax, Math.max(0, val)) / yMax) * plotH;
  const baseY = pad.t + plotH;

  // Grid Y
  ctx.strokeStyle = "#e2e8f0";
  ctx.fillStyle = "#64748b";
  ctx.font = "11px sans-serif";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const val = (yMax * i) / 5;
    const y = yAt(val);
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
    ctx.fillText(val.toFixed(2), 4, y + 4);
  }

  // Highlight anomaly columns & active hour cursor
  hours.forEach((hd, idx) => {
    const x0 = pad.l + idx * hourW;
    if (idx <= STATE.simCursor && hd.ais.is_anomaly) {
      ctx.fillStyle = "rgba(239, 68, 68, 0.14)";
      ctx.fillRect(x0, pad.t, hourW, plotH);
    }
    if (idx === STATE.simCursor) {
      ctx.save();
      ctx.strokeStyle = "#94a3b8";
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(xAt(idx), pad.t);
      ctx.lineTo(xAt(idx), baseY);
      ctx.stroke();
      ctx.restore();
    }
  });

  // Area fill + boundary curve helper
  const drawThresholdBand = (key, fillColor, strokeColor, dashed) => {
    if (!profile.length) return;
    ctx.fillStyle = fillColor;
    ctx.beginPath();
    ctx.moveTo(pad.l, baseY);
    ctx.lineTo(pad.l, yAt(profile[0][key]));
    profile.forEach((p, idx) => ctx.lineTo(xAt(idx), yAt(p[key])));
    ctx.lineTo(w - pad.r, yAt(profile[profile.length - 1][key]));
    ctx.lineTo(w - pad.r, baseY);
    ctx.closePath();
    ctx.fill();

    ctx.save();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2.0;
    if (dashed) ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.l, yAt(profile[0][key]));
    profile.forEach((p, idx) => ctx.lineTo(xAt(idx), yAt(p[key])));
    ctx.lineTo(w - pad.r, yAt(profile[profile.length - 1][key]));
    ctx.stroke();
    ctx.restore();
  };

  if (showAIS) {
    drawThresholdBand("ais_upper", "rgba(14, 165, 233, 0.20)", "#0284c7", false);
  }
  if (showZ) {
    drawThresholdBand("zscore_ucl", "rgba(249, 115, 22, 0.22)", "#ea580c", true);
  }

  if (showMean && profile.length) {
    ctx.save();
    ctx.strokeStyle = "#ca8a04";
    ctx.lineWidth = 1.6;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    profile.forEach((p, idx) => {
      const cx = xAt(idx);
      const cy = yAt(p.combined_mean);
      if (idx === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    });
    ctx.stroke();
    ctx.restore();
  }

  // Live Flow 1-Hour Line
  ctx.strokeStyle = "#0f172a";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  hours.forEach((hd, idx) => {
    if (idx > STATE.simCursor) return;
    const cx = xAt(idx);
    const cy = yAt(hd.mean_flow_lpm);
    if (idx === 0) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  });
  ctx.stroke();

  // Points & Red X Anomaly markers
  hours.forEach((hd, idx) => {
    if (idx > STATE.simCursor) return;
    const cx = xAt(idx);
    const cy = yAt(hd.mean_flow_lpm);
    if (hd.ais.is_anomaly) {
      ctx.save();
      ctx.strokeStyle = "#dc2626";
      ctx.lineWidth = 2.8;
      ctx.beginPath();
      ctx.moveTo(cx - 5.5, cy - 5.5);
      ctx.lineTo(cx + 5.5, cy + 5.5);
      ctx.moveTo(cx + 5.5, cy - 5.5);
      ctx.lineTo(cx - 5.5, cy + 5.5);
      ctx.stroke();
      ctx.fillStyle = "#b91c1c";
      ctx.font = "bold 10px sans-serif";
      ctx.fillText("ANOM", Math.min(w - pad.r - 32, cx - 14), Math.max(pad.t + 12, cy - 9));
      ctx.restore();
    } else {
      ctx.beginPath();
      ctx.arc(cx, cy, idx === STATE.simCursor ? 5 : 3.2, 0, Math.PI * 2);
      ctx.fillStyle = idx === STATE.simCursor ? "#0284c7" : "#0f172a";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
  });

  // Compact Legend Bar inside top padding
  ctx.font = "11px sans-serif";
  let lx = pad.l;
  const drawLeg = (color, text) => {
    ctx.fillStyle = color;
    ctx.fillRect(lx, 7, 12, 8);
    ctx.fillStyle = "#334155";
    ctx.fillText(text, lx + 16, 15);
    lx += ctx.measureText(text).width + 30;
  };
  drawLeg("#0f172a", "Live Flow (1-Jam)");
  if (showAIS) drawLeg("#0284c7", "AIS Batas Atas 2/4 (Biru)");
  if (showZ) drawLeg("#ea580c", "Z-Score UCL (Oranye)");
  if (showMean) drawLeg("#ca8a04", "Mean M1+M2");

  // X labels
  ctx.fillStyle = "#475569";
  ctx.font = "11px sans-serif";
  for (let hr = 0; hr < 24; hr += 2) {
    ctx.fillText(`${String(hr).padStart(2, "0")}:00`, xAt(hr) - 14, h - 10);
  }
  ctx.fillText("L/min", 4, 15);

  canvas.onclick = ev => {
    const rect = canvas.getBoundingClientRect();
    const clickX = ev.clientX - rect.left;
    const hr = Math.floor((clickX - pad.l) / hourW);
    if (hr >= 0 && hr < 24) {
      STATE.simCursor = hr;
      renderActiveHourCard(hours[hr]);
      drawSimulationChart();
      renderSimulationTable();
      drawImmunologicalSpace();
    }
  };
}

function renderSimulationTable() {
  const tbody = document.getElementById("sim-log-body");
  if (!STATE.simData || !tbody) return;
  tbody.innerHTML = STATE.simData.hours
    .slice(0, STATE.simCursor + 1)
    .map((hd, idx) => {
      const aBadge = hd.ais.is_anomaly
        ? `<span class="badge badge-anomaly">ANOMALI</span>`
        : `<span class="badge badge-normal">NORMAL</span>`;
      const mStr = `${hd.ais.max_detector_margin >= 0 ? "+" : ""}${hd.ais.max_detector_margin.toFixed(3)}`;
      return `
        <tr class="${hd.ais.is_anomaly ? "row-anom" : ""} ${idx === STATE.simCursor ? "row-highlight" : ""}">
          <td><strong>${String(hd.hour).padStart(2, "0")}:00</strong></td>
          <td>[${hd.slot_flows.map(v => v.toFixed(2)).join(", ")}]</td>
          <td><b>${hd.mean_flow_lpm.toFixed(3)}</b> <span style="color:#64748b;font-size:0.75rem;">(2/4&le;${(hd.ais_upper || 0).toFixed(2)} | 4/4&ge;${(hd.ais_leak_limit || 0).toFixed(2)})</span></td>
          <td>${hd.active_slots}/4</td>
          <td>${aBadge}</td>
          <td>d=${hd.ais.min_dist_to_self.toFixed(3)} <span style="color:#64748b;font-size:0.75rem;">(m=${mStr})</span></td>
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
    document.getElementById("btn-sim-play").textContent = "Play";
  }
  const day_name = document.getElementById("sim-weekday").value || "Saturday";
  const week_source = document.getElementById("sim-week-source").value || "week2";
  const inject_preset = document.getElementById("sim-preset").value;
  const inject_hour = parseInt(document.getElementById("sim-hour").value, 10) || 0;
  STATE.valveOpen = true;
  STATE.simData = await apiPost("/api/simulate_day", {
    day_name,
    week_source,
    inject_preset,
    inject_hour,
    auto_mode: STATE.autoMode,
  });
  STATE.simCursor = 23;
  const anyAnom = STATE.simData.hours.some(h => h.ais.is_anomaly);
  if (STATE.autoMode && anyAnom) {
    STATE.valveOpen = false;
  }
  const chartBadge = document.getElementById("sim-chart-badge");
  if (chartBadge) {
    chartBadge.className = `badge ${anyAnom ? "badge-anomaly" : "badge-normal"}`;
    chartBadge.textContent = `${STATE.simData.day_name} (${STATE.simData.w1_date_str} + ${STATE.simData.w2_date_str}) • Uji M${STATE.simData.week_number}`;
  }
  const focusIdx = inject_preset !== "none" ? inject_hour : 23;
  STATE.simCursor = 23;
  renderActiveHourCard(STATE.simData.hours[focusIdx]);
  drawSimulationChart();
  renderSimulationTable();
  drawImmunologicalSpace();
}

function togglePlayback() {
  const btn = document.getElementById("btn-sim-play");
  if (STATE.simTimer) {
    clearInterval(STATE.simTimer);
    STATE.simTimer = null;
    btn.textContent = "Play";
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
      btn.textContent = "Play";
    }
  }, 450);
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

  box.innerHTML = `
    <div style="margin-bottom:8px;">
      <strong>Slot Mingguan:</strong> ${data.day_name} pukul ${String(data.hour).padStart(2, "0")}:00–${String((data.hour + 1) % 24).padStart(2, "0")}:00
    </div>
    ${s1 ? `<div>• <b>Minggu-1 (${s1.date_str}):</b> [${s1.slot_flows.map(v => v.toFixed(2)).join(", ")}] → rata-rata <b>${s1.mean_flow_lpm.toFixed(3)} L/min</b> (${s1.active_slots}/4 aktif)</div>` : ""}
    ${s2 ? `<div>• <b>Minggu-2 (${s2.date_str}):</b> [${s2.slot_flows.map(v => v.toFixed(2)).join(", ")}] → rata-rata <b>${s2.mean_flow_lpm.toFixed(3)} L/min</b> (${s2.active_slots}/4 aktif)</div>` : ""}
    <div class="callout">
      <strong>Penggabungan ke Self-Space AIS:</strong><br/>
      • Kedua sampel di atas masuk ke himpunan <strong>336 Self-Cells</strong> sekaligus ditopang <strong>${data.ais_circadian_support_count}</strong> sel tetangga sirkadian (±1 jam lintas 14 hari, puncak wajar ${data.ais_circadian_mean_max.toFixed(3)} L/min).<br/>
      • <strong>Proyeksi Batas AIS Slot Ini:</strong> pola intermiten 2/4 ditoleransi s.d. <b>${Number(data.ais_upper_lpm).toFixed(2)} L/min</b>, sedangkan pola kontinyu 4/4 memicu detektor mulai <b>${Number(data.ais_leak_limit_lpm).toFixed(2)} L/min</b>.<br/>
      • <strong>Pembanding Kurva Oranye Grafik (&mu; + 2&sigma;):</strong> μ = ${Number(data.combined_mean).toFixed(3)}, s = ${Number(data.combined_std).toFixed(3)} → UCL visual = <b>${Number(data.zscore_ucl).toFixed(3)} L/min</b>.
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

  detPts.forEach(d => {
    if (d.mean_flow > yMax) return;
    const x = pad.l + (d.hour / 24) * plotW;
    const y = pad.t + plotH - (d.mean_flow / yMax) * plotH;
    ctx.beginPath();
    ctx.arc(x, y, 3.2, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(249, 115, 22, 0.35)";
    ctx.fill();
  });

  selfPts.forEach(s => {
    const x = pad.l + (s.hour / 24) * plotW;
    const y = pad.t + plotH - (s.mean_flow / yMax) * plotH;
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(16, 185, 129, 0.78)";
    ctx.fill();
  });

  // Highlight injected anomaly hour if any, else active cursor hour from Tab 1 Simulasi
  if (STATE.simData && STATE.simData.hours && STATE.simData.hours.length) {
    const simHours = STATE.simData.hours;
    const anomPt = simHours.find(hr => hr.injected) || simHours[STATE.simCursor] || simHours[simHours.length - 1];
    const x = pad.l + (anomPt.hour / 24) * plotW;
    const y = pad.t + plotH - (Math.min(yMax, anomPt.mean_flow_lpm) / yMax) * plotH;
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fillStyle = anomPt.ais.is_anomaly ? "#dc2626" : "#2563eb";
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
      "Hasil evaluasi AIS belum ditemukan di folder results/. Jalankan <code>python run_experiments.py</code>.";
    return;
  }
  const bs = STATE.benchData.benchmark_summary;
  const models = bs.benchmark.models;
  const holdout = bs.holdout_w1_w2;
  const lodo = STATE.benchData.lodo_cv || {};
  const perCat = bs.benchmark.per_category_performance || [];
  const vdet = models.WaterAIS_VDetector;

  document.getElementById("kpi-holdout").textContent =
    `LODO ${(lodo.overall_fpr !== undefined ? lodo.overall_fpr * 100 : holdout.fpr * 100).toFixed(2)}% FPR`;
  document.getElementById("kpi-holdout-sub").textContent =
    `Holdout W1→W2: ${holdout.false_positives}/${holdout.test_hours} (${(holdout.fpr * 100).toFixed(2)}%) | LODO: ${lodo.total_fp ?? 0}/${lodo.total_hours ?? 336} FP`;
  document.getElementById("kpi-f1").textContent =
    `F1 ${(vdet.f1_score.mean * 100).toFixed(2)}%`;
  document.getElementById("kpi-f1-sub").textContent =
    `Acc ${(vdet.accuracy.mean * 100).toFixed(2)}% | Prec ${(vdet.precision.mean * 100).toFixed(1)}% | Rec ${(vdet.recall.mean * 100).toFixed(2)}%`;

  const keys = [
    ["WaterAIS_VDetector", "WaterAIS V-Detector (Context 7D Terpadu)"],
    ["PerSlot_AIS_N2", "Ablasi: Per-Slot AIS (168 Slot Terisolasi)"],
  ];
  const valOf = (obj, m) => (typeof obj[m] === "object" ? obj[m].mean * 100 : obj[m] * 100);

  document.getElementById("bench-metrics-table").innerHTML = `
    <div style="margin-bottom:10px;">
      <strong>Validasi Holdout Minggu-1 → Minggu-2 (168 jam normal):</strong><br/>
      WaterAIS Context 7D = <b>${holdout.false_positives}/${holdout.test_hours}</b> (<b>${(holdout.fpr * 100).toFixed(2)}% FPR</b>)
      vs Per-Slot AIS = <b>${holdout.perslot_false_positives}/${holdout.test_hours}</b> (${(holdout.perslot_fpr * 100).toFixed(2)}% FPR).<br/>
      <strong>14-Fold Leave-One-Day-Out Cross-Validation (336 jam normal):</strong><br/>
      WaterAIS Context 7D = <b>${lodo.total_fp ?? 0}/${lodo.total_hours ?? 336}</b> (<b>${((lodo.overall_fpr ?? 0) * 100).toFixed(2)}% FPR</b>)
      vs Per-Slot AIS = <b>${lodo.perslot_total_fp ?? "-"}/${lodo.total_hours ?? 336}</b> (${((lodo.perslot_overall_fpr ?? 0) * 100).toFixed(2)}% FPR).
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Arsitektur AIS (20 Seed)</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>FPR Uji</th></tr></thead>
        <tbody>
          ${keys.map(([k, lbl]) => {
            const m = models[k];
            return `<tr>
              <td><strong>${lbl}</strong></td>
              <td>${valOf(m, "accuracy").toFixed(2)}%</td>
              <td>${valOf(m, "precision").toFixed(2)}%</td>
              <td>${valOf(m, "recall").toFixed(2)}%</td>
              <td><b>${valOf(m, "f1_score").toFixed(2)}%</b></td>
              <td>${valOf(m, "fpr").toFixed(2)}%</td>
            </tr>`;
          }).join("")}
        </tbody>
      </table>
    </div>
  `;

  // Bar chart: Per-category performance of WaterAIS V-Detector
  const canvas = document.getElementById("canvas-bench");
  const { ctx, w, h } = setupHiDPICanvas(canvas, 305);
  const pad = { l: 42, r: 16, t: 44, b: 46 };
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

  const shortLbl = {
    normal_all: "Normal (12)",
    slow_leak: "Bocor Halus",
    night_leak: "Bocor Malam",
    night_open_tap: "Keran Malam",
    off_schedule_burst: "Jam Sepi",
    extreme_burst: "Pipa Pecah",
    continuous_high: "Banjir 4/4",
  };
  const grpW = plotW / Math.max(1, perCat.length);
  const barW = grpW * 0.56;

  perCat.forEach((cat, gi) => {
    const val = cat.correct_rate_mean * 100;
    const bh = (val / 100) * plotH;
    const bx = pad.l + gi * grpW + (grpW - barW) / 2;
    const cx = bx + barW / 2;
    const by = pad.t + plotH - bh;
    ctx.fillStyle = cat.is_anomaly_class ? "#0284c7" : "#10b981";
    ctx.fillRect(bx, by, barW, bh);
    ctx.textAlign = "center";
    ctx.fillStyle = "#0f172a";
    ctx.font = "bold 9.5px sans-serif";
    ctx.fillText(`${val.toFixed(1)}%`, cx, by - 5);

    ctx.fillStyle = "#334155";
    ctx.font = "9.5px sans-serif";
    const lbl = shortLbl[cat.category] || cat.category;
    ctx.fillText(lbl, cx, h - 22);
    ctx.fillStyle = "#64748b";
    ctx.fillText(`(${cat.n_scenarios} sken.)`, cx, h - 8);
  });
  ctx.textAlign = "left";

  ctx.fillStyle = "#0f172a";
  ctx.font = "bold 10.5px sans-serif";
  ctx.fillText(
    `WaterAIS 7D • F1 ${(vdet.f1_score.mean * 100).toFixed(2)}% | Acc ${(vdet.accuracy.mean * 100).toFixed(2)}% | Precision ${(vdet.precision.mean * 100).toFixed(1)}%`,
    pad.l,
    15
  );

  // Scenario matrix (Pure AIS)
  const scRows = bs.benchmark.scenario_comparison || [];
  document.getElementById("bench-scenario-body").innerHTML = scRows
    .map(r => {
      const ok = r.vdetector_ais === r.expected_anomaly;
      const predTxt = r.vdetector_ais ? "ANOMALI" : "NORMAL";
      return `
        <tr class="${r.expected_anomaly ? "row-anom" : ""}">
          <td><strong>${r.scenario_id}</strong></td>
          <td>${r.name}</td>
          <td><code>${r.category}</code></td>
          <td>${r.day_name.slice(0, 3)} ${String(r.hour).padStart(2, "0")}:00</td>
          <td>[${r.slot_flows.map(v => v.toFixed(2)).join(", ")}]</td>
          <td>${r.active_slots !== undefined ? r.active_slots : "-"}/4</td>
          <td><span class="badge ${r.expected_anomaly ? "badge-anomaly" : "badge-normal"}">${r.expected_anomaly ? "ANOMALI" : "NORMAL"}</span></td>
          <td><span class="badge ${ok ? "badge-normal" : "badge-anomaly"}">${predTxt}${ok ? "" : " (Meleset)"}</span></td>
          <td style="font-size:0.78rem;">d=${r.vdetector_min_dist !== undefined ? Number(r.vdetector_min_dist).toFixed(3) : "-"} • ${r.vdetector_diagnosis || "-"}</td>
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
    await inspectWeeklySlot();
  });

  document.getElementById("btn-sim-run").addEventListener("click", runSimulation);
  document.getElementById("sim-weekday").addEventListener("change", runSimulation);
  document.getElementById("sim-week-source").addEventListener("change", runSimulation);
  document.getElementById("sim-preset").addEventListener("change", runSimulation);
  ["chk-show-zscore", "chk-show-ais", "chk-show-mean"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", drawSimulationChart);
  });
  document.getElementById("btn-sim-play").addEventListener("click", togglePlayback);
  document.getElementById("btn-sim-step").addEventListener("click", () => {
    if (!STATE.simData) return;
    STATE.simCursor = (STATE.simCursor + 1) % 24;
    if (STATE.simCursor === 0) STATE.valveOpen = true;
    renderActiveHourCard(STATE.simData.hours[STATE.simCursor]);
    drawSimulationChart();
    renderSimulationTable();
    drawImmunologicalSpace();
  });
  document.getElementById("btn-sim-reset").addEventListener("click", () => {
    if (!STATE.simData) return;
    STATE.simCursor = 0;
    STATE.valveOpen = true;
    renderActiveHourCard(STATE.simData.hours[0]);
    drawSimulationChart();
    renderSimulationTable();
    drawImmunologicalSpace();
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

  const daySel = document.getElementById("sim-weekday");
  const weekdays = STATE.initData.weekdays || [];
  daySel.innerHTML = weekdays
    .map(
      w =>
        `<option value="${w.day_name}" ${w.day_name === "Saturday" ? "selected" : ""}>${w.day_name} / ${w.day_name_id} (M1: ${w.w1_date.slice(0, 6)} • M2: ${w.w2_date.slice(0, 6)})</option>`
    )
    .join("");

  try {
    STATE.benchData = await apiGet("/api/benchmark");
    renderBenchmarkTab();
  } catch (e) {
    console.warn("Benchmark belum tersedia:", e);
  }

  await runSimulation();
  await inspectWeeklySlot();
}

bootstrap();
