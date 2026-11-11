"""
app_tkinter.py — Aplikasi Desktop Tkinter + Matplotlib untuk Deteksi Anomali Air Berbasis AIS

Mengadaptasi tata letak antarmuka IoT pada Gambar 9 Jurnal JUTISI (Control & Status,
Live vs Normal Flow Chart, Sandbox Agregator 1-Jam, dan Log Anomali Per-Jam)
dengan mesin deteksi Artificial Immune System (WaterAISVDetector) dan pembanding Z-Score.

Jalankan:
    python app_tkinter.py
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import List, Optional

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

from aggregator import aggregate_tumbling_1h, build_hourly_antigen
from ais_detector import WaterAISVDetector
from dataset_loader import DAYS_ORDER, load_water_dataset
from zscore_baseline import ZScore15MinBaseline, ZScore1HourBaseline


class WaterAISTkinterApp:
    """GUI Desktop Tkinter untuk simulasi dan uji Deteksi Anomali Air Berbasis AIS."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("WaterAIS — Deteksi Anomali Penggunaan Air (Artificial Immune System)")
        self.root.geometry("1360x860")
        self.root.minsize(1100, 700)

        # Muat dataset & latih model awal
        self.records = load_water_dataset()
        self.antigens = aggregate_tumbling_1h(self.records)
        self.z15 = ZScore15MinBaseline(k=2.0, ucl_min=0.20).fit(self.records)
        self.z1h = ZScore1HourBaseline(k=2.0, ucl_min=0.08).fit(self.antigens)
        self.ais = WaterAISVDetector(r_self=0.43, max_detectors=1500, max_attempts=25000, seed=42).fit(
            self.antigens
        )

        # State kontrol IoT
        self.auto_mode = tk.BooleanVar(value=True)
        self.valve_open = True
        self.r_self_var = tk.DoubleVar(value=0.43)
        self.max_det_var = tk.IntVar(value=1500)

        # State simulasi harian
        self.day_labels = [
            f"Hari {d+1:02d}: {self.antigens[d*24].day_name}, {self.antigens[d*24].date_str} (M{self.antigens[d*24].week_number})"
            for d in range(14)
        ]
        self.sim_day_var = tk.StringVar(value=self.day_labels[13])
        self.sim_preset_var = tk.StringVar(value="Tanpa Injeksi (Asli)")
        self.sim_hour_var = tk.IntVar(value=10)
        self.sim_cursor = 23
        self.is_playing = False
        self.play_after_id: Optional[str] = None
        self.sim_hours_data: List[dict] = []

        self._build_ui()
        self.run_day_simulation()

    def _build_ui(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Header Status Bar
        hdr = tk.Frame(self.root, bg="#0f172a", padx=14, pady=10)
        hdr.pack(fill=tk.X)

        tk.Label(
            hdr,
            text="WaterAIS — Deteksi Anomali Air Berbasis Artificial Immune System (Agregator 1-Jam)",
            fg="#ffffff",
            bg="#0f172a",
            font=("Segoe UI", 12, "bold"),
        ).pack(side=tk.LEFT)

        self.lbl_valve = tk.Label(
            hdr,
            text="Solenoid Valve: OPEN",
            fg="#ecfdf5",
            bg="#059669",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=4,
        )
        self.lbl_valve.pack(side=tk.RIGHT, padx=(8, 0))

        self.btn_mode = tk.Button(
            hdr,
            text="Mode: AUTO",
            command=self.toggle_mode,
            bg="#0284c7",
            fg="#ffffff",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=4,
        )
        self.btn_mode.pack(side=tk.RIGHT, padx=6)

        tk.Button(
            hdr,
            text="Reset / Buka Valve",
            command=self.reset_valve,
            bg="#334155",
            fg="#ffffff",
            relief=tk.FLAT,
            font=("Segoe UI", 9),
            padx=10,
            pady=4,
        ).pack(side=tk.RIGHT, padx=4)

        # Notebook tabs (mirip Gambar 9 Jurnal JUTISI)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.tab_sim = ttk.Frame(self.notebook)
        self.tab_lab = ttk.Frame(self.notebook)
        self.tab_logs = ttk.Frame(self.notebook)
        self.tab_ais = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_sim, text="  Control & Live vs Normal (24 Jam)  ")
        self.notebook.add(self.tab_lab, text="  Sandbox Agregator 1-Jam (4x15m)  ")
        self.notebook.add(self.tab_logs, text="  Anomaly Logs Per-Jam  ")
        self.notebook.add(self.tab_ais, text="  Ruang V-Detector AIS  ")

        self._build_sim_tab()
        self._build_lab_tab()
        self._build_logs_tab()
        self._build_ais_tab()

    def _build_sim_tab(self) -> None:
        left = ttk.LabelFrame(self.tab_sim, text=" Kontrol Simulasi & Parameter AIS ", padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 8), pady=4)

        ttk.Label(left, text="Pilih Hari (Dataset 2 Minggu):").pack(anchor=tk.W)
        cb_day = ttk.Combobox(left, textvariable=self.sim_day_var, values=self.day_labels, state="readonly", width=34)
        cb_day.pack(fill=tk.X, pady=(2, 8))

        ttk.Label(left, text="Skenario Injeksi Anomali:").pack(anchor=tk.W)
        presets = [
            "Tanpa Injeksi (Asli)",
            "Kebocoran Halus (0.35 x 4)",
            "Keran Terbuka Malam (1.20, 1.20, 0, 0)",
            "Pipa Pecah Ekstrem (5.10, 4.60, 0, 0)",
            "Banjir Kontinyu 1 Jam (1.80 x 4)",
            "Uji Normal: Geser Jam Mandi (0, 2.35, 0, 0)",
        ]
        cb_pre = ttk.Combobox(left, textvariable=self.sim_preset_var, values=presets, state="readonly", width=34)
        cb_pre.pack(fill=tk.X, pady=(2, 8))

        ttk.Label(left, text="Jam Injeksi (0 - 23):").pack(anchor=tk.W)
        ttk.Spinbox(left, from_=0, to=23, textvariable=self.sim_hour_var, width=10).pack(anchor=tk.W, pady=(2, 10))

        btn_row = ttk.Frame(left)
        btn_row.pack(fill=tk.X, pady=4)
        ttk.Button(btn_row, text="Jalankan 24 Jam", command=self.run_day_simulation).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))
        self.btn_play = ttk.Button(btn_row, text="Play", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3)
        ttk.Button(btn_row, text="+1 Jam", command=self.step_hour).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Parameter AIS V-Detector:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        ttk.Label(left, text="Self-Radius (r_self):").pack(anchor=tk.W, pady=(4, 0))
        ttk.Scale(left, from_=0.32, to=0.54, variable=self.r_self_var).pack(fill=tk.X)
        ttk.Label(left, text="Max V-Detectors:").pack(anchor=tk.W, pady=(4, 0))
        ttk.Spinbox(left, from_=400, to=2200, increment=100, textvariable=self.max_det_var, width=12).pack(anchor=tk.W)
        ttk.Button(left, text="Latih Ulang AIS", command=self.retrain_ais).pack(fill=tk.X, pady=8)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        self.lbl_sim_detail = tk.Label(
            left, text="", justify=tk.LEFT, wraplength=275, font=("Segoe UI", 9), fg="#1e293b"
        )
        self.lbl_sim_detail.pack(fill=tk.BOTH, expand=True, anchor=tk.NW)

        # Chart area
        right = ttk.Frame(self.tab_sim)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.fig_sim, self.ax_sim = plt.subplots(figsize=(8.5, 5.2), dpi=100)
        self.canvas_sim = FigureCanvasTkAgg(self.fig_sim, master=right)
        self.canvas_sim.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_lab_tab(self) -> None:
        frm = ttk.Frame(self.tab_lab, padding=14)
        frm.pack(fill=tk.BOTH, expand=True)

        inp = ttk.LabelFrame(frm, text=" Input 4 Slot x 15 Menit (Agregator 1-Jam) ", padding=12)
        inp.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        self.lab_day = tk.StringVar(value="Monday")
        self.lab_hour = tk.IntVar(value=10)
        self.lab_q = [tk.DoubleVar(value=0.35) for _ in range(4)]

        ttk.Label(inp, text="Hari:").pack(anchor=tk.W)
        ttk.Combobox(inp, textvariable=self.lab_day, values=list(DAYS_ORDER), state="readonly", width=22).pack(fill=tk.X, pady=(2, 8))
        ttk.Label(inp, text="Jam (0 - 23):").pack(anchor=tk.W)
        ttk.Spinbox(inp, from_=0, to=23, textvariable=self.lab_hour, width=10).pack(anchor=tk.W, pady=(2, 8))

        for i, var in enumerate(self.lab_q):
            ttk.Label(inp, text=f"Slot {i+1} ({i*15:02d}-{(i+1)*15:02d} mnt) L/min:").pack(anchor=tk.W)
            ttk.Entry(inp, textvariable=var, width=16).pack(anchor=tk.W, pady=(1, 6))

        ttk.Button(inp, text="Evaluasi AIS vs Z-Score", command=self.evaluate_sandbox).pack(fill=tk.X, pady=8)

        presets = [
            ("Kebocoran Halus (10:00)", "Monday", 10, (0.35, 0.35, 0.35, 0.35)),
            ("Keran Terbuka (03:00)", "Tuesday", 3, (1.20, 1.20, 0.00, 0.00)),
            ("Mandi Geser Jam (Min 07:00)", "Sunday", 7, (0.00, 2.35, 0.00, 0.00)),
            ("Pipa Pecah (08:00)", "Tuesday", 8, (5.20, 4.80, 0.00, 0.00)),
        ]
        for title, d, h, qs in presets:
            ttk.Button(
                inp,
                text=title,
                command=lambda d=d, h=h, qs=qs: self._load_sandbox_preset(d, h, qs),
            ).pack(fill=tk.X, pady=2)

        out = ttk.LabelFrame(frm, text=" Hasil Diagnosis Komparatif ", padding=12)
        out.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.txt_lab = tk.Text(out, font=("Consolas", 10), wrap=tk.WORD, bg="#f8fafc", relief=tk.FLAT)
        self.txt_lab.pack(fill=tk.BOTH, expand=True)
        self.evaluate_sandbox()

    def _build_logs_tab(self) -> None:
        cols = ("jam", "slots", "mean", "aktif", "ais", "z15", "z1h", "diagnosis")
        self.tree = ttk.Treeview(self.tab_logs, columns=cols, show="headings")
        headers = {
            "jam": "Jam",
            "slots": "4 x 15 Menit (L/min)",
            "mean": "Rata-rata (L/min)",
            "aktif": "Aktif",
            "ais": "AIS V-Detector",
            "z15": "Z-Score 15m",
            "z1h": "Z-Score 1h",
            "diagnosis": "Diagnosis & Aksi Valve",
        }
        widths = {"jam": 70, "slots": 190, "mean": 115, "aktif": 65, "ais": 110, "z15": 100, "z1h": 100, "diagnosis": 360}
        for c in cols:
            self.tree.heading(c, text=headers[c])
            self.tree.column(c, width=widths[c])
        self.tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def _build_ais_tab(self) -> None:
        self.fig_ais, (self.ax_trace, self.ax_space) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=100)
        self.canvas_ais = FigureCanvasTkAgg(self.fig_ais, master=self.tab_ais)
        self.canvas_ais.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._draw_ais_plots()

    def _load_sandbox_preset(self, d: str, h: int, qs: tuple) -> None:
        self.lab_day.set(d)
        self.lab_hour.set(h)
        for v, val in zip(self.lab_q, qs):
            v.set(val)
        self.evaluate_sandbox()

    def evaluate_sandbox(self) -> None:
        flows = [max(0.0, float(v.get())) for v in self.lab_q]
        ag = build_hourly_antigen(flows, int(self.lab_hour.get()) % 24, self.lab_day.get())
        r_ais = self.ais.predict_hour(ag, auto_mode=self.auto_mode.get())
        r_z15 = self.z15.predict_hour(ag)
        r_z1h = self.z1h.predict_hour(ag)

        report = [
            f"=== AGREGATOR 1-JAM ===",
            f"Waktu       : {ag.day_name} pukul {ag.hour:02d}:00-{(ag.hour+1)%24:02d}:00",
            f"4 Slot 15m  : {list(ag.slot_flows)} L/min",
            f"Rata-rata   : {ag.mean_flow_lpm:.4f} L/min  | Total Volume: {ag.total_liters:.2f} Liter/jam",
            f"Puncak/Min  : max={ag.max_flow_lpm:.2f} L/min, min={ag.min_flow_lpm:.2f} L/min, aktif={ag.active_slots}/4",
            "",
            f"--- 1. ARTIFICIAL IMMUNE SYSTEM (V-Detector Context 7D) ---",
            f"Status      : {r_ais.status_label} (Keyakinan {r_ais.confidence_pct:.1f}%)",
            f"Kategori    : {r_ais.anomaly_category}",
            f"Pemicu      : {r_ais.triggered_detectors} V-Detectors (margin {r_ais.max_detector_margin:+.4f})",
            f"Jarak Self  : {r_ais.min_dist_to_self:.4f} (r_self={self.ais.r_self:.2f}, terdekat: {r_ais.nearest_self_label})",
            f"Aksi Valve  : {r_ais.valve_action}",
            "",
            f"--- 2. BASELINE Z-SCORE 15-MENIT (Jurnal JUTISI, N=2) ---",
            f"Status      : {'ANOMALI' if r_z15.is_anomaly else 'NORMAL'}",
            f"Penjelasan  : {r_z15.reason}",
            "",
            f"--- 3. BASELINE Z-SCORE 1-JAM (Slot N=2) ---",
            f"Status      : {'ANOMALI' if r_z1h.is_anomaly else 'NORMAL'}",
            f"Penjelasan  : {r_z1h.reason}",
        ]
        self.txt_lab.delete("1.0", tk.END)
        self.txt_lab.insert(tk.END, "\n".join(report))

    def _update_valve_banner(self) -> None:
        self.btn_mode.config(text="Mode: AUTO" if self.auto_mode.get() else "Mode: MANUAL")
        if self.valve_open:
            self.lbl_valve.config(text="Solenoid Valve: OPEN (Mengalir)", bg="#059669")
        else:
            self.lbl_valve.config(text="Solenoid Valve: SHUT-OFF (Tertutup)", bg="#dc2626")

    def toggle_mode(self) -> None:
        self.auto_mode.set(not self.auto_mode.get())
        if not self.auto_mode.get():
            self.valve_open = True
        self._update_valve_banner()
        self.run_day_simulation()

    def reset_valve(self) -> None:
        self.valve_open = True
        self._update_valve_banner()

    def retrain_ais(self) -> None:
        self.ais = WaterAISVDetector(
            r_self=round(float(self.r_self_var.get()), 2),
            max_detectors=int(self.max_det_var.get()),
            max_attempts=25000,
            seed=42,
        ).fit(self.antigens)
        self._draw_ais_plots()
        self.run_day_simulation()
        self.evaluate_sandbox()
        messagebox.showinfo(
            "Pelatihan Selesai",
            f"Berhasil melatih {len(self.ais.detectors)} Mature V-Detectors dalam {self.ais.trace.train_time_sec:.3f}s.",
        )

    def run_day_simulation(self) -> None:
        day_idx = self.day_labels.index(self.sim_day_var.get()) if self.sim_day_var.get() in self.day_labels else 13
        preset = self.sim_preset_var.get()
        inj_h = int(self.sim_hour_var.get()) % 24
        day_ags = [ag for ag in self.antigens if ag.day_number == day_idx]

        self.sim_hours_data = []
        self.valve_open = True
        for ag in day_ags:
            flows = list(ag.slot_flows)
            if ag.hour == inj_h:
                if "0.35 x 4" in preset:
                    flows = [0.35, 0.35, 0.35, 0.35]
                elif "1.20, 1.20" in preset:
                    flows = [1.20, 1.20, 0.00, 0.00]
                elif "5.10, 4.60" in preset:
                    flows = [5.10, 4.60, 0.00, 0.00]
                elif "1.80 x 4" in preset:
                    flows = [1.80, 1.80, 1.80, 1.80]
                elif "2.35" in preset:
                    flows = [0.00, 2.35, 0.00, 0.00]

            cur = build_hourly_antigen(flows, ag.hour, ag.day_name, ag.date_str, ag.day_number, ag.week_number, ag.hour_index)
            r_ais = self.ais.predict_hour(cur, auto_mode=self.auto_mode.get())
            r_z15 = self.z15.predict_hour(cur)
            r_z1h = self.z1h.predict_hour(cur)
            self.sim_hours_data.append({"ag": cur, "ais": r_ais, "z15": r_z15, "z1h": r_z1h})

        self.sim_cursor = 23
        self._refresh_sim_visuals()

    def step_hour(self) -> None:
        if not self.sim_hours_data:
            return
        self.sim_cursor = (self.sim_cursor + 1) % 24
        if self.sim_cursor == 0:
            self.valve_open = True
        self._refresh_sim_visuals()

    def toggle_play(self) -> None:
        if self.is_playing:
            self.is_playing = False
            self.btn_play.config(text="Play")
            if self.play_after_id:
                self.root.after_cancel(self.play_after_id)
                self.play_after_id = None
        else:
            self.is_playing = True
            self.btn_play.config(text="Pause")
            if self.sim_cursor >= 23:
                self.sim_cursor = 0
                self.valve_open = True
            self._tick_play()

    def _tick_play(self) -> None:
        if not self.is_playing:
            return
        self._refresh_sim_visuals()
        if self.sim_cursor < 23:
            self.sim_cursor += 1
            self.play_after_id = self.root.after(450, self._tick_play)
        else:
            self.is_playing = False
            self.btn_play.config(text="Play")

    def _refresh_sim_visuals(self) -> None:
        shown = self.sim_hours_data[: self.sim_cursor + 1]
        if any(item["ais"].is_anomaly for item in shown) and self.auto_mode.get():
            self.valve_open = False
        self._update_valve_banner()

        cur = shown[-1]
        ag = cur["ag"]
        ra = cur["ais"]
        self.lbl_sim_detail.config(
            text=(
                f"Jam Aktif : {ag.hour:02d}:00 - {(ag.hour+1)%24:02d}:00\n"
                f"4x15m     : {list(ag.slot_flows)}\n"
                f"Rata-rata : {ag.mean_flow_lpm:.3f} L/min ({ag.total_liters:.1f} L)\n"
                f"Status AIS: {ra.status_label} ({ra.confidence_pct:.1f}%)\n"
                f"Diagnosis : {ra.anomaly_category}\n"
                f"Z15m/Z1h  : {'ANOM' if cur['z15'].is_anomaly else 'Norm'} / {'ANOM' if cur['z1h'].is_anomaly else 'Norm'}"
            )
        )

        # Plot 24h
        self.ax_sim.clear()
        for idx, item in enumerate(shown):
            a = item["ag"]
            col = "#ef4444" if item["ais"].is_anomaly else "#38bdf8"
            xs = [idx + s * 0.25 + 0.125 for s in range(4)]
            self.ax_sim.bar(xs, a.slot_flows, width=0.2, color=col, alpha=0.82)
            if item["ais"].is_anomaly:
                self.ax_sim.axvspan(idx, idx + 1, color="#fee2e2", alpha=0.45, zorder=0)

        hrs = [i + 0.5 for i in range(len(shown))]
        means = [it["ag"].mean_flow_lpm for it in shown]
        self.ax_sim.plot(hrs, means, color="#0f172a", marker="o", markersize=4, lw=1.8, label="Rata-rata Agregator 1-Jam")
        self.ax_sim.set_xlim(0, 24)
        self.ax_sim.set_xticks(range(0, 25, 2))
        self.ax_sim.set_xlabel("Jam (00:00 - 24:00)")
        self.ax_sim.set_ylabel("Laju Air (L/min)")
        self.ax_sim.set_title(f"Monitoring 24 Jam — {ag.day_name}, {ag.date_str}", fontweight="bold")
        self.ax_sim.grid(alpha=0.25, linestyle="--")
        self.ax_sim.legend(loc="upper right", fontsize=8)
        self.canvas_sim.draw()

        # Refresh table
        for row in self.tree.get_children():
            self.tree.delete(row)
        for item in shown:
            a = item["ag"]
            ra = item["ais"]
            self.tree.insert(
                "",
                tk.END,
                values=(
                    f"{a.hour:02d}:00",
                    f"{[round(x, 2) for x in a.slot_flows]}",
                    f"{a.mean_flow_lpm:.3f} ({a.total_liters:.1f}L)",
                    f"{a.active_slots}/4",
                    ra.status_label,
                    "ANOM" if item["z15"].is_anomaly else "Normal",
                    "ANOM" if item["z1h"].is_anomaly else "Normal",
                    f"{ra.anomaly_category} | {ra.valve_action}",
                ),
            )

    def _draw_ais_plots(self) -> None:
        self.ax_trace.clear()
        self.ax_space.clear()

        steps = self.ais.trace.history_steps if self.ais.trace else []
        if steps:
            att = [s["attempt"] for s in steps]
            self.ax_trace.plot(att, [s["detectors"] for s in steps], label="Mature V-Detectors", color="#0284c7", lw=2)
            self.ax_trace.plot(att, [s["censored"] for s in steps], label="Censored by Self", color="#dc2626", ls="--")
            self.ax_trace.plot(att, [s["redundant"] for s in steps], label="Redundant", color="#94a3b8", ls=":")
            self.ax_trace.set_title("Thymus Censorship (Training AIS)", fontweight="bold")
            self.ax_trace.set_xlabel("Kandidat Dievaluasi")
            self.ax_trace.set_ylabel("Jumlah")
            self.ax_trace.grid(alpha=0.25)
            self.ax_trace.legend(fontsize=8)

        D = self.ais.detectors
        angles = np.arctan2(D[:, 4] * 2.0 - 1.0, D[:, 5] * 2.0 - 1.0)
        d_hours = (angles / (2.0 * np.pi) * 24.0) % 24.0
        d_means = np.expm1(D[:, 0] * np.log1p(1.5))
        idx_sub = np.linspace(0, len(D) - 1, min(350, len(D)), dtype=int)

        self.ax_space.scatter(d_hours[idx_sub], d_means[idx_sub], c="#f97316", s=16, alpha=0.35, label="V-Detectors")
        self.ax_space.scatter(
            [ag.hour for ag in self.antigens],
            [ag.mean_flow_lpm for ag in self.antigens],
            c="#10b981",
            s=22,
            alpha=0.75,
            label="Self Antigens (336)",
        )
        self.ax_space.set_title("Ruang 2D Self vs V-Detectors", fontweight="bold")
        self.ax_space.set_xlabel("Jam (0 - 23)")
        self.ax_space.set_ylabel("Rata-rata Laju 1-Jam (L/min)")
        self.ax_space.set_ylim(-0.05, 2.6)
        self.ax_space.grid(alpha=0.25)
        self.ax_space.legend(fontsize=8)
        self.canvas_ais.draw()


def main() -> None:
    root = tk.Tk()
    WaterAISTkinterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

