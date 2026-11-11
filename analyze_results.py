"""
analyze_results.py — Penghasil Grafik Analisis Pure AIS & Ringkasan Markdown

Membaca hasil dari `results/` serta dataset 2 minggu, lalu menghasilkan:
- results/figures/fig1_dataset_2weeks_profile.png
- results/figures/fig2_threshold_overlay_and_activeslots.png
- results/figures/fig3_vdetector_training_and_space.png
- results/figures/fig4_ais_performance_and_ablation.png
- results/figures/fig5_sensitivity_rself_and_detectors.png
- results/benchmark_summary.md
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aggregator import aggregate_tumbling_1h, group_weekly_two_samples
from ais_detector import WaterAISVDetector
from dataset_loader import load_water_dataset
from scenarios import build_benchmark_scenarios
from zscore_baseline import ZScore1HourBaseline


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def plot_dataset_profile(antigens) -> None:
    """Gambar 1: Heatmap 14 hari x 24 jam & kurva sirkadian 24-jam."""
    matrix = np.zeros((14, 24), dtype=float)
    day_labels = [""] * 14
    for ag in antigens:
        matrix[ag.day_number, ag.hour] = ag.mean_flow_lpm
        day_labels[ag.day_number] = f"{ag.day_name[:3]} {ag.date_str[:6]} (M{ag.week_number})"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.35, 1.0]})
    im = ax1.imshow(matrix, aspect="auto", cmap="YlGnBu", interpolation="nearest")
    ax1.set_title("Peta Panas Laju Air Rata-Rata 1 Jam (14 Hari x 24 Jam)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Jam (00:00 – 23:00)")
    ax1.set_ylabel("Hari Pengamatan (Minggu 1 & 2)")
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)], fontsize=8)
    ax1.set_yticks(range(14))
    ax1.set_yticklabels(day_labels, fontsize=8)
    ax1.axhline(6.5, color="#e11d48", linestyle="--", linewidth=1.2)
    cbar = fig.colorbar(im, ax=ax1, shrink=0.88)
    cbar.set_label("Rata-rata Laju (L/min)")

    hours = np.arange(24)
    mean_by_h = np.mean(matrix, axis=0)
    max_by_h = np.max(matrix, axis=0)
    min_by_h = np.min(matrix, axis=0)

    ax2.fill_between(hours, min_by_h, max_by_h, color="#38bdf8", alpha=0.25, label="Rentang Min–Maks 14 Hari")
    ax2.plot(hours, mean_by_h, color="#0284c7", linewidth=2.2, marker="o", markersize=4, label="Rata-rata Lintas 14 Hari")
    ax2.axvspan(0, 5.5, color="#94a3b8", alpha=0.12, label="Zona Tidur (00–05)")
    ax2.axvspan(5.5, 8.5, color="#f59e0b", alpha=0.10, label="Puncak Pagi (06–08)")
    ax2.axvspan(16.5, 19.5, color="#10b981", alpha=0.10, label="Puncak Sore (17–19)")
    ax2.set_title("Profil Sirkadian Penggunaan Air Normal (336 Jam Self)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Jam Dalam Sehari")
    ax2.set_ylabel("Laju Aliran Rata-rata 1 Jam (L/min)")
    ax2.set_ylim(-0.04, 1.52)
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(alpha=0.25, linestyle="--")
    ax2.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_dataset_2weeks_profile.png"), dpi=180)
    plt.close(fig)


def plot_threshold_overlay_and_activeslots(antigens) -> None:
    """
    Gambar 2:
    Kiri  : Perbandingan visual Threshold AIS (Biru Transparan) vs Z-Score (Oranye Transparan)
            beserta batas proyeksi kebocoran kontinyu 4/4 AIS pada hari Minggu (gabungan M1+M2).
    Kanan : Distribusi slot aktif (0/4..4/4) pada 336 jam normal.
    """
    ais = WaterAISVDetector(r_self=0.43, max_detectors=1500, max_attempts=25000, seed=42).fit(antigens)
    z1h = ZScore1HourBaseline(k=2.0, ucl_min=0.08).fit(antigens)
    grouped = group_weekly_two_samples(antigens)

    day = "Sunday"
    ais_prof = ais.compute_weekday_threshold_profile(day)
    hours = np.arange(24, dtype=float) + 0.5
    x_full = np.concatenate(([0.0], hours, [24.0]))

    w1_vals = [grouped[(day, h)][0].mean_flow_lpm for h in range(24)]
    w2_vals = [grouped[(day, h)][1].mean_flow_lpm for h in range(24)]
    y_ais = np.array([p["ais_upper_lpm"] for p in ais_prof], dtype=float)
    y_leak = np.array([p["ais_leak_limit_lpm"] for p in ais_prof], dtype=float)
    y_z = np.array([z1h.slot_stats[(day, h)].ucl_final for h in range(24)], dtype=float)
    y_mean = np.array([z1h.slot_stats[(day, h)].mean for h in range(24)], dtype=float)

    y_ais_band = np.concatenate(([y_ais[0]], y_ais, [y_ais[-1]]))
    y_z_band = np.concatenate(([y_z[0]], y_z, [y_z[-1]]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.1))

    ax1.fill_between(x_full, 0, y_ais_band, color="#0ea5e9", alpha=0.22, label="Proyeksi Threshold AIS (Intermiten 2/4 — Biru)")
    ax1.plot(x_full, y_ais_band, color="#0284c7", lw=2.0)

    ax1.fill_between(x_full, 0, y_z_band, color="#f97316", alpha=0.24, label="Pembanding Threshold Z-Score (mu+2*sigma — Oranye)")
    ax1.plot(x_full, y_z_band, color="#ea580c", lw=1.9, ls="--")

    ax1.plot(hours, y_leak, color="#0f766e", lw=1.8, ls=":", marker=".", markersize=5,
             label="Batas Kebocoran Kontinyu 4/4 AIS (ais_leak_limit)")
    ax1.plot(hours, y_mean, color="#ca8a04", lw=1.4, ls="-.", label="Rata-rata Gabungan M1+M2")
    ax1.plot(hours, w1_vals, color="#1e293b", lw=1.3, marker="o", markersize=3.5, alpha=0.75, label="Data Asli M1 (04 Mei)")
    ax1.plot(hours, w2_vals, color="#4338ca", lw=1.3, marker="^", markersize=3.8, alpha=0.80, label="Data Asli M2 (11 Mei)")

    ax1.annotate(
        "Puncak Z-Score jam 08:00:\nW1=0.00, W2=1.04 -> UCL=2.00\nSementara AIS naik bertahap sirkadian",
        xy=(8.5, 2.00), xytext=(1.0, 2.08),
        arrowprops=dict(arrowstyle="->", color="#c2410c", lw=1.3),
        fontsize=7.8, color="#9a3412",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff7ed", ec="#fdba74"),
    )
    ax1.annotate(
        "Proyeksi 4/4 AIS (0.04–0.20 L/min)\njauh lebih ketat utk kebocoran konstan",
        xy=(11.5, y_leak[11]), xytext=(11.5, 0.48),
        arrowprops=dict(arrowstyle="->", color="#0f766e", lw=1.3),
        fontsize=7.8, color="#115e59",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f0fdfa", ec="#5eead4"),
    )
    ax1.set_title("Overlay Threshold AIS (Biru) vs Z-Score (Oranye) — Hari Minggu", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Jam dalam Sehari (00:00 – 24:00)")
    ax1.set_ylabel("Laju Aliran 1-Jam (L/min)")
    ax1.set_xlim(0, 24)
    ax1.set_ylim(0, max(float(np.max(y_ais)), float(np.max(y_z))) * 1.18)
    ax1.set_xticks(range(0, 25, 2))
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.legend(fontsize=7.2, loc="upper right")

    counts = [sum(1 for ag in antigens if ag.active_slots == k) for k in range(5)]
    bars = ax2.bar(
        ["0/4 Slot\n(Diam)", "1/4 Slot\n(Sesaat)", "2/4 Slot\n(Intermiten)", "3/4 Slot\n(Padat)", "4/4 Slot\n(Kontinyu)"],
        counts,
        color=["#94a3b8", "#38bdf8", "#0284c7", "#f59e0b", "#ef4444"],
    )
    for bar, c in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, c + 3, f"{c}\n({c/336*100:.1f}%)",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax2.annotate(
        "0 dari 336 jam normal aktif 4/4 slot!\nMemisahkan manifold Normal vs Kebocoran Kontinyu",
        xy=(4, 5), xytext=(2.15, 118),
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5),
        fontsize=8.3, color="#991b1b", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fef2f2", ec="#fca5a5"),
    )
    ax2.set_ylim(0, 195)
    ax2.set_title("Distribusi Slot Aktif per 1-Jam (336 Jam Self Normal)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Jumlah Blok 1-Jam")
    ax2.grid(axis="y", alpha=0.25, linestyle="--")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_threshold_overlay_and_activeslots.png"), dpi=180)
    plt.close(fig)


def plot_vdetector_space(antigens, scenarios) -> None:
    """Gambar 3: Kurva training Negative Selection & peta 2D Self vs Mature V-Detectors."""
    ais = WaterAISVDetector(r_self=0.43, max_detectors=1500, max_attempts=25000, seed=42).fit(antigens)
    trace = ais.trace
    steps = trace.history_steps

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    attempts = [s["attempt"] for s in steps]
    dets = [s["detectors"] for s in steps]
    cens = [s["censored"] for s in steps]
    reds = [s["redundant"] for s in steps]

    ax1.plot(attempts, dets, color="#0284c7", linewidth=2.2, label="Mature V-Detectors (Lolos Seleksi)")
    ax1.plot(attempts, cens, color="#ef4444", linewidth=1.8, linestyle="--", label="Censored by Self (Apoptosis)")
    ax1.plot(attempts, reds, color="#94a3b8", linewidth=1.6, linestyle=":", label="Redundant (Sudah Tercakup)")
    ax1.set_title("Dinamika Thymus Censorship & Pembentukan V-Detector", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Jumlah Kandidat Sel Imun yang Dievaluasi")
    ax1.set_ylabel("Jumlah Kumulatif")
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.legend(fontsize=8.5)

    s_hours = [ag.hour for ag in antigens]
    s_means = [ag.mean_flow_lpm for ag in antigens]
    ax2.scatter(s_hours, s_means, c="#10b981", s=28, alpha=0.72, edgecolors="#065f46", linewidths=0.4,
                label="Self Antigens (336 Jam Normal)", zorder=3)

    D = ais.detectors
    angles = np.arctan2(D[:, 4] * 2.0 - 1.0, D[:, 5] * 2.0 - 1.0)
    d_hours = (angles / (2.0 * np.pi) * 24.0) % 24.0
    d_means = np.expm1(D[:, 0] * np.log1p(1.5))
    idx_sub = np.linspace(0, len(D) - 1, min(350, len(D)), dtype=int)
    ax2.scatter(d_hours[idx_sub], d_means[idx_sub], c="#f97316", s=16, alpha=0.35,
                label="Mature V-Detectors (Non-Self)", zorder=2)

    anom_sc = [sc for sc in scenarios if sc.expected_anomaly]
    norm_sc = [sc for sc in scenarios if not sc.expected_anomaly]
    ax2.scatter([sc.antigen.hour for sc in anom_sc], [sc.antigen.mean_flow_lpm for sc in anom_sc],
                c="#dc2626", marker="X", s=75, edgecolors="white", linewidths=0.6,
                label="Skenario Anomali Uji (18)", zorder=5)
    ax2.scatter([sc.antigen.hour for sc in norm_sc], [sc.antigen.mean_flow_lpm for sc in norm_sc],
                c="#2563eb", marker="^", s=65, edgecolors="white", linewidths=0.6,
                label="Skenario Normal Uji (12)", zorder=4)

    ax2.set_title("Proyeksi 2D Ruang Self vs V-Detectors (Jam vs Rata-rata Laju)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Jam dalam Sehari (00:00 – 23:00)")
    ax2.set_ylabel("Rata-rata Laju Air 1 Jam (L/min)")
    ax2.set_xlim(-0.8, 24.0)
    ax2.set_ylim(-0.08, 2.75)
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(alpha=0.25, linestyle="--")
    ax2.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig3_vdetector_training_and_space.png"), dpi=180)
    plt.close(fig)


def plot_ais_performance_and_ablation(bench_full: Dict[str, object], lodo_data: Dict[str, object]) -> None:
    """
    Gambar 4 (Pure AIS Evaluation):
    Kiri  : Performa Klasifikasi WaterAIS V-Detector per Kategori Skenario (20 Seed).
    Kanan : Ablasi Internal Arsitektur AIS — Context-Aware 7D Self vs Naive Per-Slot AIS (Holdout & LODO CV).
    """
    bench = bench_full["benchmark"]
    holdout = bench_full["holdout_w1_w2"]
    per_cat = bench.get("per_category_performance", [])
    vdet_m = bench["models"]["WaterAIS_VDetector"]
    ps_m = bench["models"]["PerSlot_AIS_N2"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.1), gridspec_kw={"width_ratios": [1.32, 1.0]})

    # Panel 1: Performa WaterAIS per kelas skenario
    short_names = {
        "normal_all": "Normal Variatif\n(TNR / 12)",
        "slow_leak": "Bocor Halus\n4/4 (3)",
        "night_leak": "Bocor Malam\n4/4 (3)",
        "night_open_tap": "Keran Malam\nTerbuka (3)",
        "off_schedule_burst": "Lonjakan Jam\nSepi (3)",
        "extreme_burst": "Pipa Pecah\nEkstrem (3)",
        "continuous_high": "Banjir\nKontinyu (3)",
    }
    labels = [short_names.get(c["category"], c["category"]) for c in per_cat]
    rates = [c["correct_rate_mean"] * 100.0 for c in per_cat]
    stds = [c["correct_rate_std"] * 100.0 for c in per_cat]
    bar_colors = ["#10b981" if not c["is_anomaly_class"] else "#0284c7" for c in per_cat]

    x = np.arange(len(labels))
    bars = ax1.bar(x, rates, yerr=stds, capsize=3, width=0.62, color=bar_colors, edgecolor="#0f172a", linewidth=0.5)
    for b, r in zip(bars, rates):
        ax1.text(b.get_x() + b.get_width() / 2, min(103.5, r + 2.2), f"{r:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.axhline(vdet_m["f1_score"]["mean"] * 100.0, color="#6366f1", linestyle="--", lw=1.6,
                label=f"Overall F1-Score: {vdet_m['f1_score']['mean']*100:.2f}% (Acc: {vdet_m['accuracy']['mean']*100:.2f}%, Prec: {vdet_m['precision']['mean']*100:.1f}%)")
    ax1.set_ylim(0, 114)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8, fontweight="bold")
    ax1.set_ylabel("Tingkat Ketepatan Deteksi 20-Seed (%)")
    ax1.set_title("Evaluasi Kinerja WaterAIS V-Detector per Kelas Skenario (30 Uji)", fontsize=10.5, fontweight="bold")
    ax1.grid(axis="y", alpha=0.25, linestyle="--")
    ax1.legend(fontsize=8.2, loc="lower right")

    # Panel 2: Ablasi Internal AIS (Context-Aware 7D vs Per-Slot Terisolasi)
    ab_metrics = ["Holdout W1→W2\nFPR (168 jam)", "14-Fold LODO CV\nFPR (336 jam)", "Scenario Uji\nFPR (12 Normal)"]
    ctx_vals = [
        holdout["fpr"] * 100.0,
        lodo_data["overall_fpr"] * 100.0,
        vdet_m["fpr"]["mean"] * 100.0,
    ]
    ps_vals = [
        holdout["perslot_fpr"] * 100.0,
        lodo_data["perslot_overall_fpr"] * 100.0,
        ps_m["fpr"]["mean"] * 100.0,
    ]

    x2 = np.arange(len(ab_metrics))
    w = 0.34
    b1 = ax2.bar(x2 - w / 2, ps_vals, width=w, color="#f59e0b", label=f"AIS Per-Slot Terisolasi (Waktu Latih {ps_m['train_time_sec']['mean']:.2f}s)")
    b2 = ax2.bar(x2 + w / 2, ctx_vals, width=w, color="#0284c7", label=f"WaterAIS Context 7D Terpadu (Waktu Latih {vdet_m['train_time_sec']['mean']:.2f}s)")
    for b, v in zip(b1, ps_vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.2f}%", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    for b, v in zip(b2, ctx_vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.2f}%", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#0369a1")

    ax2.annotate(
        "Penggabungan 336 Self Sirkadian memangkas\nFalse Alarm Holdout & LODO hingga >94%!",
        xy=(1.17, 0.4), xytext=(0.45, 11.5),
        arrowprops=dict(arrowstyle="->", color="#0369a1", lw=1.4),
        fontsize=8, color="#0c4a6e", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#e0f2fe", ec="#7dd3fc"),
    )
    ax2.set_ylim(0, max(ps_vals + [22]) * 1.18)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(ab_metrics, fontsize=8.5, fontweight="bold")
    ax2.set_ylabel("False Positive Rate / Alarm Palsu (%) — Lebih Rendah Lebih Baik")
    ax2.set_title("Ablasi Internal AIS: Context 7D vs Per-Slot Terisolasi", fontsize=10.5, fontweight="bold")
    ax2.grid(axis="y", alpha=0.25, linestyle="--")
    ax2.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_ais_performance_and_ablation.png"), dpi=180)
    plt.close(fig)


def plot_sensitivity(sens_data: Dict[str, object]) -> None:
    """Gambar 5: Kurva sensitivitas r_self dan max_detectors pada WaterAIS V-Detector."""
    rs = sens_data["r_self_sweep"]
    ds = sens_data["max_detectors_sweep"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))
    r_vals = [x["r_self"] for x in rs]
    ax1.plot(r_vals, [x["scenario_f1_mean"] * 100 for x in rs], marker="o", lw=2.2, color="#0284c7", label="Scenario F1-Score (%)")
    ax1.plot(r_vals, [x["scenario_fpr_mean"] * 100 for x in rs], marker="s", lw=2.0, linestyle="--", color="#f59e0b", label="Scenario FPR (%)")
    ax1.plot(r_vals, [x["w2_holdout_fpr_mean"] * 100 for x in rs], marker="^", lw=2.0, linestyle=":", color="#ef4444", label="Week-2 Holdout FPR (%)")
    ax1.axvline(0.43, color="#10b981", linestyle="-.", linewidth=1.5, label="Pilihan Optimal r_self = 0.43")
    ax1.set_title("Sensitivitas Radius Toleransi Self (r_self)", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Self Radius (r_self)")
    ax1.set_ylabel("Persentase (%)")
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.legend(fontsize=8)

    m_vals = [x["max_detectors"] for x in ds]
    ax2.plot(m_vals, [x["scenario_f1_mean"] * 100 for x in ds], marker="o", lw=2.2, color="#0284c7", label="F1-Score (%)")
    ax2.plot(m_vals, [x["scenario_recall_mean"] * 100 for x in ds], marker="s", lw=2.0, linestyle="--", color="#10b981", label="Recall (%)")
    ax2_t = ax2.twinx()
    ax2_t.plot(m_vals, [x["train_time_sec_mean"] for x in ds], marker="d", lw=1.8, linestyle=":", color="#6366f1", label="Waktu Latih (s)")
    ax2_t.set_ylabel("Waktu Latih (detik)", color="#6366f1")
    ax2.set_title("Sensitivitas Populasi V-Detector (max_detectors)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Jumlah Maksimum V-Detectors")
    ax2.set_ylabel("Skor (%)")
    ax2.set_ylim(70, 104)
    ax2.grid(alpha=0.25, linestyle="--")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_t.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig5_sensitivity_rself_and_detectors.png"), dpi=180)
    plt.close(fig)


def write_markdown_summary(bench_full: Dict[str, object], lodo: Dict[str, object]) -> None:
    bench = bench_full["benchmark"]
    holdout = bench_full["holdout_w1_w2"]
    models = bench["models"]
    per_cat = bench.get("per_category_performance", [])

    def fmt(model_key, metric):
        v = models[model_key][metric]
        if isinstance(v, dict):
            return f"{v['mean']*100:.2f}% ± {v['std']*100:.2f}%"
        return f"{v*100:.2f}%"

    lines = [
        "# Ringkasan Hasil Eksperimen Deteksi Anomali Air Berbasis AIS",
        "",
        "## 1. Validasi Data Normal Asli & Ablasi Internal AIS",
        "| Arsitektur AIS | Holdout W1→W2 (168 jam) | FPR Holdout | 14-Fold LODO CV (336 jam) | FPR LODO |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **WaterAIS V-Detector (Context 7D Terpadu)** | {holdout['false_positives']}/{holdout['test_hours']} FP | **{holdout['fpr']*100:.2f}%** | {lodo['total_fp']}/{lodo['total_hours']} FP | **{lodo['overall_fpr']*100:.2f}%** |",
        f"| **Per-Slot AIS (Slot Terisolasi N=1..2)** | {holdout.get('perslot_false_positives', '-')}/{holdout['test_hours']} FP | {holdout.get('perslot_fpr', 0)*100:.2f}% | {lodo.get('perslot_total_fp', '-')}/{lodo['total_hours']} FP | {lodo.get('perslot_overall_fpr', 0)*100:.2f}% |",
        "",
        f"Waktu latih rata-rata V-Detector Holdout: `{holdout['train_time_sec']:.3f}s`.",
        "",
        "## 2. Evaluasi Keseluruhan WaterAIS (30 Skenario Uji x 20 Seed)",
        "| Model AIS | Accuracy | Precision | Recall | F1-Score | FPR | FNR |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **WaterAIS V-Detector (Context 7D)** | {fmt('WaterAIS_VDetector', 'accuracy')} | {fmt('WaterAIS_VDetector', 'precision')} | {fmt('WaterAIS_VDetector', 'recall')} | {fmt('WaterAIS_VDetector', 'f1_score')} | {fmt('WaterAIS_VDetector', 'fpr')} | {fmt('WaterAIS_VDetector', 'fnr')} |",
        f"| **Per-Slot AIS (Ablasi 168×2)** | {fmt('PerSlot_AIS_N2', 'accuracy')} | {fmt('PerSlot_AIS_N2', 'precision')} | {fmt('PerSlot_AIS_N2', 'recall')} | {fmt('PerSlot_AIS_N2', 'f1_score')} | {fmt('PerSlot_AIS_N2', 'fpr')} | {fmt('PerSlot_AIS_N2', 'fnr')} |",
        "",
        "## 3. Performa WaterAIS per Kategori Skenario",
        "| Kategori | Jumlah Skenario | Ketepatan Rata-rata (20 Seed) |",
        "| :--- | :---: | :---: |",
    ]
    for c in per_cat:
        lines.append(
            f"| {c['label']} | {c['n_scenarios']} | {c['correct_rate_mean']*100:.2f}% ± {c['correct_rate_std']*100:.2f}% |"
        )
    lines.append("")

    with open(os.path.join(RESULTS_DIR, "benchmark_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for old_png in ("fig2_n2_zscore_vs_ais_mechanism.png", "fig4_benchmark_comparison.png"):
        p = os.path.join(FIGURES_DIR, old_png)
        if os.path.exists(p):
            os.remove(p)

    with open(os.path.join(RESULTS_DIR, "benchmark_summary.json"), "r", encoding="utf-8") as f:
        bench_full = json.load(f)
    with open(os.path.join(RESULTS_DIR, "lodo_cv.json"), "r", encoding="utf-8") as f:
        lodo = json.load(f)
    with open(os.path.join(RESULTS_DIR, "sensitivity.json"), "r", encoding="utf-8") as f:
        sens = json.load(f)

    records = load_water_dataset()
    antigens = aggregate_tumbling_1h(records)
    scenarios = build_benchmark_scenarios()

    plot_dataset_profile(antigens)
    plot_threshold_overlay_and_activeslots(antigens)
    plot_vdetector_space(antigens, scenarios)
    plot_ais_performance_and_ablation(bench_full, lodo)
    plot_sensitivity(sens)
    write_markdown_summary(bench_full, lodo)

    print(f"Semua grafik Pure-AIS & ringkasan berhasil dibuat di: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
