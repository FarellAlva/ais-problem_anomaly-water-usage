"""
analyze_results.py — Penghasil Grafik Analisis & Tabel Ringkasan Markdown

Membaca hasil dari `results/` serta dataset 2 minggu, lalu menghasilkan:
- results/figures/fig1_dataset_2weeks_profile.png
- results/figures/fig2_n2_zscore_vs_ais_mechanism.png
- results/figures/fig3_vdetector_training_and_space.png
- results/figures/fig4_benchmark_comparison.png
- results/figures/fig5_sensitivity_rself_and_detectors.png
- results/benchmark_summary.md
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aggregator import aggregate_tumbling_1h, group_weekly_two_samples
from ais_detector import WaterAISVDetector
from dataset_loader import DAYS_ORDER, load_water_dataset
from scenarios import build_benchmark_scenarios


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

    # Profil sirkadian rata-rata, max 1-jam, dan rasio slot aktif
    hours = np.arange(24)
    mean_by_h = np.mean(matrix, axis=0)
    max_by_h = np.max(matrix, axis=0)
    min_by_h = np.min(matrix, axis=0)

    ax2.fill_between(hours, min_by_h, max_by_h, color="#38bdf8", alpha=0.25, label="Rentang Min-Maks 14 Hari")
    ax2.plot(hours, mean_by_h, color="#0284c7", linewidth=2.2, marker="o", markersize=4, label="Rata-rata Lintas 14 Hari")
    ax2.axvspan(0, 5.5, color="#94a3b8", alpha=0.12, label="Zona Tidur (00-05)")
    ax2.axvspan(5.5, 8.5, color="#f59e0b", alpha=0.10, label="Puncak Pagi (06-08)")
    ax2.axvspan(16.5, 19.5, color="#10b981", alpha=0.10, label="Puncak Sore (17-19)")
    ax2.set_title("Profil Sirkadian Penggunaan Air Normal (336 Jam)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Jam Dalam Sehari")
    ax2.set_ylabel("Laju Aliran Rata-rata 1 Jam (L/min)")
    ax2.set_ylim(-0.04, 1.52)
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(alpha=0.25, linestyle="--")
    ax2.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_dataset_2weeks_profile.png"), dpi=160)
    plt.close(fig)


def plot_n2_mechanism(antigens) -> None:
    """Gambar 2: Perbandingan mekanisme N=2 pada Z-Score vs AIS Sirkadian."""
    grouped = group_weekly_two_samples(antigens)
    sunday_hours = list(range(24))
    w1_sun = [grouped[("Sunday", h)][0].mean_flow_lpm for h in sunday_hours]
    w2_sun = [grouped[("Sunday", h)][1].mean_flow_lpm for h in sunday_hours]
    z_ucl_sun = []
    for a, b in zip(w1_sun, w2_sun):
        mu = 0.5 * (a + b)
        sd = float(np.std([a, b], ddof=1))
        z_ucl_sun.append(max(mu + 2.0 * sd, 0.08))

    # Global normal band dari 14 hari pada tiap jam
    all_by_h = defaultdict(list)
    for ag in antigens:
        all_by_h[ag.hour].append(ag.mean_flow_lpm)
    global_max = [max(all_by_h[h]) for h in sunday_hours]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))
    x = np.array(sunday_hours)
    ax1.bar(x - 0.18, w1_sun, width=0.35, color="#0ea5e9", label="Minggu-1 (04 Mei)")
    ax1.bar(x + 0.18, w2_sun, width=0.35, color="#6366f1", label="Minggu-2 (11 Mei)")
    ax1.plot(x, z_ucl_sun, color="#e11d48", linestyle="--", linewidth=2.0, marker="s", markersize=4,
             label="Z-Score UCL (N=2: mu + 2*sigma)")
    ax1.plot(x, global_max, color="#10b981", linestyle="-.", linewidth=1.8,
             label="Batas Wajar Lintas 14 Hari (Konteks AIS)")
    ax1.annotate("Shift jam 08:00:\nW1=0.0, W2=1.04\n-> UCL melonjak 2.0!", xy=(8, 2.00), xytext=(1.0, 1.48),
                 arrowprops=dict(arrowstyle="->", color="#e11d48", lw=1.4), fontsize=8, color="#9f1239",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fff1f2", ec="#fda4af"))
    ax1.set_title("Dampak N=2 Sampel pada Z-Score Mingguan (Contoh Hari Minggu)", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Jam (00:00 – 23:00)")
    ax1.set_ylabel("Rata-rata Laju 1 Jam (L/min)")
    ax1.set_ylim(0, 2.32)
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.legend(fontsize=7.8, loc="upper right")

    # Panel 2: Distribusi Active Slots pada 336 jam normal vs Skenario Kebocoran
    counts = [sum(1 for ag in antigens if ag.active_slots == k) for k in range(5)]
    bars = ax2.bar(["0/4 Slot\n(Diam)", "1/4 Slot\n(Sesaat)", "2/4 Slot\n(Intermiten)", "3/4 Slot\n(Padat)", "4/4 Slot\n(Kontinyu)"],
                   counts, color=["#94a3b8", "#38bdf8", "#0284c7", "#f59e0b", "#ef4444"])
    for bar, c in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, c + 3, f"{c}\n({c/336*100:.1f}%)",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax2.annotate("0 dari 336 jam normal\nyang aktif 4/4 slot!\n(Kunci deteksi kebocoran AIS)",
                 xy=(4, 5), xytext=(2.35, 118),
                 arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5),
                 fontsize=8.5, color="#991b1b", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.35", fc="#fef2f2", ec="#fca5a5"))
    ax2.set_ylim(0, 195)
    ax2.set_title("Distribusi Jumlah Slot Aktif per 1 Jam (336 Jam Data Normal)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Jumlah Jam")
    ax2.grid(axis="y", alpha=0.25, linestyle="--")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_n2_zscore_vs_ais_mechanism.png"), dpi=160)
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

    # Peta proyeksi 2D: Jam dalam hari (0..24) vs Rata-rata laju 1 jam (L/min)
    s_hours = [ag.hour for ag in antigens]
    s_means = [ag.mean_flow_lpm for ag in antigens]
    ax2.scatter(s_hours, s_means, c="#10b981", s=28, alpha=0.72, edgecolors="#065f46", linewidths=0.4,
                label="Self Antigens (336 Jam Normal)", zorder=3)

    # Konversi kembali detektor ke (hour, mean_flow_lpm) untuk visualisasi ruang
    D = ais.detectors
    angles = np.arctan2(D[:, 4] * 2.0 - 1.0, D[:, 5] * 2.0 - 1.0)
    d_hours = (angles / (2.0 * np.pi) * 24.0) % 24.0
    d_means = np.expm1(D[:, 0] * np.log1p(1.5))
    # Tampilkan sampel 350 detektor agar plot bersih dan terbaca
    idx_sub = np.linspace(0, len(D) - 1, min(350, len(D)), dtype=int)
    ax2.scatter(d_hours[idx_sub], d_means[idx_sub], c="#f97316", s=16, alpha=0.35,
                label="Mature V-Detectors (Non-Self)", zorder=2)

    # Plot skenario anomali vs normal uji
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
    fig.savefig(os.path.join(FIGURES_DIR, "fig3_vdetector_training_and_space.png"), dpi=160)
    plt.close(fig)


def plot_benchmark_comparison(bench_data: Dict[str, object], lodo_data: Dict[str, object]) -> None:
    """Gambar 4: Perbandingan Accuracy, Precision, Recall, F1, FPR, FNR & LODO FPR antar metode."""
    models = bench_data["models"]
    names = ["Z-Score 15m\n(JUTISI N=2)", "Z-Score 1h\n(Slot N=2)", "Per-Slot AIS\n(Slot N=2)", "WaterAIS\nV-Detector"]
    keys = ["ZScore_15Min_JUTISI", "ZScore_1Hour_N2", "PerSlot_AIS_N2", "WaterAIS_VDetector"]

    def get_val(k, metric):
        v = models[k][metric]
        return v["mean"] if isinstance(v, dict) else v

    metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ["#38bdf8", "#818cf8", "#34d399", "#0284c7"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw={"width_ratios": [1.35, 1.05]})
    x = np.arange(len(names))
    w = 0.19
    for idx, (m, lbl, col) in enumerate(zip(metrics, metric_labels, colors)):
        vals = [get_val(k, m) * 100.0 for k in keys]
        bars = ax1.bar(x + (idx - 1.5) * w, vals, width=w, label=lbl, color=col)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)

    ax1.set_ylim(0, 122)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9, fontweight="bold")
    ax1.set_ylabel("Skor (%)")
    ax1.set_title("Performa Uji 30 Skenario (12 Normal + 18 Anomali, 20 Seed)", fontsize=10.5, fontweight="bold")
    ax1.grid(axis="y", alpha=0.25, linestyle="--")
    ax1.legend(fontsize=8, loc="upper center", ncol=4)

    # Panel 2: Error Rates (Scenario FPR, Scenario FNR, & LODO CV FPR)
    fprs = [get_val(k, "fpr") * 100.0 for k in keys]
    fnrs = [get_val(k, "fnr") * 100.0 for k in keys]
    # LODO FPR (ZScore 15m tidak diterapkan di LODO harian slot terisolasi karena N=1 fallback lebih rapuh dari Z1h)
    lodo_z1h = lodo_data.get("zscore_1h_overall_fpr", 0.0) * 100.0
    lodo_ps = lodo_data.get("perslot_overall_fpr", 0.0) * 100.0
    lodo_ais = lodo_data.get("overall_fpr", 0.0) * 100.0
    lodo_fprs = [np.nan, lodo_z1h, lodo_ps, lodo_ais]

    w2 = 0.25
    b_fp = ax2.bar(x - w2, fprs, width=w2, color="#f59e0b", label="Scenario FPR (Alarm Palsu)")
    b_fn = ax2.bar(x, fnrs, width=w2, color="#ef4444", label="Scenario FNR (Anomali Lolos)")
    b_lo = ax2.bar(x + w2, [0.0 if np.isnan(v) else v for v in lodo_fprs], width=w2, color="#7c3aed",
                   label="14-Fold LODO FPR (Generalisasi Data Asli)")
    for b, v in zip(b_fp, fprs):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5)
    for b, v in zip(b_fn, fnrs):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5)
    for b, v in zip(b_lo, lodo_fprs):
        lbl_t = "N/A" if np.isnan(v) else f"{v:.1f}%"
        ax2.text(b.get_x() + b.get_width() / 2, (0.0 if np.isnan(v) else v) + 0.8, lbl_t, ha="center", va="bottom", fontsize=7.5)

    ax2.set_ylim(0, max(fprs + fnrs + [v for v in lodo_fprs if not np.isnan(v)] + [20]) * 1.24)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=8.5, fontweight="bold")
    ax2.set_ylabel("Tingkat Kesalahan (%)")
    ax2.set_title("Risiko Kesalahan & Generalisasi LODO (336 Jam Asli)", fontsize=10.5, fontweight="bold")
    ax2.grid(axis="y", alpha=0.25, linestyle="--")
    ax2.legend(fontsize=7.8, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_benchmark_comparison.png"), dpi=160)
    plt.close(fig)


def plot_sensitivity(sens_data: Dict[str, object]) -> None:
    """Gambar 5: Kurva sensitivitas r_self dan max_detectors."""
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
    fig.savefig(os.path.join(FIGURES_DIR, "fig5_sensitivity_rself_and_detectors.png"), dpi=160)
    plt.close(fig)


def write_markdown_summary(bench_full: Dict[str, object], lodo: Dict[str, object]) -> None:
    bench = bench_full["benchmark"]
    holdout = bench_full["holdout_w1_w2"]
    models = bench["models"]

    def fmt(model_key, metric):
        v = models[model_key][metric]
        if isinstance(v, dict):
            return f"{v['mean']*100:.2f}% ± {v['std']*100:.2f}%"
        return f"{v*100:.2f}%"

    lines = [
        "# Ringkasan Hasil Eksperimen Deteksi Anomali Air Berbasis AIS",
        "",
        "## 1. Validasi Data Normal Asli (Holdout Minggu-1 → Minggu-2 & 14-Fold LODO CV)",
        "| Metode | Holdout W1→W2 (168 jam) | FPR Holdout | 14-Fold LODO CV (336 jam) | FPR LODO |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **WaterAIS V-Detector (Context 7D)** | {holdout['false_positives']}/{holdout['test_hours']} FP | **{holdout['fpr']*100:.2f}%** | {lodo['total_fp']}/{lodo['total_hours']} FP | **{lodo['overall_fpr']*100:.2f}%** |",
        f"| **Per-Slot AIS (Slot Terisolasi)** | {holdout.get('perslot_false_positives', '-')}/{holdout['test_hours']} FP | {holdout.get('perslot_fpr', 0)*100:.2f}% | {lodo.get('perslot_total_fp', '-')}/{lodo['total_hours']} FP | {lodo.get('perslot_overall_fpr', 0)*100:.2f}% |",
        f"| **Z-Score 1-Jam (Fallback N=1)** | {holdout.get('zscore_1h_false_positives', '-')}/{holdout['test_hours']} FP | {holdout.get('zscore_1h_fpr', 0)*100:.2f}% | {lodo.get('zscore_1h_total_fp', '-')}/{lodo['total_hours']} FP | {lodo.get('zscore_1h_overall_fpr', 0)*100:.2f}% |",
        "",
        f"Waktu latih rata-rata V-Detector Holdout: `{holdout['train_time_sec']:.3f}s`.",
        "",
        "## 2. Perbandingan Komparatif (30 Skenario Uji: 12 Normal Variatif + 18 Anomali)",
        "",
        "| Metode | Accuracy | Precision | Recall | F1-Score | FPR | FNR |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Z-Score 15m (JUTISI, N=2)** | {fmt('ZScore_15Min_JUTISI', 'accuracy')} | {fmt('ZScore_15Min_JUTISI', 'precision')} | {fmt('ZScore_15Min_JUTISI', 'recall')} | {fmt('ZScore_15Min_JUTISI', 'f1_score')} | {fmt('ZScore_15Min_JUTISI', 'fpr')} | {fmt('ZScore_15Min_JUTISI', 'fnr')} |",
        f"| **Z-Score 1h (Slot N=2)** | {fmt('ZScore_1Hour_N2', 'accuracy')} | {fmt('ZScore_1Hour_N2', 'precision')} | {fmt('ZScore_1Hour_N2', 'recall')} | {fmt('ZScore_1Hour_N2', 'f1_score')} | {fmt('ZScore_1Hour_N2', 'fpr')} | {fmt('ZScore_1Hour_N2', 'fnr')} |",
        f"| **Per-Slot AIS (Slot N=2, 20 seed)** | {fmt('PerSlot_AIS_N2', 'accuracy')} | {fmt('PerSlot_AIS_N2', 'precision')} | {fmt('PerSlot_AIS_N2', 'recall')} | {fmt('PerSlot_AIS_N2', 'f1_score')} | {fmt('PerSlot_AIS_N2', 'fpr')} | {fmt('PerSlot_AIS_N2', 'fnr')} |",
        f"| **WaterAIS V-Detector (Context 7D, 20 seed)** | {fmt('WaterAIS_VDetector', 'accuracy')} | {fmt('WaterAIS_VDetector', 'precision')} | {fmt('WaterAIS_VDetector', 'recall')} | {fmt('WaterAIS_VDetector', 'f1_score')} | {fmt('WaterAIS_VDetector', 'fpr')} | {fmt('WaterAIS_VDetector', 'fnr')} |",
        "",
    ]
    with open(os.path.join(RESULTS_DIR, "benchmark_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
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
    plot_n2_mechanism(antigens)
    plot_vdetector_space(antigens, scenarios)
    plot_benchmark_comparison(bench_full["benchmark"], lodo)
    plot_sensitivity(sens)
    write_markdown_summary(bench_full, lodo)

    print(f"Semua grafik & ringkasan berhasil dibuat di: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

