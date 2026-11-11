"""
run_experiments.py — Evaluasi Empiris & Ablasi Internal Artificial Immune System (WaterAIS)

Menjalankan rangkaian eksperimen lengkap yang berfokus pada model AIS:
1. Validasi Holdout (Train Minggu-1 -> Uji Minggu-2 normal 168 jam)
2. Validasi Silang 14-Fold Leave-One-Day-Out (uji seluruh 336 jam normal asli)
3. Evaluasi Multi-Seed (20 seed) pada 30 Skenario Uji Berlabel (12 Normal + 18 Anomali):
   - WaterAISVDetector (Model Utama: Context-Aware 7D V-Detector + Boundary Probing)
   - PerSlotNSA (Ablasi Internal: AIS Per-Slot 168 x 2 sampel terisolasi)
   - Pemecahan kinerja per kategori skenario (Normal Shift/Noise/Weekend & 6 Kelas Anomali)
4. Analisis Sensitivitas Parameter AIS:
   - Variasi radius toleransi Self (r_self)
   - Variasi jumlah detektor dewasa (max_detectors)

Hasil disimpan ke results/benchmark_summary.json, results/lodo_cv.json, dan results/sensitivity.json.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

from aggregator import HourlyAntigen, aggregate_tumbling_1h
from ais_detector import PerSlotNSA, WaterAISVDetector
from dataset_loader import load_water_dataset, summarize_dataset
from scenarios import LabeledHourScenario, build_benchmark_scenarios


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def compute_classification_metrics(
    y_true: Sequence[bool],
    y_pred: Sequence[bool],
) -> Dict[str, float]:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if (not yt) and (not yp))
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if (not yt) and yp)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and (not yp))

    total = len(y_true)
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
    }


def run_holdout_week1_to_week2(
    antigens: List[HourlyAntigen],
    seed: int = 42,
) -> Dict[str, object]:
    """
    Melatih pada Minggu-1 (168 jam) dan menguji False Positive Rate pada Minggu-2 (168 jam)
    untuk WaterAISVDetector (Context 7D) vs PerSlotNSA (Slot Terisolasi N=1).
    """
    w1 = [ag for ag in antigens if ag.week_number == 1]
    w2 = [ag for ag in antigens if ag.week_number == 2]

    ais = WaterAISVDetector(r_self=0.43, max_detectors=1500, max_attempts=25000, seed=seed).fit(w1)
    preds = [ais.predict_hour(ag) for ag in w2]
    fp_list = [ag.label for ag, pr in zip(w2, preds) if pr.is_anomaly]

    ps = PerSlotNSA(r_self=0.43, n_detectors_per_slot=60, seed=seed).fit(w1)
    ps_fp = sum(1 for ag in w2 if ps.predict_hour(ag).is_anomaly)

    return {
        "train_hours": len(w1),
        "test_hours": len(w2),
        "false_positives": len(fp_list),
        "fpr": round(len(fp_list) / len(w2), 4),
        "train_time_sec": round(ais.trace.train_time_sec, 4) if ais.trace else 0.0,
        "fp_examples": fp_list[:10],
        "perslot_false_positives": ps_fp,
        "perslot_fpr": round(ps_fp / len(w2), 4),
    }


def run_leave_one_day_out(
    antigens: List[HourlyAntigen],
    seed: int = 42,
) -> Dict[str, object]:
    """
    14-Fold Leave-One-Day-Out Cross-Validation di seluruh 336 jam normal asli.
    Membandingkan ablasi internal: WaterAISVDetector (Self 7D terpadu) vs PerSlotNSA (slot terisolasi).
    """
    per_day = []
    total_fp = 0
    ps_total_fp = 0
    total_hours = 0

    for day_num in range(14):
        train_set = [ag for ag in antigens if ag.day_number != day_num]
        test_set = [ag for ag in antigens if ag.day_number == day_num]
        ais = WaterAISVDetector(r_self=0.43, max_detectors=1500, max_attempts=25000, seed=seed + day_num).fit(train_set)
        ps = PerSlotNSA(r_self=0.43, n_detectors_per_slot=60, seed=seed + day_num).fit(train_set)

        fps = []
        ps_fp_day = 0
        for ag in test_set:
            res = ais.predict_hour(ag)
            if res.is_anomaly:
                fps.append({"label": ag.label, "flows": ag.slot_flows, "margin": round(res.max_detector_margin, 4)})
            if ps.predict_hour(ag).is_anomaly:
                ps_fp_day += 1

        total_fp += len(fps)
        ps_total_fp += ps_fp_day
        total_hours += len(test_set)
        per_day.append(
            {
                "day_number": day_num,
                "date": test_set[0].date_str,
                "day_name": test_set[0].day_name,
                "hours": len(test_set),
                "fp": len(fps),
                "fpr": round(len(fps) / len(test_set), 4),
                "perslot_fp": ps_fp_day,
                "fp_details": fps,
            }
        )

    return {
        "folds": 14,
        "total_hours": total_hours,
        "total_fp": total_fp,
        "overall_fpr": round(total_fp / total_hours, 4),
        "perslot_total_fp": ps_total_fp,
        "perslot_overall_fpr": round(ps_total_fp / total_hours, 4),
        "per_day": per_day,
    }


def run_ais_evaluation(
    antigens: List[HourlyAntigen],
    scenarios: List[LabeledHourScenario],
    n_seeds: int = 20,
) -> Dict[str, object]:
    """Mengevaluasi WaterAISVDetector (Model Utama) & PerSlotNSA (Ablasi Internal) pada 20 seed."""
    y_true = [sc.expected_anomaly for sc in scenarios]

    # 1. Per-Slot NSA (Ablasi Internal AIS, 20 Seeds)
    perslot_runs = []
    perslot_seed0_preds = None
    for s in range(n_seeds):
        t0 = time.perf_counter()
        p_ais = PerSlotNSA(r_self=0.43, n_detectors_per_slot=60, seed=100 + s).fit(antigens)
        tr_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        preds = [p_ais.predict_hour(sc.antigen) for sc in scenarios]
        inf_ms = (time.perf_counter() - t1) * 1000.0 / len(scenarios)
        if s == 0:
            perslot_seed0_preds = preds
        m = compute_classification_metrics(y_true, [p.is_anomaly for p in preds])
        m["train_time_sec"] = round(tr_s, 4)
        m["inf_ms_per_hour"] = round(inf_ms, 4)
        perslot_runs.append(m)

    # 2. WaterAISVDetector (Model Utama 7D, 20 Seeds)
    vdet_runs = []
    vdet_seed0_preds = None
    vdet_seed0_trace = None
    cat_hits: Dict[str, List[float]] = defaultdict(list)

    for s in range(n_seeds):
        ais = WaterAISVDetector(r_self=0.43, max_detectors=1500, max_attempts=25000, seed=42 + s).fit(antigens)
        t1 = time.perf_counter()
        preds = [ais.predict_hour(sc.antigen) for sc in scenarios]
        inf_ms = (time.perf_counter() - t1) * 1000.0 / len(scenarios)
        if s == 0:
            vdet_seed0_preds = preds
            vdet_seed0_trace = ais.trace.to_dict() if ais.trace else {}
        m = compute_classification_metrics(y_true, [p.is_anomaly for p in preds])
        m["train_time_sec"] = round(ais.trace.train_time_sec, 4) if ais.trace else 0.0
        m["inf_ms_per_hour"] = round(inf_ms, 4)
        vdet_runs.append(m)

        # Catat akurasi/kelulusan per kategori skenario
        by_cat: Dict[str, List[bool]] = defaultdict(list)
        for sc, pr in zip(scenarios, preds):
            group = "normal_all" if not sc.expected_anomaly else sc.category
            by_cat[group].append(pr.is_anomaly == sc.expected_anomaly)
        for cat_k, ok_list in by_cat.items():
            cat_hits[cat_k].append(sum(ok_list) / len(ok_list))

    def summarize_runs(runs: List[Dict[str, float]]) -> Dict[str, object]:
        keys = ["accuracy", "precision", "recall", "f1_score", "fpr", "fnr", "train_time_sec", "inf_ms_per_hour"]
        out: Dict[str, object] = {"n_seeds": len(runs)}
        for k in keys:
            vals = [r[k] for r in runs]
            out[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }
        return out

    category_labels = {
        "normal_all": ("Pola Normal Variatif (12 Skenario)", False, 12),
        "slow_leak": ("Kebocoran Halus Kontinyu (3)", True, 3),
        "night_leak": ("Kebocoran Jam Malam (3)", True, 3),
        "night_open_tap": ("Keran Lupa Ditutup Malam (3)", True, 3),
        "off_schedule_burst": ("Lonjakan Jam Sepi (3)", True, 3),
        "extreme_burst": ("Pipa Pecah / Ekstrem (3)", True, 3),
        "continuous_high": ("Banjir Aliran Kontinyu (3)", True, 3),
    }
    per_category_summary = []
    for key, (label, is_anom, cnt) in category_labels.items():
        vals = cat_hits.get(key, [0.0])
        per_category_summary.append(
            {
                "category": key,
                "label": label,
                "is_anomaly_class": is_anom,
                "n_scenarios": cnt,
                "correct_rate_mean": round(float(np.mean(vals)), 4),
                "correct_rate_std": round(float(np.std(vals)), 4),
            }
        )

    scenario_details = []
    for idx, sc in enumerate(scenarios):
        pr = vdet_seed0_preds[idx] if vdet_seed0_preds else None
        ps_pr = perslot_seed0_preds[idx] if perslot_seed0_preds else None
        scenario_details.append(
            {
                "scenario_id": sc.scenario_id,
                "name": sc.name,
                "category": sc.category,
                "expected_anomaly": sc.expected_anomaly,
                "day_name": sc.antigen.day_name,
                "hour": sc.antigen.hour,
                "slot_flows": list(sc.antigen.slot_flows),
                "active_slots": sc.antigen.active_slots,
                "mean_flow_lpm": round(sc.antigen.mean_flow_lpm, 4),
                "perslot_ais": ps_pr.is_anomaly if ps_pr else False,
                "vdetector_ais": pr.is_anomaly if pr else False,
                "vdetector_min_dist": round(pr.min_dist_to_self, 4) if pr else 0.0,
                "vdetector_margin": round(pr.max_detector_margin, 4) if pr else 0.0,
                "vdetector_triggered": pr.triggered_detectors if pr else 0,
                "vdetector_diagnosis": pr.anomaly_category if pr else "",
                "vdetector_reason": pr.reason if pr else "",
            }
        )

    return {
        "n_scenarios": len(scenarios),
        "n_normal_scenarios": sum(1 for y in y_true if not y),
        "n_anomaly_scenarios": sum(1 for y in y_true if y),
        "models": {
            "PerSlot_AIS_N2": summarize_runs(perslot_runs),
            "WaterAIS_VDetector": summarize_runs(vdet_runs),
        },
        "per_category_performance": per_category_summary,
        "vdetector_seed0_trace": vdet_seed0_trace,
        "scenario_comparison": scenario_details,
    }


def run_sensitivity_study(
    antigens: List[HourlyAntigen],
    scenarios: List[LabeledHourScenario],
) -> Dict[str, object]:
    """Menguji sensitivitas parameter r_self dan max_detectors pada WaterAISVDetector."""
    y_true = [sc.expected_anomaly for sc in scenarios]
    w1 = [ag for ag in antigens if ag.week_number == 1]
    w2 = [ag for ag in antigens if ag.week_number == 2]

    r_self_values = [0.32, 0.36, 0.40, 0.43, 0.47, 0.52]
    r_self_results = []
    for r in r_self_values:
        f1s, fprs, holdout_fprs = [], [], []
        for s in range(5):
            ais = WaterAISVDetector(r_self=r, max_detectors=1500, max_attempts=25000, seed=42 + s).fit(antigens)
            m = compute_classification_metrics(y_true, [ais.predict_hour(sc.antigen).is_anomaly for sc in scenarios])
            f1s.append(m["f1_score"])
            fprs.append(m["fpr"])

            ais_w1 = WaterAISVDetector(r_self=r, max_detectors=1500, max_attempts=25000, seed=42 + s).fit(w1)
            w2_fp = sum(1 for ag in w2 if ais_w1.predict_hour(ag).is_anomaly)
            holdout_fprs.append(w2_fp / len(w2))

        r_self_results.append(
            {
                "r_self": r,
                "scenario_f1_mean": round(float(np.mean(f1s)), 4),
                "scenario_fpr_mean": round(float(np.mean(fprs)), 4),
                "w2_holdout_fpr_mean": round(float(np.mean(holdout_fprs)), 4),
            }
        )

    det_counts = [300, 600, 1000, 1500, 2200]
    det_results = []
    for md in det_counts:
        f1s, recs, times = [], [], []
        for s in range(5):
            ais = WaterAISVDetector(r_self=0.43, max_detectors=md, max_attempts=30000, seed=42 + s).fit(antigens)
            m = compute_classification_metrics(y_true, [ais.predict_hour(sc.antigen).is_anomaly for sc in scenarios])
            f1s.append(m["f1_score"])
            recs.append(m["recall"])
            times.append(ais.trace.train_time_sec if ais.trace else 0.0)
        det_results.append(
            {
                "max_detectors": md,
                "scenario_f1_mean": round(float(np.mean(f1s)), 4),
                "scenario_recall_mean": round(float(np.mean(recs)), 4),
                "train_time_sec_mean": round(float(np.mean(times)), 4),
            }
        )

    return {
        "r_self_sweep": r_self_results,
        "max_detectors_sweep": det_results,
    }


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    records = load_water_dataset()
    antigens = aggregate_tumbling_1h(records)
    scenarios = build_benchmark_scenarios()
    summary = summarize_dataset(records)

    print("=== Ringkasan Dataset ===")
    print(json.dumps(summary, indent=2))

    print("\n=== 1. Holdout Week 1 -> Week 2 Normal ===")
    holdout = run_holdout_week1_to_week2(antigens)
    print(json.dumps(holdout, indent=2))

    print("\n=== 2. 14-Fold Leave-One-Day-Out Cross-Validation ===")
    lodo = run_leave_one_day_out(antigens)
    print(f"Total hours={lodo['total_hours']}, Total FP={lodo['total_fp']}, Overall FPR={lodo['overall_fpr']}")

    print("\n=== 3. Evaluasi Multi-Seed WaterAIS & Ablasi Internal (20 Seeds) ===")
    bench = run_ais_evaluation(antigens, scenarios, n_seeds=20)
    print(json.dumps(bench["models"], indent=2))

    print("\n=== 4. Studi Sensitivitas r_self & max_detectors ===")
    sens = run_sensitivity_study(antigens, scenarios)
    print(json.dumps(sens, indent=2))

    full_output = {
        "dataset_summary": summary,
        "holdout_w1_w2": holdout,
        "benchmark": bench,
    }
    with open(os.path.join(RESULTS_DIR, "benchmark_summary.json"), "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "lodo_cv.json"), "w", encoding="utf-8") as f:
        json.dump(lodo, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "sensitivity.json"), "w", encoding="utf-8") as f:
        json.dump(sens, f, indent=2)

    print(f"\nSemua hasil disimpan di: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
