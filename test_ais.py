"""
test_ais.py — Suite Pengujian Otomatis (pytest) untuk Sistem Deteksi Anomali Air Berbasis AIS

Menguji:
1. Integritas parser dataset 2 minggu (1.344 interval, 14 hari, 2 minggu).
2. Integritas Agregator 1-Jam (336 blok tumbling, sliding window, 168 slot mingguan x 2 sampel).
3. Baseline Z-Score (perhitungan UCL, floor minimum, dan deteksi anomali).
4. WaterAISVDetector & PerSlotNSA (pelatihan, seleksi negatif tanpa false positive di Self,
   validasi holdout Minggu-1 -> Minggu-2, serta kemampuan menangkap kebocoran halus,
   keran malam, lonjakan ekstrem, dan toleransi pergeseran jam normal).
5. Integritas endpoint AppState pada run_web.py.
"""

from __future__ import annotations

import numpy as np

from aggregator import (
    aggregate_sliding_1h,
    aggregate_tumbling_1h,
    build_hourly_antigen,
    group_weekly_two_samples,
)
from ais_detector import PerSlotNSA, WaterAISVDetector
from dataset_loader import load_water_dataset, summarize_dataset
from run_web import AppState
from scenarios import build_benchmark_scenarios
from zscore_baseline import ZScore15MinBaseline, ZScore1HourBaseline


def test_dataset_integrity() -> None:
    records = load_water_dataset()
    assert len(records) == 1344
    summary = summarize_dataset(records)
    assert summary["total_days"] == 14
    assert len(summary["week1_dates"]) == 7
    assert len(summary["week2_dates"]) == 7


def test_aggregators_and_two_sample_grouping() -> None:
    records = load_water_dataset()
    tumbling = aggregate_tumbling_1h(records)
    assert len(tumbling) == 336  # 14 hari x 24 jam

    sliding = aggregate_sliding_1h(records)
    assert len(sliding) == 1344 - 3

    weekly = group_weekly_two_samples(tumbling)
    assert len(weekly) == 168  # 7 hari x 24 jam
    assert all(len(v) == 2 for v in weekly.values())

    # Pada seluruh 336 jam normal, tidak ada jam dengan 4/4 slot aktif sekaligus
    assert all(ag.active_slots < 4 for ag in tumbling)


def test_zscore_baselines() -> None:
    records = load_water_dataset()
    antigens = aggregate_tumbling_1h(records)

    z15 = ZScore15MinBaseline(k=2.0, ucl_min=0.20).fit(records)
    z1h = ZScore1HourBaseline(k=2.0, ucl_min=0.08).fit(antigens)

    # Cek nilai UCL selalu >= floor
    assert all(st.ucl_final >= 0.20 for st in z15.slot_stats.values())
    assert all(st.ucl_final >= 0.08 for st in z1h.slot_stats.values())

    # Cek deteksi lonjakan ekstrem jam malam
    burst = build_hourly_antigen([2.40, 0.0, 0.0, 0.0], hour=21, day_name="Monday")
    assert z15.predict_hour(burst).is_anomaly is True
    assert z1h.predict_hour(burst).is_anomaly is True


def test_ais_vdetector_zero_fp_and_high_recall() -> None:
    records = load_water_dataset()
    antigens = aggregate_tumbling_1h(records)

    # 1. Holdout Minggu-1 -> Minggu-2 (FPR sangat rendah & jauh mengungguli PerSlotNSA N=1)
    w1 = [ag for ag in antigens if ag.week_number == 1]
    w2 = [ag for ag in antigens if ag.week_number == 2]
    ais_w1 = WaterAISVDetector(r_self=0.43, max_detectors=1500, max_attempts=25000, seed=42).fit(w1)
    fp_w2 = sum(1 for ag in w2 if ais_w1.predict_hour(ag).is_anomaly)
    ps_w1 = PerSlotNSA(r_self=0.43, n_detectors_per_slot=60, seed=42).fit(w1)
    ps_fp_w2 = sum(1 for ag in w2 if ps_w1.predict_hour(ag).is_anomaly)
    assert fp_w2 <= 3  # <= ~1.8% FPR hanya dari 1 minggu latih
    assert fp_w2 * 5 < ps_fp_w2  # Jauh lebih tahan pergeseran daripada Per-Slot N=1 (~20% FPR)

    # 2. Latih pada 336 Self-Antigens penuh (0 FP di Self + cakupan V-Detector rapat)
    ais_full = WaterAISVDetector(r_self=0.43, max_detectors=1500, max_attempts=25000, seed=42).fit(antigens)
    assert len(ais_full.detectors) >= 500
    assert all(ais_full.predict_hour(ag).is_anomaly is False for ag in antigens)

    # 3. Uji pada 30 skenario berlabel
    scenarios = build_benchmark_scenarios()
    correct = sum(1 for sc in scenarios if ais_full.predict_hour(sc.antigen).is_anomaly == sc.expected_anomaly)
    assert correct >= 29  # Minimal 29/30 (Seed 42 mencapai 30/30)


def test_perslot_nsa_and_web_state() -> None:
    records = load_water_dataset()
    antigens = aggregate_tumbling_1h(records)
    perslot = PerSlotNSA(r_self=0.43, n_detectors_per_slot=80, seed=42).fit(antigens)
    assert len(perslot.slot_models) == 168

    leak = build_hourly_antigen([0.45, 0.45, 0.45, 0.45], hour=2, day_name="Thursday")
    assert perslot.predict_hour(leak).is_anomaly is True

    state = AppState()
    init_p = state.get_init_payload()
    assert len(init_p["days"]) == 14
    assert len(init_p["heatmap"]) == 14
    ev = state.evaluate_hour([0.35, 0.35, 0.35, 0.35], hour=10, day_name="Monday")
    assert ev["ais_vdetector"]["is_anomaly"] is True
    sim = state.simulate_day(day_number=13, inject_preset="slow_leak", inject_hour=10)
    assert len(sim["hours"]) == 24
    assert sim["hours"][10]["ais"]["is_anomaly"] is True

