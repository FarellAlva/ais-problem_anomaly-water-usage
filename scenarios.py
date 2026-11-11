"""
scenarios.py — Suite Skenario Uji Berlabel (Normal Variatif & 6 Kelas Anomali Air)

Digunakan untuk menguji performa deteksi berbasis Agregator 1 Jam secara adil antara:
- Z-Score 15-Menit (Jurnal JUTISI)
- Z-Score 1-Jam
- AIS Per-Slot (168 x 2 sampel)
- AIS Context-Aware V-Detector (Model Utama)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from aggregator import HourlyAntigen, build_hourly_antigen


@dataclass(frozen=True)
class LabeledHourScenario:
    scenario_id: str
    name: str
    category: str          # e.g. "normal_shift", "slow_leak", "night_leak", etc.
    expected_anomaly: bool
    antigen: HourlyAntigen
    description: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "category": self.category,
            "expected_anomaly": self.expected_anomaly,
            "description": self.description,
            "antigen": self.antigen.to_dict(),
        }


def build_benchmark_scenarios() -> List[LabeledHourScenario]:
    """
    Menghasilkan 30 skenario uji berlabel (12 Normal Variatif + 18 Anomali Beragam)
    berbasis karakteristik riil dataset 2 minggu.
    """
    specs: List[Tuple[str, str, str, bool, str, int, Tuple[float, float, float, float], str]] = [
        # --- 12 SKENARIO NORMAL VARIATIF (Expected: False) ---
        ("N-01", "Pergeseran Mandi Minggu 07:00", "normal_shift", False, "Sunday", 7, (0.00, 2.35, 0.00, 0.00),
         "Mandi pagi bergeser 15-30 menit dari pola Minggu historis."),
        ("N-02", "Pergeseran Mandi Minggu 08:00", "normal_shift", False, "Sunday", 8, (2.40, 0.00, 0.00, 0.00),
         "Mandi pagi pukul 08:00, wajar pada akhir pekan."),
        ("N-03", "Mandi Pagi Senin 06:00", "normal_normal", False, "Monday", 6, (2.25, 0.00, 0.85, 0.00),
         "Pola mandi pagi jam sibuk hari kerja."),
        ("N-04", "Mandi Sore Selasa 18:00", "normal_normal", False, "Tuesday", 18, (0.00, 1.95, 0.00, 0.00),
         "Mandi sore rutin pulang kerja."),
        ("N-05", "Aktivitas Siang Rabu 13:00", "normal_shift", False, "Wednesday", 13, (0.85, 0.00, 0.30, 0.00),
         "Cuci piring / aktivitas siang wajar (di hari lain normal, namun di Rabu kebetulan rendah)."),
        ("N-06", "Aktivitas Siang Kamis 15:00", "normal_normal", False, "Thursday", 15, (0.00, 0.72, 0.00, 0.40),
         "Penggunaan air sedang di sore hari."),
        ("N-07", "Malam Minggu Aktivitas 22:00", "normal_weekend", False, "Saturday", 22, (0.00, 0.15, 1.25, 0.20),
         "Aktivitas larut malam hari Sabtu sesuai pola kedua minggu data asli."),
        ("N-08", "Tetesan Kecil Dini Hari 02:00", "normal_noise", False, "Tuesday", 2, (0.00, 0.18, 0.00, 0.00),
         "Pemakaian toilet / tetesan ringan di bawah ambang fisik 0.20 L/min."),
        ("N-09", "Tetesan Kecil Malam 00:00", "normal_noise", False, "Thursday", 0, (0.14, 0.00, 0.00, 0.00),
         "Fluktuasi kecil tengah malam (setara data normal 0.05-0.24 L/min)."),
        ("N-10", "Jam Tidur Tenang 04:00", "normal_zero", False, "Friday", 4, (0.00, 0.00, 0.00, 0.00),
         "Kondisi nol aliran pada jam tidur."),
        ("N-11", "Cuci Sore Jumat 16:00", "normal_normal", False, "Friday", 16, (0.00, 0.00, 0.82, 0.00),
         "Penggunaan 1 slot wajar menjelang sore."),
        ("N-12", "Pergeseran Mandi Sore Kamis 19:00", "normal_shift", False, "Thursday", 19, (0.00, 1.80, 0.00, 0.65),
         "Mandi sore bergeser ke pukul 19:00."),

        # --- 18 SKENARIO ANOMALI NYATA (Expected: True) ---
        # 1) Kebocoran Halus Kontinyu (Slow Continuous Leak)
        ("A-01", "Kebocoran Halus Siang 10:00", "slow_leak", True, "Monday", 10, (0.35, 0.35, 0.35, 0.35),
         "Pipa/keran menetes deras terus-menerus 0.35 L/min selama 1 jam penuh (21 L/jam)."),
        ("A-02", "Kebocoran Sedang Sore 17:00", "slow_leak", True, "Wednesday", 17, (0.50, 0.50, 0.50, 0.50),
         "Kebocoran kontinyu 0.50 L/min pada jam sibuk sore (30 L/jam, kerap lolos Z-Score rata-rata!)."),
        ("A-03", "Kebocoran Halus Siang 12:00", "slow_leak", True, "Friday", 12, (0.32, 0.36, 0.30, 0.34),
         "Kebocoran konstan 4/4 slot di siang hari."),

        # 2) Kebocoran Jam Malam (Night Continuous Leak)
        ("A-04", "Kebocoran Pipa Malam 02:00", "night_leak", True, "Thursday", 2, (0.45, 0.45, 0.45, 0.45),
         "Aliran bocor terus-menerus pada jam tidur dini hari."),
        ("A-05", "Kebocoran Pipa Dini Hari 04:00", "night_leak", True, "Sunday", 4, (0.35, 0.40, 0.38, 0.36),
         "Kebocoran halus konsisten saat rumah seharusnya sepi."),
        ("A-06", "Kebocoran Awal Malam 23:00", "night_leak", True, "Tuesday", 23, (0.55, 0.52, 0.50, 0.54),
         "Aliran kontinyu 4/4 slot menjelang tengah malam hari kerja."),

        # 3) Keran Lupa Ditutup Malam Hari (Night Tap Left Open)
        ("A-07", "Keran Lupa Ditutup 03:00", "night_open_tap", True, "Tuesday", 3, (1.20, 1.20, 0.00, 0.00),
         "Keran terbuka 30 menit penuh di dini hari (36 liter terbuang)."),
        ("A-08", "Aliran Malam Sedang 01:00", "night_open_tap", True, "Friday", 1, (0.80, 0.60, 0.00, 0.00),
         "Penggunaan tidak wajar selama 30 menit pada pukul 01:00."),
        ("A-09", "Aliran Malam Berulang 05:00", "night_open_tap", True, "Wednesday", 5, (0.95, 1.10, 0.85, 0.00),
         "Aliran tinggi 3 slot berturut-turut sebelum pukul 06:00."),

        # 4) Aktivitas Mendadak di Jam Sepi (Off-Schedule Burst)
        ("A-10", "Lonjakan Mendadak Jam 21:00", "off_schedule_burst", True, "Monday", 21, (2.40, 0.00, 0.00, 0.00),
         "Debit tinggi 2.40 L/min di jam sepi malam hari kerja (normal maks 0.27 L/min)."),
        ("A-11", "Lonjakan Tengah Malam 00:00", "off_schedule_burst", True, "Wednesday", 0, (1.85, 0.00, 0.00, 0.00),
         "Debit shower tinggi tepat tengah malam hari kerja."),
        ("A-12", "Lonjakan Jam 20:00 Hari Kerja", "off_schedule_burst", True, "Tuesday", 20, (2.10, 1.50, 0.00, 0.00),
         "Pemakaian berat berkepanjangan pada jam istirahat malam."),

        # 5) Lonjakan Debit Ekstrem (Extreme Pipe Burst)
        ("A-13", "Pipa Pecah Jam Sibuk Pagi 08:00", "extreme_burst", True, "Tuesday", 8, (5.20, 4.80, 0.00, 0.00),
         "Debit air melampaui kapasitas keran normal (> 2.6 L/min)."),
        ("A-14", "Lonjakan Ekstrem Siang 11:00", "extreme_burst", True, "Monday", 11, (4.40, 3.90, 0.00, 0.00),
         "Lonjakan drastis 4+ L/min pada jam aktivitas siang."),
        ("A-15", "Lonjakan Ekstrem Sore 18:00", "extreme_burst", True, "Saturday", 18, (5.50, 0.00, 0.00, 0.00),
         "Semburan sangat tinggi pada sore akhir pekan."),

        # 6) Banjir / Aliran Besar Kontinyu (Continuous High Flow)
        ("A-16", "Pipa Bocor Besar Siang 14:00", "continuous_high", True, "Thursday", 14, (1.80, 1.80, 1.80, 1.80),
         "Aliran deras terus-menerus selama 1 jam penuh (108 L/jam)."),
        ("A-17", "Aliran Tinggi Kontinyu Pagi 07:00", "continuous_high", True, "Friday", 7, (2.20, 2.30, 2.15, 2.25),
         "Pipa terbuka penuh tanpa henti selama 4 slot di jam pagi (133.5 L/jam)."),
        ("A-18", "Aliran Sedang-Tinggi Kontinyu 16:00", "continuous_high", True, "Sunday", 16, (1.10, 1.15, 1.05, 1.20),
         "Kebocoran sedang-tinggi 4/4 slot penuh di sore hari (67.5 L/jam)."),
    ]

    scenarios: List[LabeledHourScenario] = []
    for idx, (sid, name, cat, exp, day_name, hour, flows, desc) in enumerate(specs):
        ag = build_hourly_antigen(
            slot_flows=flows,
            hour=hour,
            day_name=day_name,
            date_str=f"Scenario-{sid}",
            day_number=14 + (idx % 7),
            week_number=3,
            hour_index=idx,
        )
        scenarios.append(
            LabeledHourScenario(
                scenario_id=sid,
                name=name,
                category=cat,
                expected_anomaly=exp,
                antigen=ag,
                description=desc,
            )
        )
    return scenarios

