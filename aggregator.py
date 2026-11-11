"""
aggregator.py — Agregator Penggunaan Air 1 Jam & Ekstraktor Antigen (Self/Non-Self)

Menyediakan dua mode agregasi 1 jam (4 interval x 15 menit):
1. Tumbling 1-Hour Aggregator:
   Membagi 1.344 interval menjadi 336 blok jam penuh (14 hari x 24 jam).
   Pada tingkat mingguan (7 hari x 24 jam = 168 slot mingguan), tiap slot tepat
   memiliki 2 sampel data (Minggu-1 dan Minggu-2).
2. Sliding 1-Hour Aggregator (Real-Time IoT Loop):
   Pada setiap tick 15 menit baru, mengagregasi 4 interval terakhir (1 jam berjalan)
   untuk deteksi anomali dini tanpa menunggu pergantian jam bulat.

Ekstraksi Vektor Antigen (7 Dimensi):
  x = [log1p(mean_flow), log1p(max_flow), active_ratio, log1p(min_flow),
       circadian_sin, circadian_cos, is_weekend]
yang dinormalisasi dengan skala domain tetap agar jarak Euclidean berbobot merepresentasikan
kemiripan perilaku penggunaan air secara adil antara jam sibuk dan jam tidur.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from dataset_loader import FlowIntervalRecord, DAYS_ORDER


FEATURE_NAMES: Tuple[str, ...] = (
    "mean_flow_log",
    "max_flow_log",
    "active_ratio",
    "min_flow_log",
    "circadian_sin",
    "circadian_cos",
    "is_weekend",
)

FEATURE_LABELS_ID: Tuple[str, ...] = (
    "Rata-rata Laju 1 Jam",
    "Puncak 15-Menit",
    "Rasio Slot Aktif",
    "Aliran Minimum Kontinyu",
    "Konteks Jam (Sin)",
    "Konteks Jam (Cos)",
    "Akhir Pekan",
)

# Skala normalisasi berbasis domain fisik rumah tangga
FEATURE_SCALES = np.array(
    [
        math.log1p(1.5),   # ~0.9163
        math.log1p(3.0),   # ~1.3863
        1.0,               # [0, 1]
        math.log1p(1.0),   # ~0.6931
        1.0,               # [0, 1]
        1.0,               # [0, 1]
        1.0,               # {0, 1}
    ],
    dtype=float,
)

# Bobot kepentingan imunologis tiap dimensi pada ruang Antigen
FEATURE_WEIGHTS = np.array(
    [
        1.40,  # mean_flow_log: volume total 1 jam
        1.30,  # max_flow_log: lonjakan sesaat
        0.85,  # active_ratio: durasi keterlibatan slot dalam 1 jam
        2.20,  # min_flow_log: penanda kuat kebocoran kontinyu (4/4 slot > 0)
        2.20,  # circadian_sin: konteks waktu 24 jam
        2.20,  # circadian_cos: konteks waktu 24 jam
        0.25,  # is_weekend: perbedaan halus hari kerja vs akhir pekan
    ],
    dtype=float,
)


@dataclass(frozen=True)
class HourlyAntigen:
    """Representasi hasil agregasi 1 jam sebagai 1 Antigen dalam AIS."""

    hour_index: int              # 0..335 untuk tumbling, atau tick index untuk sliding
    day_name: str
    day_index: int               # 0..6
    date_str: str
    day_number: int              # 0..13
    week_number: int             # 1 atau 2
    hour: int                    # 0..23
    weekly_hour_slot: int        # 0..167 (day_index * 24 + hour)
    slot_flows: Tuple[float, float, float, float]  # 4 nilai 15-menit [q1, q2, q3, q4]
    mean_flow_lpm: float
    max_flow_lpm: float
    min_flow_lpm: float
    std_flow_lpm: float
    active_slots: int            # 0..4
    active_ratio: float          # 0.0, 0.25, 0.50, 0.75, 1.0
    total_liters: float          # sum(slot_flows) * 15 menit
    is_weekend: bool
    features_norm: np.ndarray    # vektor 7D ternormalisasi

    @property
    def label(self) -> str:
        return f"{self.day_name}, {self.date_str} {self.hour:02d}:00-{(self.hour + 1) % 24:02d}:00"

    def to_dict(self) -> Dict[str, object]:
        return {
            "hour_index": self.hour_index,
            "day_name": self.day_name,
            "day_index": self.day_index,
            "date_str": self.date_str,
            "day_number": self.day_number,
            "week_number": self.week_number,
            "hour": self.hour,
            "weekly_hour_slot": self.weekly_hour_slot,
            "slot_flows": [round(float(x), 3) for x in self.slot_flows],
            "mean_flow_lpm": round(float(self.mean_flow_lpm), 4),
            "max_flow_lpm": round(float(self.max_flow_lpm), 4),
            "min_flow_lpm": round(float(self.min_flow_lpm), 4),
            "std_flow_lpm": round(float(self.std_flow_lpm), 4),
            "active_slots": int(self.active_slots),
            "active_ratio": float(self.active_ratio),
            "total_liters": round(float(self.total_liters), 2),
            "is_weekend": bool(self.is_weekend),
            "label": self.label,
        }


def extract_antigen_vector(
    slot_flows: Sequence[float],
    hour: int,
    is_weekend: bool,
) -> np.ndarray:
    """
    Mengubah 4 laju aliran 15-menit + konteks waktu menjadi vektor fitur 7D ternormalisasi.

    Menggunakan transformasi log1p pada laju aliran agar perbedaan kecil di jam malam
    (mis. 0.00 -> 0.45 L/min) tetap proporsional dibanding variasi wajar di jam sibuk pagi.
    """
    arr = np.asarray(slot_flows, dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"Agregator 1-jam wajib berisi 4 slot 15-menit, diterima {arr.shape}.")

    mean_f = float(np.mean(arr))
    max_f = float(np.max(arr))
    min_f = float(np.min(arr))
    active_ratio = float(np.sum(arr > 0.0) / 4.0)

    angle = 2.0 * math.pi * (hour % 24) / 24.0
    circ_sin = 0.5 * (math.sin(angle) + 1.0)
    circ_cos = 0.5 * (math.cos(angle) + 1.0)
    wknd = 1.0 if is_weekend else 0.0

    raw_vec = np.array(
        [
            math.log1p(max(0.0, mean_f)),
            math.log1p(max(0.0, max_f)),
            active_ratio,
            math.log1p(max(0.0, min_f)),
            circ_sin,
            circ_cos,
            wknd,
        ],
        dtype=float,
    )
    return raw_vec / FEATURE_SCALES


def build_hourly_antigen(
    slot_flows: Sequence[float],
    hour: int,
    day_name: str = "Monday",
    date_str: str = "Simulated",
    day_number: int = 0,
    week_number: int = 1,
    hour_index: int = 0,
) -> HourlyAntigen:
    """Membangun objek HourlyAntigen lengkap dari 4 nilai 15-menit dan metadata waktu."""
    arr = np.asarray(slot_flows, dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"Harus berisi tepat 4 nilai 15-menit, diterima {len(slot_flows)}.")
    if day_name not in DAYS_ORDER:
        raise ValueError(f"Nama hari tidak valid: {day_name}")

    day_idx = DAYS_ORDER.index(day_name)
    is_wknd = day_name in ("Saturday", "Sunday")
    feat_norm = extract_antigen_vector(arr, hour, is_wknd)

    return HourlyAntigen(
        hour_index=hour_index,
        day_name=day_name,
        day_index=day_idx,
        date_str=date_str,
        day_number=day_number,
        week_number=week_number,
        hour=hour % 24,
        weekly_hour_slot=day_idx * 24 + (hour % 24),
        slot_flows=(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])),
        mean_flow_lpm=float(np.mean(arr)),
        max_flow_lpm=float(np.max(arr)),
        min_flow_lpm=float(np.min(arr)),
        std_flow_lpm=float(np.std(arr, ddof=0)),
        active_slots=int(np.sum(arr > 0.0)),
        active_ratio=float(np.sum(arr > 0.0) / 4.0),
        total_liters=float(np.sum(arr) * 15.0),
        is_weekend=is_wknd,
        features_norm=feat_norm,
    )


def aggregate_tumbling_1h(records: List[FlowIntervalRecord]) -> List[HourlyAntigen]:
    """
    Mengagregasi rekaman 15-menit menjadi blok jam tetap (Tumbling 1-Hour Window).
    Dari 1.344 record (2 minggu) dihasilkan tepat 336 HourlyAntigen (14 hari x 24 jam).
    """
    buckets: Dict[Tuple[int, str, str, int, int], List[FlowIntervalRecord]] = defaultdict(list)
    ordered_keys: List[Tuple[int, str, str, int, int]] = []

    for r in records:
        key = (r.day_number, r.date_str, r.day_name, r.week_number, r.hour)
        if key not in buckets:
            ordered_keys.append(key)
        buckets[key].append(r)

    ordered_keys.sort(key=lambda k: (k[0], k[4]))

    antigens: List[HourlyAntigen] = []
    for idx, key in enumerate(ordered_keys):
        day_num, date_str, day_name, week_num, hour = key
        items = sorted(buckets[key], key=lambda r: r.minute)
        if len(items) != 4:
            raise ValueError(
                f"Bucket {date_str} jam {hour:02d}:00 memiliki {len(items)} interval (harus 4)."
            )
        flows = [it.flow_rate_lpm for it in items]
        antigens.append(
            build_hourly_antigen(
                slot_flows=flows,
                hour=hour,
                day_name=day_name,
                date_str=date_str,
                day_number=day_num,
                week_number=week_num,
                hour_index=idx,
            )
        )
    return antigens


def aggregate_sliding_1h(records: List[FlowIntervalRecord]) -> List[HourlyAntigen]:
    """
    Agregator 1-jam bergerak (Sliding Window setiap 15 menit):
    Mulai dari indeks ke-3, setiap interval baru mengambil 4 interval terakhir
    [t-3, t-2, t-1, t] sehingga sistem IoT dapat mengevaluasi jendela 1 jam
    setiap 15 menit secara real-time.
    """
    if len(records) < 4:
        return []

    antigens: List[HourlyAntigen] = []
    for end_idx in range(3, len(records)):
        win = records[end_idx - 3 : end_idx + 1]
        cur = win[-1]
        flows = [r.flow_rate_lpm for r in win]
        antigens.append(
            build_hourly_antigen(
                slot_flows=flows,
                hour=cur.hour,
                day_name=cur.day_name,
                date_str=cur.date_str,
                day_number=cur.day_number,
                week_number=cur.week_number,
                hour_index=end_idx,
            )
        )
    return antigens


def group_weekly_two_samples(antigens: List[HourlyAntigen]) -> Dict[Tuple[str, int], List[HourlyAntigen]]:
    """
    Mengelompokkan agregasi 1-jam ke dalam 168 slot mingguan (7 hari x 24 jam).
    Pada data 2 minggu, setiap kunci (day_name, hour) berisi 2 sampel:
    [sampel Minggu-1, sampel Minggu-2].
    """
    grouped: Dict[Tuple[str, int], List[HourlyAntigen]] = defaultdict(list)
    for ag in antigens:
        grouped[(ag.day_name, ag.hour)].append(ag)
    return dict(grouped)


def weighted_antigen_distance(a: np.ndarray, b_matrix: np.ndarray) -> np.ndarray:
    """Jarak Euclidean berbobot antara satu antigen `a` (D,) dan matriks `b_matrix` (N, D)."""
    diff = (b_matrix - a) * FEATURE_WEIGHTS
    return np.sqrt(np.sum(diff * diff, axis=1))


def weighted_pairwise_distance_matrix(a_mat: np.ndarray, b_mat: np.ndarray) -> np.ndarray:
    """Matriks jarak Euclidean berbobot antara (M, D) dan (N, D)."""
    diff = (a_mat[:, None, :] - b_mat[None, :, :]) * FEATURE_WEIGHTS[None, None, :]
    return np.sqrt(np.sum(diff * diff, axis=2))
