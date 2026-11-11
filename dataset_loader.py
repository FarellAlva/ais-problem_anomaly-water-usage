"""
dataset_loader.py — Parser Dataset Aliran Air 2 Minggu (Model-dataset-2 weeks.txt)

Membaca 1.344 interval pengukuran 15 menit (04 Mei 2025 – 17 Mei 2025) untuk
studi IoT Deteksi Anomali Penggunaan Air, beserta struktur indeks hari,
minggu (Minggu-1 vs Minggu-2), jam, dan slot waktu mingguan.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple


DAYS_ORDER: Tuple[str, ...] = (
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
)

DAYS_ID_MAP: Dict[str, str] = {
    "Sunday": "Minggu",
    "Monday": "Senin",
    "Tuesday": "Selasa",
    "Wednesday": "Rabu",
    "Thursday": "Kamis",
    "Friday": "Jumat",
    "Saturday": "Sabtu",
}

DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Model-dataset-2 weeks.txt",
)

_LINE_RE = re.compile(
    r"^(\w+),\s+(\d+\s+\w+\s+\d+)\s+(\d{2}):(\d{2})\s*-\s*(\d{2}):(\d{2})$"
)


@dataclass(frozen=True)
class FlowIntervalRecord:
    """Satu baris catatan sensor aliran air pada interval 15 menit."""

    index: int
    day_name: str
    day_index: int          # 0=Sunday .. 6=Saturday
    date_str: str           # e.g., "04 May 2025"
    day_number: int         # 0..13 across the 2 weeks
    week_number: int        # 1 (days 0..6) or 2 (days 7..13)
    hour: int               # 0..23
    minute: int             # 0, 15, 30, 45
    slot_15m_in_day: int    # 0..95
    weekly_slot_15m: int    # 0..671
    flow_rate_lpm: float    # Average Flow Rate (L/min)
    raw_label: str

    @property
    def is_weekend(self) -> bool:
        return self.day_name in ("Saturday", "Sunday")

    @property
    def day_name_id(self) -> str:
        return DAYS_ID_MAP.get(self.day_name, self.day_name)

    @property
    def volume_liters(self) -> float:
        """Perkiraan volume air pada interval 15 menit (L/min * 15 menit)."""
        return self.flow_rate_lpm * 15.0


def load_water_dataset(filepath: str = DEFAULT_DATASET_PATH) -> List[FlowIntervalRecord]:
    """
    Memuat dataset Model-dataset-2 weeks.txt menjadi daftar terurut FlowIntervalRecord.

    Validasi:
    - Memastikan setiap baris memiliki format timestamp 'Day, DD Mon YYYY HH:MM - HH:MM'
      dan nilai float laju aliran (L/min).
    - Menetapkan indeks hari 0..13 dan minggu 1 atau 2 secara otomatis.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file tidak ditemukan: {filepath}")

    records: List[FlowIntervalRecord] = []
    seen_dates: Dict[str, int] = {}

    with open(filepath, "r", encoding="utf-8") as fh:
        header = fh.readline()
        if not header:
            raise ValueError("File dataset kosong.")

        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Format baris tidak valid (harus 2 kolom tab): {stripped!r}")

            ts_str, val_str = parts[0].strip(), parts[1].strip()
            match = _LINE_RE.match(ts_str)
            if not match:
                raise ValueError(f"Format waktu tidak dikenali: {ts_str!r}")

            day_name, date_str, sh_s, sm_s, _eh_s, _em_s = match.groups()
            if day_name not in DAYS_ORDER:
                raise ValueError(f"Nama hari tidak dikenali: {day_name!r}")

            if date_str not in seen_dates:
                seen_dates[date_str] = len(seen_dates)
            day_number = seen_dates[date_str]
            week_number = 1 if day_number < 7 else 2

            hour = int(sh_s)
            minute = int(sm_s)
            day_index = DAYS_ORDER.index(day_name)
            slot_in_day = hour * 4 + (minute // 15)
            weekly_slot = day_index * 96 + slot_in_day
            flow_rate = float(val_str)

            records.append(
                FlowIntervalRecord(
                    index=len(records),
                    day_name=day_name,
                    day_index=day_index,
                    date_str=date_str,
                    day_number=day_number,
                    week_number=week_number,
                    hour=hour,
                    minute=minute,
                    slot_15m_in_day=slot_in_day,
                    weekly_slot_15m=weekly_slot,
                    flow_rate_lpm=flow_rate,
                    raw_label=ts_str,
                )
            )

    return records


def summarize_dataset(records: List[FlowIntervalRecord]) -> Dict[str, object]:
    """Menghasilkan ringkasan deskriptif statistik dataset 2 minggu."""
    flows = [r.flow_rate_lpm for r in records]
    dates = []
    for r in records:
        if r.date_str not in dates:
            dates.append(r.date_str)

    nonzero = [f for f in flows if f > 0.0]
    total_liters = sum(r.volume_liters for r in records)

    return {
        "total_15m_slots": len(records),
        "total_days": len(dates),
        "dates": dates,
        "week1_dates": dates[:7],
        "week2_dates": dates[7:14],
        "nonzero_15m_slots": len(nonzero),
        "zero_15m_slots": len(records) - len(nonzero),
        "max_15m_flow_lpm": max(flows) if flows else 0.0,
        "mean_15m_flow_lpm": sum(flows) / len(flows) if flows else 0.0,
        "total_volume_liters_2w": round(total_liters, 2),
        "avg_daily_liters": round(total_liters / max(1, len(dates)), 2),
    }

