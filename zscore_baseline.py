"""
zscore_baseline.py — Implementasi Baseline Z-Score Statistik (Referensi Jurnal JUTISI)

Mengimplementasikan pendekatan Z-Score dari:
  "Sistem Deteksi Anomali Penggunaan Air Berbasis IoT dengan Pendekatan Z-Score Statistik"
  (Farell Alvaro Theriono et al., JuTISI)

Menyediakan dua varian:
1. ZScore15MinBaseline (672 slot mingguan = 7 hari x 96 slot 15-menit):
   - Menghitung mu dan sigma (ddof=1) tiap slot (N=2 pada data 2 minggu).
   - UCL_final = max(mu + k * sigma, ucl_min), dengan k=2.0 dan ucl_min=0.20 L/min.
   - Aturan agregator 1-jam: jika >= 1 dari 4 slot 15-menit melebihi UCL_final,
     maka jam tersebut diklasifikasikan sebagai ANOMALI.
2. ZScore1HourBaseline (168 slot mingguan = 7 hari x 24 jam):
   - Menerapkan rumus Z-Score yang sama pada rata-rata laju air 1 jam (N=2).
   - UCL_final = max(mu + k * sigma, ucl_min_hourly), dengan k=2.0 dan ucl_min=0.08 L/min.

Catatan Matematis N=2:
Jika data pelatihan hanya 1 minggu (N=1 per slot), sample std (n-1) tidak terdefinisi
sehingga jatuh ke fallback sigma=0 (hanya mengandalkan ucl_min). Dan pada N=2,
s = |x1 - x2| / sqrt(2) membuat batas atas sangat tidak stabil ketika dua minggu
memiliki nilai kembar (s=0 -> UCL sempit) atau bergeser jam (s besar -> UCL terlalu lebar).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from aggregator import HourlyAntigen
from dataset_loader import FlowIntervalRecord


@dataclass(frozen=True)
class SlotStat:
    mean: float
    std: float
    ucl_raw: float
    ucl_final: float
    n_samples: int


@dataclass
class ZScorePrediction:
    is_anomaly: bool
    score: float            # Rasio maksimum (observed / ucl_final) atau Z-score
    observed_value: float
    ucl_final: float
    triggered_slots: List[int]
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "is_anomaly": self.is_anomaly,
            "score": round(float(self.score), 4),
            "observed_value": round(float(self.observed_value), 4),
            "ucl_final": round(float(self.ucl_final), 4),
            "triggered_slots": self.triggered_slots,
            "reason": self.reason,
        }


class ZScore15MinBaseline:
    """Metode Z-Score Jurnal JUTISI pada 672 slot mingguan (interval 15 menit)."""

    def __init__(self, k: float = 2.0, ucl_min: float = 0.20) -> None:
        self.k = float(k)
        self.ucl_min = float(ucl_min)
        self.slot_stats: Dict[Tuple[str, int, int], SlotStat] = {}

    def fit(self, records: Sequence[FlowIntervalRecord]) -> "ZScore15MinBaseline":
        buckets: Dict[Tuple[str, int, int], List[float]] = defaultdict(list)
        for r in records:
            buckets[(r.day_name, r.hour, r.minute)].append(r.flow_rate_lpm)

        self.slot_stats.clear()
        for key, vals in buckets.items():
            n = len(vals)
            mu = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if n >= 2 else 0.0
            ucl_raw = mu + self.k * sd
            ucl_final = max(ucl_raw, self.ucl_min)
            self.slot_stats[key] = SlotStat(mu, sd, ucl_raw, ucl_final, n)
        return self

    def predict_hour(self, antigen: HourlyAntigen) -> ZScorePrediction:
        triggered: List[int] = []
        max_ratio = 0.0
        worst_obs = 0.0
        worst_ucl = self.ucl_min

        for idx, val in enumerate(antigen.slot_flows):
            minute = idx * 15
            stat = self.slot_stats.get((antigen.day_name, antigen.hour, minute))
            ucl = stat.ucl_final if stat is not None else self.ucl_min
            ratio = val / max(ucl, 1e-9)
            if ratio > max_ratio:
                max_ratio = ratio
                worst_obs = val
                worst_ucl = ucl
            if val > ucl:
                triggered.append(idx)

        is_anom = len(triggered) > 0
        if is_anom:
            reason = (
                f"Z-Score 15m: slot {triggered} melebihi UCL "
                f"({worst_obs:.2f} > {worst_ucl:.2f} L/min)"
            )
        else:
            reason = f"Normal Z-Score 15m (maks rasio {max_ratio:.2f}x UCL)"
        return ZScorePrediction(
            is_anomaly=is_anom,
            score=max_ratio,
            observed_value=worst_obs,
            ucl_final=worst_ucl,
            triggered_slots=triggered,
            reason=reason,
        )


class ZScore1HourBaseline:
    """Adaptasi Z-Score Jurnal JUTISI langsung ke 168 slot agregasi 1 jam (N=2)."""

    def __init__(self, k: float = 2.0, ucl_min: float = 0.08) -> None:
        self.k = float(k)
        self.ucl_min = float(ucl_min)
        self.slot_stats: Dict[Tuple[str, int], SlotStat] = {}

    def fit(self, antigens: Sequence[HourlyAntigen]) -> "ZScore1HourBaseline":
        buckets: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        for ag in antigens:
            buckets[(ag.day_name, ag.hour)].append(ag.mean_flow_lpm)

        self.slot_stats.clear()
        for key, vals in buckets.items():
            n = len(vals)
            mu = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if n >= 2 else 0.0
            ucl_raw = mu + self.k * sd
            ucl_final = max(ucl_raw, self.ucl_min)
            self.slot_stats[key] = SlotStat(mu, sd, ucl_raw, ucl_final, n)
        return self

    def predict_hour(self, antigen: HourlyAntigen) -> ZScorePrediction:
        stat = self.slot_stats.get((antigen.day_name, antigen.hour))
        ucl = stat.ucl_final if stat is not None else self.ucl_min
        obs = antigen.mean_flow_lpm
        ratio = obs / max(ucl, 1e-9)
        is_anom = obs > ucl
        reason = (
            f"Z-Score 1h: rata-rata {obs:.3f} > UCL {ucl:.3f} L/min"
            if is_anom
            else f"Normal Z-Score 1h ({obs:.3f} <= UCL {ucl:.3f} L/min)"
        )
        return ZScorePrediction(
            is_anomaly=is_anom,
            score=ratio,
            observed_value=obs,
            ucl_final=ucl,
            triggered_slots=[0, 1, 2, 3] if is_anom else [],
            reason=reason,
        )

