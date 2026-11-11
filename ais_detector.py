"""
ais_detector.py — Implementasi Artificial Immune System (AIS) From Scratch
untuk Deteksi Anomali Penggunaan Air Berbasis Agregator 1 Jam

Mengimplementasikan dua model AIS berbasis Negative Selection untuk menjelaskan dan
menyelesaikan tantangan data 2 minggu (2 sampel per slot mingguan):

1. WaterAISVDetector (Model Utama — Context-Aware Real-Valued V-Detector AIS):
   - Mengubah seluruh jendela 1-jam normal selama 2 minggu (336 antigen) menjadi
     Self-Space terpadu berdimensi 7 dengan encoding waktu sirkadian 24-jam.
   - Dengan konteks sirkadian + kemiripan lintas 14 hari, keterbatasan 2 sampel per
     hari-jam teratasi secara alami (jam 07:00 mendapat dukungan tetangga jam 06:00-08:00
     dari 14 hari, bukan hanya 2 angka terisolasi).
   - Fase Thymus Censorship (Negative Selection):
     Membangkitkan kandidat sel imun pada manifold fisik aliran 4-slot + konteks jam,
     menghapus kandidat yang masuk ke dalam bola toleransi Self (d_self <= r_self),
     dan menetapkan radius variabel r_d = d_self - r_self bagi detektor Non-Self dewasa.
   - Fase Boundary-Shell Probing (Clonal Maturation):
     Menghasilkan detektor halus tepat di luar permukaan Self agar anomali moderat
     seperti kebocoran tipis atau keran lupa ditutup malam hari ikut tercover.
   - Fase Inferensi & Diagnosis Imunologis:
     Menentukan status NORMAL (Self) vs ANOMALI (Non-Self), skor afinitas/margin,
     fitur pemicu dominan (kebocoran kontinyu, lonjakan malam, lonjakan ekstrem, dll.),
     serta keputusan aktuator Solenoid Valve (OPEN / SHUT-OFF) pada mode Auto.

2. PerSlotNSA (Model Pembanding — AIS Per-Slot 168 x 2 Sampel):
   - Menerapkan Negative Selection secara terpisah pada tiap slot mingguan (day, hour)
     yang hanya memiliki 2 sampel Self (Minggu-1 & Minggu-2).
   - Disediakan agar pengguna dapat mengamati dan membandingkan secara langsung mengapa
     pendekatan per-slot dengan N=2 kurang fleksibel terhadap pergeseran jadwal normal.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from aggregator import (
    FEATURE_LABELS_ID,
    FEATURE_SCALES,
    FEATURE_WEIGHTS,
    HourlyAntigen,
    extract_antigen_vector,
    weighted_antigen_distance,
)


@dataclass
class AISTrainingTrace:
    """Rekaman proses seleksi negatif pada fase pelatihan AIS."""

    n_self_cells: int
    r_self: float
    max_detectors: int
    total_attempts: int
    detectors_created: int
    censored_by_self: int
    discarded_redundant: int
    train_time_sec: float
    radius_min: float
    radius_median: float
    radius_max: float
    history_steps: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_self_cells": self.n_self_cells,
            "r_self": round(self.r_self, 4),
            "max_detectors": self.max_detectors,
            "total_attempts": self.total_attempts,
            "detectors_created": self.detectors_created,
            "censored_by_self": self.censored_by_self,
            "discarded_redundant": self.discarded_redundant,
            "train_time_sec": round(self.train_time_sec, 4),
            "radius_min": round(self.radius_min, 4),
            "radius_median": round(self.radius_median, 4),
            "radius_max": round(self.radius_max, 4),
            "history_steps": self.history_steps,
        }


@dataclass
class AISDetectionResult:
    """Hasil klasifikasi dan diagnosis imunologis untuk satu agregasi 1-jam."""

    is_anomaly: bool
    status_label: str              # "ANOMALI" atau "NORMAL"
    decision_source: str           # "V-DETECTOR" atau "SELF-TOLERANCE"
    triggered_detectors: int
    max_detector_margin: float     # > 0 berarti berada dalam radius detektor Non-Self
    min_dist_to_self: float
    self_margin: float             # min_dist_to_self - r_self
    confidence_pct: float          # 0..100% keyakinan keputusan
    anomaly_category: str          # Kategori diagnosis manusiawi
    dominant_feature: str          # Fitur pemicu utama
    nearest_self_label: str        # Sampel normal historis paling mirip
    valve_action: str              # "CLOSE_VALVE" (mode auto anomali) atau "KEEP_OPEN"
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "is_anomaly": self.is_anomaly,
            "status_label": self.status_label,
            "decision_source": self.decision_source,
            "triggered_detectors": self.triggered_detectors,
            "max_detector_margin": round(float(self.max_detector_margin), 4),
            "min_dist_to_self": round(float(self.min_dist_to_self), 4),
            "self_margin": round(float(self.self_margin), 4),
            "confidence_pct": round(float(self.confidence_pct), 1),
            "anomaly_category": self.anomaly_category,
            "dominant_feature": self.dominant_feature,
            "nearest_self_label": self.nearest_self_label,
            "valve_action": self.valve_action,
            "reason": self.reason,
        }


class WaterAISVDetector:
    """
    Context-Aware Real-Valued Negative Selection Algorithm (V-Detector)
    untuk Deteksi Anomali Penggunaan Air 1-Jam.
    """

    def __init__(
        self,
        r_self: float = 0.43,
        max_detectors: int = 1500,
        max_attempts: int = 25000,
        seed: int = 42,
    ) -> None:
        self.r_self = float(r_self)
        self.max_detectors = int(max_detectors)
        self.max_attempts = int(max_attempts)
        self.seed = int(seed)

        self.self_matrix: Optional[np.ndarray] = None
        self.self_antigens: List[HourlyAntigen] = []
        self.detectors: Optional[np.ndarray] = None
        self.radii: Optional[np.ndarray] = None
        self.trace: Optional[AISTrainingTrace] = None

    def _fast_extract_physical(
        self,
        q0: float,
        q1: float,
        q2: float,
        q3: float,
        hour: int,
        is_wknd: bool,
    ) -> np.ndarray:
        """Ekstraksi cepat vektor 7D ternormalisasi pada manifold fisik 4-slot + sirkadian."""
        mean_f = 0.25 * (q0 + q1 + q2 + q3)
        max_f = max(q0, q1, q2, q3)
        min_f = min(q0, q1, q2, q3)
        n_act = (1 if q0 > 0.0 else 0) + (1 if q1 > 0.0 else 0) + (1 if q2 > 0.0 else 0) + (1 if q3 > 0.0 else 0)
        ang = 2.0 * math.pi * (hour % 24) / 24.0
        return np.array(
            [
                math.log1p(mean_f) / FEATURE_SCALES[0],
                math.log1p(max_f) / FEATURE_SCALES[1],
                (0.25 * n_act) / FEATURE_SCALES[2],
                math.log1p(min_f) / FEATURE_SCALES[3],
                (0.5 * (math.sin(ang) + 1.0)) / FEATURE_SCALES[4],
                (0.5 * (math.cos(ang) + 1.0)) / FEATURE_SCALES[5],
                (1.0 if is_wknd else 0.0) / FEATURE_SCALES[6],
            ],
            dtype=float,
        )

    def _sample_candidate(self, rng: np.random.Generator, S: np.ndarray) -> np.ndarray:
        """
        Membangkitkan kandidat sel imun murni pada manifold fisik 4-slot + konteks sirkadian.
        Menggabungkan Clonal Boundary-Shell Probing di sekitar Self-Antigen dengan eksplorasi
        burst rendah-menengah, kebocoran kontinyu 4/4 slot, dan lonjakan debit ekstrem.
        """
        rt = rng.random()
        if rt < 0.30:
            # Boundary-Shell Physical Probing: mulai dari sel Self nyata lalu mutasi fisik
            ag = self.self_antigens[int(rng.integers(0, len(self.self_antigens)))]
            dh = (-2, -1, 0, 0, 1, 2)[int(rng.integers(0, 6))]
            hr = (ag.hour + dh) % 24
            wk = ag.is_weekend if rng.random() < 0.60 else bool(rng.integers(0, 2))
            q = [ag.slot_flows[0], ag.slot_flows[1], ag.slot_flows[2], ag.slot_flows[3]]
            pm = rng.random()
            if pm < 0.40:
                drip = float(rng.uniform(0.18, 0.85))
                q = [max(v, drip) + float(rng.uniform(0.0, 0.15)) for v in q]
            elif pm < 0.75:
                k = int(rng.integers(1, 4))
                for _ in range(k):
                    j = int(rng.integers(0, 4))
                    q[j] += float(rng.uniform(0.45, 2.2))
            else:
                mult = float(rng.uniform(1.4, 3.0))
                bump = float(rng.uniform(0.3, 1.5)) if max(q) < 0.2 else 0.0
                q = [v * mult + bump for v in q]
            return self._fast_extract_physical(q[0], q[1], q[2], q[3], hr, wk)

        hr = int(rng.integers(0, 24))
        wk = bool(rng.integers(0, 2))

        if rt < 0.58:
            # Skenario burst rendah-menengah (1..3 slot aktif; peka jam tidur/malam)
            k_act = int(rng.integers(1, 4))
            q = [0.0, 0.0, 0.0, 0.0]
            for idx in rng.choice(4, size=k_act, replace=False):
                q[int(idx)] = float(rng.uniform(0.30, 2.9))
            return self._fast_extract_physical(q[0], q[1], q[2], q[3], hr, wk)

        if rt < 0.78:
            # Skenario aliran kontinyu (4/4 slot aktif -> penanda kuat kebocoran)
            base = float(np.exp(rng.uniform(math.log(0.20), math.log(3.4))))
            q = [
                min(6.0, max(0.12, base + float(rng.normal(0.0, 0.10))))
                for _ in range(4)
            ]
            return self._fast_extract_physical(q[0], q[1], q[2], q[3], hr, wk)

        # Skenario burst ekstrem / pipa pecah (1..4 slot aktif)
        k_act = int(rng.integers(1, 5))
        q = [0.0, 0.0, 0.0, 0.0]
        for idx in rng.choice(4, size=k_act, replace=False):
            q[int(idx)] = float(rng.uniform(2.5, 6.5))
        return self._fast_extract_physical(q[0], q[1], q[2], q[3], hr, wk)

    def fit(self, self_antigens: Sequence[HourlyAntigen]) -> "WaterAISVDetector":
        """
        Melatih V-Detector AIS menggunakan himpunan Self-Antigen normal 1-jam.
        Memanfaatkan pra-perkalian bobot fitur dan identitas ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        agar seleksi negatif dan eliminasi redundansi berjalan cepat di NumPy/BLAS.
        """
        if len(self_antigens) == 0:
            raise ValueError("Self-Antigen untuk pelatihan AIS tidak boleh kosong.")

        self.self_antigens = list(self_antigens)
        S = np.vstack([ag.features_norm for ag in self.self_antigens])
        self.self_matrix = S

        S_w = S * FEATURE_WEIGHTS
        S_norm2 = np.sum(S_w * S_w, axis=1)
        r_self_sq = self.r_self * self.r_self

        rng = np.random.default_rng(self.seed)
        t0 = time.perf_counter()

        D_buf = np.empty((self.max_detectors, S.shape[1]), dtype=float)
        D_w_buf = np.empty((self.max_detectors, S.shape[1]), dtype=float)
        D_norm2 = np.empty(self.max_detectors, dtype=float)
        R_buf = np.empty(self.max_detectors, dtype=float)
        R_sq_buf = np.empty(self.max_detectors, dtype=float)
        n_det = 0
        censored = 0
        redundant = 0
        attempts = 0
        history_steps: List[Dict[str, float]] = []

        checkpoint_step = max(250, self.max_attempts // 20)

        for i in range(1, self.max_attempts + 1):
            attempts = i
            if n_det >= self.max_detectors:
                break

            cand = self._sample_candidate(rng, S)
            cw = cand * FEATURE_WEIGHTS
            cn2 = float(np.dot(cw, cw))
            d2_self = float(np.min(np.maximum(0.0, S_norm2 + cn2 - 2.0 * (S_w @ cw))))

            # Seleksi Negatif (Thymus Censorship): gugurkan jika masuk radius Self
            if d2_self <= r_self_sq:
                censored += 1
            else:
                # Eliminasi redundansi jika titik sudah tercakup detektor dewasa
                is_covered = False
                if n_det > 0:
                    d2_det = np.maximum(0.0, D_norm2[:n_det] + cn2 - 2.0 * (D_w_buf[:n_det] @ cw))
                    if np.any(d2_det <= R_sq_buf[:n_det]):
                        is_covered = True

                if is_covered:
                    redundant += 1
                else:
                    d_self = math.sqrt(d2_self)
                    r_d = d_self - self.r_self
                    D_buf[n_det] = cand
                    D_w_buf[n_det] = cw
                    D_norm2[n_det] = cn2
                    R_buf[n_det] = r_d
                    R_sq_buf[n_det] = r_d * r_d
                    n_det += 1

            if i % checkpoint_step == 0 or n_det == self.max_detectors or i == self.max_attempts:
                history_steps.append(
                    {
                        "attempt": i,
                        "detectors": n_det,
                        "censored": censored,
                        "redundant": redundant,
                        "censorship_rate": round(censored / i, 4),
                    }
                )

        dt = time.perf_counter() - t0
        self.detectors = D_buf[:n_det].copy()
        self.radii = R_buf[:n_det].copy()

        r_min = float(np.min(self.radii)) if n_det > 0 else 0.0
        r_med = float(np.median(self.radii)) if n_det > 0 else 0.0
        r_max = float(np.max(self.radii)) if n_det > 0 else 0.0

        self.trace = AISTrainingTrace(
            n_self_cells=len(self.self_antigens),
            r_self=self.r_self,
            max_detectors=self.max_detectors,
            total_attempts=attempts,
            detectors_created=n_det,
            censored_by_self=censored,
            discarded_redundant=redundant,
            train_time_sec=dt,
            radius_min=r_min,
            radius_median=r_med,
            radius_max=r_max,
            history_steps=history_steps,
        )
        return self

    def _diagnose_anomaly(
        self,
        antigen: HourlyAntigen,
        nearest_self: HourlyAntigen,
    ) -> Tuple[str, str]:
        """Menghasilkan kategori diagnosis & fitur pemicu dominan yang mudah dibaca."""
        diff_weighted = np.abs(antigen.features_norm - nearest_self.features_norm) * FEATURE_WEIGHTS
        dom_idx = int(np.argmax(diff_weighted[:4]))
        dom_label = FEATURE_LABELS_ID[dom_idx]

        if antigen.active_slots == 4 and antigen.min_flow_lpm > 0.05:
            if antigen.hour < 6 or antigen.hour >= 21:
                return "Kebocoran Kontinyu Jam Malam", dom_label
            return "Kebocoran Kontinyu (4/4 Slot Aktif)", dom_label

        if antigen.max_flow_lpm >= 3.2:
            return "Lonjakan Debit Ekstrem", dom_label

        if antigen.hour < 6 or antigen.hour >= 20:
            return "Aktivitas Air Abnormal Jam Tidur/Malam", dom_label

        return "Pola Penggunaan di Luar Kebiasaan Normal", dom_label

    def predict_hour(
        self,
        antigen: HourlyAntigen,
        auto_mode: bool = True,
    ) -> AISDetectionResult:
        """
        Mengevaluasi satu jendela agregasi 1-jam terhadap populasi Mature V-Detectors.
        """
        if self.self_matrix is None or self.detectors is None or self.radii is None:
            raise RuntimeError("Model AIS belum dilatih. Panggil .fit(...) terlebih dahulu.")

        x = antigen.features_norm

        # Hitung jarak ke seluruh Self untuk explainability & nearest historical match
        self_dists = weighted_antigen_distance(x, self.self_matrix)
        nearest_idx = int(np.argmin(self_dists))
        min_d_self = float(self_dists[nearest_idx])
        nearest_self = self.self_antigens[nearest_idx]
        self_margin = min_d_self - self.r_self

        # Cek aktivasi Mature V-Detectors (Non-Self Recognition)
        if len(self.detectors) > 0:
            det_dists = weighted_antigen_distance(x, self.detectors)
            margins = self.radii - det_dists
            triggered_mask = margins > 0.0
            n_trig = int(np.sum(triggered_mask))
            max_det_margin = float(np.max(margins))
        else:
            n_trig = 0
            max_det_margin = -1.0

        is_anomaly = n_trig > 0
        if is_anomaly:
            category, dom_feat = self._diagnose_anomaly(antigen, nearest_self)
            # Skor keyakinan berbasis kombinasi margin detektor & jarak keluar dari Self
            raw_conf = 55.0 + 45.0 * math.tanh(max(0.0, max_det_margin) * 2.5 + max(0.0, self_margin) * 1.2)
            confidence = float(min(99.9, max(50.0, raw_conf)))
            valve_action = "CLOSE_VALVE" if auto_mode else "MANUAL_OVERRIDE_OPEN"
            reason = (
                f"{category}: memicu {n_trig} V-Detector AIS "
                f"(margin=+{max_det_margin:.3f}, jarak Self={min_d_self:.3f} > r_self={self.r_self:.2f})"
            )
            return AISDetectionResult(
                is_anomaly=True,
                status_label="ANOMALI",
                decision_source="V-DETECTOR",
                triggered_detectors=n_trig,
                max_detector_margin=max_det_margin,
                min_dist_to_self=min_d_self,
                self_margin=self_margin,
                confidence_pct=confidence,
                anomaly_category=category,
                dominant_feature=dom_feat,
                nearest_self_label=nearest_self.label,
                valve_action=valve_action,
                reason=reason,
            )

        # Normal (Self)
        raw_conf = 55.0 + 45.0 * math.tanh(max(0.0, -self_margin) * 2.5)
        confidence = float(min(99.9, max(50.0, raw_conf)))
        reason = (
            f"Normal (Self-Space): tidak memicu detektor Non-Self "
            f"(jarak Self={min_d_self:.3f} <= r_self={self.r_self:.2f}; terdekat: {nearest_self.label})"
        )
        return AISDetectionResult(
            is_anomaly=False,
            status_label="NORMAL",
            decision_source="SELF-TOLERANCE",
            triggered_detectors=0,
            max_detector_margin=max_det_margin,
            min_dist_to_self=min_d_self,
            self_margin=self_margin,
            confidence_pct=confidence,
            anomaly_category="Normal / Pola Wajar",
            dominant_feature="-",
            nearest_self_label=nearest_self.label,
            valve_action="KEEP_OPEN",
            reason=reason,
        )


class PerSlotNSA:
    """
    Model Pembanding: AIS Negative Selection Per-Slot Mingguan (168 Slot x 2 Sampel).

    Digunakan untuk menjawab pertanyaan mekanisme "bagaimana jika 2 sampel per hari-jam
    dipakai secara terpisah tanpa konteks sirkadian lintas hari".
    Pada tiap kunci (day_name, hour), hanya 2 titik Self (Minggu-1 & Minggu-2) yang
    digunakan dalam fitur aliran 4D [mean, max, active_ratio, min].
    """

    def __init__(self, r_self: float = 0.43, n_detectors_per_slot: int = 120, seed: int = 42) -> None:
        self.r_self = float(r_self)
        self.n_detectors_per_slot = int(n_detectors_per_slot)
        self.seed = int(seed)
        self.slot_models: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def fit(self, self_antigens: Sequence[HourlyAntigen]) -> "PerSlotNSA":
        buckets: Dict[Tuple[str, int], List[np.ndarray]] = defaultdict(list)
        for ag in self_antigens:
            # Ambil hanya 4 fitur aliran (karena waktu sudah dikunci ke slot exact)
            buckets[(ag.day_name, ag.hour)].append(ag.features_norm[:4])

        rng = np.random.default_rng(self.seed)
        self.slot_models.clear()
        w4 = FEATURE_WEIGHTS[:4]
        w4_sq = w4 * w4

        for key, vecs in buckets.items():
            S = np.vstack(vecs)
            D_buf = np.empty((self.n_detectors_per_slot, 4), dtype=float)
            R_buf = np.empty(self.n_detectors_per_slot, dtype=float)
            n_det = 0
            max_tries = self.n_detectors_per_slot * 5

            # Bangkitkan seluruh kandidat dalam batch vektor agar sangat cepat
            half = max_tries // 2
            cands = np.empty((max_tries, 4), dtype=float)
            for i in range(max_tries):
                if i < half:
                    k_act = int(rng.integers(1, 5))
                    slots = np.zeros(4)
                    idx = rng.choice(4, size=k_act, replace=False)
                    slots[idx] = rng.uniform(0.15, 5.5, size=k_act)
                else:
                    base = float(np.exp(rng.uniform(math.log(0.15), math.log(3.0))))
                    slots = np.clip(base + rng.normal(0.0, 0.12, size=4), 0.05, 5.5)
                cands[i] = extract_antigen_vector(slots, key[1], False)[:4]

            # Jarak seluruh kandidat ke Self slot (biasanya hanya N=2 titik!)
            d_self_all = np.sqrt(np.min(np.sum(((cands[:, None, :] - S[None, :, :]) ** 2) * w4_sq, axis=2), axis=1))
            for i in range(max_tries):
                if n_det >= self.n_detectors_per_slot:
                    break
                d_self = float(d_self_all[i])
                if d_self <= self.r_self:
                    continue
                cand = cands[i]
                if n_det > 0:
                    d_det = np.sqrt(np.sum(((D_buf[:n_det] - cand) ** 2) * w4_sq, axis=1))
                    if np.any(d_det <= R_buf[:n_det]):
                        continue
                D_buf[n_det] = cand
                R_buf[n_det] = d_self - self.r_self
                n_det += 1

            self.slot_models[key] = (S, D_buf[:n_det].copy(), R_buf[:n_det].copy())
        return self

    def predict_hour(self, antigen: HourlyAntigen) -> AISDetectionResult:
        key = (antigen.day_name, antigen.hour)
        if key not in self.slot_models:
            return AISDetectionResult(
                is_anomaly=False,
                status_label="NORMAL",
                decision_source="NO-SLOT-MODEL",
                triggered_detectors=0,
                max_detector_margin=-1.0,
                min_dist_to_self=0.0,
                self_margin=-self.r_self,
                confidence_pct=50.0,
                anomaly_category="Normal",
                dominant_feature="-",
                nearest_self_label="-",
                valve_action="KEEP_OPEN",
                reason="Slot mingguan tidak ditemukan",
            )

        S, D_slot, R_slot = self.slot_models[key]
        x4 = antigen.features_norm[:4]
        w4 = FEATURE_WEIGHTS[:4]
        d_self = float(np.min(np.sqrt(np.sum(((S - x4) * w4) ** 2, axis=1))))

        if len(D_slot) > 0:
            d_det = np.sqrt(np.sum(((D_slot - x4) * w4) ** 2, axis=1))
            margins = R_slot - d_det
            n_trig = int(np.sum(margins > 0.0))
            max_m = float(np.max(margins))
        else:
            n_trig = 0
            max_m = -1.0

        is_anom = n_trig > 0
        return AISDetectionResult(
            is_anomaly=is_anom,
            status_label="ANOMALI" if is_anom else "NORMAL",
            decision_source="PER-SLOT-NSA",
            triggered_detectors=n_trig,
            max_detector_margin=max_m,
            min_dist_to_self=d_self,
            self_margin=d_self - self.r_self,
            confidence_pct=85.0 if is_anom else 80.0,
            anomaly_category="Anomali Slot (N=2)" if is_anom else "Normal Slot (N=2)",
            dominant_feature="Aliran Slot" if is_anom else "-",
            nearest_self_label=f"{antigen.day_name} {antigen.hour:02d}:00 (2 sampel)",
            valve_action="CLOSE_VALVE" if is_anom else "KEEP_OPEN",
            reason=(
                f"Per-Slot NSA ({len(S)} sampel): dist={d_self:.3f}, triggered={n_trig}"
            ),
        )
