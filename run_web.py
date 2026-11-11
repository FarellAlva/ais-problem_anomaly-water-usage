"""
run_web.py — Server Web & API JSON Lokal untuk Dashboard Deteksi Anomali Air Berbasis AIS

Menjalankan HTTP server lokal yang melayani:
- Antarmuka web statis di folder `web/` (index.html, style.css, app.js)
- Endpoint REST JSON API berbasis mesin Python asli (`aggregator.py`, `ais_detector.py`,
  `scenarios.py`, utilitas overlay Z-Score grafik, serta hasil evaluasi di `results/`):
  - GET  /api/init             -> Ringkasan dataset, data 14 hari, kurva sirkadian, status model AIS awal
  - POST /api/train            -> Melatih ulang WaterAISVDetector dengan (r_self, max_detectors, seed)
  - POST /api/evaluate_hour    -> Evaluasi API 1 jam (4x15m + hari + jam) pada AIS V-Detector
  - POST /api/simulate_day     -> Simulasi 24 jam model mingguan gabungan + injeksi anomali + overlay threshold
  - GET  /api/weekly_slot      -> Inspeksi 2 sampel (Minggu-1 & Minggu-2) + dukungan sirkadian AIS
  - GET  /api/benchmark        -> Mengembalikan hasil evaluasi AIS & ablasi internal di `results/`

Jalankan:
    python run_web.py
"""

from __future__ import annotations

import json
import mimetypes
import os
import socket
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

from aggregator import (
    HourlyAntigen,
    aggregate_tumbling_1h,
    build_hourly_antigen,
    group_weekly_two_samples,
)
from ais_detector import PerSlotNSA, WaterAISVDetector
from dataset_loader import DAYS_ID_MAP, DAYS_ORDER, load_water_dataset, summarize_dataset
from scenarios import build_benchmark_scenarios
from zscore_baseline import ZScore15MinBaseline, ZScore1HourBaseline


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


class AppState:
    """Menyimpan state dataset dan model AIS/Z-Score yang sedang aktif di memori."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.records = load_water_dataset()
        self.antigens = aggregate_tumbling_1h(self.records)
        self.dataset_summary = summarize_dataset(self.records)
        self.weekly_groups = group_weekly_two_samples(self.antigens)
        self.scenarios = build_benchmark_scenarios()

        self.z15 = ZScore15MinBaseline(k=2.0, ucl_min=0.20).fit(self.records)
        self.z1h = ZScore1HourBaseline(k=2.0, ucl_min=0.08).fit(self.antigens)
        self.perslot_ais = PerSlotNSA(r_self=0.43, n_detectors_per_slot=60, seed=42).fit(self.antigens)
        self.ais = WaterAISVDetector(r_self=0.43, max_detectors=1500, max_attempts=25000, seed=42).fit(
            self.antigens
        )

    def retrain_ais(self, r_self: float, max_detectors: int, seed: int = 42) -> Dict[str, Any]:
        with self.lock:
            self.ais = WaterAISVDetector(
                r_self=r_self,
                max_detectors=max_detectors,
                max_attempts=max(12000, max_detectors * 18),
                seed=seed,
            ).fit(self.antigens)
            return self.get_ais_payload()

    def get_ais_payload(self) -> Dict[str, Any]:
        D = self.ais.detectors
        R = self.ais.radii
        angles = np.arctan2(D[:, 4] * 2.0 - 1.0, D[:, 5] * 2.0 - 1.0)
        d_hours = (angles / (2.0 * np.pi) * 24.0) % 24.0
        d_means = np.expm1(D[:, 0] * np.log1p(1.5))

        step = max(1, len(D) // 350)
        sample_dets = [
            {
                "hour": round(float(d_hours[i]), 2),
                "mean_flow": round(float(d_means[i]), 3),
                "radius": round(float(R[i]), 3),
                "active_ratio": round(float(D[i, 2]), 2),
            }
            for i in range(0, len(D), step)
        ]

        return {
            "r_self": self.ais.r_self,
            "max_detectors": self.ais.max_detectors,
            "seed": self.ais.seed,
            "trace": self.ais.trace.to_dict() if self.ais.trace else {},
            "detector_projection": sample_dets,
            "radii_sample": [round(float(r), 4) for r in R[::max(1, len(R)//200)]],
        }

    def evaluate_hour(
        self,
        slot_flows: List[float],
        hour: int,
        day_name: str,
        auto_mode: bool = True,
    ) -> Dict[str, Any]:
        with self.lock:
            ag = build_hourly_antigen(
                slot_flows=slot_flows,
                hour=hour,
                day_name=day_name,
                date_str="Uji-Interaktif",
                day_number=14,
                week_number=3,
                hour_index=0,
            )
            res_ais = self.ais.predict_hour(ag, auto_mode=auto_mode)
            res_perslot = self.perslot_ais.predict_hour(ag)
            res_z15 = self.z15.predict_hour(ag)
            res_z1h = self.z1h.predict_hour(ag)

            # Data historis 2 sampel pada slot (day_name, hour)
            pair = self.weekly_groups.get((day_name, hour % 24), [])
            two_samples = [
                {
                    "week": p.week_number,
                    "date_str": p.date_str,
                    "slot_flows": list(p.slot_flows),
                    "mean_flow_lpm": round(p.mean_flow_lpm, 4),
                    "max_flow_lpm": round(p.max_flow_lpm, 4),
                    "active_slots": p.active_slots,
                }
                for p in pair
            ]

            return {
                "antigen": ag.to_dict(),
                "ais_vdetector": res_ais.to_dict(),
                "ais_perslot": res_perslot.to_dict(),
                "zscore_15m": res_z15.to_dict(),
                "zscore_1h": res_z1h.to_dict(),
                "two_samples_slot": two_samples,
            }

    def _build_threshold_profile_unlocked(self, day_name: str) -> List[Dict[str, Any]]:
        ais_prof = self.ais.compute_weekday_threshold_profile(day_name)
        profile: List[Dict[str, Any]] = []
        for h in range(24):
            pair = self.weekly_groups.get((day_name, h), [])
            w1_ag = next((p for p in pair if p.week_number == 1), None)
            w2_ag = next((p for p in pair if p.week_number == 2), None)
            z_st = self.z1h.slot_stats.get((day_name, h))
            profile.append(
                {
                    "hour": h,
                    "w1_mean": round(w1_ag.mean_flow_lpm, 4) if w1_ag else 0.0,
                    "w2_mean": round(w2_ag.mean_flow_lpm, 4) if w2_ag else 0.0,
                    "combined_mean": round(z_st.mean, 4) if z_st else 0.0,
                    "combined_std": round(z_st.std, 4) if z_st else 0.0,
                    "zscore_ucl": round(z_st.ucl_final, 4) if z_st else 0.08,
                    "ais_upper": ais_prof[h]["ais_upper_lpm"],
                    "ais_leak_limit": ais_prof[h]["ais_leak_limit_lpm"],
                }
            )
        return profile

    def simulate_day(
        self,
        day_number: Optional[int] = None,
        day_name: Optional[str] = None,
        week_source: str = "week2",
        inject_preset: str = "none",
        inject_hour: int = 10,
        auto_mode: bool = True,
    ) -> Dict[str, Any]:
        with self.lock:
            if day_name is not None and day_name in DAYS_ORDER:
                target_week = 1 if week_source == "week1" else 2
                day_hours = [
                    ag for ag in self.antigens
                    if ag.day_name == day_name and ag.week_number == target_week
                ]
            else:
                dn = 13 if day_number is None else int(day_number)
                day_hours = [ag for ag in self.antigens if ag.day_number == dn]
                week_source = "week1" if (day_hours and day_hours[0].week_number == 1) else "week2"

            if not day_hours:
                day_hours = [ag for ag in self.antigens if ag.day_number == 0]

            target_day = day_hours[0].day_name
            threshold_profile = self._build_threshold_profile_unlocked(target_day)
            pair0 = self.weekly_groups.get((target_day, 0), [])
            w1_date = next((p.date_str for p in pair0 if p.week_number == 1), "-")
            w2_date = next((p.date_str for p in pair0 if p.week_number == 2), "-")

            sim_hours: List[Dict[str, Any]] = []
            for ag in day_hours:
                flows = list(ag.slot_flows)
                injected = False
                if inject_preset != "none" and ag.hour == inject_hour:
                    injected = True
                    if inject_preset == "slow_leak":
                        flows = [0.35, 0.35, 0.35, 0.35]
                    elif inject_preset == "night_open_tap":
                        flows = [1.20, 1.20, 0.00, 0.00]
                    elif inject_preset == "extreme_burst":
                        flows = [5.10, 4.60, 0.00, 0.00]
                    elif inject_preset == "continuous_high":
                        flows = [1.80, 1.80, 1.80, 1.80]
                    elif inject_preset == "schedule_shift":
                        flows = [0.00, 2.35, 0.00, 0.00]

                cur_ag = build_hourly_antigen(
                    slot_flows=flows,
                    hour=ag.hour,
                    day_name=ag.day_name,
                    date_str=ag.date_str,
                    day_number=ag.day_number,
                    week_number=ag.week_number,
                    hour_index=ag.hour_index,
                )
                r_ais = self.ais.predict_hour(cur_ag, auto_mode=auto_mode)
                r_z15 = self.z15.predict_hour(cur_ag)
                r_z1h = self.z1h.predict_hour(cur_ag)
                tp = threshold_profile[cur_ag.hour]

                sim_hours.append(
                    {
                        "hour": cur_ag.hour,
                        "label": cur_ag.label,
                        "injected": injected,
                        "slot_flows": [round(float(x), 3) for x in cur_ag.slot_flows],
                        "mean_flow_lpm": round(cur_ag.mean_flow_lpm, 4),
                        "max_flow_lpm": round(cur_ag.max_flow_lpm, 4),
                        "active_slots": cur_ag.active_slots,
                        "total_liters": round(cur_ag.total_liters, 2),
                        "combined_mean": tp["combined_mean"],
                        "zscore_ucl": tp["zscore_ucl"],
                        "ais_upper": tp["ais_upper"],
                        "ais_leak_limit": tp["ais_leak_limit"],
                        "ais": r_ais.to_dict(),
                        "zscore_15m": r_z15.to_dict(),
                        "zscore_1h": r_z1h.to_dict(),
                    }
                )

            return {
                "day_number": day_hours[0].day_number,
                "day_name": target_day,
                "day_name_id": DAYS_ID_MAP.get(target_day, target_day),
                "date_str": day_hours[0].date_str,
                "week_number": day_hours[0].week_number,
                "week_source": week_source,
                "w1_date_str": w1_date,
                "w2_date_str": w2_date,
                "inject_preset": inject_preset,
                "inject_hour": inject_hour,
                "auto_mode": auto_mode,
                "threshold_profile": threshold_profile,
                "hours": sim_hours,
            }

    def get_weekly_slot(self, day_name: str, hour: int) -> Dict[str, Any]:
        with self.lock:
            h = hour % 24
            pair = self.weekly_groups.get((day_name, h), [])
            z1h_stat = self.z1h.slot_stats.get((day_name, h))
            ais_prof = self.ais.compute_weekday_threshold_profile(day_name)[h]
            neighbors = [
                ag for ag in self.antigens
                if min((ag.hour - h) % 24, (h - ag.hour) % 24) <= 1
            ]
            return {
                "day_name": day_name,
                "hour": h,
                "samples": [p.to_dict() for p in pair],
                "combined_mean": round(z1h_stat.mean, 4) if z1h_stat else 0.0,
                "combined_std": round(z1h_stat.std, 4) if z1h_stat else 0.0,
                "zscore_ucl": round(z1h_stat.ucl_final, 4) if z1h_stat else 0.08,
                "ais_upper_lpm": ais_prof["ais_upper_lpm"],
                "ais_leak_limit_lpm": ais_prof["ais_leak_limit_lpm"],
                "ais_circadian_support_count": len(neighbors),
                "ais_circadian_mean_max": round(
                    max((n.mean_flow_lpm for n in neighbors), default=0.0), 4
                ),
            }

    def get_init_payload(self) -> Dict[str, Any]:
        with self.lock:
            # Matriks 14 x 24 untuk heatmap
            days_meta = []
            heatmap = []
            for d in range(14):
                day_ags = [ag for ag in self.antigens if ag.day_number == d]
                days_meta.append(
                    {
                        "day_number": d,
                        "day_name": day_ags[0].day_name,
                        "day_name_id": day_ags[0].day_name_id if hasattr(day_ags[0], "day_name_id") else day_ags[0].day_name,
                        "date_str": day_ags[0].date_str,
                        "week_number": day_ags[0].week_number,
                        "total_liters": round(sum(a.total_liters for a in day_ags), 1),
                    }
                )
                heatmap.append([ag.to_dict() for ag in day_ags])

            # Profil sirkadian 24 jam
            circadian = []
            for h in range(24):
                hs = [ag for ag in self.antigens if ag.hour == h]
                circadian.append(
                    {
                        "hour": h,
                        "mean_flow": round(float(np.mean([a.mean_flow_lpm for a in hs])), 4),
                        "max_hourly_mean": round(float(np.max([a.mean_flow_lpm for a in hs])), 4),
                        "max_15m_flow": round(float(np.max([a.max_flow_lpm for a in hs])), 4),
                        "active_days": int(sum(1 for a in hs if a.active_slots > 0)),
                    }
                )

            # Self projection (336 titik)
            self_proj = [
                {
                    "hour": ag.hour,
                    "mean_flow": round(ag.mean_flow_lpm, 4),
                    "max_flow": round(ag.max_flow_lpm, 4),
                    "active_slots": ag.active_slots,
                    "label": ag.label,
                }
                for ag in self.antigens
            ]

            weekdays_meta = []
            for d_name in DAYS_ORDER:
                w1_ags = [ag for ag in self.antigens if ag.day_name == d_name and ag.week_number == 1]
                w2_ags = [ag for ag in self.antigens if ag.day_name == d_name and ag.week_number == 2]
                weekdays_meta.append(
                    {
                        "day_name": d_name,
                        "day_name_id": DAYS_ID_MAP.get(d_name, d_name),
                        "w1_date": w1_ags[0].date_str if w1_ags else "-",
                        "w2_date": w2_ags[0].date_str if w2_ags else "-",
                        "w1_liters": round(sum(a.total_liters for a in w1_ags), 1),
                        "w2_liters": round(sum(a.total_liters for a in w2_ags), 1),
                    }
                )

            return {
                "dataset_summary": self.dataset_summary,
                "days": days_meta,
                "weekdays": weekdays_meta,
                "heatmap": heatmap,
                "circadian": circadian,
                "self_projection": self_proj,
                "scenarios": [sc.to_dict() for sc in self.scenarios],
                "ais": self.get_ais_payload(),
            }

    def get_benchmark_payload(self) -> Dict[str, Any]:
        bench_path = os.path.join(RESULTS_DIR, "benchmark_summary.json")
        lodo_path = os.path.join(RESULTS_DIR, "lodo_cv.json")
        sens_path = os.path.join(RESULTS_DIR, "sensitivity.json")
        out: Dict[str, Any] = {}
        if os.path.exists(bench_path):
            with open(bench_path, "r", encoding="utf-8") as f:
                out["benchmark_summary"] = json.load(f)
        if os.path.exists(lodo_path):
            with open(lodo_path, "r", encoding="utf-8") as f:
                out["lodo_cv"] = json.load(f)
        if os.path.exists(sens_path):
            with open(sens_path, "r", encoding="utf-8") as f:
                out["sensitivity"] = json.load(f)
        return out


STATE: Optional[AppState] = None


class AISRequestHandler(BaseHTTPRequestHandler):
    """Handler HTTP untuk file statis web dan REST API Python."""

    def log_message(self, format: str, *args: Any) -> None:
        # Suppres log HTTP per-request agar terminal bersih
        pass

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/init":
            self._send_json(STATE.get_init_payload())
            return

        if path == "/api/weekly_slot":
            qs = parse_qs(parsed.query)
            day_name = qs.get("day", ["Sunday"])[0]
            if day_name not in DAYS_ORDER:
                day_name = "Sunday"
            hour = int(qs.get("hour", ["6"])[0]) % 24
            self._send_json(STATE.get_weekly_slot(day_name, hour))
            return

        if path == "/api/benchmark":
            self._send_json(STATE.get_benchmark_payload())
            return

        # File statis dari web/
        rel = path.lstrip("/")
        if rel == "" or rel == "web" or rel == "web/":
            rel = "index.html"
        elif rel.startswith("web/"):
            rel = rel[4:]

        target = os.path.normpath(os.path.join(WEB_DIR, rel))
        if not target.startswith(os.path.normpath(WEB_DIR)) or not os.path.isfile(target):
            self.send_error(HTTPStatus.NOT_FOUND, "File tidak ditemukan")
            return

        mime, _ = mimetypes.guess_type(target)
        with open(target, "rb") as fh:
            content = fh.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime or 'application/octet-stream'}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            body = self._read_json_body()
        except Exception as exc:
            self._send_json({"error": f"JSON tidak valid: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return

        if path == "/api/train":
            r_self = float(body.get("r_self", 0.43))
            max_det = int(body.get("max_detectors", 1500))
            seed = int(body.get("seed", 42))
            r_self = max(0.25, min(0.75, r_self))
            max_det = max(100, min(3000, max_det))
            res = STATE.retrain_ais(r_self=r_self, max_detectors=max_det, seed=seed)
            self._send_json(res)
            return

        if path == "/api/evaluate_hour":
            flows = body.get("slot_flows", [0.0, 0.0, 0.0, 0.0])
            if not isinstance(flows, list) or len(flows) != 4:
                self._send_json({"error": "slot_flows harus array 4 angka"}, status=HTTPStatus.BAD_REQUEST)
                return
            flows = [max(0.0, float(x)) for x in flows]
            hour = int(body.get("hour", 10)) % 24
            day_name = str(body.get("day_name", "Monday"))
            if day_name not in DAYS_ORDER:
                day_name = "Monday"
            auto_mode = bool(body.get("auto_mode", True))
            self._send_json(STATE.evaluate_hour(flows, hour, day_name, auto_mode=auto_mode))
            return

        if path == "/api/simulate_day":
            raw_dn = body.get("day_number")
            day_number = int(raw_dn) if raw_dn is not None else None
            day_name = body.get("day_name")
            if day_name is not None and day_name not in DAYS_ORDER:
                day_name = "Saturday"
            week_source = str(body.get("week_source", "week2"))
            inject_preset = str(body.get("inject_preset", "none"))
            inject_hour = int(body.get("inject_hour", 10)) % 24
            auto_mode = bool(body.get("auto_mode", True))
            self._send_json(
                STATE.simulate_day(
                    day_number=day_number,
                    day_name=day_name,
                    week_source=week_source,
                    inject_preset=inject_preset,
                    inject_hour=inject_hour,
                    auto_mode=auto_mode,
                )
            )
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Endpoint API tidak ditemukan")


def find_free_port(start_port: int = 8050, max_port: int = 8150) -> int:
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def main() -> None:
    global STATE
    print("Memuat dataset 2 minggu & melatih model V-Detector AIS awal...")
    STATE = AppState()
    port = find_free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), AISRequestHandler)
    url = f"http://127.0.0.1:{port}/"
    print("=" * 74)
    print("DASHBOARD DETEKSI ANOMALI PENGGUNAAN AIR BERBASIS ARTIFICIAL IMMUNE SYSTEM")
    print("=" * 74)
    print(f"Server aktif di : {url}")
    print(f"V-Detectors     : {len(STATE.ais.detectors)} sel dewasa (r_self={STATE.ais.r_self})")
    print("Tekan Ctrl+C untuk menghentikan server.")
    print("=" * 74)

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer dihentikan.")
        server.server_close()


if __name__ == "__main__":
    main()

