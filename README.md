# WaterAIS — Deteksi Anomali Penggunaan Air Berbasis Artificial Immune System (AIS) & Agregator 1-Jam

Implementasi *from-scratch* **Artificial Immune System (AIS)** menggunakan **Context-Aware Real-Valued V-Detector Negative Selection** untuk mendeteksi anomali penggunaan air IoT pada **jendela agregasi 1 jam** (`4 × 15 menit`), lengkap dengan perbandingan empiris terhadap **Z-Score Statistik (Jurnal JUTISI)** serta dua aplikasi interaktif (**Web Dashboard** dan **Desktop GUI Tkinter**).

---

## 1. Struktur Dataset 2 Minggu (`Model-dataset-2 weeks.txt`)

Dataset pengamatan laju aliran air normal berlangsung selama **14 hari (04 Mei 2025 – 17 Mei 2025)** dengan resolusi 15 menit:
- Total rekaman 15-menit: **`1.344`** slot (`287` aktif `> 0 L/min`, `1.057` bernilai `0.00 L/min`).
- Volume total 14 hari: **`3.493,35 Liter`** (rata-rata **`249,53 Liter/hari`**), laju puncak **`2,61 L/min`**.
- **Agregator 1-Jam (`Tumbling Window`)**: mengelompokkan setiap `4` slot berurutan menjadi **`336` blok jam** (`14 hari × 24 jam`).
- Fakta fisik penting dari 336 jam normal:
  - `0/4` slot aktif (diam): **112 jam (33,3%)**
  - `1/4` slot aktif: **163 jam (48,5%)**
  - `2/4` slot aktif: **59 jam (17,6%)**
  - `3/4` slot aktif: **2 jam (0,6%)**
  - **`4/4` slot aktif kontinyu**: **`0 jam (0,0%)`** — seluruh data normal selalu memiliki setidaknya satu interval jeda `0.00 L/min`, sehingga aliran `4/4` slot penuh menjadi ciri fisik kuat kebocoran pipa/keran.

---

## 2. Pertanyaan Kunci: Apakah Z-Score Bisa Diterapkan & Bagaimana Mekanisme AIS pada Data 2 Minggu?

### A. Mengapa Z-Score Jurnal JUTISI dengan $N=2$ Sampel Per Hari-Jam Rapuh?
Pada data 2 minggu yang dipetakan ke jadwal mingguan (`7 hari × 24 jam = 168 slot jam`, atau `672 slot 15-menit`), setiap pasangan `(Hari, Jam)` tepat hanya memiliki **$N = 2$ sampel** (*Minggu-1* dan *Minggu-2*). Secara statistik, rumus $UCL = \mu + k \cdot s$ dengan $N=2$ mengalami tiga masalah mendasar:
1. **Standar Deviasi $N=2$ Tidak Stabil**: $s = |x_1 - x_2| / \sqrt{2}$. Jika Minggu-1 dan Minggu-2 sama-sama `0`, maka $s = 0$ (ambang menjadi sangat kaku dan rentan alarm palsu). Sebaliknya jika penghuni rumah menggeser jam mandi (misalnya *Minggu 08:00* pada W1 = `0.00` dan W2 = `1.04 L/min`), maka $UCL$ melonjak hingga `~2.0 L/min` sehingga kebocoran sedang (`0.50 L/min × 4`) lolos begitu saja.
2. **Gagal Jika $N=1$ (Holdout / Leave-One-Out)**: Pada validasi *1-minggu* atau *Leave-One-Day-Out*, `s` (`ddof=1`) tidak terdefinisi (`0/0`) dan memicu false positive tinggi (**35,71%** di Holdout W1→W2, **27,98%** di 14-Fold LODO CV).
3. **Rentan Pergeseran Jadwal 15 Menit**: Pada uji 30 skenario, Z-Score 15-menit (`JUTISI`) memicu **50,0% False Positive Rate** karena pergeseran mandi 15–30 menit dianggap anomali.

### B. Mekanisme AIS Mengatasi Keterbatasan 2 Sampel
Alih-alih mengisolasi setiap `(Hari, Jam)` menjadi `168` pulau terpisah yang hanya berisi 2 angka (*Per-Slot AIS* gagal **19,64% FPR** pada Holdout dan **17,86% FPR** pada LODO CV), **WaterAIS V-Detector** bekerja dengan mekanisme berikut:
1. **Agregator 1-Jam → Antigen 7D Terpadu**: Setiap jendela 1 jam (`4 × 15 menit`) diekstrak menjadi vektor 7 dimensi:
   $$\mathbf{x} = \big[\log(1+\bar{q}),\; \log(1+q_{\max}),\; r_{\text{aktif}},\; \log(1+q_{\min}),\; \sin_{\text{jam}},\; \cos_{\text{jam}},\; \mathbb{I}_{\text{akhir pekan}}\big]$$
   Seluruh `336` blok jam dari 14 hari membentuk **Self-Space** terpadu. Setiap jam tidak lagi berdiri sendiri dari 2 angka, melainkan mendapat dukungan kontekstual dari ~42 tetangga sirkadian (`±1 jam` lintas 14 hari).
2. **Thymus Censorship (Negative Selection)**: Sistem membangkitkan kandidat sel imun di manifold fisik aliran air 4-slot + waktu sirkadian. Kandidat yang berjarak $\le r_{\text{self}}$ (`0.43`) dari sel *Self* manapun langsung digugurkan (*apoptsis*).
3. **Mature V-Detectors Non-Self**: Kandidat yang lolos menjadi detektor dewasa beradius variabel $r_d = d_{\text{self}} - r_{\text{self}}$, membungkus rapat komplemen ruang *Self* (termasuk seluruh rezim aliran kontinyu `4/4`, aktivitas tak wajar jam tidur `00:00–05:59` & `20:00–23:59`, serta lonjakan debit ekstrem).
4. **Inferensi 1-Jam Real-Time (`Tumbling` & `Sliding Window`)**: Bila sebuah jendela 1 jam mengaktifkan $\ge 1$ V-Detector, jam tersebut diklasifikasikan **ANOMALI**, didiagnosis penyebabnya, dan pada mode **AUTO** memicu sinyal penutupan *Solenoid Valve*.

---

## 3. Ringkasan Hasil Evaluasi Empiris

### A. Validasi Data Normal Asli (Generalisasi Jadwal Tanpa Data Anomali Sintetis)
| Metode | Holdout Minggu-1 → Minggu-2 (`168 jam`) | FPR Holdout | 14-Fold Leave-One-Day-Out (`336 jam`) | FPR LODO |
| :--- | :---: | :---: | :---: | :---: |
| **WaterAIS V-Detector (Context 7D)** | `2 / 168` FP | **1,19%** | `0 / 336` FP | **0,00%** |
| **Per-Slot AIS (Slot Terisolasi N=1..2)** | `33 / 168` FP | 19,64% | `60 / 336` FP | 17,86% |
| **Z-Score 1-Jam (Fallback N=1)** | `60 / 168` FP | 35,71% | `94 / 336` FP | 27,98% |

### B. Benchmark Komparatif (30 Skenario Uji: 12 Normal Variatif + 18 Anomali 6 Kelas)
| Metode | Accuracy | Precision | Recall | F1-Score | FPR | FNR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Z-Score 15m (JUTISI, N=2)** | 80,00% | 75,00% | 100,00% | 85,71% | 50,00% | 0,00% |
| **Z-Score 1h (Slot N=2)** | 90,00% | 94,12% | 88,89% | 91,43% | 8,33% | 11,11% |
| **Per-Slot AIS (Slot N=2, 20 seed)** | 99,83% ± 0,73% | 99,74% | 100,00% | 99,86% | 0,42% | 0,00% |
| **WaterAIS V-Detector (Context 7D, 20 seed)** | **98,67% ± 1,94%** | **100,00%** | **97,78%** | **98,85% ± 1,69%** | **0,00%** | 2,22% |

> **Catatan:** Pada `seed=42`, WaterAIS V-Detector mencapai **30/30 (100,0%)**. Grafik lengkap tersimpan di `results/figures/` (`fig1` s.d. `fig5`).

---

## 4. Cara Menjalankan Aplikasi

Pastikan Python 3.9+ dan dependensi (`numpy`, `scipy`, `matplotlib`, `pytest`) sudah terpasang:
```bash
cd d:\biocomputing-7\ais-problem
pip install -r requirements.txt
```

### 1) Dashboard Web Interaktif (Direkomendasikan)
Menjalankan server HTTP lokal beserta REST API Python dan langsung membuka browser:
```bash
python run_web.py
```
Atau klik ganda **`run_web.bat`** di Windows Explorer.
Server otomatis memilih port bebas mulai `http://127.0.0.1:8050/` dan menyediakan 5 tab interaktif:
1. **Simulasi Harian & Agregator 1-Jam**: putar 24 jam, injeksi skenario kebocoran/pergeseran jam, kontrol `Mode: AUTO/MANUAL` & `Solenoid Valve`.
2. **Lab Uji Mandiri (`4×15 Menit`)**: ketik angka aliran 15-menitan sendiri dan bandingkan putusan 4 metode sekaligus.
3. **Data 2 Minggu & Mekanisme N=2**: heatmap interaktif `14 × 24` yang dapat diklik untuk membandingkan sampel Minggu-1 vs Minggu-2.
4. **Ruang Imunologis & V-Detectors**: kurva *Thymus Censorship* & proyeksi 2D Self vs V-Detectors yang dapat dilatih ulang secara *real-time*.
5. **Benchmark AIS vs Z-Score**: tabel & grafik evaluasi lengkap.

### 2) Aplikasi Desktop Tkinter GUI
```bash
python app_tkinter.py
```
Atau klik ganda **`run_tkinter.bat`**.

### 3) Menjalankan Ulang Eksperimen & Grafik Publikasi
```bash
python run_experiments.py
python analyze_results.py
```
Output JSON dan 5 grafik PNG resolusi tinggi disimpan di `results/` dan `results/figures/`.

### 4) Pengujian Otomatis (`pytest`)
```bash
pytest -v test_ais.py
```

---

## 5. Struktur Berkas

```text
ais-problem/
├── Model-dataset-2 weeks.txt                # Dataset asli 1.344 interval (14 hari x 96)
├── Jurnal-Deteksi_Anomali_Air_IoT_ZSkor_JUTISI.pdf
├── dataset_loader.py                        # Parser dataset 15-menit & metadata mingguan
├── aggregator.py                            # Agregator 1-jam (tumbling/sliding) & vektor antigen 7D
├── ais_detector.py                          # WaterAISVDetector (7D) & PerSlotNSA (168x2)
├── zscore_baseline.py                       # Baseline Z-Score 15m (JUTISI) & 1h
├── scenarios.py                             # 30 skenario uji berlabel (12 normal + 18 anomali)
├── run_experiments.py                       # Holdout W1->W2, 14-Fold LODO, 20-seed & sensitivitas
├── analyze_results.py                       # Generator 5 grafik PNG & benchmark_summary.md
├── run_web.py / run_web.bat                 # Server Web + REST API JSON lokal
├── web/ (index.html, style.css, app.js)     # Antarmuka Dashboard Web interaktif
├── app_tkinter.py / run_tkinter.bat         # Aplikasi Desktop Tkinter + Matplotlib
├── test_ais.py                              # Suite pengujian pytest (5 test)
├── requirements.txt                         # Daftar dependensi Python
├── results/                                 # JSON metrik, benchmark_summary.md, & figures/*.png
└── README.md
```

