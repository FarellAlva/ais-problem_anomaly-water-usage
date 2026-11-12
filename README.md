# WaterAIS — Deteksi Anomali Penggunaan Air Berbasis Artificial Immune System (AIS) & Agregator 1-Jam

Implementasi *from-scratch* **Artificial Immune System (AIS)** menggunakan **Context-Aware Real-Valued V-Detector Negative Selection** untuk mendeteksi anomali penggunaan air IoT pada **jendela agregasi 1 jam** (`4 × 15 menit`), dilengkapi dua aplikasi interaktif (**Web Dashboard** dan **Desktop GUI Tkinter**) serta overlay pembanding visual **Threshold AIS (Biru)** vs **Threshold Z-Score (Oranye)** pada grafik 24-jam.

- **Repositori Source Code**: https://github.com/FarellAlva/ais-problem_anomaly-water-usage
- **Repositori Sumber Dataset**: https://github.com/FarellAlva/Dataset-Sistem-Deteksi-Anomali-Penggunaan-Air-Berbasis-IoT-dengan-Pendekatan-Z-Score-Statistik-

---

## 1. Struktur Dataset 2 Minggu (`Model-dataset-2 weeks.txt`)

Dataset pengamatan laju aliran air normal berlangsung selama **14 hari (04 Mei 2025 – 17 Mei 2025)** dengan resolusi 15 menit:
- Total rekaman 15-menit: **`1.344`** slot (`287` aktif `> 0 L/min`, `1.057` bernilai `0.00 L/min`).
- Volume total 14 hari: **`3.493,35 Liter`** (rata-rata **`249,53 Liter/hari`**), laju puncak **`2,61 L/min`**.
- **Agregator 1-Jam (`Tumbling Window`)**: mengelompokkan setiap `4` slot berurutan menjadi **`336` blok jam** (`14 hari × 24 jam`).
- Karakteristik fisik penting dari 336 jam normal:
  - `0/4` slot aktif (diam): **112 jam (33,3%)**
  - `1/4` slot aktif: **163 jam (48,5%)**
  - `2/4` slot aktif: **59 jam (17,6%)**
  - `3/4` slot aktif: **2 jam (0,6%)**
  - **`4/4` slot aktif kontinyu**: **`0 jam (0,0%)`** — seluruh data normal selalu memiliki setidaknya satu interval jeda `0.00 L/min`, sehingga aliran `4/4` slot penuh menjadi ciri fisik kuat kebocoran pipa/keran.

---

## 2. Mekanisme WaterAIS pada Data 2 Minggu & Proyeksi Threshold

### A. Mengatasi Keterbatasan 2 Sampel Per Hari-Jam
Pada data berdurasi 2 minggu, setiap slot `(Hari, Jam)` tepat memiliki 2 pengamatan (*Minggu-1* dan *Minggu-2*). Bila tiap slot diisolasi terpisah (*Per-Slot AIS N=2*), model mudah memicu false alarm ketika penghuni menggeser jam mandi (`19,64%` FPR di Holdout W1→W2 dan `17,86%` FPR di 14-Fold LODO CV).

**WaterAIS V-Detector** menggabungkan seluruh **`336` blok jam** ke dalam **Self-Space 7 dimensi terpadu** menggunakan vektor fitur ternormalisasi:
$$\mathbf{x} = \Phi(\mathbf{q}, h, D) = \left[\frac{\ln(1+\bar{q})}{s_0},\; \frac{\ln(1+q_{\max})}{s_1},\; \frac{r_{\text{aktif}}}{s_2},\; \frac{\ln(1+q_{\min})}{s_3},\; \frac{\sin_h}{s_4},\; \frac{\cos_h}{s_5},\; \frac{\mathbb{I}_{\text{wknd}}}{s_6}\right]$$
dengan vektor bobot imunologis $\mathbf{W} = [1.40, 1.30, 0.85, 2.20, 3.20, 3.20, 0.25]$.

1. **Thymus Censorship (Negative Selection)**: Kandidat detektor yang berjarak Euclidean berbobot $\le r_{\text{self}}$ (`0.43`) dari sel *Self* manapun mengalami apoptosis. Kandidat yang lolos menjadi *Mature V-Detector* dengan radius variabel $r_k = \min_{\mathbf{s}_j \in S} d_w(\mathbf{c}_k, \mathbf{s}_j) - r_{\text{self}}$.
2. **Keputusan Klasifikasi Sejati (Ruang 7D)**: Blok 1-jam dinyatakan **ANOMALI** jika dan hanya jika berada dalam bola detektor ($\exists k: d_w(\mathbf{x}, \mathbf{c}_k) < r_k$), memicu diagnosis kategori anomali dan penutupan *Solenoid Valve* secara otomatis pada mode `AUTO`.
3. **Proyeksi Dual-Manifold pada Grafik 24-Jam (`ais_upper_lpm` vs `ais_leak_limit_lpm`)**:
   - **Kurva Biru Transparan (`ais_upper_lpm`)**: Memproyeksikan batas atas wajar pada irisan pola **intermiten 2/4 slot** ($\mathbf{q}_{\text{inter}}=[2\bar{q}, 2\bar{q}, 0, 0]$).
   - **Batas Kebocoran Kontinyu (`ais_leak_limit_lpm`)**: Memproyeksikan ambang pemicu detektor pada pola **aliran kontinyu 4/4 slot** ($\mathbf{q}_{\text{leak}}=[\bar{q}, \bar{q}, \bar{q}, \bar{q}]$), sekitar `0,04–0,22 L/min`. Inilah alasan kebocoran halus `0.35 × 4` tetap terdeteksi **`X` Merah** walaupun titik rata-ratanya (`0.35 L/min`) berada di bawah kurva intermiten biru.
   - **Overlay Oranye Transparan (`Z-Score UCL = mu + 2*sigma`)**: Disediakan opsional pada grafik 24-jam sebagai pembanding geometri ambang statistik mingguan.

---

## 3. Ringkasan Evaluasi Empiris (Pure AIS, 20 Seed)

| Arsitektur / Pengujian | Holdout W1→W2 (168 jam) | 14-Fold LODO CV (336 jam) | 30 Skenario Uji (Accuracy / Precision / Recall / F1) |
| :--- | :---: | :---: | :---: |
| **WaterAIS V-Detector (Context 7D)** | `2/168` (**1,19% FPR**) | `0/336` (**0,00% FPR**) | **98,67%** / **100,00%** / **97,78%** / **98,85%** (FPR **0,00%**) |
| **Per-Slot AIS (Ablasi 168×2 Terisolasi)** | `33/168` (19,64% FPR) | `60/336` (17,86% FPR) | 99,83% / 99,74% / 100,00% / 99,86% (rentan geser jadwal nyata) |

---

## 4. Cara Menjalankan Aplikasi

```bash
cd d:\biocomputing-7\ais-problem
pip install -r requirements.txt
```

### 1) Dashboard Web Interaktif
```bash
python run_web.py
```
Atau klik ganda **`run_web.bat`**. Menyediakan 4 tab interaktif:
1. **Simulasi Harian & Threshold AIS**: Pilih model hari mingguan gabungan (`Sunday–Saturday`), uji terhadap `Minggu-1` / `Minggu-2` atau skenario injeksi, dan nyalakan/matikan overlay transparan **Threshold AIS (Biru)** vs **Threshold Z-Score (Oranye)**.
2. **Data 2 Minggu & Penggabungan Self**: Heatmap `14 × 24` interaktif + inspektor sampel M1/M2 dan dukungan sirkadian AIS.
3. **Ruang Imunologis & V-Detectors**: Kurva *Thymus Censorship*, proyeksi 2D Self vs V-Detectors, dan formulasi ambang AIS 7D.
4. **Evaluasi & Validasi Kinerja AIS**: Evaluasi per kategori skenario, ablasi internal AIS, dan matriks 30 skenario uji.

### 2) Aplikasi Desktop Tkinter GUI
```bash
python app_tkinter.py
```
Atau klik ganda **`run_tkinter.bat`**.

### 3) Menjalankan Ulang Eksperimen, Grafik, Makalah Word, & Pengujian
```bash
python run_experiments.py
python analyze_results.py
python build_ais_paper_docx.py
pytest -v test_ais.py
```
