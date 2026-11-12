# Ringkasan Hasil Eksperimen Deteksi Anomali Air Berbasis AIS

## 1. Validasi Data Normal Asli & Ablasi Internal AIS
| Arsitektur AIS | Holdout W1→W2 (168 jam) | FPR Holdout | 14-Fold LODO CV (336 jam) | FPR LODO |
| :--- | :---: | :---: | :---: | :---: |
| **WaterAIS V-Detector (Context 7D Terpadu)** | 2/168 FP | **1.19%** | 0/336 FP | **0.00%** |
| **Per-Slot AIS (Slot Terisolasi N=1..2)** | 33/168 FP | 19.64% | 60/336 FP | 17.86% |

Waktu latih rata-rata V-Detector Holdout: `1.620s`.

## 2. Evaluasi Keseluruhan WaterAIS (30 Skenario Uji x 20 Seed)
| Model AIS | Accuracy | Precision | Recall | F1-Score | FPR | FNR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **WaterAIS V-Detector (Context 7D)** | 98.67% ± 1.94% | 100.00% ± 0.00% | 97.78% ± 3.24% | 98.85% ± 1.69% | 0.00% ± 0.00% | 2.22% ± 3.24% |
| **Per-Slot AIS (Ablasi 168×2)** | 99.83% ± 0.73% | 99.74% ± 1.15% | 100.00% ± 0.00% | 99.86% ± 0.59% | 0.42% ± 1.82% | 0.00% ± 0.00% |

## 3. Performa WaterAIS per Kategori Skenario
| Kategori | Jumlah Skenario | Ketepatan Rata-rata (20 Seed) |
| :--- | :---: | :---: |
| Pola Normal Variatif (12 Skenario) | 12 | 100.00% ± 0.00% |
| Kebocoran Halus Kontinyu (3) | 3 | 100.00% ± 0.00% |
| Kebocoran Jam Malam (3) | 3 | 100.00% ± 0.00% |
| Keran Lupa Ditutup Malam (3) | 3 | 86.67% ± 19.44% |
| Lonjakan Jam Sepi (3) | 3 | 100.00% ± 0.00% |
| Pipa Pecah / Ekstrem (3) | 3 | 100.00% ± 0.00% |
| Banjir Aliran Kontinyu (3) | 3 | 100.00% ± 0.00% |
