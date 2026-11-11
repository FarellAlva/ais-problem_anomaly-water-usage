# Ringkasan Hasil Eksperimen Deteksi Anomali Air Berbasis AIS

## 1. Validasi Data Normal Asli (Holdout Minggu-1 → Minggu-2 & 14-Fold LODO CV)
| Metode | Holdout W1→W2 (168 jam) | FPR Holdout | 14-Fold LODO CV (336 jam) | FPR LODO |
| :--- | :---: | :---: | :---: | :---: |
| **WaterAIS V-Detector (Context 7D)** | 2/168 FP | **1.19%** | 0/336 FP | **0.00%** |
| **Per-Slot AIS (Slot Terisolasi)** | 33/168 FP | 19.64% | 60/336 FP | 17.86% |
| **Z-Score 1-Jam (Fallback N=1)** | 60/168 FP | 35.71% | 94/336 FP | 27.98% |

Waktu latih rata-rata V-Detector Holdout: `1.221s`.

## 2. Perbandingan Komparatif (30 Skenario Uji: 12 Normal Variatif + 18 Anomali)

| Metode | Accuracy | Precision | Recall | F1-Score | FPR | FNR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Z-Score 15m (JUTISI, N=2)** | 80.00% | 75.00% | 100.00% | 85.71% | 50.00% | 0.00% |
| **Z-Score 1h (Slot N=2)** | 90.00% | 94.12% | 88.89% | 91.43% | 8.33% | 11.11% |
| **Per-Slot AIS (Slot N=2, 20 seed)** | 99.83% ± 0.73% | 99.74% ± 1.15% | 100.00% ± 0.00% | 99.86% ± 0.59% | 0.42% ± 1.82% | 0.00% ± 0.00% |
| **WaterAIS V-Detector (Context 7D, 20 seed)** | 98.67% ± 1.94% | 100.00% ± 0.00% | 97.78% ± 3.24% | 98.85% ± 1.69% | 0.00% ± 0.00% | 2.22% ± 3.24% |
