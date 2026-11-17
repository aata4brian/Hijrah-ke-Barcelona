# Hasil fusi fitur V4.6

**375 baris uji nyata**, 5 fold, 7 laga unik.
Known controlled retest, bukan holdout baru. Fold/skenario bukan sampel independen.

| Metode | SpellF1 | MacroF1 |
| --- | ---: | ---: |
| calibration_selected_v46 | 0.648356 | 0.910764 |
| control_old_features | 0.648356 | 0.910764 |
| expanded_tracking_features | 0.648356 | 0.910764 |
| frozen_deep_v45 | 0.647762 | 0.910062 |
| frozen_ensemble_deep_v45 | 0.644764 | 0.912702 |

## Perbandingan berpasangan

| Kandidat vs pembanding | Delta SpellF1 per laga | Naik/turun/seri | Terburuk |
| --- | ---: | --- | ---: |
| expanded_tracking_features vs control_old_features | +0.000000 | 0/0/75 | +0.000000 |
| calibration_selected_v46 vs frozen_deep_v45 | +0.000594 | 21/8/46 | -0.012556 |
| calibration_selected_v46 vs frozen_ensemble_deep_v45 | +0.003592 | 43/32/0 | -0.034542 |

## Pilihan sebelum test

- Fold 0, calibration J03WN1: specialist=old, tree_weight=0.25, specialist_weight=0.0.
- Fold 1, calibration J03WOH: specialist=old, tree_weight=0.0, specialist_weight=0.0.
- Fold 2, calibration J03WOY: specialist=old, tree_weight=0.0, specialist_weight=0.0.
- Fold 3, calibration J03WPY: specialist=old, tree_weight=0.25, specialist_weight=0.0.
- Fold 4, calibration J03WMX: specialist=old, tree_weight=0.0, specialist_weight=0.0.

## Interpretasi dan batas

- Kontrol 203 fitur dan expanded 281 fitur memakai sampel/anggaran pohon identik.
- Deep V4.5 dan ensemble deep adalah baseline beku; skor uji direproduksi hingga toleransi 1e-12.
- Pemilihan suhu dan campuran hanya dari calibration. Tidak memilih fold dari test.
- Jangan menyebut campuran paling baik hanya karena ditambah fitur; baca delta terhadap kontrol.
- Audio/flag telah tersedia sebagai sidecar, tetapi bobotnya belum dipelajari: data eksternal sejajar tidak tersedia.
- Tidak ada training deep baru, pembacaan GT target, promosi otomatis, atau perubahan submission.
- Ringkasan memakai bobot skenario lalu rerata laga. Pengulangan dua retest bukan laga baru.
- Precision/recall peluit/flag dan SpellF1 target tidak diukur.
