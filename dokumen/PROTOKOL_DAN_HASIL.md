# Protokol eksperimen dan pembacaan hasil

## Unit pemisahan adalah pertandingan

Tujuh laga IDSSE yang dipakai: J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ,
dan J03WR9. Lima laga pertama membentuk rotasi development. Pada fold f, satu
laga menjadi test development, laga berikutnya kalibrasi, dan tiga lainnya latih.
J03WQQ dan J03WR9 adalah **known retest**, bukan holdout yang masih baru.

| Fold | Latih | Kalibrasi | Test development | Known retest |
|---|---|---|---|---|
| 0 | WOH, WOY, WPY | WN1 | WMX | WQQ, WR9 |
| 1 | WMX, WOY, WPY | WOH | WN1 | WQQ, WR9 |
| 2 | WMX, WN1, WPY | WOY | WOH | WQQ, WR9 |
| 3 | WMX, WN1, WOH | WPY | WOY | WQQ, WR9 |
| 4 | WN1, WOH, WOY | WMX | WPY | WQQ, WR9 |

Singkatan tabel diberi awalan J03. Split dijalankan oleh `data.split()` dan
divalidasi agar tidak tumpang tindih dalam satu fold. Dua retest sudah berulang
kali diamati pada putaran sebelumnya; menyebutnya validasi independen baru
akan melebih-lebihkan bukti. Mengacak frame akan membocorkan konteks temporal
antarpartisi, sehingga tidak digunakan.

## Gangguan dan bobot

Lima skenario adalah original, short_gaps, long_gaps, camera_gaps, dan drift.
Bobot agregasi skenario masing-masing 0,10; 0,15; 0,30; 0,30; 0,15. Kode gangguan
memakai generator acak ber-seed; ini simulasi, bukan rekonstruksi statistik
missingness kamera pertandingan lomba. Hasil di setiap laga dirata-ratakan
dengan bobot skenario, lalu diagregasi menurut protokol laporan.

375 baris V4.6 berasal dari 5 fold × 3 laga uji × 5 skenario × 5 metode. Baris
bukan 375 pertandingan independen. Pengulangan retest dan gangguan pada sumber
laga yang sama tidak boleh diperlakukan sebagai ukuran sampel baru. Tidak
ada interval kepercayaan atau uji signifikansi yang kami klaim dari tabel ini.

## Hasil terukur

Sumber angka: `hasil/v46/summary.csv`, `evaluation_rows.csv`, `by_match.csv`,
`paired_deltas.csv`, dan `verification.json`.

| Metode evaluasi per-fold | SpellF1 | MacroF1 |
|---|---:|---:|
| Calibration-selected V4.6 | 0,648356 | 0,910764 |
| Control old 203 features | 0,648356 | 0,910764 |
| Expanded tracking features | 0,648356 | 0,910764 |
| Frozen deep V4.5 | 0,647762 | 0,910062 |
| Frozen ensemble deep V4.5 | 0,644764 | 0,912702 |

Nilai presisi penuh ada di CSV, bukan angka dibulatkan di tabel ini. Ringkasan
calibration-selected SpellF1 = 0.6483558219452347; frozen deep = 0.6477615693349609.
150 baris baseline beku direproduksi sampai toleransi absolut 1e-12 dalam
verifikasi laporan terdahulu. Itu verifikasi numerik baseline, bukan pengukuran
skor privat target atau jaminan generalisasi.

## Ablation dan hasil negatif

Control 203 dan expanded 281 memakai sampel serta anggaran pohon yang sama.
Keduanya akhirnya memilih specialist weight nol pada semua fold. Kesamaan
hasil karena jalur specialist tidak aktif; **bukan bukti bahwa 281 fitur
tidak mengandung informasi**. Klaim yang didukung lebih sempit: protokol ini
tidak menemukan manfaat yang cukup untuk mengaktifkan specialist.

Dibanding deep V4.5, rerata delta SpellF1 calibration-selected sekitar +0,000594.
Dari 75 pasangan fold/laga/skenario, 21 naik, 8 turun, 46 seri; delta terburuk
-0,012556. Dibanding frozen ensemble deep, 43 naik dan 32 turun, delta terburuk
-0,034542. Ini trade-off, bukan perbaikan seragam pada setiap kasus.

MacroF1 dan SpellF1 dapat bergerak berbeda. Frame panjang dapat mendominasi
akurasi status, sedangkan fragmentasi singkat menambah atau menghilangkan spell.
Kami memakai scorer panitia, tidak membuat metrik pengganti agar hasil terlihat
lebih baik. Interpretasi tepat metrik mengikuti `EVALUATION.md`.

## Apa yang belum diukur

- Rata-rata lima fold yang dipakai CSV rilis belum dievaluasi sebagai ensemble
  pada holdout eksternal yang sama sekali tidak ikut melatih salah satu komponennya.
- Decoder rilis berbeda dari decoder per-fold evaluasi, dan fitur target mengikuti
  kontrak float16→float32. Tabel ini bukan estimasi tak bias skor CSV tersebut.
- Skor privat pertandingan lomba tidak diketahui. Tidak ada leaderboard publik
  yang memberi bukti peringkat atau angka target.
- Precision/recall peluit dan bendera, manfaat video recovery, serta keandalan
  lintas kamera belum dibuktikan untuk konfigurasi rilis.
- Unit test sintetis membuktikan kontrak kode tertentu, bukan kualitas taktik
  atau kepatuhan seluruh riwayat eksperimen secara otomatis.

## Pekerjaan lanjutan

Prioritas ilmiah adalah laga baru yang belum digunakan, evaluasi ensemble yang
benar-benar terpisah, analisis ketidakpastian terkalibrasi, dan benchmark cue
dengan label dari pertandingan eksternal. Data target tidak boleh diubah menjadi
validation set melalui catatan manusia. Klaim negatif dan keterbatasan tetap
disimpan agar keputusan berikutnya tidak sekadar mengulang pencarian yang sama.
