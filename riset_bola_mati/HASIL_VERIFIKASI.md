# Hasil verifikasi detektor peluit dan kandidat bendera

Tanggal: **17 November 2026**. Versi: **0.1.0**, eksperimen lokal.

## Kesimpulan

Program dapat membaca video asli, memindai audio penuh, mencari kandidat
bendera, dan menggabungkan bukti tanpa mengubah submission possession.
**Ini belum merupakan detektor dead/live yang tervalidasi akurasinya.**
Tidak ada pengukuran precision, recall, F1, atau SpellF1; nilainya tetap
`null` di manifest dan kosong di ledger performa, bukan nol.

## Hasil yang benar-benar dijalankan

| Pemeriksaan | Cakupan | Hasil | Waktu proses |
| --- | --- | --- | --- |
| Tes sintetis dan integrasi | 28 tes | 28 lulus, 0 gagal, 0 error, 0 skip | 7,018 detik |
| Audio penuh | Video 0–7972,76 detik; audio sampai 7972,693333 detik | 399 chunk, 127.563.093 sampel stereo pada 16 kHz; **0 kandidat peluit** | 28,832 detik |
| Audio + visual | 450–456 detik, 2 sampel/detik | 12 sampel visual; 8 observasi `LOWERED_CANDIDATE` pada 7 sampel; tidak ada raised/supported | 12,662 detik |
| Fusi uji visual | 12 titik waktu | Seluruhnya `UNKNOWN`, tidak ada bukti stoppage yang cukup | Termasuk waktu di atas |

Delapan observasi bukan delapan bendera unik atau delapan deteksi benar.
Runtime satu kali pengukuran pada laptop integrator, bukan rata-rata benchmark
berulang. Jalur audio jauh lebih ringan daripada visual; visual uji ini
memerlukan sekitar 2,11 kali durasi cuplikan, **belum real-time**. Belum
dijalankan pemindaian visual seluruh pertandingan.

Audio berakhir **0,066667 detik sebelum video**. Status scan penuh adalah
`complete` dengan `audio_status=processed_partial_coverage`; ekor tanpa audio
tidak dipadding menjadi silence. Seluruh 15.946 titik fusi scan audio tetap
`UNKNOWN`; ini abstention sistem, bukan skor klasifikasi.

## Mengapa nol kandidat peluit belum berarti berhasil

- Nol kandidat tidak membuktikan tidak ada peluit dan tidak mengukur recall.
- Pemeriksaan tanpa label pada audio 450–458 detik menghasilkan RMS per kanal
  sekitar **−55,08 dBFS**, sedangkan gate energi per jendela default
  `min_rms_db=-50`. Ini indikasi risiko sinyal lirih terlewat, bukan bukti
  penyebab seluruh hasil nol: RMS cuplikan berbeda dari RMS jendela STFT.
- Threshold tidak diubah untuk memaksakan deteksi pada target. Perubahan
  sebaiknya dikalibrasi memakai rekaman eksternal berlabel, dengan evaluasi
  terpisah per pertandingan dan sensitivitas terhadap level audio.
- Sinyal tonal tidak menetapkan identitas wasit; peluit juga bisa berarti
  restart, berasal dari penonton, atau bukan peluit sama sekali.

## Keandalan yang diperiksa

Tes mencakup stereo antiphase, silence/noise, sinyal tonal sintetis, batas
chunk, ketepatan seek AAC, audio lebih pendek dari video, alignment decode
frame, homography, geometri warna/pose, occlusion, perpindahan asosiasi,
fusi konservatif, CLI, serta penolakan menimpa output lama.

Parser PowerShell `JALANKAN.ps1` lulus pemeriksaan sintaks. Launcher
double-click end-to-end dan instalasi bersih di komputer lain **belum diuji**.
Model ONNX terverifikasi hash dan dapat diinferensikan di environment lokal.
Hash source runtime dan CSV output kedua run final cocok saat audit akhir.

Paket 107 file rekan cocok dengan manifest aslinya. SHA256 submission V4.4
tetap `2c0032f0f17009953eb9a7cec2786b561ad412a579b82dd40b06a3a67427b9bc`.
Tidak ada ekspor ulang submission, training berat, atau upload detektor baru.

## Bukti dan cara mencoba

Semua path berikut relatif terhadap `12_Deteksi_Bola_Mati/`:

- `outputs/audio_full_verified_20261117/manifest.json`: runtime, coverage,
  konfigurasi, hash source/output, dan jumlah kandidat scan penuh.
- `outputs/visual_verified_20261117/`: `flags.csv`, `visual_samples.csv`,
  `deadball_evidence.csv`, `whistles.csv`, dan manifest uji visual final.
- `outputs/verification_20261117/tests.json` dan `tests.log`: hasil 28 tes.
- `outputs/verification_20261117/diagnostik_runtime.csv`: dua baris numerik
  runtime/coverage/abstention untuk ledger pusat, **bukan metrik akurasi**.

Klik `COBA_30_DETIK.bat` untuk mencoba 450–480 detik, atau dari PowerShell:

```powershell
.\JALANKAN.ps1 -Mode all -Mulai 450 -Durasi 30
.\JALANKAN.ps1 -Mode audio -SeluruhVideo
```

Output baru masuk folder unik di `outputs/`; hasil lama tidak ditimpa.
Model pose generik bukan classifier hakim garis. Label `SUPPORTED` hanya
dukungan persistensi geometri/fusi, bukan konfirmasi role atau probabilitas.
Bendera turun/tidak ada peluit tidak pernah otomatis dianggap bola hidup.

## Langkah berikutnya yang paling bernilai

1. Evaluasi audio pada laga eksternal berlabel: precision/recall event,
   false alarm per jam, galat onset, serta ketahanan terhadap volume rendah.
2. Evaluasi role/flag track dan tingkat abstention visual pada laga eksternal;
   bandingkan audio-only, visual-only, dan fusi sebelum integrasi possession.
3. Tentukan kebutuhan classifier role dan optimasi visual setelah diketahui
   jenis kegagalannya, bukan langsung training besar atau scan video dua jam.

Jangan melabel manual target BAR–ATH untuk tuning/evaluasi ini. Bobot model
berlisensi AGPL-3.0; review kewajiban lisensi sebelum distribusi/deployment.
