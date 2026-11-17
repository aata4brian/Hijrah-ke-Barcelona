# Reproduksi kandidat rilis equal5

## Jalur utama: data resmi ke CSV yang sama

Gunakan Python **3.12.10**, Windows x64, dan environment terpisah. Daftar paket
berasal dari environment yang benar-benar dipakai, bukan hasil perkiraan versi.
Wheel PyTorch CUDA 12.1 memerlukan indeks tambahan:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements-release.txt
python -m riset_v4.release --data-dir "D:\MTAH\data" --output "outputs\equal5\Hijrah ke Barcelona.csv"
python evaluation/validate_submission.py "outputs\equal5\Hijrah ke Barcelona.csv"
Get-FileHash "outputs\equal5\Hijrah ke Barcelona.csv" -Algorithm SHA256
```

`--data-dir` menunjuk folder induk tactical/broadcast milik peserta berwenang.
Perintah hanya memerlukan metadata dan raw tracking tactical; tidak perlu
mengunduh data ke repo. `--device auto` memakai CUDA bila tersedia, selain itu
CPU. Kesamaan byte telah dibuktikan pada environment rilis; lintas CPU/GPU,
driver, dan versi paket belum dijamin. Jangan mengubah decoder ketika hasil
berbeda: hash mismatch sengaja menghentikan promosi output.

Hash CSV rilis yang diwajibkan:

```text
a38b30c361cacb08dcf3d60173eb137ff28f1517d8d463dd03ad666641599ae7
```

File mempunyai 199.319 baris data berurutan, frame 0–199318, ditambah header
`frame,label`. Jalur rilis memeriksa hash model sebelum deserialisasi, jumlah
fitur, rentang probabilitas, kontrak CSV, dan hash akhir. File tujuan yang sudah
ada tidak ditimpa. Bila hash akhir berbeda, `.pending` dipertahankan untuk
inspeksi dan tidak dipromosikan menjadi hasil yang dinyatakan benar.

Catatan verifikasi raw-to-CSV ada di `hasil/rilis/reproduction.json`. File
verification yang ditulis entrypoint berisi hash input, tidak berisi data input.
Rilis tidak memerlukan sidecar fitur/probabilitas target, label, cue, maupun
path absolut pengembang.

## Pengujian terarah

```powershell
python -m unittest riset_v4.test_release riset_v4.test_artifact_checks riset_v4.test_fusion_features riset_v4.test_fusion_artifacts -v
node viewer/test_dual_viewer.cjs
```

Tes dengan data eksperimen lokal yang tidak ikut repo dapat berstatus skipped.
Jangan menganggap skipped sebagai pengukuran ulang. Pengujian GPU luas sebelumnya
pernah mengalami error backend; tes terarah atau reproduksi inferensi yang lulus
tidak membuktikan seluruh suite training GPU sehat. Status aktual tes disimpan
terpisah dalam catatan rilis, tidak diubah menjadi klaim semua tes lulus.

## Mengulang pelatihan bukan sekadar inferensi

Data eksternal diambil dengan `python -m riset_v4.fetch_public --output PATH`,
atau unduh XML dari record primer IDSSE, lalu gunakan parser
`night_tracking.save_external_cache`. Jangan masukkan label pertandingan lomba.
V4.4 dijalankan melalui `riset_v4.run`; deep refinement melalui
`riset_v4.local_deep`; pemilihan fusi melalui `riset_v4.fusion_experiment`.
Gunakan `--help` dan manifest per fold untuk parameter run yang sebenarnya.

Source dan metadata eksperimen disediakan untuk mengaudit metode. **Rilis ini
menjamin sasaran reproduksi inferensi dari bobot beku, bukan klaim pelatihan
ulang akan menghasilkan binary identik pada semua mesin.** Pelatihan ulang
memerlukan data eksternal dan checkpoint pendahulu yang sesuai; jangan menukar
warm start secara diam-diam. Bobot rilis tidak diperbarui otomatis oleh benchmark.

## Viewer dan menu lama

Viewer dua CSV hanya membaca video/CSV lokal yang dipilih pengguna. Tidak ada
video tertanam di halaman. Uji Node memeriksa logika viewer, bukan bukti playback
nyata di seluruh browser. Menu `eksperimen/run.py` 1–15 dipertahankan sebagai
arsip baseline; nomor 7 **bukan** perintah regenerasi kandidat equal5 ini.
