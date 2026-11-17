# Hijrah ke Barcelona

## Rilis pengumpulan: equal5 V4.6

Kandidat rilis adalah **calibration_fusion_v46_equal5**, bukan menu 7 di arsip
eksperimen. Raw tracking tactical dan metadata resmi diproses menjadi 203 fitur,
lima deep temporal model, campuran pohon terkalibrasi, dan satu decoder tetap.
Tidak ada koreksi label manual.

```powershell
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements-release.txt
python -m riset_v4.release --data-dir "D:\MTAH\data" --output "outputs\equal5\Hijrah ke Barcelona.csv"
python evaluation/validate_submission.py "outputs\equal5\Hijrah ke Barcelona.csv"
```

Python 3.12.10; gunakan environment virtual. Reproduksi dari raw ATD sudah
dibuktikan **byte-identik**, 199.319 frame, SHA256
`a38b30c361cacb08dcf3d60173eb137ff28f1517d8d463dd03ad666641599ae7`.
Bukti ada di [hasil/rilis](hasil/rilis/). Data berlisensi tetap di komputer
peserta; repo tidak menyediakan video, tracking target, atau cache fitur target.

Mulai dari [dokumen teknis singkat](Hijrah%20ke%20Barcelona.md),
[panduan reproduksi](dokumen/REPRODUKSI.md), dan
[indeks pembahasan akademik](dokumen/README.md).
Hasil eksternal V4.6 sekitar SpellF1 0,648356 adalah evaluasi **per fold** pada
tujuh laga yang sudah dikenal, bukan skor ensemble rilis atau skor privat lomba.
Expanded features/cue belum terbukti memberi manfaat dan tidak diaktifkan.

Catatan manual target sempat dibaca pada persiapan akhir, tetapi tidak dipakai
untuk mengubah, memilih, melatih, atau memvalidasi kandidat yang sudah beku.
Lihat [provenance](dokumen/DATA_DAN_PROVENANCE.md) dan
[batas kepatuhan](dokumen/KEPATUHAN.md). Review manusia bukan sumber label
pengumpulan, termasuk bila dokumen kerja lama pernah mengusulkannya.

Pekerjaan rekan dan penelitian terdahulu dipertahankan di branch
`team-dispatch-frame-review-20261116` dan `riset/v4-switch-contact-tcn-20261112`.
Cache branch rekan berasal dari IDSSE eksternal CC BY 4.0; lihat
`DATA_SUMBER_TIM.md` di branch tersebut. Tidak ada klaim bahwa seluruh
eksperimen arsip dipakai dalam submission atau telah diuji ulang pada rilis.

## Arsip menu eksperimen V1–V3

Bagian berikut dipertahankan sebagai catatan kerja sebelumnya. Penyebutan
"kandidat saat ini" pada bagian arsip tidak menggantikan entrypoint rilis di atas.

Ruang kerja tim untuk fase Ball Possession di More Than a Hack 2027. Setiap percobaan menghasilkan CSV `frame,label` dengan 199.319 baris untuk kamera tactical. Nomor **07** mereproduksi kandidat `flat/play_bias` yang sudah diverifikasi.

## Jalankan percobaan

1. Instal **Python 3.12** dengan Python Launcher di Windows. Ambil paket data dari tautan resmi panitia yang diterima anggota terdaftar.
2. Ekstrak paket dengan susunan `data/tactical/` dan `data/broadcast/`. Kalau data disimpan di folder lain, pilih folder induk `tactical/` ketika diminta.
3. Buka **`JALANKAN.bat`**, lalu ketik nomor **1–15**. Saat pertama dibuka, launcher memasang dependensi yang dipatok versinya.
4. Ambil `Hijrah ke Barcelona.csv` dan `run.json` dari subfolder baru di `outputs/`. Terminal akan memberi tahu apakah CSV lolos validator panitia.

Di terminal Windows, perintah yang sama bisa dijalankan dari root repo.

```powershell
.\eksperimen\JALANKAN.bat
.\eksperimen\JALANKAN.bat --method 7 --data-dir "D:\MTAH\data"
```

Untuk Python 3.12 dengan lingkungan virtual yang sudah aktif:

```bash
python -m pip install -r eksperimen/requirements-lock.txt
python eksperimen/run.py --list
python eksperimen/run.py --method 7 --data-dir /path/to/data
```

## Pembagian lima anggota

| Anggota | Nomor |
|---|---|
| Affan | 1, 6, 11 |
| Mosi | 2, 7, 12 |
| Khansa | 3, 8, 13 |
| Faiq | 4, 9, 14 |
| Brian | 5, 10, 15 |

Keempat baseline panitia, dua versi tim terdahulu, kandidat utama, dan delapan variasi model atau decoder tersedia dari menu yang sama. Rincian tiap nomor ada di [panduan eksperimen](eksperimen/README.md). Hasil eksternal yang sudah selesai ada di [hasil](hasil/README.md).

## Susunan repo

| Lokasi | Isi |
|---|---|
| [games.py](games.py) | Konstanta pertandingan dan pemetaan frame dari panitia |
| [starter/](starter/) | Contoh tracking, tagging, dan baseline panitia |
| [evaluation/](evaluation/) | Scorer dan validator panitia |
| [data/](data/) | Petunjuk lokasi paket resmi di komputer anggota |
| [eksperimen/](eksperimen/) | Menu 1–15, kode pipeline, dan bobot dari data publik IDSSE |
| [hasil/](hasil/) | Ringkasan dan tabel evaluasi pertandingan eksternal |
| [viewer/](viewer/) | Viewer HTML untuk memeriksa sinkronisasi video lokal dan prediksi |
| `outputs/` | CSV prediksi dan catatan run lokal |

Panduan resmi [CHALLENGE](CHALLENGE.md), [DATA](DATA.md), [EVALUATION](EVALUATION.md), [SUBMISSION](SUBMISSION.md), dan [RULES](RULES.md) disalin dari [repo panitia](https://github.com/metrica-sports/more-than-a-hack-2027/tree/f8f2629ff1e0985f81d7e59d8413678514ee1530).

Video, tracking, homography, dan tagging Barcelona vs Athletic berasal dari paket berlisensi panitia. Setiap anggota membuka berkas itu di komputernya sendiri. Hasil per frame untuk pengumpulan disimpan di `outputs/` dan diserahkan lewat formulir panitia. Lima menit label publik dipakai untuk pengecekan format, indeks, dan sinkronisasi. Pemilihan metode memakai pertandingan eksternal.

## Kandidat dan dokumen pengumpulan

Nomor 7 mereproduksi kandidat saat ini. [Dokumen teknis tim](Hijrah%20ke%20Barcelona.md) menjelaskan data latih, aturan, hasil, dan keterbatasan. Dokumen versi 1 disimpan sebagai arsip di dokumen/. Repo kode mengikuti susunan panitia; CSV pengumpulan tetap di keluaran lokal. Lima belas pilihan merupakan konfigurasi percobaan yang sudah dapat dijalankan, termasuk baseline pembanding. Untuk mengulang pelatihan, lihat [panduan pelatihan](eksperimen/PELATIHAN.md).

## Buka video dengan prediksi

Buka **BUKA_VIEWER.bat** untuk memuat video tactical, CSV prediksi, referensi publik, dan indikator tracking bola secara lokal. Video dibaca langsung dari folder data masing-masing anggota. Pilih folder lewat --data-dir, MTAH_DATA_DIR, atau konfigurasi pribadi .mtah-local.json. [Petunjuk viewer](viewer/README.md) menjelaskan pemilihan CSV dan cara membuka data dari Drive untuk desktop.
