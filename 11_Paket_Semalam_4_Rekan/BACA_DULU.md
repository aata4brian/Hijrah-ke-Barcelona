# Empat rekan, empat komputer, empat eksperimen semalam

Ini paket kerja untuk **empat rekan di luar integrator**. Masing-masing
menjalankan tugas berbeda di komputernya sendiri; bukan empat proses berat
di laptop integrator. Integrator tetap memegang kandidat V4.4 beku.

| Rekan | Launcher | Eksperimen | Perangkat |
|---|---|---|---|
| 1 | `REKAN_1_DECODER.bat` | Pencarian bobot ensemble dan biaya batas; audit GT publik hanya untuk satu kandidat beku | CPU bisa; CUDA mempercepat inferensi |
| 2 | `REKAN_2_TCN.bat` | TCN 5 vs 20 epoch, seed dan data sama | NVIDIA CUDA disarankan; jangan dipindah ke CPU diam-diam |
| 3 | `REKAN_3_POHON.bat` | Pohon standar vs kapasitas lebih besar, data sama | CPU, RAM 16 GB disarankan |
| 4 | `REKAN_4_GAP.bat` | Pohon standar: tiga vs lima skenario augmentasi, kapasitas sama | CPU, RAM 16 GB disarankan |

Setiap komputer mengantrekan fold secara berurutan, menyimpan checkpoint,
CSV skor, manifest/hash, dan log. Delapan jam membatasi peluncuran job baru,
bukan janji selesai semua fold atau penghentian paksa training.
Paket tidak mengunggah data, tidak mengirim submission, dan tidak otomatis
mengganti kandidat. Target SpellF1 >0,9 adalah sasaran, bukan hasil terukur.

## Yang dibagikan

**Copy seluruh folder ini** ke masing-masing komputer, jangan launcher saja.
Payload berisi source, tujuh cache IDSSE publik (~211 MiB), checkpoint baseline
lima fold, aturan, dan scorer. Tidak memerlukan video atau tracking BAR-ATH.
GT publik hanya di `audit_publik`, dipakai sekali oleh rekan 1 dengan satu CSV
beku yang hash-nya dikunci. Jangan masukkan GT itu ke training/tuning.
Tidak perlu membuka folder `01_Submission` maupun `10_Kandidat_Final_V44`.

## Persiapan Windows

1. Pasang Python **3.12 x64**. Taruh paket di drive lokal, bukan folder sinkronisasi aktif.
2. Rekan 2 adalah pemilik NVIDIA CUDA; tiga rekan lain boleh memakai CPU.
3. Buka PowerShell di folder paket. Instal environment lokal **sekali**:

```powershell
# Rekan 1, 3, 4:
powershell -ExecutionPolicy Bypass -File .\PASANG_ENV.ps1 -Device cpu
# Hanya rekan 2 (NVIDIA):
powershell -ExecutionPolicy Bypass -File .\PASANG_ENV.ps1 -Device cuda
```

Unduhan Torch dapat besar; lakukan sebelum malam. Installer tidak memasang
paket global. Jika instalasi terputus, environment parsial tidak dihapus;
lanjutkan perintah pip dari `PASANG_ENV.ps1` menggunakan `.venv/Scripts/python.exe`.
Instalasi baru di komputer rekan **belum** dianggap lulus sebelum tes preflight.

4. Jalankan pemeriksaan sesuai nomor tugas, misalnya:

```powershell
powershell -ExecutionPolicy Bypass -File .\JALANKAN.ps1 -Rekan 3 -CekSaja
```

Preflight memeriksa hash payload, tes kecil, serta CUDA untuk rekan 2.
Tidak meluncurkan training panjang atau membaca GT target pada mode ini.
Rekomendasi: RAM 16 GB; sediakan **25 GiB kosong setelah instalasi** untuk
rekan 2/3 dan **35 GiB** untuk rekan 4. Rekan 1 sebaiknya 8 GiB kosong.
Runner menghitung kebutuhan konservatif tiap training dan menolak bila tidak
cukup; tidak menghapus bukti lama. Perkiraan ini bukan hasil pengukuran peak RAM.

## Mulai malam

**Masing-masing hanya menjalankan satu file, sesuai nomor tugas:**

| Nomor | File yang di-double-click | Pengaturan otomatis |
|---|---|---|
| 1 | `REKAN_1_DECODER.bat` | Audit satu CSV beku, lalu 192 kandidat decoder/mixture per fold |
| 2 | `REKAN_2_TCN.bat` | Lanjutkan checkpoint epoch 5 sampai 20, 120 langkah/epoch, seed 331 |
| 3 | `REKAN_3_POHON.bat` | Tree 90 vs 240 iterasi; 25 vs 41 leaf, data tiga skenario sama |
| 4 | `REKAN_4_GAP.bat` | Tree 90 iterasi tetap; tambah short_gaps dan long_gaps pada training |

Urutan fold semua tugas: **0, 2, 1, 3, 4**. Masing-masing fold: tiga laga
latih, satu kalibrasi, tiga uji terpisah; lima skenario uji dengan seed **432**.
Tujuh laga ini sudah dikenal: controlled retest, bukan holdout baru.
Semua jalur menyertakan baseline beku yang dinilai ulang secara berpasangan.
TCN dan tree tidak boleh dipilih berdasarkan skor GT lima menit atau laga uji.

Colok charger, matikan sleep otomatis lewat pengaturan komputer, jangan tutup
lid laptop. Runner mencoba mencegah idle sleep Windows selama antrean aktif,
tetapi tidak dapat mencegah restart OS, mati listrik, atau proteksi termal.
Jangan jalankan dua tugas berat dalam satu komputer. Jangan ubah kode saat run.

Untuk anggaran lain: `JALANKAN.ps1 -Rekan 3 -Jam 10`. Delapan jam dihitung
ulang setiap launcher dijalankan. Job yang telah mulai boleh melewati anggaran.
Lama selesai semua fold **belum diukur pada komputer rekan**, tidak dijanjikan.

## Status, berhenti, dan resume

- `STATUS.ps1` menampilkan `queue_status.json`; log rinci di
  `09_Riset_V4/outputs/queue_rekan_N_semalam_20261116/fold_K.log`.
- Hentikan halus: `STOP_SETELAH_JOB.ps1 -Rekan N`. Fold aktif diselesaikan,
  lalu tidak ada fold baru. **Jangan paksa tutup** jika masih bisa menunggu.
- Jika terputus, jalankan launcher yang sama; checkpoint epoch/model/skenario
  dilanjutkan. Fold lengkap diverifikasi hash-nya, tidak dilatih ulang.
- Untuk lanjut setelah stop, hapus **hanya file `STOP_AFTER_JOB`** pada antrean
  sendiri setelah memastikan proses berhenti. Jangan hapus folder output.
- Konfigurasi/kode berubah harus memakai sesi baru (`-Sesi nama_baru`).
- Kegagalan, disk tidak cukup, dan hasil parsial tetap tercatat; bukan skor nol.

## Pagi: kirim hasil, bukan screenshot

Setelah antrean berhenti, jalankan:

```powershell
powershell -ExecutionPolicy Bypass -File .\KUMPULKAN.ps1 -Rekan 3
```

Ganti angka sesuai tugas. Kirim **`HASIL_REKAN_N_....zip`** lokal yang dicetak
ke integrator. Arsip berisi CSV rinci, ringkasan, kandidat model, manifest/hash,
dan log; tidak berisi environment, GT target, video, atau cache fitur besar.
Checkpoint training tetap di komputer rekan agar bisa resume. Hasil parsial
boleh dikirim, dengan status parsial. Jangan edit angka CSV secara manual.
Jika melanjutkan run setelah mengirim snapshot parsial, kirim ke integrator
sebagai snapshot baru untuk diperiksa, bukan menimpa hasil lama secara paksa.

Integrator mengikuti `UNTUK_INTEGRATOR.md`; hasil masuk CSV pusat yang sama.
Perubahan kandidat final tetap keputusan eksplisit setelah memeriksa pasangan
fold/laga/skenario lengkap, bukan otomatis mengambil angka tertinggi.

## Batas hasil

Tes sintetis membuktikan jalur program, **bukan peningkatan SpellF1 nyata**.
Bandingkan pasangan baseline-varian dalam run yang sama; tampilkan penurunan
original, gap, dan per laga juga. Oracle di file diagnostik bukan model deployable.
Skor audit publik bukan ranking dan tidak dipakai memilih kandidat.
Tidak ada jaminan target privat SpellF1 >0,9. Lihat `VERIFIKASI.md` untuk
yang benar-benar diuji dan yang masih perlu diverifikasi di komputer rekan.
