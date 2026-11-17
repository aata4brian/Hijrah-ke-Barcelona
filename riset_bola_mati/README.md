# Peluit + bendera hakim garis → bukti bola mati

**Implementasi eksperimental lokal. Tidak mengubah submission V4.4/V4.5.**
Ini dua detektor independen dan satu penggabung bukti, bukan sistem yang sudah
terbukti akurat menentukan seluruh fase dead/live. Tidak ada label manual,
GT privat, atau tuning terhadap GT lima menit BAR–ATH.

## Hasil verifikasi terbaru

Lihat **[HASIL_VERIFIKASI.md](HASIL_VERIFIKASI.md)** untuk bukti dan batasnya.
Pada 17 November 2026: **28 tes lulus**; scan audio 132,88 menit selesai
**28,83 detik** dengan **0 kandidat peluit**; uji audio+visual 6 detik selesai
**12,66 detik**, seluruh fusi `UNKNOWN`. Nol deteksi bukan bukti tidak ada
peluit. Akurasi belum diukur dan jalur visual belum real-time.

## Menjalankan pada laptop integrator

Klik **`COBA_30_DETIK.bat`**, atau buka PowerShell di folder ini:

```powershell
# Uji singkat: default salinan video lokal dan homography dari konfigurasi tim.
.\JALANKAN.ps1 -Mode all -Mulai 450 -Durasi 30
# Peluit saja: hemat CPU, tidak menjalankan model pose.
.\JALANKAN.ps1 -Mode audio -SeluruhVideo
# Visual saja, 2 sampel/detik; 5 Hz lebih peka terhadap gestur pendek, lebih mahal.
.\JALANKAN.ps1 -Mode flags -Mulai 450 -Durasi 60 -SampleHz 5
# Seluruh video: bisa lama; bukan default double-click.
.\JALANKAN.ps1 -Mode all -SeluruhVideo
```

Gunakan `-Video "path\video.mp4" -Homography "path\homography.csv"` untuk
menentukan sumber. Video tactical 1920×1080, 25 fps, 199.319 frame,
audio AAC stereo 48 kHz ditemukan pada salinan lokal integrator.
Google Drive harus tersedia offline bila membacanya langsung dari H:.

Program menolak folder output yang sudah ada. Hasil baru di `outputs/`.
`-Overlay` menyimpan video diagnosis **lokal**, dipadatkan waktunya menjadi
sampel yang dianalisis, **bukan** video berkecepatan asli untuk anotasi.

## Output dan maknanya

| File | Isi |
|---|---|
| `whistles.csv` | Awal/akhir kandidat peluit, frekuensi, kanal, skor heuristik |
| `flags.csv` | Asosiasi orang+bendera; kandidat/persistensi naik atau turun |
| `visual_samples.csv` | Semua frame yang diperiksa, termasuk abstain, kualitas H, jumlah tile |
| `deadball_evidence.csv` | Bukti gabungan, default sekitar 2 Hz, **bukan** CSV submission |
| `manifest.json` | Status run, konfigurasi, versi/hash, coverage, runtime, batas validasi |

- `RAISED_CANDIDATE` / `LOWERED_CANDIDATE`: pose + warna bendera + dekat
  touchline cocok, tetapi identitas sebagai asisten wasit belum terverifikasi.
- `RAISED_SUPPORTED` / `LOWERED_SUPPORTED`: geometri konsisten ≥0,6 detik dan
  ≥2 observasi. Ini **bukan** probabilitas terkalibrasi atau ground truth.
- `STOPPAGE_CANDIDATE`: peluit baru atau kandidat bendera naik.
- `STOPPAGE_SUPPORTED`: event peluit yang sudah selesai dan bendera naik
  persisten pada observasi yang sama. Masih perlu konteks permainan.
- `UNKNOWN`: bukti tidak cukup/kedaluwarsa/tidak terlihat. **Bukan LIVE**.

Timestamp detik dihitung dari awal file tactical; frame 0-based. Peluit memakai
interval awal inklusif/akhir eksklusif. Bukti peluit berlaku dua detik sesudah
event (setelah jeda penutupan minimal 96 ms), bukan menandai seluruh jeda. Flag tidak dibawa melewati occlusion.
Tidak ada estimasi restart atau pengisian label D ke submission otomatis.
Output adalah hasil analisis **offline**, bukan jaminan latensi streaming realtime.
Durasi stream audio dicatat terpisah: audio tactical ini lebih pendek sekitar
67 ms dari video. Ekor yang tidak memiliki audio tidak diisi silence palsu.

## Mengapa desain ini hemat

- Audio streaming per chunk 20 detik lewat **satu proses FFmpeg**, stereo tetap terpisah agar tidak saling
  menghapus. STFT 32 ms / hop 10 ms; gate tonalitas, energi, SNR, dan durasi.
  Sampel sisa dibawa tepat melintasi batas chunk. Output seeking menjaga
  alignment AAC; tidak melakukan seek ulang setiap chunk. Tidak menyimpan
  seluruh waveform atau spectrogram pertandingan. Kandidat hop disimpan
  hingga akhir scan, sehingga memori kandidat bertambah dengan durasi/banyaknya tone.
- Model pose nano 13,5 MB, ONNX melalui OpenCV CPU; **tanpa memasang Ultralytics**
  dan tanpa mengganggu environment riset lama. Empat thread, GPU tetap bebas.
- Visual memproses 2 Hz secara default, memakai warna kuning/oranye untuk
  proposal, proyeksi homography untuk touchline, maksimal empat tile native
  384×384 per sampel. Tidak menjalankan pose di semua frame full-resolution.
- Decoder tetap membaca frame antar-sampel secara berurutan; sampling menghemat
  inferensi, bukan menghilangkan biaya decode H.264. Tidak menjanjikan real-time.

## Instalasi di komputer lain

Python 3.12, FFmpeg/ffprobe pada PATH. Buat environment terpisah; jangan
menimpa environment training lama. Instalasi berikut **belum diuji bersih**:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_model.py
.\.venv\Scripts\python.exe run.py all --video "video.mp4" --homography "homography.csv" --start 450 --duration 30 --out "outputs\percobaan_1"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`setup_model.py` hanya mengunduh satu model dengan revisi dan SHA256 dikunci.
Source: https://huggingface.co/Xenova/yolov8n-pose . Model generik COCO pose,
**bukan model khusus bendera/wasit**. Lisensi model **AGPL-3.0**; review
kewajibannya sebelum distribusi/deployment, jangan menganggap lisensi repo
kompetisi otomatis mencakup bobot. File model dan output diabaikan Git.

## Batas penting dan rencana validasi

Peluit penonton, musik bernada tinggi, dan siulan pemain bisa lolos detektor.
Peluit wasit dapat berarti restart; bendera naik bisa diabaikan/advantage.
Warna kostum/iklan kuning-oranye bisa menjadi false positive; bendera putih,
occlusion, figur kecil, dan gerakan singkat bisa terlewat. Asosiasi role masih
heuristik, bukan pengenal hakim garis terlatih. Maksimal empat proposal juga
dapat memangkas objek relevan. Angka skor bukan probabilitas.

Untuk sistem dead/live yang layak dipromosikan: evaluasi pada laga eksternal
yang boleh diberi label, pisah train/calibration/test **per pertandingan**,
ukur event precision/recall/F1, false alarms per jam, toleransi onset, flag
track recall/abstention, serta dead/live F1 dan galat waktu restart. Bandingkan
audio-only, flag-only, dan fusi dengan baseline tracking tim. Jangan memberi
label manual pada BAR–ATH untuk melatih atau mengevaluasi sistem ini.

Semua video/audio/overlay/fitur target tetap lokal. Tidak ada upload GitHub,
API berbayar, pemasangan dependensi, atau training berat otomatis.
