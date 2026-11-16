# Verifikasi paket empat rekan — 16 November 2026

## Bukti yang sudah ada

- Lima belas tes awal decoder dan tim lulus pada environment workspace.
- Smoke sintetis empat jalur lulus dalam 117,328 detik; decoder 30 baris,
  masing-masing jalur training 60 baris, complete-resume, pemuatan model
  di interpreter baru, dan penolakan perubahan konfigurasi.
- Bukti: `09_Riset_V4/team_smoke_verification_v2.log` pada workspace integrator.
- Percobaan smoke pertama terkena timeout 120 detik, bukan sukses.

## Pemeriksaan final

- Payload portabel sudah dibangun: source, tujuh cache eksternal, baseline
  lima fold, scorer, launcher, dan manifest SHA-256; sekitar 231 MiB.
- Seluruh 58 tes modul riset lulus dalam 35,564 detik.
- Dua puluh tes decoder/tim/integritas lulus pula pada environment
  `.venv_repro_v44` yang terisolasi; `pip check` lulus. Environment ini
  sudah tersedia sebelumnya, bukan fresh-install baru paket empat rekan.
- Smoke v3 lulus dalam 111,437 detik: keempat jalur sengaja diinterupsi
  setelah checkpoint skenario pertama, dilanjutkan sampai matriks lengkap,
  lalu diuji complete-resume dan pemuatan kandidat di interpreter baru.
- Output nyata dari fixture empat jalur berhasil diekspor/diimpor dua kali
  secara idempoten pada folder sementara; tidak masuk ledger kompetisi.
- Tes menolak hash berubah, path traversal, matriks complete yang berlubang,
  konflik snapshot, dan konfigurasi berubah. Snapshot parsial berikutnya
  tidak otomatis menimpa snapshot lama; simpan arsip dan tinjau dahulu.
- Seluruh skrip PowerShell berhasil di-parse. `-CekSaja` nomor 1–4 lulus
  pula menggunakan environment terisolasi integrator; nomor 2 mendeteksi
  RTX 2050. Kendala stderr Windows PowerShell diperbaiki; exit code tes
  tetap diperiksa, bukan diabaikan.
- Sepuluh model baseline berhasil dimuat dari source payload; lima checkpoint
  semuanya epoch 5 dengan history lengkap. Ketujuh cache cocok dengan hash
  baseline; split lima fold cocok. Tidak menjalankan training data nyata.
- Seluruh 61 hash paket final, CSV kandidat, model kandidat, dan CSV legacy
  tetap cocok dengan bukti beku sebelumnya.
- Lima belas tes ledger lulus. Salinan aman sumber nyata menghasilkan
  3.765 catatan dari 15 sumber, 167 ringkasan; refresh kedua tidak menambah
  atau merevisi hasil dan CSV byte-identik. Tidak ada skor sintetis baru.
- Refresh langsung CSV pusat sempat gagal mengganti ringkasan karena
  `Ringkasan_Performa_Semua.csv` terkunci (WinError 5). Tidak dipaksa
  ditutup atau ditimpa. Tutup file itu lalu jalankan `PERBARUI_PERFORMA.bat`
  sebelum menggabungkan hasil rekan. Status sukses lama bukan bukti refresh baru.
- Bukti workspace: `team_regression_final.log`, `team_isolated_tests.log`,
  `team_smoke_verification_v3.log`, dan `team_dispatch_verification.json`
  di `09_Riset_V4`; log preflight tersedia
  di folder paket. Log lokal bukan bagian payload yang dikunci hash.

Paket siap disalin untuk instalasi dan preflight pada komputer rekan. Ini
bukan sertifikasi bahwa empat perangkat lain sudah kompatibel. Jalankan
`verify_payload.py` dan `JALANKAN.ps1 -Rekan N -CekSaja` pada tiap komputer.

## Batas verifikasi

- Fixture sintetis bukan skor penelitian dan tidak dimasukkan ke CSV pusat.
- Belum menjalankan eksperimen semalam nyata atau audit target publik nyata.
- Instalasi bersih dan kompatibilitas empat komputer rekan belum diuji.
- Durasi penuh, peak RAM/disk, dan selesainya lima fold belum diukur.
- Pengguna tetap integrator; empat rekan menjalankan empat komputer berbeda.
- Tidak ada promosi model, publikasi, atau pengiriman submission otomatis.
- SpellF1 >0,9 masih sasaran, bukan hasil terukur.
