# Bukti rilis

- `reproduction.json`: inferensi dari raw ATD dan metadata menghasilkan CSV
  byte-identik kandidat beku (199.319 frame). Hash input tidak membocorkan isinya.
- `tests.json`: hasil pengujian terarah rilis; 24 test ditemukan, 23 lulus dan
  1 skipped karena artefak eksperimen lokal memang tidak ikut paket publik.
- `audit.json`: cakupan audit berkas lokal, riwayat remote, dan atribusi.

Validator resmi lulus pada **berkas submission yang dipasangkan**, bukan hanya
pada fixture. Uji Node viewer lulus untuk logika; playback nyata belum diuji.
Tidak ada klaim bahwa suite training GPU luas sudah seluruhnya lulus.
Prediksi serta data berlisensi tidak diletakkan di folder ini.
