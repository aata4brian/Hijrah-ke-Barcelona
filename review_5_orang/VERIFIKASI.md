# Verifikasi persiapan unggahan

- Dua tes pembagian lulus: semua 158.857 frame jendela resmi tercakup tepat
  sekali, tidak melintasi jeda di dalam segmen, selisih beban paling banyak
  satu frame. Tidak membaca video atau label target.
- Verifikasi SHA256 payload: **107 file / 242.209.703 byte**, cocok dengan
  paket integrator. Tidak menyertakan venv, cache fitur training, log pribadi
  atau keluaran eksperimen baru.
- Preflight source paket: **20 tes lulus**, 7,587 detik di environment
  integrator yang sudah tersedia. Data/label yang dipakai tes adalah fixture
  sintetis di folder sementara. Bukan audit nyata target atau training rekan.
- Instalasi, preflight dan training pada empat komputer rekan tetap harus
  dijalankan masing-masing; hasil itu belum tersedia.
- Pemeriksaan ini tidak mengklaim akurasi possession video atau skor privat.
- Berkas asli payload tidak diedit; `.gitattributes` mencegah konversi newline
  pada folder payload agar pemeriksaan hash tetap berlaku setelah clone.
