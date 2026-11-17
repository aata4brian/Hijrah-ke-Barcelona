# Batas kepatuhan dan publikasi

Rujukan otoritatif adalah RULES.md dan SUBMISSION.md panitia, bukan checklist
buatan tim. Dokumen ini menjelaskan pengendalian yang benar-benar diterapkan.

- Kandidat berasal dari model dan konfigurasi yang dibekukan sebelum pembacaan
  catatan evaluasi manual target. Catatan itu tidak menjadi input kode rilis,
  training, validasi, seleksi, threshold, maupun koreksi frame.
- Tidak ada asumsi bahwa frame yang tidak ditandai pengguna adalah benar.
  Kami tidak mengukur akurasi target dari catatan tersebut. Paparan catatan
  dijelaskan terbuka di DATA_DAN_PROVENANCE.md.
- Hanya raw ATD dan metadata tactical dibaca saat inferensi rilis. Dataset
  supervisi berasal dari tujuh pertandingan IDSSE eksternal yang dapat diakses.
- Video, tracking target, homography, tagging, cache fitur target, anotasi pribadi,
  konfigurasi komputer, serta cookie/token tidak menjadi berkas rilis publik.
- Prediksi submission disimpan lokal untuk formulir; repo menyediakan kode, model,
  hash, dan bukti reproduksi, bukan ekspor tracking target.
- LICENSE dan atribusi starter dipertahankan. Lisensi dataset eksternal dibedakan
  dari lisensi kode dan aset pertandingan lomba.
- Tidak ada klaim skor privat, peringkat, peningkatan seragam, atau manfaat cue
  yang belum diuji. Kandidat ensemble tidak diberi skor tabel per-fold.

Branch lama berisi pembagian **inspeksi teknis** lima orang yang secara eksplisit
melarang anotasi possession, koreksi prediksi, dan konstanta khusus frame. Itu
bukan bukti bahwa anotasi manual pernah menjadi instruksi branch tersebut.
Catatan evaluasi manual yang disediakan terpisah tetap tidak boleh digunakan
untuk training, validasi, seleksi, maupun perubahan hasil. Pembagian tugas
teknis bukan label target atau hasil penilaian akurasi.

Audit memeriksa perubahan commit remote yang terjangkau, isi sumber lokal yang
cocok, file besar, pola kredensial, sumber cache eksternal, dan hash artefak.
Pencarian pola tidak membuktikan ketiadaan seluruh jenis rahasia, dan nama file
saja tidak membuktikan lisensi. Temuan yang belum jelas harus menjadi penghalang
publikasi, bukan diubah menjadi status aman secara otomatis.

Formulir pengumpulan tidak dikirim otomatis oleh pekerjaan pengemasan ini.
Satu anggota terdaftar perlu memeriksa nama tim, dua berkas, serta URL publik
sebelum mengirim respons. Status publikasi dan tes aktual ada pada catatan rilis;
dokumen ini tidak boleh dipakai sebagai sertifikasi kelayakan oleh panitia.
