# Integrator (anggota kelima)

1. Bagikan seluruh folder `11_Paket_Semalam_4_Rekan` yang memiliki
   `payload_manifest.json`. Rekan dengan NVIDIA menerima tugas 2; nomor lain
   boleh ditukar **sebelum** mulai. Tidak perlu membagikan video/data target.
2. Pastikan masing-masing mengirim bukti `-CekSaja` lulus, ruang kosong, dan
   nomor tugas. Empat komputer boleh berjalan paralel, satu antrean per komputer.
3. Biarkan `10_Kandidat_Final_V44` dan submission lama tetap beku.
4. Terima arsip hasil ke folder lokal terpisah, jangan mengekstrak sembarang
   ZIP ke atas source/model final. Gunakan importer dari workspace utama:

```powershell
cd '09_Riset_V4'
.\.venv\Scripts\python.exe -m riset_v4.team_results import --archive 'D:\hasil\HASIL_REKAN_3_xxx.zip'
```

Importer memeriksa daftar berkas, path aman, SHA-256, metadata, key evaluasi,
kelengkapan run bertanda complete; **tidak melakukan joblib.load**. Hasil
masuk `09_Riset_V4/outputs/rekan_N_sesi_fold_K/`. CSV pusat diperbarui dengan
`perbarui_performa.py` yang sama. Import ulang snapshot identik aman; konflik
dengan file yang sudah ada ditolak, bukan ditimpa. Untuk snapshot parsial
yang kemudian bertambah, simpan kedua arsip, tinjau perbedaannya; importer
sengaja tidak menimpa file lama. Jangan melewati penolakan dengan menghapus bukti.

5. Tutup CSV pusat di Excel bila terkunci. Jika sinkronisasi gagal, data hasil
   tetap tersimpan; jalankan `PERBARUI_PERFORMA.bat` dari workspace utama.
6. Nilai **selisih berpasangan** pada fold/laga/skenario yang sama. Run satu
   fold tidak boleh disebut skor lima fold. Seed/fold berulang bukan laga baru.
7. Cek minimal: SpellF1, MacroF1, recall episode <2 s, jumlah possession,
   original vs gap, dan penurunan per laga. Kalibrasi saja yang memilih decoder.
8. Jangan gabungkan checkpoint lintas fold lalu menilai pada laga yang pernah
   melatih salah satunya. Jangan masukkan oracle atau audit publik ke ranking.
9. Bila kandidat dipromosikan: simpan keputusan, inferensi target tanpa GT,
   validator, reproduksi byte-identik, dan paket sumber baru. Tidak ada push,
   publikasi, atau pengiriman formulir otomatis.

## Isi tugas

| Rekan | Hipotesis yang diuji | Pembanding yang adil |
|---|---|---|
| 1 | Salah biaya transisi/campuran menyebabkan episode terlewat | Ensemble V4.4 50:50 dan decoder beku; pemilihan hanya pada kalibrasi |
| 2 | TCN belum konvergen dalam 5 epoch | Checkpoint seed331/data sama diteruskan 20 epoch; baseline 5 epoch |
| 3 | Kapasitas pohon terlalu rendah | Tiga skenario/seed sama; paket kapasitas lebih besar, bukan isolasi satu hyperparameter |
| 4 | Distribusi gap training belum cukup bervariasi | Parameter tree tetap; tiga skenario vs lima skenario |

Kenaikan nyata mungkin kecil atau negatif. Tes software tidak membuktikan
hipotesis, dan target >0,9 tidak boleh ditulis sebagai hasil sebelum terukur.
