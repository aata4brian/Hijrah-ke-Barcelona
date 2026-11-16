# Pembagian inspeksi teknis video target — lima orang

**Bukan tugas anotasi possession.** Tidak boleh membuat label H/A/D, mencatat
siapa memiliki bola sebagai acuan, mengubah prediksi, atau membuat konstanta
khusus frame dari hasil menonton. Lihat [RULES](../RULES.md).

Pembagian ini hanya untuk memeriksa pemuatan video, indeks frame, overlay,
format CSV, penanganan data hilang, dan bug program yang dapat direproduksi.
Jangan memakai hasil menonton sebagai validasi label atau pemilihan model.

## Cakupan

- Referensi **tactical**, bukan broadcast, 25 fps, 199.319 frame total.
- Jendela babak yang diumumkan `games.py`: **11185–90651** dan
  **110808–190197**, total **158.857 frame**. Ini bukan bocoran mask GT privat;
  hanya rentang resmi yang sudah dipublikasikan. Video sebelum/sesudah babak
  dan jeda tidak dibagi. CSV submission tetap wajib 199.319 baris.
- Frame **0-based**, batas awal/akhir frame **inklusif**. Timestamp adalah
  waktu file video, **bukan jam pertandingan**; akhir waktu ditulis eksklusif.
- Lihat [`pembagian_frame.csv`](pembagian_frame.csv) untuk titik lompat tepat.
  Khansa memperoleh dua segmen karena pembagian melewati jeda babak.

| Orang | Jumlah frame |
|---|---:|
| Affan | 31.772 |
| Mosi | 31.772 |
| Khansa | 31.771 |
| Faiq | 31.771 |
| Brian | 31.771 |

Nama mengikuti daftar anggota di README tim. Nomor tugas training dipilih
terpisah sesuai perangkat; pemilik CUDA mendapat tugas TCN.

## Cara mengakses

1. Ambil video dari tautan resmi peserta dan simpan lokal. **Tidak ada video,
   gambar, cuplikan, tracking atau tautan Drive privat dalam GitHub ini.**
2. Buka `BUKA_VIEWER.bat` dari root repo; petunjuk [viewer](../viewer/README.md).
3. Lompat ke timestamp dalam CSV. Jangan menafsirkan broadcast sebagai tactical.
4. Laporkan **bug teknis**: langkah reproduksi, error program, versi commit,
   dan jenis bug. Jangan sertakan label possession, screenshot/video, koreksi
   tracking, atau aturan khusus kejadian/frame. Hasil pemeriksaan tidak berarti
   akurasi possession telah divalidasi.

Daftar ini hanya pembagian pekerjaan, **bukan hasil review yang sudah dikerjakan**.
Generator `buat_pembagian.py` hanya membaca konstanta resmi, tidak membaca
video, prediksi, GT publik ataupun GT privat. Ulangi dengan:

```powershell
py -3.12 review_5_orang/buat_pembagian.py
py -3.12 -m unittest discover -s review_5_orang -p "test_*.py" -v
```
