# Atribusi data paket empat rekan

## IDSSE: data eksternal, bukan pertandingan lomba

Bassek, Manuel; Rein, Robert; Weber, Hendrik; Memmert, Daniel (2025).
*An integrated dataset of spatiotemporal and event data in elite soccer*.
Figshare, v1. DOI: https://doi.org/10.6084/m9.figshare.28196177.v1

Artikel: https://doi.org/10.1038/s41597-025-04505-y

Lisensi dataset: **Creative Commons Attribution 4.0 International (CC BY 4.0)**,
https://creativecommons.org/licenses/by/4.0/ . Lisensi dikonfirmasi dari metadata
API sumber: https://api.figshare.com/v2/articles/28196177 .

Pemilik data: Deutsche Fußball Liga (DFL); pengumpulan Sportec Solutions/TRACAB.
Mirror publik: https://huggingface.co/datasets/pysport/idsse-data .
Metadata Figshare sumber menjadi rujukan daftar tujuh laga.

File `11_Paket_Semalam_4_Rekan/09_Riset_V4/external_cache/*.npz` merupakan
**data yang telah diubah**: XML sumber dikonversi pipeline tim menjadi array
koordinat pemain/bola dan label pertandingan eksternal, dipisah per babak,
kemudian disimpan sebagai NPZ. Bukan file XML asli. Skrip transformasi:
`eksperimen/pipeline/night_tracking.py` di dalam paket.
ID laga: J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9.
Hash berkas ada dalam `payload_manifest.json`. Tidak ada dukungan/endorsement
dari pemilik data yang tersirat. Hak sumber tetap mengikuti CC BY 4.0.

## BAR–ATH: hanya aset publik resmi dan prediksi algoritmik

Kode/aturan/scorer dan GT lima menit berasal dari repository panitia:
https://github.com/metrica-sports/more-than-a-hack-2027/tree/f8f2629ff1e0985f81d7e59d8413678514ee1530 .
Lisensi kode/dokumentasi upstream ada di [`LICENSE`](LICENSE).

Lisensi kode **tidak mencakup** video/tracking pertandingan berlisensi.
Data itu harus diperoleh anggota terdaftar dari panitia, tidak diunggah ulang.
GT publik hanya untuk pemeriksaan pipeline, bukan training/tuning/seleksi.
Prediksi beku adalah keluaran algoritmik tim; bukan human ground truth.
