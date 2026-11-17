# Data, label, dan provenance

## Data pertandingan lomba

Pertandingan target berasal dari paket berlisensi More Than a Hack 2027. Hak
pakai aset itu tidak berubah menjadi MIT hanya karena kode starter memakai MIT.
Pengguna berwenang mengambil paket dari panitia sendiri. Repository tidak
mendistribusikan video, ATD, homography, Smart Tagging, potongan video, fitur
per-frame target, atau sidecar probabilitas yang merekonstruksi aset tersebut.

Rilis menggunakan raw ATD dan metadata tactical. Dua SHA256 input dicatat dalam
`hasil/rilis/reproduction.json`; hash berfungsi mengidentifikasi input tanpa
menerbitkan isinya. Nama file dan blok babak mengikuti `games.py` panitia.
Label publik lima menit hanya boleh dipakai mengecek format, sinkronisasi,
dan arti label. Entry point rilis tidak membacanya.

## Data eksternal untuk supervisi

Seluruh label pelatihan yang dipakai model berasal dari tujuh laga IDSSE: data
resmi Bundesliga divisi pertama/kedua dalam *An integrated dataset of
spatiotemporal and event data in elite soccer*, Bassek, Rein, Weber, dan Memmert
(2025), DOI dataset `10.6084/m9.figshare.28196177.v1`; artikel
`10.1038/s41597-025-04505-y`. Record primer menyatakan lisensi CC BY 4.0.

Kode lama mengunduh metadata dan observed positions dari mirror publik
`pysport/idsse-data` di Hugging Face. Sumber primer Figshare menjadi rujukan
asal-usul dan lisensi; tersedia tujuh XML metadata dan tujuh XML posisi yang
dibutuhkan loader. XML event tidak dibutuhkan oleh jalur pembentukan label ini.

`night_tracking.idsse_halves()` membaca `BallStatus` dan `BallPossession` pada
FrameSet BALL. BallStatus=1 dengan owner 1/2 dipetakan H/A; BallStatus=0 ke D;
owner atau status tidak tersedia dipetakan X. Koordinat eksternal dipindahkan
dari pusat lapangan ke origin yang sama dengan representasi internal.

Kami **tidak membuat anotasi manual baru** untuk tujuh laga itu. Label adalah
atribut yang sudah disediakan dataset; kami bukan anotatornya dan tidak
mengklaim mengetahui identitas tiap operator anotasi atau seluruh prosedur QA
pemasok di luar yang dijelaskan sumber aslinya. Jumlah laga adalah tujuh; jumlah
frame yang masuk setiap run bergantung pada parsing/mask dan dicatat oleh
artefak eksperimen, bukan angka baru yang dikarang di dokumen.

Cache NPZ eksternal pada branch kerja rekan adalah **adaptasi IDSSE**, bukan
tracking pertandingan Barcelona–Athletic. Adaptasi meliputi parsing XML, pemisahan
babak, koordinat meter, mask kosong, serta pengodean H/A/D/X. Atribusi dan tautan
lisensi asal tetap diperlukan. Rilis inferensi tidak memerlukan cache itu.

## Model dan sumber

Sepuluh model pada `models/equal5/` terdiri atas lima deep V4.5 dan lima head pohon
V4.4. Semua dikemas base64 dari artefak beku; ini encoding, bukan enkripsi atau
pelatihan ulang. Manifest menyimpan ukuran dan SHA256 binary asli. Loader
memeriksa keduanya **sebelum** `joblib.load`. Tidak ada dependency API
komersial atau layanan model jarak jauh saat inferensi rilis.

`joblib` dapat mengeksekusi kode saat deserialisasi. Pemeriksaan hash hanya
memeriksa integritas terhadap manifest terpercaya, bukan membuktikan file pihak
ketiga aman. Jangan mengganti model dengan berkas yang tidak dipercaya.

## Catatan review manusia pada target

Catatan evaluasi manual target sempat tersedia dan dibaca saat persiapan akhir.
Catatan itu **tidak dipakai** untuk training, validasi, pemilihan kandidat,
penetapan threshold, aturan frame, maupun pengubahan CSV rilis. Kandidat yang
dipaketkan sudah dibekukan sebelumnya. Kami tidak menyatakan bahwa catatan
tersebut belum pernah terlihat, dan bagian yang tidak dikoreksi bukan ground truth.

Berkas catatan, salinan, dan rentang koreksinya tidak ikut repository. Batas
pemisahan ini penting untuk kejujuran proses; audit kode dan hash bukan
sertifikasi panitia atas keseluruhan aktivitas tim. Bila panitia menilai
paparan catatan perlu penjelasan tambahan, tim harus menjelaskannya apa adanya.

## Bantuan perangkat AI

Asisten AI dipakai dalam pengembangan kode, troubleshooting, pengemasan,
pengujian, dan penyusunan dokumentasi. Keluaran possession akhir tetap hasil
inferensi program dengan model beku, bukan anotasi LLM atas video atau
penyalinan label manusia. Tim tetap bertanggung jawab memeriksa sumber, hasil
tes, lisensi, dan isi pengumpulan.
