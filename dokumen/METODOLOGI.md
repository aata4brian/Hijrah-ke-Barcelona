# Metodologi: dari tracking ke rangkaian possession

## 1. Pertanyaan dan pendekatan

Kami memperlakukan possession sebagai segmentasi urutan, bukan sekadar mencari
pemain terdekat pada setiap frame. Bola yang bergerak menuju penerima dapat jauh
dari semua pemain, sedangkan kedekatan sesaat saat duel belum tentu berarti
pergantian penguasaan. Karena itu, informasi spasial perlu dibaca bersama konteks
gerak dan kestabilan urutan.

Keluaran model mempunyai tiga kelas internal: H, A, dan D. X tersedia dalam kontrak
CSV panitia, tetapi kandidat ini tidak mengeluarkan abstention. Tidak adanya X
bukan jaminan keyakinan tinggi; model tetap dapat salah pada observasi yang hilang.
Inferensi bersifat **offline/nonkausal**: fitur, konteks konvolusi, dan decoder dapat
memakai informasi sesudah frame yang sedang diprediksi. Kami tidak mengklaim
sistem ini siap untuk keputusan siaran langsung.

## 2. Masukan dan koordinat

Jalur rilis memanggil `fusion_bundle.local_tracking()`, yang meminta hanya metadata
ATD dan raw ATD tactical melalui `games.require`. Metadata menghubungkan slot
tracking dengan dua tim. Koordinat dinormalisasi ke meter memakai dimensi lapangan
resmi dalam `games.py`; urutan frame, jumlah pemain per baris, dan panjang timeline
diperiksa. Satu baris ATD bernomor mulai 1 dipetakan ke indeks video mulai 0.

Data dibagi menurut dua blok babak yang didefinisikan panitia. Decoder tidak
menyambung possession melewati jeda babak. Frame di luar blok babak diberi D
sebagai placeholder timeline, **bukan hasil inferensi bola mati**.

Tidak ada pembacaan Smart Tagging, label target, video, audio, homography, atau
cache fitur target pada jalur rilis. Impor modul utilitas lama tidak sama dengan
memanggil loader tagging; fungsi `local_halves()` lama bukan entrypoint rilis.

## 3. Fitur dan observasi hilang

`features.pair()` menghasilkan 203 kolom untuk dua orientasi tim. Representasi
mempertahankan fitur legacy dan menambahkan konteks kontak serta gerak relatif
pada 25 fps. Kelompoknya meliputi keberadaan bola, lama sejak/sebelum observasi,
posisi relatif terhadap lapangan, kecepatan dan percepatan bola, jarak pemain
terdekat, jumlah pemain terlihat, sebaran tim, pendekatan radial, gerak bersama,
serta durasi kontak. Sebagian statistik memakai jendela 5, 25, atau 75 frame.

Kecepatan tidak dihitung menyeberangi observasi kosong atau lompatan slot yang
melampaui batas kode. Slot pemain tidak diasumsikan identitas persisten sepanjang
laga. Fitur tim menekankan geometri dan pemain terdekat, bukan menghafal nomor slot.

Missingness dipertahankan sebagai NaN, bukan diganti koordinat nol. Transformasi
deep memakai normalisasi beku, nilai nol untuk fitur tidak terlihat setelah
normalisasi, dan indikator observasi eksplisit. Angka nol itu bukan klaim bahwa
bola ada di suatu lokasi. Model pohon memakai perlakuan missingness implementasinya.

Untuk mereproduksi kandidat yang telah dibekukan, fitur float32 dikonversi ke
float16 lalu kembali ke float32 sebelum inferensi. Ini meniru presisi penyimpanan
kandidat asal. Menghapus langkah tersebut terlihat seperti peningkatan presisi,
tetapi dapat mengubah CSV. Evaluasi eksternal yang dicatat mengekstrak float32;
perbedaan presisi ini merupakan keterbatasan transfer evaluasi ke kandidat rilis.

## 4. Simetri tim

Pasangan fitur kedua menukar H dan A, termasuk tanda selisih fitur tim. Prediksi
pada orientasi kedua dipetakan kembali sebelum dirata-ratakan. Kelas D tetap D;
enam arah batas juga dipermutasi sesuai pasangan asal-tujuan. Prinsip ini mencegah
model bergantung pada posisi tim di array, tetapi bukan bukti bebas bias lintas
kompetisi atau stadion. Pengujian sintetis memeriksa konsistensi pertukaran tim.

## 5. Model temporal

Deep V4.5 memperluas TCN pendahulu: enam blok pretrained ditambah dua blok dilasi
64 dan 128, lalu refiner enam blok pada probabilitas tiga status dan enam batas.
Cabang tambahan diinisialisasi agar keluaran awal mempertahankan baseline.
Halo 636 frame dipakai ketika memotong inferensi menjadi chunk agar konteks
di sekitar batas chunk tetap tersedia. Detail arsitektur ada di `deep_temporal.py`
dan `tcn.py`; ini implementasi tim yang terinspirasi literatur temporal convolution
dan refinement, bukan reproduksi persis arsitektur MS-TCN [3,4].

Loss menggabungkan cross-entropy status, binary cross-entropy batas, dan smoothing
log-probabilitas. Mask label tidak dikenal menghilangkan kontribusi supervisi yang
tidak tersedia. Smoothing tidak diterapkan menyeberangi batas label nyata. Konfigurasi
V4.5 yang tercatat memakai 12 epoch, 160 step per epoch, batch 4, core 512, dan
validasi setiap 3 epoch; manifest per fold menjadi rujukan parameter sebenarnya.

Pembanding pohon adalah histogram gradient boosting dengan head status dan head
batas. `switch_context()` memakai konteks sebelum/sesudah 25 frame, mempertahankan
jendela satu detik dari grid legacy. Model ini memberi inductive bias berbeda
dari jaringan temporal, tanpa harus selalu meningkatkan hasil.

## 6. Kalibrasi dan penggabungan

Suhu status dan transformasi logit batas dipelajari pada pertandingan kalibrasi
yang terpisah di masing-masing fold. Parameter pasangan H/A diikat untuk menjaga
simetri. Tidak ada kalibrasi dari label pertandingan lomba.

Untuk fold f, probabilitas gabungan adalah `(1-w_f) p_deep + w_f p_tree`; rumus
yang sama dipakai untuk enam probabilitas batas. Bobot pohon adalah 0,25 pada
fold 0 dan 3, serta 0 pada fold 1, 2, dan 4. Specialist tambahan berbobot nol di
semua fold. Rata-rata aritmetika lima fold menjadi masukan satu decoder tetap.
Fold tidak dipilih berdasarkan skor terbaik pada target.

## 7. Decoder dan aturan possession

Viterbi memaksimalkan jumlah log-emission dikurangi biaya transisi yang dimodulasi
proposal batas. Konfigurasi rilis: team=2, dead=4, boundary_gain=2, emission_scale=0,2,
dead_bias=-0,18, proposal_threshold=0,1. Konfigurasi ini **berbeda** dari decoder
terpilih per fold dalam evaluasi eksternal (boundary_gain=1; threshold=0,15).
Tidak boleh menyamakan skor tabel per-fold dengan skor ensemble rilis.

Aturan operasional kami adalah kontinuitas yang didukung observasi, bukan
nearest-player voting. Loose ball dan duel tidak memiliki state tersendiri;
H/A/D diputuskan model serta biaya pergantian. Restart dan bola mati juga
diinferensikan dari tracking dan konteks, bukan daftar peluit atau event target.
Model belum mengkodekan seluruh hukum sepak bola secara simbolik. Karena itu,
bola diam saat permainan masih berlangsung dan bola mati di area padat dapat
tertukar. Tidak ada tabel override, daftar frame koreksi, atau pengeditan CSV.

## 8. Alternatif yang tidak masuk rilis

Expanded tracking 281 kolom, fitur cue sampai 290 kolom, eksperimen peluit/bendera,
serta pemulihan bola dari video tersedia sebagai riset terpisah. Ablation yang
ada belum membuktikan manfaat specialist, dan cue tidak mempunyai evaluasi
eksternal tersinkron yang memadai. Menambah modul karena tersedia justru akan
melemahkan keterlacakan. Rilis menggunakan 203 fitur dan nol bobot cue/specialist.

Referensi bernomor mengikuti [daftar sumber](REFERENSI.md).
