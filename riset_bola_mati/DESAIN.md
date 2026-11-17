# Rancangan deteksi peluit, bendera, dan indikasi bola mati

Status awal: implementasi eksperimental, bukan akurasi terukur pada pertandingan.

1. **Audio**: PCM stereo 16 kHz, STFT pendek, bukti tonal sempit, SNR dan
   kontinuitas temporal. Setiap kanal diperiksa; tidak merata-ratakan kanal
   yang mungkin berlawanan fase. Keluaran adalah kandidat peluit, bukan
   kepastian bahwa sumbernya wasit. Pemrosesan dibatasi per chunk.
2. **Visual**: model pose ONNX lokal opsional, asosiasi orang dengan bendera
   kuning/oranye dan posisi dekat touchline dari homography. Tangan di atas
   bahu saja tidak cukup. Tanpa bukti identitas/visibilitas, hasil UNKNOWN.
   Baseline warna tanpa pose hanya menghasilkan kandidat, bukan hakim garis
   terkonfirmasi. Tidak melatih atau memberi label manual pada BAR–ATH.
3. **Temporal**: bendera perlu konsisten beberapa observasi; identitas track
   dan waktu diamati dipertahankan. Tidak tampak bukan bendera turun.
4. **Fusi**: gabungkan event peluit dan flag di sekitar waktu yang sama.
   Bendera tidak otomatis menghentikan permainan; peluit juga bisa untuk
   restart atau berasal dari penonton. Tidak adanya keduanya bukan bukti LIVE.
   Keluaran konservatif STOPPAGE_CANDIDATE/STOPPAGE_SUPPORTED/UNKNOWN,
   dengan kedaluwarsa bukti; tidak mengarang waktu restart.
5. **Audit**: konfigurasi, hash source/model, jangkauan media, frame/timestamp,
   runtime dan jumlah abstain dicatat. Tidak mengganti CSV submission lama.
6. **Validasi**: uji audio sintetis, geometri pose, occlusion, batas chunk,
   alignment dan fusi; smoke nyata untuk I/O saja. Akurasi harus diuji pada
   pertandingan eksternal berlabel, dipisah per pertandingan.

## Temuan engineering yang sudah ditangani

- Input seeking FFmpeg pada AAC menggeser onset sinyal sintetis. Decoder
  memakai output seeking dan satu proses streaming, diverifikasi terhadap
  potongan waveform referensi; tidak menganggap waktu packet = onset peluit.
- Audio dan video tidak selalu sama durasinya. Coverage memakai durasi stream
  audio aktual; selisih akhir tidak ditambal silence.
- Warna sepatu/kaus dapat menyerupai bendera. Ada eksklusi anatomi generik,
  penolakan satu blob untuk beberapa orang, penolakan geometri bertentangan,
  dan reset persistensi bila asosiasi flag melompat. Ini bukan classifier role.
- Gate touchline untuk proposal dihitung secara vektor, satu invers matriks
  per sampel, bukan ratusan invers untuk semua blob warna.

Semua keluaran media/fitur tetap lokal, diabaikan Git; tidak ada upload.
