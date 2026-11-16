# Empat eksperimen terpisah dari laptop integrator

Paket ini tersedia di branch **`team-dispatch-frame-review-20261116`**, bukan `main`.
Pilih branch tersebut sebelum **Code → Download ZIP**, lalu ekstrak; atau:

```powershell
git clone --branch team-dispatch-frame-review-20261116 --single-branch https://github.com/aata4brian/more-than-a-hack.git
```

Buka folder **`11_Paket_Semalam_4_Rekan`**, bukan launcher menu lama di root.

| Tugas | Launcher | Komputer |
|---|---|---|
| Rekan 1 | `REKAN_1_DECODER.bat` | Decoder/mixture; audit satu kandidat beku hanya untuk pengecekan pipeline |
| Rekan 2 | `REKAN_2_TCN.bat` | TCN 5→20 epoch; wajib pemilik NVIDIA CUDA |
| Rekan 3 | `REKAN_3_POHON.bat` | Kapasitas pohon; CPU/RAM memadai |
| Rekan 4 | `REKAN_4_GAP.bat` | Augmentasi tiga→lima skenario; CPU/RAM memadai |

Empat rekan adalah Affan, Mosi, Khansa, Faiq; **pilih nomor tugas berdasarkan
perangkat**, khususnya tugas 2 diberikan kepada pemilik CUDA. Brian integrator.
Pembagian inspeksi video di folder `review_5_orang` tidak menentukan pemilik GPU.

1. Pasang Python 3.12 x64.
2. Ikuti [`BACA_DULU.md`](11_Paket_Semalam_4_Rekan/BACA_DULU.md) untuk instalasi,
   ruang disk, preflight, stop/resume, dan pengumpulan ZIP hasil.
3. Jalankan `verify_payload.py` sebelum memuat model. Payload asli 107 file
   dipertahankan hash-nya; Git tidak mengonversi akhir baris folder payload.
4. Jangan commit environment, output training, atau data lomba. Kirim hasil
   menggunakan `KUMPULKAN.ps1`, kemudian integrator memeriksa manifest/hash.

Baseline, model dan checkpoint adalah berkas tim tepercaya; **jangan memuat
joblib/pickle/checkpoint dari sumber tak dikenal**. Tes integrator bukan bukti
instalasi/training berhasil di komputer rekan. Tidak ada jaminan SpellF1 >0,9.

## Data yang boleh dan tidak boleh dibagikan

- Tujuh NPZ adalah transformasi **dataset eksternal IDSSE CC BY 4.0**, bukan
  tracking BAR–ATH. Lihat [atribusi data](DATA_SUMBER_TIM.md).
- `audit_publik/BAR-ATH_possession_gt_public.csv` adalah referensi publik resmi,
  bukan GT privat; hanya untuk satu pemeriksaan pipeline beku rekan 1.
- `audit_publik/frozen_candidate.csv` adalah **prediksi algoritmik V4.4 beku**,
  bukan GT dan bukan kandidat V4.5. Tidak boleh dikoreksi manual.
- Tidak ada video/gambar/tracking/homography/tagging BAR–ATH di paket ini.
- Repo utama tetap berisi dokumentasi/menu lama; unggahan ini tidak mengubah
  kandidat submission atau mengklaim hasil V4.5 sudah tersedia dari menu lama.
