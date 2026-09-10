# ReserveandResourceAPP

Estimasi sumberdaya batubara dari data lubang bor (Excel) dan topografi (DXF),
memakai model stratigrafi ber-grid.

Masukan: **satu Excel** berisi collar + interval seam + kualitas, dan **satu DXF**
topografi. Keluaran: tabel sumberdaya per seam per kelas, peta, grid ASCII untuk
GIS, dan laporan exception validasi.

---

## Cara pakai

```bash
pip install -r requirements.txt

# Bangkitkan data contoh untuk mencoba
python scripts/make_synthetic_data.py --out sample_data

PYTHONPATH=src python -m coalres.cli \
    --holes sample_data/drillholes.xlsx \
    --topo  sample_data/topo.dxf \
    --config sample_data/config.yml \
    --out   output/sample
```

Pipeline berhenti dengan kode keluar 1 bila validasi menemukan ERROR. Gunakan
`--allow-errors` untuk tetap melanjutkan — tapi angka yang keluar tidak layak
dipakai sebelum ERROR-nya diselesaikan.

---

## Format Excel yang diharapkan

Nama kolom **tidak perlu persis**; pemetaan alias ada di `config/default.yml`
bagian `columns` dan bisa ditambah tanpa mengubah kode. Perbandingan mengabaikan
kapital, spasi, dan underscore.

### Sheet `Collar` — satu baris per lubang

| Kolom | Wajib | Keterangan |
|---|---|---|
| `HOLE_ID` | ✅ | ID lubang |
| `EASTING` | ✅ | Koordinat X (UTM) |
| `NORTHING` | ✅ | Koordinat Y (UTM) |
| `RL` | ✅ | Elevasi collar — **pakai hasil Total Station, bukan GPS handheld** |
| `TOTAL_DEPTH` | | Dipakai untuk memeriksa interval yang melebihi TD |
| `BLOCK` | | Nama blok/prospek |

### Sheet `Seam` — satu baris per interval seam per lubang

| Kolom | Wajib | Keterangan |
|---|---|---|
| `HOLE_ID` | ✅ | |
| `SEAM` | ✅ | Kode seam, mis. `S10A` |
| `FROM` | ✅ | Kedalaman roof (m) |
| `TO` | ✅ | Kedalaman floor (m) |
| `TM` | | Total moisture, basis **ar** (%) |
| `IM` | | Inherent moisture, basis **adb** (%) |
| `ASH` `VM` `FC` `TS` | | Basis **adb** (%) |
| `CV_ADB` `CV_AR` | | Gross calorific value (cal/g) |
| `RD` | | **Apparent** relative density, basis air-dried |

Satu seam boleh punya beberapa baris (ply); ply akan dikompositkan otomatis
dengan pembobotan massa. Kolom kualitas boleh juga ditaruh di sheet terpisah
yang di-join pada `(HOLE_ID, SEAM)` — atur lewat `input.quality_sheet`.

### DXF topografi

Kontur (`LWPOLYLINE`/`POLYLINE`), spot height (`POINT`), atau TIN
(`3DFACE`/`MESH`). Semua simpul ber-Z ditarik lalu diinterpolasi menjadi DTM.
Titik ber-Z = 0 dibuang bila mayoritas titik lain ber-Z bukan nol — kontur yang
lupa diberi elevasi adalah kesalahan lazim yang membuat lubang datar di DTM.

---

## Yang perlu diketahui sebelum mempercayai angkanya

### 1. Tonase memakai ARD in-situ, bukan ARD lab

ARD dari sertifikat lab diukur pada kondisi air-dried (IM). Batubara di alam
mengandung total moisture (TM) yang jauh lebih tinggi. Air (RD 1,0) menarik
densitas curah ke arah 1,0, sehingga **ARD in-situ lebih rendah dari ARD lab**.

Memakai ARD lab apa adanya **melebihkan tonase** — untuk batubara Sumatera
Selatan tipikal (TM 42%, IM 18%, ARD 1,36) sekitar **10%**:

| | ARD lab | ARD in-situ | Dampak |
|---|---|---|---|
| S10A | 1,36 | 1,228 | −9,7% |
| S10B | 1,35 | 1,218 | −9,8% |

Konversi memakai Preston & Sanders (1993), diterapkan per seam dari TM dan IM.
Atur lewat `density.insitu_method`.

Ini hanya berlaku untuk **apparent** relative density (ASTM D167, diukur pada
bongkah utuh sehingga pori ikut terhitung). Bila lab melaporkan *true/real
density* dari piknometer pada sampel digerus, angka itu tidak boleh dipakai untuk
tonase sama sekali. Validator menandainya lewat cek `ARD_NOT_APPARENT`.

### 2. Volume tidak dikoreksi cos(dip)

```
Volume = luas sel DALAM PETA × ketebalan VERTIKAL (roof RL − floor RL)
```

Prisma vertikal mengisi ruang antar dua permukaan secara persis, berapa pun
dip-nya, dan lubang vertikal mengukur tepat besaran itu. Dip mengecilkan
ketebalan **dan** membesarkan luas bidang seam; keduanya saling meniadakan.

Menerapkan koreksi cos(dip) pada volume **mengurangi tonase secara keliru**.
Koreksi true thickness berlaku untuk bor **miring** (interval sepanjang lubang ≠
ketebalan vertikal), dan untuk hal yang memang bergantung pada tebal tegak lurus:
cutoff ketebalan minimum dan parameter penambangan. Keduanya diperlakukan
terpisah di `resource.apply_cutoffs`.

Dip tetap diturunkan dari gradien grid struktur, dan dipakai untuk cutoff serta
sebagai pembanding terhadap dip yang diukur di core — perbandingan itu adalah uji
independen terhadap kualitas korelasi.

### 3. Klasifikasi di sini bukan klasifikasi final

> **KCMI tidak memuat tabel radius.** Ia kode *pelaporan* berbasis prinsip, satu
> keluarga dengan JORC, dan menuntut klasifikasi dijustifikasi serta diungkapkan
> dasarnya oleh Competent Person. Angka jarak yang lazim dipakai berasal dari
> **SNI 5015**, yang berstatus pedoman — dan harus diverifikasi ke dokumen standar
> versi terkini sebelum dipakai untuk pelaporan.

Modul ini menilai **spasi titik data**. Ia tidak dapat menilai kualitas korelasi
seam, kerapatan struktur, atau kecukupan QAQC — padahal ketiganya adalah bagian
dari klasifikasi. Keluarannya adalah titik awal untuk penilaian Competent Person.

Dua pengaman ditegakkan karena keduanya sering dilanggar implementasi
berbasis buffer:

- **Kriteria spasi, bukan buffer.** Sel hanya naik kelas bila ada cukup banyak
  titik dalam radius (`classification.min_points`), bukan sekadar satu titik
  terdekat. Satu lubang terisolasi tidak membuktikan kontinuitas apa pun.
- **Klasifikasi per seam.** Seam tipis yang sulit dikorelasi tidak mewarisi kelas
  dari seam utama di lubang yang sama.

Selain itu, `require_quality_for_measured` menahan kelas Terukur di sel yang
lubang pendukungnya tidak punya data kualitas: klasifikasi mencerminkan keyakinan
pada tonase **dan** kualitas, bukan hanya geometri.

`classification.geological_condition` adalah keputusan yang paling menentukan
hasil akhir — menggesernya dari `moderate` ke `simple` melipatgandakan luas
Terukur tanpa satu pun angka di laporan yang terlihat salah. Buktikan pilihannya
dari data, jangan sekadar menyatakannya.

---

## Validasi yang dijalankan

| Kode | Severity | Yang diperiksa |
|---|---|---|
| `COLLAR_DUPLICATE` `COLLAR_MISSING` | ERROR | ID ganda, koordinat kosong |
| `COLLAR_OUTLIER` | WARNING | Koordinat jauh dari sebaran (X/Y tertukar) |
| `TOPO_DATUM_BIAS` | ERROR | Offset sistematis RL collar vs topo — masalah datum |
| `COLLAR_VS_TOPO` | WARNING | Selisih RL per lubang (kasus GPS vs Total Station) |
| `INTERVAL_INVALID` `INTERVAL_OVERLAP` `INTERVAL_BEYOND_TD` | ERROR | Geometri interval |
| `SEAM_ORPHAN` `SEAM_UNKNOWN` | ERROR | Interval tanpa collar; seam di luar skema |
| `STRAT_OUT_OF_ORDER` | ERROR | Urutan seam bertentangan dengan stratigrafi |
| `STRUCTURE_RESIDUAL` | WARNING | Residual RL floor besar — sesar atau salah korelasi |
| `MASS_BALANCE` | ERROR | IM+Ash+VM+FC ≠ 100 (adb) |
| `CV_CONVERSION` | ERROR | CV(ar) tidak konsisten dengan CV(adb), TM, IM |
| `ARD_RANGE` `ARD_NOT_APPARENT` | WARNING | ARD di luar rentang wajar; bukan apparent density |
| `ARD_ASH_OUTLIER` `ARD_ASH_SLOPE` | WARNING | Menyimpang dari tren ARD–Ash |
| `QUALITY_COVERAGE` | ERROR/WARNING/INFO | Cakupan data kualitas per seam |
| `NO_QAQC` | WARNING | Tidak ada duplikat/CRM/umpire/blank |
| `PARTING_EXCEEDS_CUTOFF` | WARNING | Parting melebihi cutoff — seam semestinya dipecah |

---

## Struktur

```
config/default.yml         parameter: pemetaan kolom, cutoff, grid, klasifikasi
src/coalres/
  config.py                pemuatan & validasi konfigurasi
  io/excel.py              import Excel, pemetaan alias, normalisasi hole_id
  io/dxf.py                import topografi DXF
  validate.py              mesin QC -> laporan exception
  quality.py               konversi basis, Preston-Sanders, compositing
  grid.py                  Grid, interpolasi (IDW/nearest/linear), dip
  model.py                 stratmodel: structure + isopach, clip topo, subcrop
  classify.py              klasifikasi berbasis spasi
  resource.py              cutoff, volume, tonase, kualitas terbobot
  report.py                tabel, peta, grid ASCII
  pipeline.py              orkestrasi
  cli.py                   antarmuka baris perintah
scripts/make_synthetic_data.py   dataset uji bergaya BGG, dengan cacat disengaja
tests/                     44 tes, termasuk terhadap sertifikat lab nyata
```

Model dibangun dengan strategi **structure + isopach**: floor tiap seam di-grid
sebagai permukaan struktur, ketebalan vertikal di-grid sebagai isopach, lalu roof
diturunkan (`roof = floor + isopach`). Isopach selalu ≥ 0 dan lebih halus
daripada struktur, sehingga lebih stabil untuk diinterpolasi. Menginterpolasi
roof dan floor secara terpisah kerap menghasilkan ketebalan negatif atau
permukaan yang saling memotong di area ekstrapolasi.

---

## Belum ada (dan diperlukan sebelum pelaporan)

- **Batas IUP / poligon batas** — ekstrapolasi bisa keluar konsesi. Saat ini
  hanya dibatasi `grid.max_extrapolation_m`.
- **Pengurangan sungai, jalan, permukiman, kawasan lindung.**
- **Penanganan sesar** — interpolasi belum dipisah per blok sesar. `STRUCTURE_RESIDUAL`
  akan menandai lokasi yang mencurigakan bila sesar ternyata ada.
- **Kriging** — saat ini IDW/nearest/linear. Kriging baru bermakna bila jumlah
  titik per domain cukup untuk membuat variogram.
- **QAQC lab** — tidak bisa dibuat surut; harus dibangun untuk pengiriman berikutnya.

## Tes

```bash
PYTHONPATH=src python -m pytest tests/ -q
```
