# Usulan Struktur Paket dan Skema Config

Dokumen ini adalah §13 langkah 2. Ia **usulan**, belum dibangun. Kode model dan
estimasi tidak disentuh sampai dokumen ini dikonfirmasi.

Dua gerbang masih terbuka dan memengaruhi isi dokumen ini — **G1** (apa isi 23
lubang ber-seam `A`) dan **G7** (proksimat menutup 86%, bukan 100%). Bagian yang
bergantung pada keduanya ditandai `[G1]` / `[G7]`.

---

## 1. Pipeline 11 tahap

Tanda `[GATE]` berarti tahap itu dapat menghentikan run. Tahap 5 satu-satunya
yang berputar kembali.

| # | Tahap | Modul | Status |
|---|---|---|---|
| 0 | Project setup | `project.py` | **baru** |
| 1 | Input | `io/` | ada |
| 2 | Audit `[GATE]` | `audit/` | **selesai** |
| 3 | Seam database | `seams.py`, `weathering.py` | sebagian |
| 4 | Pemodelan | `model.py` | **baru** |
| 5 | Validasi `[GATE]` → 4 | `validate.py` | **baru** |
| 6 | Titik observasi | `observation.py` | **baru** |
| 7 | Kompleksitas geologi | `complexity.py` | **baru** |
| 8 | Radius (lookup murni) | `radius.py` | **baru** |
| 9 | Batas + RPEEE | `limits.py` | sebagian |
| 10 | Estimasi + pelaporan | `estimate.py`, `export/` | sebagian |

Tahap 6 dan 7 saling bebas dan boleh berjalan paralel; keduanya masuk ke tahap 8.

### Mengapa tahap 5 berputar kembali ke 4

Validasi membandingkan model terhadap lubang bor (Tabel 5.3–5.30 di laporan
rujukan: deviasi ketebalan dan deviasi roof per seam). Bila deviasinya di luar
toleransi, yang salah adalah modelnya, bukan datanya — jadi jalurnya kembali ke
tahap 4, bukan maju ke 6. Tanpa loop ini, deviasi besar akan lolos menjadi angka
sumberdaya.

---

## 2. Peta modul: yang ada, yang berubah, yang baru

### Dipakai ulang apa adanya

| Modul | Baris | Peran |
|---|---|---|
| `config.py` | 446 | skema config, seluruh gerbang |
| `io/minex.py` | 213 | pembaca flat file |
| `io/excel.py` | 452 | pembaca workbook BGG |
| `audit/` | 1.723 | tahap 2, selesai |
| `density.py` | 159 | Preston & Sanders |
| `topo.py` | 196 | `Surface`, TIN, subcrop |
| `grdout.py` | 136 | Surfer `.grd` |
| `dxfout.py` | 230 | kontur DXF |
| `palette.py`, `errors.py`, `logging_setup.py` | 163 | penunjang |

### Berubah

| Modul | Perubahan |
|---|---|
| `seams.py` | menerima `collision_resolution` `[G1]`; cutoff parting 0,10 m |
| `estimate.py` | radius dari tahap 8, bukan dari config langsung |
| `rpeee.py` | diserap `limits.py` |
| `bow.py` | diserap `weathering.py`; rekap Excel tetap |
| `pipeline.py` | 808 baris dipecah menurut tahap; tinggal orkestrator |
| `report.py` | dipecah ke `export/` |

### Dihapus

| Modul | Alasan |
|---|---|
| `webui/` | FastAPI diganti Streamlit (§spec) |

### Baru

```
src/coalres/
├── project.py          # tahap 0: run root, manifest, register asumsi
├── weathering.py       # tahap 3: BOW → pemotongan seam
├── model.py            # tahap 4: structure+isopach stacking, flag dukungan
├── validate.py         # tahap 5: deviasi model-vs-bor, gerbang
├── observation.py      # tahap 6: lubang mana yang jadi titik observasi
├── complexity.py       # tahap 7: formulir 8 parameter
├── radius.py           # tahap 8: fungsi murni
├── limits.py           # tahap 9: subcrop, kedalaman, pelapukan, RPEEE
├── export/
│   ├── bab5.py         # Tabel 5-x, Gambar 5.x, DOCX
│   ├── handover.py     # uncut/ dan limited/
│   ├── charts.py       # PDF vektor + PNG 300 dpi + CSV di samping tiap grafik
│   └── tables.py       # Excel + CSV
└── ui/                 # Streamlit
```

---

## 3. Keputusan desain yang perlu dikonfirmasi

### 3.1 Tahap 8 fungsi murni

```python
def radius_m(condition: GeologicalCondition, klass: ResourceClass) -> float:
    """Tabel SNI 5015:2019. Tanpa config, tanpa override, tanpa kelas bawaan."""
```

Tidak menerima `Config`. Konsekuensinya tabel radius pindah dari config ke
konstanta modul — **ini mengubah perilaku yang ada sekarang**, di mana
`classification_radii_m` dapat diisi bebas di YAML. Alasannya: radius adalah isi
standar, bukan preferensi proyek; dapat-diisi-bebas berarti dapat-diisi-salah
tanpa gejala.

Peringatan `RADII_UNVERIFIED_WARNING` tetap dicetak sampai angkanya diverifikasi
ke teks SNI. Saya **belum** memverifikasinya.

> **Perlu keputusan:** tabel radius jadi konstanta kode (usulan), atau tetap di
> config?

### 3.2 Batubara di luar radius Tereka

Dikeluarkan, tidak pernah dilipat menjadi Tereka. Luas dan tonasenya tetap
dilaporkan sebagai baris terpisah "di luar radius" supaya tidak hilang diam-diam.

### 3.3 Flag dukungan per sel

Setiap sel grid membawa:

| Flag | Arti |
|---|---|
| `interpolated` | di dalam hull titik data |
| `extrapolated` | di luar hull |
| `n_support` | jumlah lubang dalam radius pengaruh |

**Sel `extrapolated` tidak pernah Terukur.** Ini aturan keras, bukan setelan.

### 3.4 Swauji peta

Setiap titik observasi yang memenuhi syarat wajib jatuh di dalam poligon kelas
tertinggi seam-nya. Bila tidak, peta dan tabel tidak sinkron — itu bug, dan
run berhenti. Dijalankan otomatis di tahap 10.

### 3.5 Formulir kompleksitas (tahap 7)

Mengikuti Tabel 5-32 laporan rujukan: 8 parameter, tiap parameter diberi skor
sederhana/moderat/kompleks, tiap skor menuntut justifikasi ≥40 karakter.

Dua peringatan **wajib** tercetak, tidak bisa dimatikan:

1. **Bias pembobotan kelompok.** Kelompok Tektonik memuat 4 dari 8 parameter,
   Sedimentasi 3, Kualitas 1. Menghitung skor tanpa bobot memiringkan hasil ke
   kelompok yang parameternya paling banyak. Di laporan rujukan PT SGM skornya
   6 sederhana lawan 2 moderat — disimpulkan *sederhana*.
2. **Margin tepi jurang.** Selisih skor dan seberapa dekat ke kelas sebelah.
   6–2 tidak dekat; 5–3 akan membalik kelas dan **menggandakan/membagi radius**.
   Angka marginnya dicetak apa pun hasilnya.

### 3.6 Batas IUP tidak diterapkan

Keputusan proyek yang tercatat. `limits.iup_boundary: null` tetap ada di skema
supaya niatnya terlihat, bukan terlupa. Dicatat di **tepat tiga tempat**:
dokumen asumsi, laporan QA, dan header ringkasan sumberdaya.

### 3.7 Konvensi volume

`luas datar × ketebalan vertikal`. **Tanpa koreksi cos(dip).** Sudah dikunci
uji regresi `test_no_cosine_dip_correction_is_applied`; tidak berubah.

---

## 4. Tambahan skema config

Blok yang sudah ada tidak diubah kecuali disebut. Yang di bawah ini **baru**.

```yaml
project:
  name: "SGM"                      # masuk ke nama run root
  reporting_year: 2023
  competent_person: ""             # kosong = draf, dicetak sebagai draf

model:
  cell_size_m: 25.0
  # Roof/floor TIDAK diinterpolasi bebas satu sama lain: floor dibangun dari
  # roof dikurangi isopach, supaya ketebalan tidak pernah negatif.
  method: structure_isopach_stacking
  reference_surface: roof
  extrapolation:
    allow: true
    max_distance_m: 250.0          # di luar ini: sel kosong, bukan tebakan
    never_measured: true           # aturan keras, tidak dapat dimatikan

validation:                        # menambah blok yang sudah ada
  thickness_deviation_tolerance_m: 0.30
  roof_deviation_tolerance_m: 1.00
  max_holes_outside_tolerance_frac: 0.10
  barren_hole_conflict: stop       # roof model di atas TD lubang barren

complexity:                        # tahap 7 — formulir, bukan satu kata
  sedimentasi:
    variasi:          {skor: sederhana, justifikasi: ""}
    kesinambungan:    {skor: sederhana, justifikasi: ""}
    percabangan:      {skor: moderat,   justifikasi: ""}
  tektonik:
    sesar:            {skor: sederhana, justifikasi: ""}
    lipatan:          {skor: sederhana, justifikasi: ""}
    intrusi:          {skor: sederhana, justifikasi: ""}
    kemiringan:       {skor: sederhana, justifikasi: ""}
  kualitas:
    variasi_kualitas: {skor: moderat,   justifikasi: ""}

observation_point:                 # SUDAH DIISI
  requires_quality: true

limits:
  iup_boundary: null               # keputusan proyek tercatat — lihat 3.6
  apply_weathering: true
  apply_subcrop: true
  max_depth_m: 100.0

export:
  bab5:      {enabled: true, docx: true, chart_dpi: 300, csv_beside_charts: true}
  handover:  {enabled: true, grid_format: esri_ascii, also_geotiff: true,
              also_csv_xyz: true, dxf_3d_polylines: true}
```

### Yang **tidak** masuk config

| Hal | Alasan |
|---|---|
| Tabel radius `[3.1]` | isi standar, bukan preferensi proyek |
| `never_measured` untuk sel ekstrapolasi | aturan keras |
| Dua peringatan kompleksitas `[3.5]` | wajib, tidak dapat dimatikan |
| Swauji peta `[3.4]` | bug detector, bukan opsi |

---

## 5. Pohon keluaran

```
run_SGM_20260912_1430/
├── 01_bab5_laporan/
│   ├── tabel/       Tabel_5-1.xlsx … Tabel_5-36.xlsx (+ .csv)
│   ├── gambar/      Gambar_5-1.pdf + .png (300 dpi) + .csv
│   └── draf_bab5.docx
├── 02_reserve_handover/
│   ├── uncut/       SG{seam}{SR,SF,ST}_uncut.asc + .tif + .csv
│   ├── limited/     SG{seam}{SR,SF,ST}_ltd.asc + .tif + .csv
│   ├── kontur/      Seam_A_Roof.dxf (POLYLINE 3D ber-Z)
│   ├── batas/       subcrop.shp, depth_limit.shp
│   └── manifest.json
└── 03_audit_and_provenance/
    ├── audit.md, audit.xlsx
    ├── asumsi.md            ← pelapukan, RD, batas IUP
    ├── qa.md
    └── input_digests.json
```

Aturan penamaan: penanda `_uncut` / `_ltd` **wajib** di setiap nama berkas grid.
Grid uncut dan limited terlihat identik saat dibuka; tertukar, hasilnya salah
tanpa gejala.

**Tidak pernah menulis biner Minescape proprietary.** ESRI ASCII Grid primer.

### Kelas isi laporan Bab V

| Kelas | Isi | Cakupan |
|---|---|---|
| A | dihasilkan penuh dari data | Tabel 5-1…5-30, 5-33, 5-35, 5-36; Gambar 5.1, 5.4–5.19, 5.23, 5.25–5.33 |
| B | butuh masukan ekonomi | Tabel 5-34 (BESR), Gambar 5.24 (design pit) |
| C | slot kosong bernarasi | Gambar 5.2, 5.3, 5.20–5.22 |

Kelas C dicetak sebagai slot bertanda "**BUTUH ISI MANUAL**", bukan dikosongkan
diam-diam.

---

## 6. Urutan pembangunan

| Langkah | Isi | Bergantung |
|---|---|---|
| 1 | `project.py`, pecah `pipeline.py` | — |
| 2 | `weathering.py`, `seams.py` | **G1** |
| 3 | `model.py` | langkah 2 |
| 4 | `validate.py` | langkah 3 |
| 5 | `observation.py`, `complexity.py`, `radius.py` | — |
| 6 | `limits.py`, `estimate.py` | langkah 4, 5 |
| 7 | `export/handover.py` | langkah 6 |
| 8 | `export/bab5.py` | langkah 6, **G7** |
| 9 | `ui/` Streamlit | langkah 8 |

Langkah 5 tidak bergantung pada gerbang mana pun dan dapat dikerjakan lebih dulu
sambil G1 dan G7 menunggu jawaban.
