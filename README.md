# ReserveandResourceAPP

Estimasi sumberdaya batubara in-situ dengan metode **poligon pengaruh (Voronoi)**
dibatasi radius klasifikasi, dengan metode **titik observasi sirkular** sebagai
alternatif, diklasifikasikan menurut **SNI 5015:2019**.

Referensi data: PT. Budi Gema Gempita, Blok Lawai 1, Muara Lawai, Sumatera Selatan.

**Program lengkap.** Begitu data masuk, cukup isi konfigurasi lalu jalankan.

```bash
pip install -r requirements.txt

# 1. Audit data (gerbang keras) - selalu jalankan ini dulu
PYTHONPATH=src python -m coalres.cli audit --config config/template.yaml

# 2. Estimasi penuh - menolak berjalan sampai audit lulus
PYTHONPATH=src python -m coalres.cli run --config config/template.yaml
```

Mencoba tanpa data produksi:

```bash
python scripts/make_synthetic_dataset.py --holes 25 --out sample_data
PYTHONPATH=src python -m coalres.cli run --config sample_data/config.yaml
```

Dataset sintetis ditulis dalam **format workbook BGG yang sebenarnya**, lengkap
dengan header merged bertingkat, sehingga resolver header ikut teruji dan bukan
dilewati.

Kode keluar: `0` audit lulus · `1` konfigurasi/masukan tidak sah · `2` ada gerbang STOP.

---

## Phase 0 sebagai gerbang keras

Audit memeriksa 11 hal dan **berhenti** pada kondisi yang tidak boleh
diselesaikan oleh kode — hanya manusia yang boleh memutuskan, dan keputusannya
masuk ke konfigurasi agar ikut tercatat pada keluaran.

| # | Pemeriksaan | Berhenti bila |
|---|---|---|
| 1 | Peta header teratasi + sampel baris | — (wajib dikonfirmasi) |
| 2 | Konflik koordinat Collar vs BHC | `authoritative_coordinate_source` tidak ada |
| 3 | Basis kedalaman (reconciled vs wellsite) | ada lubang tanpa log terekonsiliasi |
| 4 | Konflik litologi vs sampling | `coal_thickness_source` tidak ada |
| 5 | Cakupan kualitas per seam | tabel kualitas tidak dipasok |
| 6 | Basis RD | `RD_basis` = `unknown` |
| 7 | Open hole vs cored, core recovery | — |
| 8 | Integritas, LAS, topografi | overlap interval, TD, topo tidak ada |
| 9 | Jumlah lubang per seam | seam ditembus < 3 lubang |
| 10 | Justifikasi kondisi geologi | justifikasi kosong / < 100 karakter |
| 11 | Batasan RPEEE | `max_depth_m` diisi tanpa `max_depth_basis` |

### Resolver header

Header workbook BGG membentang beberapa baris dan memakai merged cell, sehingga
`header=0` selalu salah. Resolver membuka sheet lewat openpyxl dan
**mengembangkan setiap merged range secara eksak**, lalu menutup blok header
dengan aturan "baris header bebas angka" dan memetakan hasilnya ke nama kanonik
lewat alias.

Alternatif yang ditolak:

- `read_excel(header=[6,7,8])` — menuntut nomor baris dihardcode per sheet, dan
  diam-diam salah bila tata letak bergeser satu baris.
- Forward-fill horizontal — mengisi melewati ujung merge, sehingga kolom tak
  berjudul mewarisi judul tetangganya.
- "Baris data = baris pertama dengan ≥3 angka" — **dicoba dan gagal**: baris data
  SLL hanya membawa dua angka (Depth From, Depth To), sehingga ambang itu
  melompati delapan baris data pertama dan menariknya menjadi header.

Alias diperlukan karena header berbeda antar workbook untuk kolom yang sama:
DH09_05C1 menulis `Lithologi` dan `Continuity`, DH11_01 menulis `Lith` dan
`Hole Type`. Kode litologi (`C3`, `KL`, `XC`, …) **dibaca dari sheet
`Library SLL`**, tidak dihardcode.

### Kurva LAS adalah cacah mentah

`LD` dan `SD` bersatuan **CPS**, bukan bulk density g/cc. Keduanya sah untuk
verifikasi pick seam dan rekonsiliasi kedalaman, dan **diblokir dari jalur
tonase**. Kurva densitas terkalibrasi dikenali dari mnemonic *dan* satuannya
(`RHOB` + `g/cc`), tidak pernah ditebak dari besaran nilainya.

### Aturan inventori (8.4)

Bila `max_depth_m` kosong, run tetap selesai tetapi **setiap keluaran dilabeli
Inventori Batubara**, bukan Sumberdaya, dan kolom kelas menjadi Inventori
Terukur / Tertunjuk / Tereka. Aturan ini tidak dapat dilewati setelan lain.

---

## Koreksi terhadap spesifikasi: catatan RD 1,94

Spesifikasi menyatakan bahwa RD 1,36 air-dried, dikonversi dengan TM 42,25% dan
M adb 17,55%, menghasilkan ~1,94 t/m³, dan menyimpulkan basis RD bukan
air-dried. **Angka itu berasal dari rumus yang salah.**

1,9417 = 1,36 × (100 − 17,55) / (100 − 42,25) adalah konversi **kadar** antar
basis moisture (ash, CV, sulphur), diterapkan pada densitas. Kadar adalah fraksi
massa; densitas adalah massa per **volume**, dan volume ikut bertambah ketika air
masuk. Rumus kadar mengabaikan penambahan volume itu.

Konversi densitas yang benar adalah **Preston & Sanders (1993)**:

```
K     = (100 − M_adb) / (100 − TM)
RD_is = K · RD_ad / (1 + RD_ad · (K − 1))      →  1,2276 t/m³
```

1,2276 wajar untuk batubara dengan CV ar 3373 kcal/kg, jadi 1,36 **konsisten**
dengan ARD air-dried. Uji pendukung: matriks kering yang tersirat adalah 1,47 —
rentang wajar untuk batubara ash ~11%.

Gerbang `RD_basis` tetap ditegakkan (basis wajib dinyatakan lab, tidak boleh
disimpulkan dari nilainya), tetapi atas dasar itu — bukan atas dasar 1,94.
Keduanya ada di `density.py` dengan nama eksplisit dan diuji di
`tests/test_las_and_density.py`.

Arah efeknya penting: RD in-situ **lebih rendah** dari RD air-dried, sehingga
memakai RD lab apa adanya **melebihkan** tonase sekitar 10%.

---

## Struktur

```
config/template.yaml       template; field gerbang sengaja kosong
src/coalres/
  config.py                skema pydantic, gerbang justifikasi & RPEEE
  errors.py                kesalahan eksplisit; tidak ada default diam-diam
  logging_setup.py         logging terstruktur
  density.py               Preston-Sanders + konversi kadar (bernama, berarah)
  io/excel.py              resolver header merged, pustaka litologi
  io/las.py                LAS; deteksi cacah mentah vs densitas terkalibrasi
  io/dxf.py                topografi; laporan jenis entitas
  io/quality_table.py      skema tabel kualitas yang dipasok pengguna
  seams.py                 interseksi seam, cutoff parting, perlakuan core loss
  quality.py               cakupan komposit, rata-rata terbobot tonase per basis
  topo.py                  permukaan TIN, subcrop, grid kedalaman
  estimate.py              Voronoi & sirkular, pemotongan, tonase
  classify.py              pita radius kumulatif, label kelas
  rpeee.py                 batasan ekonomi berurutan, aturan label 8.4
  logs.py                  plot log per lubang
  palette.py               token warna dan gaya sumbu bersama
  report.py                Excel, vektor, grid, peta kontur, QA/QC, run log
  pipeline.py              orkestrasi
  audit/checks.py          11 pemeriksaan Phase 0
  audit/render.py          laporan teks + Markdown
  cli.py                   coalres audit | run
scripts/make_synthetic_dataset.py   dataset uji dalam format workbook BGG
tests/                     110 tes
```

## Dua format masukan

`input_format` memilih pembaca:

| Format | Berkas | Catatan |
|---|---|---|
| `bgg_workbook` | satu `.xlsx` per lubang + LAS + CSV kualitas + DXF | header merged bertingkat, diresolusi otomatis |
| `minex_flat` | `surv` · `lit` · `qual` · `faults` · `topo` | **tidak berheader** — arti kolom dideklarasikan di konfigurasi |

Berkas Minex tidak berheader, jadi urutan kolom yang salah **tidak memunculkan
kesalahan apa pun** — tebal dan kualitas tetap terbaca sebagai angka yang wajar.
Karena itu `minex.*_columns` dan `minex.quality_rd_basis` adalah gerbang audit,
dan tabel deklarasi kolom dicetak untuk dikonfirmasi manusia.

Berkas `lit` Minex memuat interval **seam**, bukan seluruh kolom litologi — jadi
tidak ada parting atau core loss untuk dikurangkan, dan logika parting jalur BGG
tidak diterapkan di sini.

## Hitung ganda seam terpecah

Seam induk dan anaknya (`seam_splits: {A: [A1, A2]}`) adalah batubara yang **sama**
yang direpresentasikan berbeda di lubang berbeda. Membentuk tesselasi Voronoi
terpisah untuk masing-masing membuat domainnya bertindih.

Pada dataset dummy, kesalahan itu bernilai **26,9% — 13,4 dari 49,9 juta ton** —
dan tidak memunculkan gejala apa pun di peta maupun tabel.

Perbaikannya di akar: **satu tesselasi atas gabungan seluruh lubang dalam satu
satuan stratigrafi**. Tiap sel dimiliki tepat satu lubang, dan lubang itu
menyumbang representasi yang memang ia punya. Tumpang tindih menjadi mustahil
menurut konstruksi. Anak yang bertumpuk (A1 di atas A2) tetap bertindih dalam
peta — itu benar, keduanya seam pada kedudukan stratigrafi berbeda.
`plan_overlap_report()` ikut terbit di setiap run sebagai jaring pengaman.

## Deliverable

| # | Keluaran | Isi |
|---|---|---|
| 1 | `resource_estimate.xlsx` | 11 sheet: sampul berlabel, asumsi & batasan, ringkasan seam x kelas, per seam, total, kualitas terbobot, intercept per lubang, rekonsiliasi RPEEE, rekonsiliasi tebal-kualitas, sensitivitas RD, temuan audit |
| 2 | `vector/` | GeoJSON + Shapefile poligon klasifikasi, beratribut seam, kelas, luas, tebal, RD, tonase, lubang sumber, cakupan kualitas, label |
| 3 | `grids/` | ASCII grid + GeoTIFF per seam: roof RL, floor RL, tebal batubara, kedalaman di bawah permukaan — plus sidecar berisi metode interpolasi, spasi grid, dan sejauh mana permukaan didukung data bor |
| 4 | `grd/` | **Grid Surfer (.grd)** per seam: `{seam}_uncut.grd`, `{seam}_cut.grd`, roof, floor, depth, dan `{seam}_qual_{ATRIBUT}.grd` — masing-masing dengan berkas keterangan `.txt` |
| 5 | `dxf/` | **Kontur struktur roof & floor per seam ke DXF**, interval dapat diatur (bawaan 2 m), layer `Seam A Roof`, `Seam A Floor`, dst. Satu berkas gabungan + satu per seam |
| 6 | `maps/` | **Peta kontur struktur roof & floor** (garis kontur berlabel + subcrop + batas blok), peta isopach kontur, peta permukaan terisi, peta klasifikasi dengan subcrop dan kontur batas kedalaman |
| 7 | `logs/` | Plot per lubang: kurva GR dan densitas LAS di samping litologi dan pick seam |
| 8 | `qaqc_report.md` | Temuan audit, pengecualian, justifikasi kondisi geologi verbatim, rekonsiliasi RPEEE, rekonsiliasi jumlah lubang |
| 9 | `run_log.json` | Konfigurasi terpakai, SHA-256 setiap berkas masukan, timestamp, versi pustaka |

### Uncut vs cut

`{seam}_uncut.grd` adalah ketebalan **geologi** in-situ: amplop roof ke floor,
**tanpa** ketebalan minimum, **tanpa** pengecualian parting, **tanpa** dilusi,
**tanpa** cutoff. `{seam}_cut.grd` adalah ketebalan setelah aturan penambangan.

Perbedaannya bukan kosmetik: grid uncut dibangun dari interseksi **pra-cutoff**,
sehingga lubang yang lebih tipis dari `min_seam_thickness_m` ikut mendukung
interpolasi. Pada dataset dummy, seam A uncut turun sampai 0,96 m melawan cut
3,02 m, dan cakupannya 112 sel lebih luas.

Membangun grid uncut dari interseksi yang sudah lolos cutoff menghasilkan
berkas bernama uncut yang isinya cut — dan tidak ada cara membedakannya dari
berkas yang benar. Peringatan itu tercetak di tiap berkas keterangan.

Grid RD (`{seam}_qual_rd_t_per_m3.grd`) ikut ditulis: ia faktor ketiga pada
`tonase = luas × tebal × RD`, jadi estimasi cadangan di hilir membutuhkannya
bersama grid ketebalan.

Sel tanpa data ditulis sebagai `1.70141e+38` (Surfer blank), bukan nol — sel
bernilai nol dan sel tanpa data adalah dua hal berbeda.

### Ekspor DXF

Tiga bentuk berkas ditulis sekaligus, pilih yang sesuai alur kerja Anda:

| Berkas | Isi |
|---|---|
| `Seam A Roof.dxf`, `Seam A Floor.dxf`, … | **satu permukaan per berkas** — untuk memuat satu permukaan ke CAD |
| `seam_A_contours.dxf`, … | satu seam per berkas, roof dan floor menyatu |
| `seam_contours.dxf` | semua seam, semua permukaan |

Tiap berkas memuat layer terpisah untuk roof dan floor, plus layer kontur
indeks, garis subcrop, dan titik bor:

```
Seam A Roof          kontur 2 m
Seam A Roof Index    kontur 10 m (2 m x 5), lebih tebal
Seam A Floor
Seam A Floor Index
Seam A Subcrop       POLYLINE 3D, DI-DRAPE ke topografi
Boreholes            POINT pada RL collar
Borehole Labels      TEXT hole_id
```

Tiga hal yang menentukan kebenarannya:

- **Kontur dipotong oleh subcrop.** Permukaan seam di dalam model membentang
  sampai batas dukungan data, termasuk ke area di mana seam berada di atas
  topografi — di sana batubaranya sudah tererosi. Mengekspor kontur di situ
  menggambarkan seam yang tidak ada.
- **Elevasi dibawa tiap polyline** lewat atribut `elevation`, bukan hanya
  tersirat dari nama layer, sehingga CAD dan GIS dapat membacanya dan memberi
  label sendiri.
- **Subcrop di-drape ke topografi.** Ia adalah garis tempat seam memotong
  permukaan tanah; menulisnya pada elevasi 0 membuatnya melayang di tempat yang
  salah pada model 3D.

`$INSUNITS` disetel 6 (meter).

Peta kontur struktur digambar dengan `linestyles="solid"` secara eksplisit:
matplotlib menggambar level negatif sebagai garis putus-putus secara bawaan, dan
pada peta struktur batubara RL negatif itu lazim sementara garis putus-putus
berarti "perkiraan". Garis subcrop dihaluskan sebesar satu sel grid — cukup
menghilangkan gerigi rasterisasi, tidak pernah menggeser garis lebih jauh dari
resolusi yang mendasarinya.

## Aturan yang dikunci di kode dan diuji

**Tonase.**

```
Tonnes = luas poligon DALAM PETA (m2) x tebal VERTIKAL (m) x RD in-situ (t/m3)
```

Tanpa koreksi cosinus dip. Luas dalam peta dikalikan tebal vertikal sudah memberi
volume prisma yang sebenarnya: dip mengecilkan tebal tegak lurus DAN membesarkan
luas bidang seam, dan keduanya saling meniadakan. `test_no_cosine_dip_correction_is_applied`
menjaga agar koreksi itu tidak pernah disisipkan.

**Tebal dari lubangnya sendiri.** Di bawah Voronoi, poligon adalah daerah
pengaruh satu lubang; interpolasi tebal antar poligon bertentangan dengan asumsi
itu. Permukaan interpolasi hanya MEMOTONG poligon (subcrop, batas kedalaman) —
mempengaruhi luas, tidak pernah tebal.

**Densitas.** Tidak pernah mengganti RD yang hilang dengan nilai bawaan.
Pilihannya hanya konstanta dari konfigurasi (dicap sebagai asumsi di setiap
tabel) atau seam dikeluarkan. Konversi air-dried -> in-situ memakai Preston &
Sanders (1993), dan nilai masukan maupun hasilnya dicetak pada setiap record.

**Kualitas.** Dibobot tonase, tidak pernah lintas basis analitik, jumlah sampel
ikut di setiap angka rata-rata, dan interval yang benar-benar tercakup lab
dilaporkan bersama nilainya.

**Anti hitung-ganda.** Sel Voronoi saling eksklusif menurut konstruksi. Metode
sirkular menyelesaikan tumpang tindih dengan memberikan kelas TERTINGGI, lalu
mengurangi area yang sudah diklaim kelas di atasnya.

**Segmentasi lingkaran.** 128 segmen. Luas poligon-n beraturan adalah
`(n/2pi)*sin(2pi/n)` kali lingkaran sejati: bawaan shapely (16 segmen) meleset
-2,58%, dan galat itu terbawa langsung ke tonase.

## Keputusan yang sudah dikunci

| Setelan | Nilai | Alasan |
|---|---|---|
| `coal_thickness_source` | `lithology` | Konvensi Minex — agar sebanding dengan model terdahulu |
| `core_loss_treatment` | `as_coal` | Konvensi Minex |

**Konsekuensi yang dilaporkan, bukan disembunyikan.** Minex menerima tebal dari
tabel litologi dan kualitas dari tabel komposit tanpa memeriksa bahwa keduanya
menutupi interval yang sama; ketidakcocokan itu tidak memunculkan peringatan di
sana. Modul `seams` menghitung angka yang sama dengan Minex, lalu menerbitkan
selisihnya lewat `thickness_quality_reconciliation()`.

**Core loss dibatasi pada atribusi geolog.** `as_coal` hanya berlaku untuk core
loss yang kolom `Seam`-nya dinamai. Core loss di luar amplop seam selalu waste.
Pada DH09_05C1 ada 0,445 m core loss di batuan penutup dan interburden
(70,070–70,270 dan 83,470–83,715) yang akan salah terhitung sebagai batubara
tanpa batasan ini.

## Tes

```bash
PYTHONPATH=src:tests python -m pytest tests/ -q
```
