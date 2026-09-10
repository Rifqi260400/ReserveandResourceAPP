# ReserveandResourceAPP

Estimasi sumberdaya batubara in-situ dengan metode **poligon pengaruh (Voronoi)**
dibatasi radius klasifikasi, dengan metode **titik observasi sirkular** sebagai
alternatif, diklasifikasikan menurut **SNI 5015:2019**.

Referensi data: PT. Budi Gema Gempita, Blok Lawai 1, Muara Lawai, Sumatera Selatan.

> **Status: Phase 0 (audit data) selesai. Modul estimasi belum dibangun.**
> `coalres run` menolak berjalan sampai audit lulus.

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m coalres.cli audit --config config/template.yaml
```

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
  audit/checks.py          11 pemeriksaan Phase 0
  audit/render.py          laporan teks + Markdown
  cli.py                   coalres audit | run
tests/                     41 tes terhadap workbook, LAS, dan sertifikat asli
```

Belum dibangun (menunggu Phase 0 dikonfirmasi): `seams`, `quality`, `topo`,
`estimate`, `classify`, `rpeee`, `report`.

## Tes

```bash
PYTHONPATH=src:tests python -m pytest tests/ -q
```
