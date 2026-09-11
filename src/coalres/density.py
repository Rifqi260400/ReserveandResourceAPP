"""Konversi basis densitas.

Disertakan pada Phase 0 karena gerbang audit item 6 (basis RD) tidak dapat
dijelaskan konsekuensinya tanpa rumus konversinya. Jalur tonase belum dibangun.

CATATAN TERHADAP SPESIFIKASI. Spesifikasi menyatakan bahwa RD 1,36 air-dried
yang dikonversi dengan TM 42,25% dan M adb 17,55% menghasilkan ~1,94 t/m3, dan
menyimpulkan basisnya bukan air-dried. Angka 1,94 berasal dari

    RD x (100 - M_adb) / (100 - TM)  =  1,36 x 82,45 / 57,75  =  1,9417

yang merupakan rumus konversi KADAR (ash, CV, sulphur) antar basis moisture,
bukan konversi densitas. Kadar adalah fraksi massa dan berskala terhadap massa
total; densitas adalah massa dibagi VOLUME, dan volume ikut bertambah ketika air
masuk. Menerapkan rumus kadar pada densitas mengabaikan penambahan volume itu,
sehingga menghasilkan angka yang memang mustahil.

Konversi yang benar adalah Preston & Sanders (1993), diimplementasikan di bawah.
Untuk angka yang sama ia menghasilkan 1,2276 t/m3, yang wajar untuk batubara
dengan CV ar 3373 kcal/kg. Jadi 1,36 KONSISTEN dengan ARD air-dried, dan
"inkonsistensi" yang dikutip spesifikasi adalah artefak rumus, bukan bukti
tentang basis. Gerbang RD_basis tetap ditegakkan - basis wajib dinyatakan lab,
tidak boleh disimpulkan dari nilainya - tetapi atas dasar itu, bukan atas dasar
angka 1,94.
"""
from __future__ import annotations

import numpy as np

from .errors import MissingDataError


def preston_sanders_insitu_ard(
    ard_air_dried_t_per_m3: float,
    total_moisture_ar_pct: float,
    inherent_moisture_adb_pct: float,
) -> float:
    """ARD air-dried -> ARD in-situ (Preston & Sanders, 1993).

        K      = (100 - M_adb) / (100 - TM_ar)
        RD_is  = K * RD_ad / (1 + RD_ad * (K - 1))

    Model fisiknya: volume matriks kering tetap, dan air tambahan antara kondisi
    air-dried dan in-situ menempati volumenya sendiri (RD air = 1,0).

    Arah efeknya penting. Untuk batubara peringkat rendah dengan TM jauh di atas
    M_adb, air yang ditambahkan menarik densitas curah ke arah 1,0, sehingga
    RD in-situ LEBIH RENDAH dari RD air-dried. Memakai RD lab apa adanya
    MELEBIHKAN tonase - pada contoh BGG sekitar 10%.

    Hanya berlaku untuk APPARENT relative density (ASTM D167, diukur pada
    bongkah utuh sehingga pori ikut terhitung). True/real density dari
    piknometer pada sampel digerus tidak boleh dipakai untuk tonase.

    Referensi: Preston, K.B. & Sanders, R.H. (1993), "Estimating the in-situ
    relative density of coal", Australian Coal Geology 9, 22-26.
    """
    for name, value in (
        ("ard_air_dried_t_per_m3", ard_air_dried_t_per_m3),
        ("total_moisture_ar_pct", total_moisture_ar_pct),
        ("inherent_moisture_adb_pct", inherent_moisture_adb_pct),
    ):
        if value is None or not np.isfinite(value):
            raise MissingDataError(
                f"konversi ARD in-situ memerlukan {name}; tidak ada nilai pengganti."
            )
    if not 0 <= total_moisture_ar_pct < 100 or not 0 <= inherent_moisture_adb_pct < 100:
        raise MissingDataError("moisture harus dalam persen, 0 <= nilai < 100")

    k = (100.0 - inherent_moisture_adb_pct) / (100.0 - total_moisture_ar_pct)
    return k * ard_air_dried_t_per_m3 / (1.0 + ard_air_dried_t_per_m3 * (k - 1.0))


def mass_fraction_basis_conversion(
    value: float,
    total_moisture_ar_pct: float,
    inherent_moisture_adb_pct: float,
    direction: str,
) -> float:
    """Konversi KADAR antar basis moisture (ash, CV, sulphur) - BUKAN densitas.

        adb_to_ar : value x (100 - TM) / (100 - M_adb)
        ar_to_adb : value x (100 - M_adb) / (100 - TM)

    Diberi nama dan arah eksplisit supaya tidak pernah lagi terpakai untuk
    densitas. Diterapkan pada RD 1,36 dengan arah `ar_to_adb`, rumus ini
    menghasilkan 1,9417 - angka yang dikutip spesifikasi sebagai bukti bahwa
    basis RD bukan air-dried. Ia sebenarnya hanya memperlihatkan bahwa rumus
    kadar tidak berlaku untuk densitas: kadar adalah fraksi massa, sedangkan
    densitas adalah massa per VOLUME, dan volume ikut bertambah saat air masuk.
    """
    dry_ar = 100.0 - total_moisture_ar_pct
    dry_adb = 100.0 - inherent_moisture_adb_pct
    if direction == "adb_to_ar":
        return value * dry_ar / dry_adb
    if direction == "ar_to_adb":
        return value * dry_adb / dry_ar
    raise ValueError(f"arah tidak dikenal: {direction!r}")


def dry_matter_ard(ard_air_dried_t_per_m3: float, inherent_moisture_adb_pct: float) -> float:
    """ARD matriks kering yang tersirat dari ARD air-dried dan M_adb.

    Uji kewajaran: untuk batubara dengan ash 5-15% angkanya biasanya 1,40-1,55.
    Hasil di luar itu menandakan nilai yang dilaporkan kemungkinan bukan
    apparent density, atau basisnya bukan air-dried.
    """
    return (100.0 - inherent_moisture_adb_pct) / (
        100.0 / ard_air_dried_t_per_m3 - inherent_moisture_adb_pct
    )


def resolve_in_situ_rd(
    rd_value: float | None,
    rd_basis: str,
    total_moisture_ar_pct: float | None,
    inherent_moisture_adb_pct: float | None,
    assumed_rd_t_per_m3: float | None,
) -> tuple[float, bool, str]:
    """Tentukan RD in-situ untuk satu interseksi seam.

    TIDAK PERNAH mengganti RD yang hilang dengan nilai bawaan. Bila tidak ada
    hasil lab, pilihannya hanya dua dan keduanya keputusan eksplisit pengguna:
    nilai konstan dari konfigurasi (yang lalu dicap sebagai asumsi pada setiap
    tabel), atau seam itu dikeluarkan dari estimasi tonase.

    Mengembalikan (rd, is_assumed, catatan). `catatan` menampilkan nilai masukan
    DAN hasil konversinya, sehingga peninjau melihat koreksinya, bukan
    menyimpulkannya.
    """
    if rd_value is not None and np.isfinite(rd_value):
        if rd_basis == "in_situ":
            return float(rd_value), False, f"RD {rd_value:.3f} t/m3 (basis in_situ, dipakai apa adanya)"
        if rd_basis in {"air_dried", "as_received"}:
            converted = preston_sanders_insitu_ard(
                rd_value, total_moisture_ar_pct, inherent_moisture_adb_pct
            )
            return (
                float(converted), False,
                f"RD {rd_value:.3f} t/m3 ({rd_basis}) -> in-situ {converted:.4f} t/m3 "
                f"(Preston & Sanders 1993; TM {total_moisture_ar_pct:.2f}%, "
                f"M_adb {inherent_moisture_adb_pct:.2f}%)"
            )
        raise MissingDataError(
            f"RD_basis '{rd_basis}' tidak dapat dikonversi. Basis harus dinyatakan "
            "laboratorium; ia tidak boleh disimpulkan dari nilainya."
        )

    if assumed_rd_t_per_m3 is not None:
        return (
            float(assumed_rd_t_per_m3), True,
            f"TIDAK ADA HASIL RD - memakai nilai asumsi {assumed_rd_t_per_m3} t/m3 "
            "dari konfigurasi"
        )
    raise MissingDataError(
        "tidak ada hasil RD dan assumed_rd_t_per_m3 tidak diisi. Pilihannya: "
        "isi assumed_rd_t_per_m3 (dicap sebagai asumsi pada setiap keluaran), "
        "atau keluarkan seam ini dari estimasi. Tidak ada nilai bawaan."
    )
