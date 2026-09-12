"""Tahap 8: radius klasifikasi menurut kondisi geologi.

Pencariannya FUNGSI MURNI: masuk kondisi geologi dan kelas, keluar meter. Tidak
ada `Config`, tidak ada kelas bawaan, tidak ada jalan pintas. Yang boleh diubah
manusia adalah hasil PENILAIAN kompleksitas di tahap 7 - bukan angka radius yang
menempel pada tiap kelas, dan bukan kelas bawaan ketika penilaian belum ada.

Pemisahan itu penting: mengubah penilaian meninggalkan jejak (skor, justifikasi,
dan alasan penimpaan), sedangkan mengubah angka radius tidak meninggalkan apa
pun yang dapat diperiksa.
"""
from __future__ import annotations

from typing import Literal

GeologicalCondition = Literal["sederhana", "moderat", "kompleks"]
ResourceClass = Literal["terukur", "tertunjuk", "tereka"]

# Radius kumulatif dari titik observasi, dalam meter, menurut SNI 5015:2019.
#
# PERINGATAN: angka di bawah ini BELUM diverifikasi terhadap teks standar. Ia
# disalin dari konfigurasi kerja proyek, bukan dari dokumen SNI. Jangan
# perlakukan sebagai otoritatif sampai diverifikasi.
SNI_5015_2019_RADII_M: dict[str, dict[str, float]] = {
    "sederhana": {"terukur": 500.0, "tertunjuk": 1000.0, "tereka": 1500.0},
    "moderat": {"terukur": 250.0, "tertunjuk": 500.0, "tereka": 1000.0},
    "kompleks": {"terukur": 100.0, "tertunjuk": 200.0, "tereka": 400.0},
}

TABLE_UNVERIFIED_WARNING = (
    "Tabel radius SNI_5015_2019_RADII_M BELUM diverifikasi terhadap teks SNI "
    "5015:2019. Verifikasi ke dokumen standar sebelum hasil dipakai untuk RKAB "
    "atau laporan Competent Person."
)


def radius_m(condition: GeologicalCondition, klass: ResourceClass) -> float:
    """Radius kumulatif, dalam meter. Fungsi murni."""
    try:
        return SNI_5015_2019_RADII_M[condition][klass]
    except KeyError as exc:
        raise ValueError(
            f"kombinasi tidak dikenal: kondisi='{condition}', kelas='{klass}'. "
            f"Kondisi yang sah: {sorted(SNI_5015_2019_RADII_M)}; "
            f"kelas: {sorted(next(iter(SNI_5015_2019_RADII_M.values())))}."
        ) from exc


def radii_for(condition: GeologicalCondition) -> dict[str, float]:
    """Ketiga radius untuk satu kondisi geologi."""
    return dict(SNI_5015_2019_RADII_M[condition])


def compare_to_table(condition: GeologicalCondition,
                     configured: dict[str, float]) -> list[str]:
    """Selisih antara radius di konfigurasi dan tabel standar.

    Konfigurasi boleh menimpa - kadang ada dasar teknis untuk itu - tetapi
    penimpaan yang tidak disadari adalah cara paling mudah menghasilkan angka
    yang salah tanpa gejala. Tiap selisih dikembalikan untuk dicetak.
    """
    table = SNI_5015_2019_RADII_M[condition]
    return [
        f"radius {klass} di konfigurasi {configured[klass]:g} m, tabel "
        f"{table[klass]:g} m (selisih {configured[klass] - table[klass]:+g} m)"
        for klass in ("terukur", "tertunjuk", "tereka")
        if klass in configured and abs(configured[klass] - table[klass]) > 1e-9
    ]
