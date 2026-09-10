"""Kesalahan eksplisit.

Tidak ada default diam-diam di mana pun. Nilai yang tidak disediakan oleh data
atau konfigurasi harus memunculkan kesalahan, bukan jatuh ke nilai cadangan.
Setiap kelas di bawah menandai satu gerbang yang harus dilewati manusia.
"""
from __future__ import annotations


class CoalResError(Exception):
    """Akar semua kesalahan tool ini."""


class ConfigError(CoalResError):
    """Konfigurasi tidak lengkap atau tidak konsisten."""


class AuditGateError(CoalResError):
    """Gerbang audit Phase 0 tidak terlewati.

    Dinaikkan hanya untuk kondisi yang tidak boleh diselesaikan oleh kode:
    sumber koordinat, basis kedalaman, sumber ketebalan, basis RD, jumlah
    lubang minimum, dan justifikasi kondisi geologi.
    """

    def __init__(self, gate: str, message: str, remedy: str) -> None:
        self.gate = gate
        self.remedy = remedy
        super().__init__(f"[{gate}] {message}\n  Tindakan: {remedy}")


class SchemaError(CoalResError):
    """Struktur file masukan tidak sesuai yang diharapkan."""


class MissingDataError(CoalResError):
    """Data yang dibutuhkan tidak ada, dan tidak ada penggantinya yang sah."""
