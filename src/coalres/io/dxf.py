"""Import topografi dari DXF.

Topo survei umumnya datang sebagai salah satu dari:
  - kontur (LWPOLYLINE/POLYLINE dengan elevasi), atau
  - titik spot height (POINT), atau
  - permukaan TIN (3DFACE / MESH / POLYFACE).

Modul ini menarik semua simpul ber-Z dari entitas apa pun yang dikenali,
lalu menyerahkannya sebagai awan titik untuk di-grid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # ezdxf bersifat opsional sampai benar-benar dipakai
    import ezdxf
except ImportError:  # pragma: no cover
    ezdxf = None

_SUPPORTED = {"POINT", "LWPOLYLINE", "POLYLINE", "LINE", "3DFACE", "MESH"}


def load_topo_points(
    path: str | Path, layers: list[str] | None = None
) -> tuple[np.ndarray, dict]:
    """Baca DXF, kembalikan array titik (N, 3) dan ringkasan entitas.

    Titik dengan Z = 0 dibuang bila mayoritas titik lain ber-Z bukan nol:
    kontur yang lupa diberi elevasi adalah kesalahan lazim dan akan
    membuat lubang datar di DTM.
    """
    if ezdxf is None:  # pragma: no cover
        raise ImportError("ezdxf belum terpasang: pip install ezdxf")

    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()

    coords: list[tuple[float, float, float]] = []
    counts: dict[str, int] = {}
    skipped: dict[str, int] = {}

    for entity in msp:
        etype = entity.dxftype()
        if layers and entity.dxf.layer not in layers:
            continue
        if etype not in _SUPPORTED:
            skipped[etype] = skipped.get(etype, 0) + 1
            continue

        before = len(coords)
        if etype == "POINT":
            p = entity.dxf.location
            coords.append((p.x, p.y, p.z))
        elif etype == "LINE":
            for p in (entity.dxf.start, entity.dxf.end):
                coords.append((p.x, p.y, p.z))
        elif etype == "LWPOLYLINE":
            # LWPOLYLINE adalah entitas 2D; elevasinya ada di atribut elevation.
            z = float(entity.dxf.elevation)
            for x, y, *_ in entity.get_points():
                coords.append((x, y, z))
        elif etype == "POLYLINE":
            for vertex in entity.vertices:
                p = vertex.dxf.location
                coords.append((p.x, p.y, p.z))
        elif etype == "3DFACE":
            for name in ("vtx0", "vtx1", "vtx2", "vtx3"):
                p = getattr(entity.dxf, name)
                coords.append((p.x, p.y, p.z))
        elif etype == "MESH":
            for p in entity.vertices:
                coords.append((p[0], p[1], p[2]))
        counts[etype] = counts.get(etype, 0) + (len(coords) - before)

    if not coords:
        raise ValueError(
            f"tidak ada simpul ber-koordinat yang terbaca dari {Path(path).name}. "
            f"Entitas yang dilewati: {skipped or 'tidak ada'}"
        )

    points = np.asarray(coords, dtype=float)
    points = np.unique(points, axis=0)

    zero_z = np.isclose(points[:, 2], 0.0)
    dropped_zero = 0
    if 0 < zero_z.sum() < len(points) * 0.5:
        dropped_zero = int(zero_z.sum())
        points = points[~zero_z]

    summary = {
        "n_points": len(points),
        "entities": counts,
        "skipped": skipped,
        "dropped_zero_z": dropped_zero,
        "z_min": float(points[:, 2].min()),
        "z_max": float(points[:, 2].max()),
        "extent": (
            float(points[:, 0].min()),
            float(points[:, 0].max()),
            float(points[:, 1].min()),
            float(points[:, 1].max()),
        ),
    }
    return points, summary


def topo_grid_from_dxf(path: str | Path, grid, layers: list[str] | None = None):
    """Baca DXF lalu interpolasikan ke Grid yang diberikan."""
    from ..grid import interpolate_to_grid

    points, summary = load_topo_points(path, layers=layers)
    surface = interpolate_to_grid(
        grid,
        points[:, 0],
        points[:, 1],
        points[:, 2],
        method="linear",
        max_extrapolation=None,   # DTM diisi penuh; pemotongan dilakukan di hilir
    )
    return surface, summary
