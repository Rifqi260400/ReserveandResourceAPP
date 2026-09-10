"""Pembaca topografi DXF.

Melaporkan jenis entitas yang ditemukan dan jumlahnya, sesuai permintaan audit.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import ezdxf
import numpy as np

from ..errors import SchemaError

SUPPORTED = {"POINT", "LWPOLYLINE", "POLYLINE", "LINE", "3DFACE", "MESH"}


@dataclass
class TopoPoints:
    path: Path
    points: np.ndarray                       # (N, 3)
    entity_counts: dict[str, int] = field(default_factory=dict)
    skipped_entities: dict[str, int] = field(default_factory=dict)
    layers: dict[str, int] = field(default_factory=dict)
    dropped_zero_z: int = 0

    @property
    def extent(self) -> tuple[float, float, float, float]:
        p = self.points
        return (float(p[:, 0].min()), float(p[:, 0].max()),
                float(p[:, 1].min()), float(p[:, 1].max()))

    @property
    def z_range(self) -> tuple[float, float]:
        return (float(self.points[:, 2].min()), float(self.points[:, 2].max()))


def load_topography(path: str | Path, layers: list[str] | None = None) -> TopoPoints:
    path = Path(path)
    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()

    coords: list[tuple[float, float, float]] = []
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    layer_counts: Counter[str] = Counter()

    for entity in msp:
        etype = entity.dxftype()
        layer = str(getattr(entity.dxf, "layer", ""))
        if layers and layer not in layers:
            continue
        if etype not in SUPPORTED:
            skipped[etype] += 1
            continue

        before = len(coords)
        if etype == "POINT":
            p = entity.dxf.location
            coords.append((p.x, p.y, p.z))
        elif etype == "LINE":
            for p in (entity.dxf.start, entity.dxf.end):
                coords.append((p.x, p.y, p.z))
        elif etype == "LWPOLYLINE":
            # LWPOLYLINE bersifat 2D; elevasinya ada di atribut elevation.
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
                coords.append((float(p[0]), float(p[1]), float(p[2])))

        added = len(coords) - before
        counts[etype] += 1
        layer_counts[layer] += added

    if not coords:
        raise SchemaError(
            f"{path.name}: tidak ada simpul berkoordinat yang terbaca. "
            f"Entitas yang dilewati: {dict(skipped) or 'tidak ada'}"
        )

    points = np.unique(np.asarray(coords, dtype=float), axis=0)

    # Kontur yang lupa diberi elevasi adalah kesalahan lazim dan akan membuat
    # bidang datar palsu di DTM. Dibuang hanya bila jelas minoritas.
    zero = np.isclose(points[:, 2], 0.0)
    dropped = 0
    if 0 < zero.sum() < len(points) * 0.5:
        dropped = int(zero.sum())
        points = points[~zero]

    return TopoPoints(
        path=path, points=points, entity_counts=dict(counts),
        skipped_entities=dict(skipped), layers=dict(layer_counts),
        dropped_zero_z=dropped,
    )
