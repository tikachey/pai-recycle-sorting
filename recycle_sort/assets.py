"""Procedural assets: lathe (surface-of-revolution) meshes."""

from __future__ import annotations

import numpy as np


def lathe_mesh(profile: list[tuple[float, float]], segments: int = 32):
    """Return (vertices, faces) of a closed surface of revolution.

    profile: list of (radius, z) from bottom to top. The first and last
    points are capped with a center vertex.
    """
    verts = []
    for r, z in profile:
        for k in range(segments):
            a = 2 * np.pi * k / segments
            verts.append((r * np.cos(a), r * np.sin(a), z))
    n_ring = len(profile)
    bottom = len(verts)
    verts.append((0.0, 0.0, profile[0][1]))
    top = len(verts)
    verts.append((0.0, 0.0, profile[-1][1]))

    faces = []
    for i in range(n_ring - 1):
        for k in range(segments):
            a = i * segments + k
            b = i * segments + (k + 1) % segments
            c = (i + 1) * segments + k
            d = (i + 1) * segments + (k + 1) % segments
            faces.append((a, b, d))
            faces.append((a, d, c))
    for k in range(segments):
        faces.append((bottom, (k + 1) % segments, k))
        base = (n_ring - 1) * segments
        faces.append((top, base + k, base + (k + 1) % segments))
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)
