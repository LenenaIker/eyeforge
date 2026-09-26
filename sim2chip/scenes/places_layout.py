"""Procedural places for the generic scene (scene.contexts): a building front with balconies, window sills and
the street in front, and a room. Pure Python (no Isaac Sim), so layouts, placement and cameras can be tested
anywhere; scenes/places.py turns the boxes into meshes.

Frame of reference: z up, meters. The building front is the plane y = 0, the building is behind it (y > 0)
and the street in front (y < 0). The room stands apart, around ROOM_AT. Every box is axis aligned and listed
under its material; surfaces are where objects can stand, each of one support kind (spec.CONTEXTS).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

ROOM_AT = (80.0, 0.0)   # the room is built away from the building: only one of them is shown per frame
TILE_M = 2.0            # texture repeat on walls and floors


@dataclass
class Surface:
    kind: str                       # "ground", "balcony_floor", "railing", "window_sill", "floor", "furniture"
    x: tuple[float, float]
    y: tuple[float, float]
    z: float
    hang: bool = False              # hung on the outside of a railing: the object's back touches y[0]
    group: int = -1                 # balcony index, to keep a close balcony view on its own balcony


@dataclass
class Layout:
    boxes: dict[str, list] = field(default_factory=dict)  # material -> [((cx, cy, cz), (sx, sy, sz))]
    surfaces: list[Surface] = field(default_factory=list)
    balconies: list[dict] = field(default_factory=list)   # {"x", "w", "d", "z"} per balcony
    size: tuple[float, float] = (0.0, 0.0)                # width and height of the building front or room

    def add(self, material: str, center, size) -> None:
        self.boxes.setdefault(material, []).append((tuple(center), tuple(size)))


def building(rng, railings: list[str] | None = None) -> Layout:
    """A block of flats: 3 to 6 floors and 3 to 6 bays; each upper bay has a balcony, a window with a sill,
    or a bare window. Balconies have a bar or a solid railing and sometimes an awning."""
    lay = Layout()
    floors, h = rng.randint(3, 6), rng.uniform(2.7, 3.2)
    bays, w = rng.randint(3, 6), rng.uniform(3.0, 4.5)
    W, H = bays * w, floors * h
    lay.size = (W, H)
    style = rng.choice(railings or ["bars", "solid"])
    p_balcony = rng.uniform(0.2, 0.8)
    lay.add("wall", (0, 0.15, (H + 0.6) / 2), (W, 0.3, H + 0.6))
    lay.surfaces.append(Surface("ground", (-W / 2 + 0.4, W / 2 - 0.4), (-1.6, -0.25), 0.0))
    for b in range(bays):
        bx = -W / 2 + (b + 0.5) * w
        # Ground floor: a door or a shop window.
        lay.add("glass", (bx, -0.02, 1.2), (rng.uniform(1.0, w - 0.8), 0.04, 2.3))
        for f in range(1, floors):
            z = f * h
            if rng.random() < p_balcony:
                d, bw = rng.uniform(1.0, 2.2), w - rng.uniform(0.2, 0.6)
                i = len(lay.balconies)
                lay.balconies.append({"x": bx, "w": bw, "d": d, "z": z})
                lay.add("slab", (bx, -d / 2, z - 0.1), (bw, d, 0.2))
                lay.add("glass", (bx, -0.02, z + 1.1), (rng.uniform(1.0, bw - 0.6), 0.04, 2.2))
                lay.surfaces.append(Surface("balcony_floor", (bx - bw / 2 + 0.2, bx + bw / 2 - 0.2), (-d + 0.25, -0.2),
                                            z, group=i))
                _railing(lay, style, bx, bw, d, z, i)
                if rng.random() < 0.35:  # awning: a sloping roof of fabric, as a flat top and a hanging front
                    ad = min(d, rng.uniform(0.8, 1.6))
                    lay.add("fabric", (bx, -ad / 2, z + 2.45), (bw * 0.9, ad, 0.03))
                    lay.add("fabric", (bx, -ad, z + 2.3), (bw * 0.9, 0.02, 0.3))
            else:
                ww = rng.uniform(1.0, min(1.8, w - 0.8))
                lay.add("glass", (bx, -0.02, z + 0.9 + 0.7), (ww, 0.04, 1.4))
                if rng.random() < 0.8:
                    sd = rng.uniform(0.18, 0.32)
                    lay.add("sill", (bx, -sd / 2, z + 0.87), (ww + 0.2, sd, 0.06))
                    lay.surfaces.append(Surface("window_sill", (bx - ww / 2, bx + ww / 2), (-sd + 0.02, -0.04), z + 0.9))
    return lay


def _railing(lay: Layout, style: str, bx: float, bw: float, d: float, z: float, group: int) -> None:
    x0, x1, top = bx - bw / 2, bx + bw / 2, z + 1.0
    if style == "solid":
        lay.add("railing", (bx, -d + 0.06, z + 0.5), (bw, 0.12, 1.0))
        for x in (x0 + 0.06, x1 - 0.06):
            lay.add("railing", (x, -d / 2, z + 0.5), (0.12, d, 1.0))
        # Pots stand on top of a solid railing (the object's centre stays on its 12 cm).
        lay.surfaces.append(Surface("railing", (x0 + 0.15, x1 - 0.15), (-d + 0.06, -d + 0.06), top, group=group))
        return
    lay.add("railing", (bx, -d + 0.02, top), (bw, 0.05, 0.05))
    lay.add("railing", (bx, -d + 0.02, z + 0.1), (bw, 0.04, 0.04))
    n = max(2, int(bw / 0.14))
    for k in range(n + 1):
        lay.add("railing", (x0 + k * bw / n, -d + 0.02, z + 0.55), (0.02, 0.02, 0.9))
    for x in (x0, x1):
        lay.add("railing", (x, -d / 2, top), (0.05, d, 0.05))
    # Planters hang on the outside of bar railings, their rim near the top rail.
    lay.surfaces.append(Surface("railing", (x0 + 0.2, x1 - 0.2), (-d, -d), top - 0.35, hang=True, group=group))


def room(rng) -> Layout:
    """A room without ceiling (the sky map lights it): floor, three walls, a window with an inner sill and
    one or two pieces of furniture against the walls."""
    lay = Layout()
    ox, oy = ROOM_AT
    sx, sy, h = rng.uniform(3.5, 7.0), rng.uniform(3.5, 7.0), 2.7
    lay.size = (sx, sy)
    lay.add("floor", (ox, oy, 0.01), (sx, sy, 0.02))
    lay.add("wall", (ox, oy + sy / 2 + 0.1, h / 2), (sx + 0.4, 0.2, h))          # back
    lay.add("wall", (ox - sx / 2 - 0.1, oy, h / 2), (0.2, sy, h))                # left
    lay.add("wall", (ox + sx / 2 + 0.1, oy, h / 2), (0.2, sy, h))                # right
    lay.surfaces.append(Surface("floor", (ox - sx / 2 + 0.3, ox + sx / 2 - 0.3), (oy - sy / 2 + 0.3, oy + sy / 2 - 0.3),
                                0.02))
    ww = rng.uniform(1.0, min(2.2, sx - 1.2))
    wx = ox + rng.uniform(-(sx - ww) / 2 + 0.3, (sx - ww) / 2 - 0.3)
    lay.add("glass", (wx, oy + sy / 2 - 0.01, 1.6), (ww, 0.04, 1.3))
    lay.add("sill", (wx, oy + sy / 2 - 0.15, 0.92), (ww + 0.2, 0.3, 0.05))
    lay.surfaces.append(Surface("window_sill", (wx - ww / 2, wx + ww / 2), (oy + sy / 2 - 0.28, oy + sy / 2 - 0.05),
                                0.945))
    for _ in range(rng.randint(1, 2)):
        fw, fd, fh = rng.uniform(0.8, 1.8), rng.uniform(0.4, 0.8), rng.uniform(0.45, 1.0)
        side = rng.choice(["left", "right"])
        fx = ox - sx / 2 + fd / 2 if side == "left" else ox + sx / 2 - fd / 2
        fy = oy + rng.uniform(-sy / 2 + fw / 2 + 0.2, sy / 2 - fw / 2 - 0.4)
        lay.add("furniture", (fx, fy, fh / 2), (fd, fw, fh))
        lay.surfaces.append(Surface("furniture", (fx - fd / 2 + 0.1, fx + fd / 2 - 0.1), (fy - fw / 2 + 0.1, fy + fw / 2 - 0.1),
                                    fh))
    return lay


def in_view(eye, look, p, hfov_deg: float, aspect: float, margin: float = 0.9) -> bool:
    """Whether point p is inside the camera's field of view (look direction from eye to look, z up)."""
    f = _norm([look[i] - eye[i] for i in range(3)])
    r = _norm([f[1], -f[0], 0.0]) if abs(f[2]) < 0.999 else [1.0, 0.0, 0.0]
    u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]
    v = [p[i] - eye[i] for i in range(3)]
    depth = sum(v[i] * f[i] for i in range(3))
    if depth <= 0.1:
        return False
    th = math.tan(math.radians(hfov_deg) / 2) * margin
    return abs(sum(v[i] * r[i] for i in range(3))) <= th * depth and \
        abs(sum(v[i] * u[i] for i in range(3))) <= th / aspect * depth


def _norm(v):
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return [c / n for c in v]


def place(surfaces: list[Surface], items: list[tuple[float, float]], rng, tries: int = 40) -> list:
    """Spots for items (footprint radius, height) on the surfaces, never overlapping. The first item goes on
    the first surface. Returns (x, y, z) per item, or None where no free spot was found."""
    taken: list[tuple[float, float, float, float]] = []  # x, y, z, radius
    out = []
    for n, (r, _h) in enumerate(items):
        pos = None
        for _ in range(tries):
            s = surfaces[0] if n == 0 else rng.choice(surfaces)
            if s.x[1] - s.x[0] < 2 * r and s.kind != "railing":
                continue
            x = rng.uniform(s.x[0] + r, s.x[1] - r) if s.x[1] - s.x[0] > 2 * r else (s.x[0] + s.x[1]) / 2
            if s.hang:
                y = s.y[0] - r - 0.02
            elif s.y[1] - s.y[0] > 2 * r:
                y = rng.uniform(s.y[0] + r, s.y[1] - r)
            else:
                y = (s.y[0] + s.y[1]) / 2
            if all(abs(z - s.z) > 0.3 or math.hypot(x - ox, y - oy) > r + orad + 0.05 for ox, oy, z, orad in taken):
                pos = (x, y, s.z)
                taken.append((x, y, s.z, r))
                break
        out.append(pos)
    return out


def facade_shot(lay: Layout, supports: list[str], distance: tuple[float, float], hfov: float, aspect: float, rng):
    """Camera on the street looking at a spot of the building front. Returns eye, look and the surfaces in
    view (of the wanted supports), the one looked at first."""
    cands = [s for s in lay.surfaces if s.kind in supports] or lay.surfaces
    focus = rng.choice(cands)
    fx, fy, fz = (focus.x[0] + focus.x[1]) / 2, (focus.y[0] + focus.y[1]) / 2, focus.z
    d = rng.uniform(*distance)
    eye = (fx + rng.uniform(-0.4, 0.4) * d, fy - d, rng.uniform(1.4, 1.8))
    look = (fx + rng.uniform(-0.5, 0.5), 0.0, fz + rng.uniform(-0.5, 0.8))
    seen = [s for s in cands if s is not focus and in_view(eye, look, _mid(s), hfov, aspect)]
    return eye, look, [focus] + seen


def balcony_shot(lay: Layout, supports: list[str], rng):
    """Camera standing on a balcony, looking out to its railing and floor (or along it)."""
    if not lay.balconies:
        return None
    i = rng.randrange(len(lay.balconies))
    b = lay.balconies[i]
    eye = (b["x"] + rng.uniform(-0.3, 0.3) * b["w"], -0.35, b["z"] + rng.uniform(1.3, 1.7))
    if rng.random() < 0.7:
        look = (b["x"] + rng.uniform(-0.3, 0.3) * b["w"], -b["d"], b["z"] + rng.uniform(0.0, 0.6))
    else:
        look = (b["x"] + rng.choice([-1, 1]) * b["w"] / 2, -b["d"] / 2, b["z"] + 0.2)
    surf = [s for s in lay.surfaces if s.group == i and s.kind in supports] or \
        [s for s in lay.surfaces if s.group == i]
    return eye, look, surf


def room_shot(lay: Layout, supports: list[str], rng):
    """Camera standing in the room near the open side, looking at a spot of the wanted supports."""
    ox, oy = ROOM_AT
    sx, sy = lay.size
    cands = [s for s in lay.surfaces if s.kind in supports] or lay.surfaces
    focus = rng.choice(cands)
    eye = (ox + rng.uniform(-0.35, 0.35) * sx, oy - sy / 2 + rng.uniform(0.1, 0.8), rng.uniform(1.3, 1.7))
    look = ((focus.x[0] + focus.x[1]) / 2, (focus.y[0] + focus.y[1]) / 2, focus.z)
    return eye, look, [focus] + [s for s in cands if s is not focus]


def _mid(s: Surface):
    return ((s.x[0] + s.x[1]) / 2, (s.y[0] + s.y[1]) / 2, s.z)


def mesh(boxes: list) -> tuple[list, list, list, list]:
    """One mesh for many boxes: points, face vertex counts, indices and texture coordinates in TILE_M units
    (each face has its own four corners, so textures keep their real-world scale on every face)."""
    pts, counts, idx, uv = [], [], [], []
    for (cx, cy, cz), (sx, sy, sz) in boxes:
        x0, x1, y0, y1, z0, z1 = cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, cz - sz / 2, cz + sz / 2
        faces = [  # corners counter-clockwise seen from outside, with their (u, v) in meters
            ([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]),  # front
            ([(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)], [(x1, z0), (x0, z0), (x0, z1), (x1, z1)]),  # back
            ([(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)], [(y1, z0), (y0, z0), (y0, z1), (y1, z1)]),  # left
            ([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], [(y0, z0), (y1, z0), (y1, z1), (y0, z1)]),  # right
            ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]),  # top
            ([(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)], [(x0, y1), (x1, y1), (x1, y0), (x0, y0)]),  # bottom
        ]
        for corners, st in faces:
            base = len(pts)
            pts += corners
            uv += [(u / TILE_M, v / TILE_M) for u, v in st]
            counts.append(4)
            idx += [base, base + 1, base + 2, base + 3]
    return pts, counts, idx, uv
