"""The street's bands filled along it (flow_section.py measured them on the photo): vegetation bands with their
NVIDIA asset at the measured spacing, height and cover, kept off the road, the houses and walls and the trees the
photo already placed; grass and sidewalk bands as ground strips of their colour. Used by env_add.py over the stretch
the photo sees and by street_template.py over the whole street. Pure numpy over veg_catalog.py's proxies.
"""
import numpy as np

import veg_catalog

STRIP_LIFT = 0.03  # ground strips this far over the floor, so they do not flicker against it
GROUND_KINDS = ('grass', 'sidewalk')


def pose(base, height, proxy, up, yaw, lean=None):
    """4x4 from the proxy's frame (metres, Z up, trunk base at the origin) to the place: turned by yaw about up,
    scaled to height, base on the given point; lean (a horizontal direction, an angle in radians) tips its top
    that way about its base."""
    e1 = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    k = float(np.clip(height / float(proxy['height_m']), 0.3, 3.0))
    ca, sa = np.cos(yaw), np.sin(yaw)
    m = np.eye(4)
    m[:3, :3] = np.stack([ca * e1 + sa * e2, -sa * e1 + ca * e2, up], 1) * k
    if lean is not None and lean[1]:
        axis = np.cross(up, lean[0])
        axis /= np.linalg.norm(axis)
        kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rot = np.eye(3) + np.sin(lean[1]) * kx + (1 - np.cos(lean[1])) * kx @ kx  # Rodrigues
        m[:3, :3] = rot @ m[:3, :3]
    m[:3, 3] = base
    return m


def crown(proxy):
    """Half the proxy's width across, metres, at scale 1."""
    p = proxy['points']
    return float(max(np.ptp(p[:, 0]), np.ptp(p[:, 1]))) / 2


def frame(section, up, floor):
    """The road's centre, along and across axes, its edge on each side and at(u, v): the floor point u metres
    across and v along."""
    road = section['road']
    c, a, b = (np.asarray(road[k], float) for k in ('centre', 'along', 'across'))
    edge = {-1: road['half'][0], 1: road['half'][1]}

    def at(u, v):
        x = c + u * b + v * a
        return x + (float(floor(x)[0]) - x @ up) * up
    at.across, at.centre = b, c
    return edge, at


def band_span(band, edge):
    """A band's inner and outer distance from the road's centre line."""
    lo = edge[band['side']] + max(band['from_m'], 0.0)
    return lo, max(edge[band['side']] + band['to_m'], lo + 0.3)


def strips(section, up, floor, along):
    """The grass and sidewalk bands as ground quads of their colour from along[0] to along[1]."""
    edge, at = frame(section, up, floor)
    out = []
    for band in section['bands']:
        if band['kind'] not in GROUND_KINDS:
            continue
        side = band['side']
        lo, hi = band_span(band, edge)
        corners = [at(side * u, v) + STRIP_LIFT * up for u, v in
                   ((lo, along[0]), (lo, along[1]), (hi, along[1]), (hi, along[0]))]
        out.append({'points': [x.tolist() for x in corners], 'uv': [[0, 0]] * 4, 'texture': None,
                    'colour': [x / 255 for x in band['colour']], 'band': band['kind']})
    return out


GROUP = {'tree': 'tree', 'hedge': 'shrub', 'shrub': 'shrub'}  # band kind -> section['image_cover'] key


def poisson(lo_u, hi_u, lo_v, hi_v, d, rng, tries=30):
    """Points (u, v) in the rectangle at least d apart, in random order (dart throwing on a grid of cells d/sqrt 2):
    the first n of them are an even scatter of n, and n + k keeps those n."""
    cell = d / np.sqrt(2)
    grid = {}
    pts = []
    for _ in range(int(tries * (hi_u - lo_u) * (hi_v - lo_v) / (d * d)) + tries):
        p = (rng.uniform(lo_u, hi_u), rng.uniform(lo_v, hi_v))
        gi, gj = int((p[0] - lo_u) // cell), int((p[1] - lo_v) // cell)
        if any(np.hypot(p[0] - q[0], p[1] - q[1]) < d for i in range(gi - 2, gi + 3) for j in range(gj - 2, gj + 3)
               if (q := grid.get((i, j))) is not None):
            continue
        grid[(gi, gj)] = p
        pts.append(p)
    return pts


LOOK = {'back': 1.4, 'tilt': 4.0, 'shrub': 1.0}  # fill()'s defaults; env_add.py tunes them to the photo
# Crowns reach over the road at most this share of its half width: the middle stays open to the far end of the
# street, its vanishing point (what a line-following model learns from; the user, 2026-10-13).
OVERHANG = 0.35


def clear_corridor(place, proxy, side, edge, at, up, tilt, rng_u, base_at):
    """A tree's pose and base whose crown keeps out of the road's open middle (OVERHANG): straightened first, then
    moved away from the road. base_at(u) gives the base for an across distance u; rng_u its first one."""
    u = rng_u
    for _ in range(12):
        m = place(base_at(u), tilt)
        pts = proxy['points'] @ m[:3, :3].T + m[:3, 3]
        inner = float(np.min(side * ((pts - at.centre) @ at.across)))
        limit = edge[side] * (1 - OVERHANG)
        if inner >= limit:
            return m
        if tilt > 0:
            tilt = max(0.0, tilt - 3.0)
        else:
            u += limit - inner + 0.2
    return m


def fill(section, up, floor, along, veg_dir, seed=0, avoid=(), avoid_m=1.5, look=None):
    """Vegetation and ground strips over the street from along[0] to along[1] metres (v, from the road's centre
    point), laid as a set dresser would (user, 2026-10-13): a tree band is a front row at the photo's height and
    spacing by the road, and behind it a back row of half as many, look['back'] times taller and leaning
    look['tilt'] degrees over the road, which closes the canopy with few trees (163 scattered ones did not, and
    left the road in the sun). A hedge is a row of touching shrubs with gaps by its cover; other shrubs a Poisson
    scatter by their cover times look['shrub'] (3 at most). floor(X) gives the floor's up coordinate under each
    point; avoid: points no stem stands within avoid_m (plus its crown; a tree, 0.7 of its spacing) of. Returns
    ([{name, url, place, proxy, band, row}], [ground quads]); the same seed gives the same plants."""
    edge, at = frame(section, up, floor)
    avoid = np.asarray(avoid, float).reshape(-1, 3)
    flat = lambda X: X - np.outer(np.atleast_2d(X) @ up, up)  # noqa: E731
    av = flat(avoid) if len(avoid) else avoid
    cat = {x['name']: x for x in veg_catalog.catalog(veg_dir)}
    look = {**LOOK, **(look or {})}

    objects = []
    for bi, band in enumerate(section['bands']):
        if not band.get('pick') or band['kind'] not in GROUP:
            continue
        rng = np.random.default_rng([seed, bi])
        side = band['side']
        lo, hi = band_span(band, edge)
        names = [n for n in band.get('picks') or [band['pick']] if n in cat]
        proxies = {n: np.load(veg_catalog.proxy(veg_dir, cat[n])) for n in names}
        height = max(band['height_m'], 0.5)
        r = max(crown(z) * float(np.clip(height / float(z['height_m']), 0.3, 3.0)) for z in proxies.values())
        lo = max(lo, edge[side] + 0.3)
        # (u, v, height, lean angle in degrees, row)
        if band['kind'] == 'tree':
            spacing = band.get('spacing_m') or 2 * r
            far = max(hi, edge[side] + band.get('depth_m', band['to_m']) + r, lo + 2 * r)
            spots = [(lo + rng.uniform(0.2, 1.0), v, height * rng.uniform(0.85, 1.1), 0.0, 'front')
                     for v in np.arange(along[0] + rng.uniform(0, spacing), along[1], spacing)]
            behind = lo + max(r, 2.0)
            spots += [(behind + rng.uniform(0, max(far - behind, 1.0)), v,
                       height * look['back'] * rng.uniform(0.9, 1.1), look['tilt'] * rng.uniform(0.7, 1.0), 'back')
                      for v in np.arange(along[0] + spacing / 2, along[1], 2 * spacing)]
            gap = 0.7 * spacing
        elif band['kind'] == 'hedge':  # a row of shrubs touching, gaps where the photo shows none
            d = max(1.4 * r, 0.6)
            spots = [(rng.uniform(lo, max(hi, lo + 0.1)), v, height * rng.uniform(0.8, 1.2), 0.0, 'hedge')
                     for v in np.arange(along[0], along[1], d) if rng.uniform() < band['cover']]
            gap = avoid_m + r
        else:
            far = max(hi, edge[side] + band.get('depth_m', band['to_m']) + r, lo + 2 * r)
            d = max(1.2 * r, 1.0)
            n = 100 * band['cover'] / (4 * d * d) * min(look['shrub'], 3.0) * (far - lo) * (along[1] - along[0]) / 100
            spots = [(u, v, height * rng.uniform(0.8, 1.2), 0.0, 'shrub')
                     for u, v in poisson(lo, far, along[0], along[1], d, rng)[:int(round(n))]]
            gap = avoid_m + r
        toward_road = -side * at.across
        for k, (u, v, hgt, tilt, row) in enumerate(spots):
            base = at(side * u, v)
            if len(av) and np.min(np.linalg.norm(av - flat(base), axis=1)) <= gap:
                continue
            name = names[k % len(names)]
            yaw = rng.uniform(0, 2 * np.pi)
            m = clear_corridor(lambda b0, t, name=name, hgt=hgt, yaw=yaw: pose(
                b0, hgt, proxies[name], up, yaw, (toward_road, np.radians(t))), proxies[name], side, edge, at, up,
                tilt, u, lambda uu, v=v: at(side * uu, v))
            objects.append({'name': name, 'url': cat[name]['url'], 'band': band['kind'], 'row': row,
                            'place': m, 'proxy': proxies[name]})
    return objects, strips(section, up, floor, along)
