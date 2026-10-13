"""The street's bands filled along it (flow_section.py measured them on the photo): vegetation bands with their
NVIDIA asset at the measured spacing, height and cover, kept off the road, the houses and walls and the trees the
photo already placed; grass and sidewalk bands as ground strips of their colour. Used by env_add.py over the stretch
the photo sees and by street_template.py over the whole street. Pure numpy over veg_catalog.py's proxies.
"""
import numpy as np

import veg_catalog

STRIP_LIFT = 0.03  # ground strips this far over the floor, so they do not flicker against it
GROUND_KINDS = ('grass', 'sidewalk')


def pose(base, height, proxy, up, yaw):
    """4x4 from the proxy's frame (metres, Z up, trunk base at the origin) to the place: turned by yaw about up,
    scaled to height, base on the given point."""
    e1 = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    k = float(np.clip(height / float(proxy['height_m']), 0.3, 3.0))
    ca, sa = np.cos(yaw), np.sin(yaw)
    m = np.eye(4)
    m[:3, :3] = np.stack([ca * e1 + sa * e2, -sa * e1 + ca * e2, up], 1) * k
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


def fill(section, up, floor, along, veg_dir, rng, avoid=(), avoid_m=1.5):
    """Vegetation and ground strips over the street from along[0] to along[1] metres (v, from the road's centre
    point). floor(X) gives the floor's up coordinate under each point; avoid: points no stem stands within
    avoid_m (plus its crown) of, horizontally. Returns ([{name, url, place, proxy}], [ground quads])."""
    edge, at = frame(section, up, floor)
    avoid = np.asarray(avoid, float).reshape(-1, 3)
    flat = lambda X: X - np.outer(np.atleast_2d(X) @ up, up)  # noqa: E731
    av = flat(avoid) if len(avoid) else avoid
    cat = {x['name']: x for x in veg_catalog.catalog(veg_dir)}

    objects = []
    for band in section['bands']:
        side = band['side']
        lo, hi = band_span(band, edge)
        if not band.get('pick') or band['kind'] not in ('tree', 'hedge', 'shrub'):
            continue
        proxy = np.load(veg_catalog.proxy(veg_dir, cat[band['pick']]))
        height = max(band['height_m'], 0.5)
        k = float(np.clip(height / float(proxy['height_m']), 0.3, 3.0))
        r = crown(proxy) * k
        if band['kind'] == 'tree':
            step = band.get('spacing_m') or max(2 * r, 6.0)
            step = max(step, 1.2 * r)
            keep = 1.0
        elif band['kind'] == 'hedge':  # a row of shrubs touching, gaps where the photo shows none
            step, keep = max(1.4 * r, 0.6), band['cover']
        else:
            step, keep = max(3 * r, 2.0), band['cover']
        v = along[0] + rng.uniform(0, step)
        while v < along[1]:
            if rng.uniform() < keep:
                u = side * rng.uniform(max(lo, edge[side] + 0.3), max(hi, lo + 0.1))
                base = at(u, v)
                # Stems apart as the photo's own are (a tree's spacing), not by whole crowns: crowns overlap in a row.
                gap = 0.8 * step if band['kind'] == 'tree' else avoid_m + r
                if not len(av) or np.min(np.linalg.norm(av - flat(base), axis=1)) > gap:
                    hgt = height * rng.uniform(0.85, 1.15)
                    objects.append({'name': band['pick'], 'url': cat[band['pick']]['url'],
                                    'place': pose(base, hgt, proxy, up, rng.uniform(0, 2 * np.pi)),
                                    'proxy': proxy, 'band': band['kind']})
            v += step * rng.uniform(0.75, 1.25)
    return objects, strips(section, up, floor, along)
