"""A straight street from one photo, so the place can be seen from anywhere along it (template `straight_street`
of the scene flow; crossroad.py is the corner one).

The street's direction is the long axis of the road pixels on the ground plane, which holds whether the photo
looks along the street or straight at a facade across it. The surroundings meshes (env_add.py) are split by the
side of the street they stand on. A side the photo does not show is the shown side turned 180 degrees about the
street's centre line (turned, not mirrored: text keeps its handedness). Each side is repeated along the street,
every copy with another brick (crossroad.rebrick). Sidewalks take the photo's ground material, the road is
asphalt a curb lower. Views are cameras on the road or the sidewalks, looking along the street or at a side.
Run in the `trellis2` env:

    python street_template.py --scene data/street_test/flow/basque_street/place \
        --photo-scene data/street_test/flow/basque_street --out data/street_test/flow/basque_street/street

Writes <out>/isaac/{scene.json,meshes.npz}, <out>/photo.png and <out>/template.json.
"""
import argparse
import json
import math
import os
import re
import shutil

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import section_fill
from crossroad import (ASPHALT, BRICKS, ROAD_WORDS, _axis_rotation, affine, lifts, look, photo_keys, quad, rebrick,
                       usd_keys, usd_of)

SEE_CONE, SEE_M, SEE_SHARE = 30.0, 40.0, 0.02  # a view must have this much of the scene ahead (sees_enough)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True, help='an env_add.py scene (the place from the photo camera)')
    ap.add_argument('--photo-scene', required=True, help='its flow directory (depth, ground masks, pieces)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--repeats', type=int, default=2, help='copies of each side before and after the original')
    ap.add_argument('--curb', type=float, default=0.12)
    ap.add_argument('--reach', type=float, default=40.0, help='sidewalk depth beyond the curb, metres')
    ap.add_argument('--views', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    src = os.path.join(args.scene, 'isaac')
    scene = json.load(open(os.path.join(src, 'scene.json')))
    arr = np.load(os.path.join(src, 'meshes.npz'))
    up = np.asarray(scene['up'], float)
    up /= np.linalg.norm(up)
    names = {e['index']: e['name'] for e in scene.get('env', [])}
    floor = scene['pieces'][0]
    level = float(np.mean(np.asarray(floor['points']) @ up))

    # Ground points of the photo, the road ones apart when a ground phrase names a road.
    dp = np.load(os.path.join(args.photo_scene, 'depth_background.npz'))
    g = np.load(os.path.join(args.photo_scene, 'ground.npz'))
    phrases = [str(p) for p in g['phrases']]
    road_sel = [i for i, p in enumerate(phrases) if any(w in p for w in ROAD_WORDS)]
    road = np.any(g['masks'][road_sel], 0) if road_sel else np.any(g['masks'], 0)
    pts = dp['points'][road & (dp['mask'] > 0)]
    pts = pts[np.isfinite(pts).all(1)]
    pts = pts[np.linalg.norm(pts, axis=1) < 40]  # far road pixels are few and noisy
    flat = pts - (pts @ up)[:, None] * up
    c = flat.mean(0)
    # The street's direction: the road's long axis on the ground (PCA of its points).
    ev, evec = np.linalg.eigh(np.cov((flat - c).T))
    a = evec[:, np.argmax(ev)]
    a -= (a @ up) * up
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    if b @ (-c) < 0:  # b points from the street's centre towards the camera's side
        b = -b
    across = (flat - c) @ b
    lo_b, hi_b = np.percentile(across, [5, 95])
    width = float(np.clip(hi_b - lo_b, 3.5, 14.0)) if road_sel else 7.0
    centre = c + (lo_b + hi_b) / 2 * b if road_sel else c
    centre = centre - (centre @ up) * up + level * up

    # Surroundings by side; their bases set on the sidewalk level (far ones come out floating or sunk).
    sides = {1: [], -1: []}
    moves = lifts(scene, arr, up, level)  # a building on a raised lot moves with it
    for i, t in scene['views'][0]['objects']:
        if i not in names:
            continue
        p, move = moves[i]
        s = 1 if float((p.mean(0) - centre) @ b) >= 0 else -1
        sides[s].append((p, arr[f'obj{i}_faces'], arr[f'obj{i}_colors'], names[i], photo_keys(arr, i),
                         usd_of(arr, i, move @ np.asarray(t))))
    # Facade planes env_add.py kept where no mesh stands: they go with their side, like the meshes.
    facades = {1: [], -1: []}
    for p in scene['pieces']:
        if re.search(r'tex_wall\d+\.png$', p['texture'] or ''):
            q = np.asarray(p['points'])
            facades[1 if float((q.mean(0) - centre) @ b) >= 0 else -1].append(p)
    shown = [s for s in (1, -1) if sides[s] or facades[s]]
    if not shown:
        raise SystemExit('no surroundings mesh to build a street from')
    if len(shown) == 1:  # the other side: the shown one turned 180 degrees about the centre line's vertical
        s0 = shown[0]
        # The road's own pixels rarely show its whole width (a parked car, the camera on one lane): the camera
        # stands in the street, so the centre line goes under it and the road reaches to the shown facades less
        # a sidewalk. A centre line from the road pixels put the turned copy on top of the camera (basque_street).
        pts_s = np.concatenate([p for p, *_ in sides[s0]] + [np.asarray(fp['points']) for fp in facades[s0]])
        # The facade line: the block's points nearest the street (signed b coordinate from the old centre).
        facade_b = s0 * float(np.percentile(s0 * ((pts_s - centre) @ b), 2))
        cam_b = float((np.zeros(3) - centre) @ b)
        gap = abs(cam_b - facade_b)
        centre = centre + cam_b * b
        half = float(np.clip(gap - 1.5, 2.0, 7.0))
        width = 2 * half
        rot = np.asarray(_axis_rotation(up, math.pi))
        # The turned side is a copy: rebricked, without the photo's texture.
        turn = affine(rot, centre - rot @ centre)
        sides[-s0] = [((p - centre) @ rot.T + centre, f, col, n, {},
                       usd and {'url': usd['url'], 'pose': turn @ usd['pose']})
                      for p, f, col, n, _, usd in sides[s0]]
        facades[-s0] = [{**fp, 'points': ((np.asarray(fp['points']) - centre) @ rot.T + centre).tolist()}
                        for fp in facades[s0]]

    def along_street(fp):
        c = np.asarray(fp['points'])
        n = np.cross(c[1] - c[0], c[3] - c[0])
        return abs(n @ a) < 0.35 * np.linalg.norm(n)

    out_arr, objects = {}, 0
    lengths, facade_quads = {}, []
    originals = [x for x in (1, -1) if x in shown]

    def clear_of_road(p):
        """A copied object stands off the road and away from the photo's camera. The street's axis comes from the
        road's pixels, and a side lane twisted it some 30 degrees on the roads photo: copies moved "along" it
        crossed the road and a shrub stood on the camera, view 0 black (2026-10-13)."""
        h = p @ up
        base = p[h < h.min() + 0.5]
        if np.abs((base - centre) @ b).min() < width / 2 + 0.3:
            return False
        return float(np.linalg.norm(p, axis=1).min()) > 2.0  # the photo's camera is the origin

    for s, block in sides.items():
        allp = np.concatenate([p for p, *_ in block] + [np.asarray(fp['points']) for fp in facades[s]])
        along = (allp - centre) @ a
        a0, a1 = np.percentile(along, [1, 99])
        L = float(max(a1 - a0, 8.0))
        lengths[s] = L
        for k in range(-args.repeats, args.repeats + 1):
            brick = BRICKS[0] if k == 0 else BRICKS[1 + rng.integers(len(BRICKS) - 1)]
            shift = k * L * a
            for fp in facades[s]:
                # Copies along the street only for a facade that runs along it: an oblique one (a far house seen
                # at an angle), moved by a block's length, crossed the road in front of the camera (basque_street).
                if k and not along_street(fp):
                    continue
                facade_quads.append({**fp, 'points': (np.asarray(fp['points']) + shift).tolist()})
            for p, f, col, n, photo, usd in block:
                if (k or s not in originals) and not clear_of_road(p + shift):
                    continue
                if any(w in n for w in ('house', 'wall', 'building')):
                    col = rebrick(col, brick)
                out_arr[f'obj{objects}_points'] = (p + shift).astype(np.float32)
                out_arr[f'obj{objects}_faces'] = f
                out_arr[f'obj{objects}_colors'] = np.asarray(col, np.float32)
                out_arr.update(usd_keys(objects, usd, affine(move=shift)))
                # The block where the photo has it keeps the photo's texture; copies are rebricked. A tiled
                # texture (a lot's wall, env_add.wall_tile) fits anywhere.
                if k == 0 or 'tile' in photo:
                    out_arr.update({f'obj{objects}_{key}': v for key, v in photo.items()})
                objects += 1

    # Ground: the road along the street, a sidewalk with the photo's material on each side, curb faces.
    span = (args.repeats + 1) * max(lengths.values()) + 20
    half = width / 2
    road_pts = [centre + x * span * a + y * (half + args.reach) * b - args.curb * up
                for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    # The photo's light on its ground (flow_photometry.py baked it into the photo's own ground): the template's
    # road, sidewalks and land take it too. Without it a sunlit road came out white past the photo's own patch
    # (roads photo 12, 2026-10-14).
    gp = next((p for p in scene['pieces'] if p.get('ground_photo')), None)
    lit = np.asarray((gp or {}).get('baked_gain') or [1.0, 1.0, 1.0], float)
    lit_colour = lambda c: section_fill.to_srgb(section_fill.to_linear(c) * lit).tolist()  # noqa: E731
    lit_tag = 'lit_' + '_'.join(f'{x:.3f}' for x in lit)
    pieces = [quad(road_pts, colour=lit_colour(ASPHALT))] + facade_quads
    size = json.load(open(os.path.join(args.photo_scene, 'pieces.json')))['materials']['floor'].get('size_m', [1, 1])
    sec_path = os.path.join(args.photo_scene, 'section.json')
    section = json.load(open(sec_path)) if os.path.exists(sec_path) else None
    for s in (1, -1):
        c0 = centre + s * half * b
        # With the photo's street bands (flow_section.py) the sidewalk is as wide as the photo's and the land past
        # it takes its outer bands' colour: a 60 m sidewalk of the photo's asphalt read as a square (roads,
        # 2026-10-13).
        walk, land = args.reach, None
        if section:
            same = int(np.sign(b @ np.asarray(section['road']['across'])))  # this template's b against the section's
            mine = [x for x in section['bands'] if x['side'] == s * same]
            walks = [x['to_m'] for x in mine if x['kind'] == 'sidewalk' and x['to_m'] > 0.3]
            walk = float(np.clip(max(walks) if walks else 1.0, 0.5, 4.0))
            outer = [x for x in mine if x['kind'] in ('grass', 'hedge', 'shrub', 'tree')]
            land = (np.mean([x['colour'] for x in outer], 0) / 255 * 0.8).tolist() if outer else [0.32, 0.36, 0.24]
        side = [c0 - span * a, c0 + span * a, c0 + span * a + s * walk * b, c0 - span * a + s * walk * b]
        uv = [[0, 0], [2 * span / size[0], 0], [2 * span / size[0], walk / size[1]], [0, walk / size[1]]]
        tex = floor['texture'] and (section_fill.tinted(floor['texture'], lit, lit_tag) if np.any(lit != 1)
                                    else floor['texture'])
        pieces.append(quad(side, uv, tex, lit_colour(floor['colour'])))
        pieces[-1]['gain'] = floor.get('gain')  # the photometry gain of the photo's ground
        if land is not None:
            c1 = c0 + s * walk * b
            pieces.append(quad([c1 - span * a, c1 + span * a, c1 + span * a + s * (args.reach - walk) * b,
                                c1 - span * a + s * (args.reach - walk) * b], colour=land))
            # The land past the sidewalk is the photo's grass when it has some: textured, not a flat green.
            pieces[-1]['colour'] = lit_colour(land)
            tex = section_fill.ground_texture(section, 'grass', lit_colour(land))
            if tex:
                k = 2 * span / section_fill.TILE_M, (args.reach - walk) / section_fill.TILE_M
                pieces[-1].update(texture=tex, uv=[[0, 0], [k[0], 0], [k[0], k[1]], [0, k[1]]])
        pieces.append(quad([c0 - span * a, c0 + span * a, c0 + span * a - args.curb * up,
                            c0 - span * a - args.curb * up], colour=lit_colour([0.55, 0.54, 0.52])))
    # The photo's own ground seen from above (piece_build.ground_photo) where the photo stands: its side lane and
    # its cracks. The template's ground is level and the road a curb lower: lifted to stay over both.
    for p in scene['pieces']:
        if p.get('ground_photo'):
            q = np.asarray(p['points'], float)
            pieces.append({**p, 'points': (q + (level + 0.06 - q @ up)[:, None] * up).tolist()})
    # The photo's street bands (flow_section.py): its asphalt's colour, grass and sidewalk strips the whole length.
    if section:
        pieces[0]['colour'] = lit_colour([x / 255 for x in section['road']['colour']])
        lit_band = lambda x: {**x, 'colour': [v * 255 for v in lit_colour([c / 255 for c in x['colour']])]}  # noqa
        lit_section = {**section, 'bands': [lit_band(x) for x in section['bands']]}
        pieces += section_fill.strips(lit_section, up, lambda X: np.full(len(np.atleast_2d(X)), level), (-span, span))

    # Views: the photo's camera, then cameras over the road and sidewalks looking along the street or at a side.
    def quad_points(fp, n=300):
        c = np.asarray(fp['points'])
        uv = rng.uniform(0, 1, (n, 2))
        return c[0] + uv[:, :1] * (c[1] - c[0]) + uv[:, 1:] * (c[3] - c[0])
    env_pts = np.concatenate([quad_points(fp) for fp in facade_quads] + [trimesh.sample.sample_surface(
        trimesh.Trimesh(out_arr[f'obj{i}_points'], out_arr[f'obj{i}_faces'], process=False), 2000, seed=0)[0]
        for i in range(objects)])
    tree = cKDTree(env_pts)
    clearance = float(np.clip(half, 1.5, 5.0))  # narrow lanes leave little room
    views, tries = [{'pose': np.eye(4).tolist(), 'objects': [], 'lamp': None}], 0
    reach_a = (args.repeats + 0.5) * min(lengths.values())

    def sees_enough(pos, fwd):
        """Houses, walls or plants in the middle of the view: a share of the scene's surface samples within
        SEE_CONE degrees of the view axis and SEE_M metres."""
        d = env_pts - pos
        dist = np.linalg.norm(d, axis=1)
        cos = (d @ fwd) / np.maximum(dist, 1e-6)
        return np.mean((cos > math.cos(math.radians(SEE_CONE))) & (dist < SEE_M)) >= SEE_SHARE
    while len(views) <= args.views and tries < 5000:
        tries += 1
        # Mostly a person on the road looking along it, so the far end of the street (its vanishing point) and the
        # lines lead the eye; some views turn to a side. Cameras up to 4 m and pitched 3 to 18 degrees down mostly
        # saw the ground, and side views over the land past the sidewalk a flat green band (pots, 2026-10-14).
        along = rng.random() < 0.7
        across = rng.uniform(-half, half) if along else rng.uniform(-half - 2, half + 2)
        pos = centre + rng.uniform(-reach_a, reach_a) * a + across * b + rng.uniform(1.4, 2.2) * up
        if tree.query(pos)[0] < clearance:
            continue
        if along:  # either way
            yaw = (0 if rng.random() < 0.5 else math.pi) + math.radians(rng.uniform(-20, 20))
        else:  # at one side
            yaw = (math.pi / 2 if rng.random() < 0.5 else -math.pi / 2) + math.radians(rng.uniform(-40, 40))
        pitch = math.radians(rng.uniform(-4, 8))
        fwd = math.cos(pitch) * (math.cos(yaw) * a + math.sin(yaw) * b) - math.sin(pitch) * up
        if not sees_enough(pos, fwd):
            continue
        views.append({'pose': look(pos, fwd, up).tolist(), 'objects': [], 'lamp': None})
    for v in views:
        v['objects'] = [[i, np.eye(4).tolist()] for i in range(objects)]

    os.makedirs(os.path.join(args.out, 'isaac'), exist_ok=True)
    scene.update(objects=objects, views=views, pieces=pieces, env=[], indoor=False, lamp=None)
    scene.pop('floor_plane', None)  # the photo's floor; the template's ground is level
    json.dump(scene, open(os.path.join(args.out, 'isaac', 'scene.json'), 'w'))
    np.savez_compressed(os.path.join(args.out, 'isaac', 'meshes.npz'), **out_arr)
    shutil.copy(os.path.join(args.scene, 'photo.png'), args.out)
    rep = {'width_m': round(width, 2), 'road_phrases': [phrases[i] for i in road_sel],
           'sides_shown': shown, 'block_length_m': {str(k): round(v, 1) for k, v in lengths.items()},
           'meshes': objects, 'views': len(views), 'tries': tries, 'clearance_m': round(clearance, 2)}
    json.dump(rep, open(os.path.join(args.out, 'template.json'), 'w'), indent=1)
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
