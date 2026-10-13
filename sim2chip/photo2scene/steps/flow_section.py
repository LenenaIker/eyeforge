"""Step 2.1c of the scene flow, the street's cross-section (user, 2026-10-13): professional scene builders lay the
road first and fill the bands beside it (verge, hedge, trees, houses) by masks, so one photo is read the same way.
SAM 3 segments every band kind on the erased photo; MoGe's points, over piece_build's floor and true vertical, give
the road's axis and width and, on each side, where each band starts and ends from the road's edge, how tall it is,
how much of the street's length it covers and, for trees, how far apart they stand. The VLM picks an NVIDIA asset
for each tree, shrub or hedge band (flow_vegetation.py). env_add.py and street_template.py fill the bands from it.
Run in the `teacher-sam3` env after piece_build.py:

    python sim2chip/photo2scene/steps/flow_section.py --dir <flow dir> --catalog <cache>/vegetation

Writes <dir>/section.json ({"road": {"centre", "along", "across", "half", "visible"}, "image_cover": {"tree",
"shrub"}, "bands": [{"side", "kind", "from_m", "to_m", "depth_m", "height_m", "cover", "spacing_m", "colour",
"pick", "picks"}]}) and <dir>/section.jpg (the photo by band and the bands seen from above).
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import Sam3Model, Sam3Processor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from sim2chip.llm import Client, ask_json, image_part, map_within  # noqa: E402
from bg_prepare import segment  # noqa: E402
from flow_segment import picture  # noqa: E402
from flow_vegetation import MODEL, PROMPT, sheet  # noqa: E402

ROAD_WORDS = ('road', 'street', 'asphalt', 'lane', 'cobble')
# Band kinds and their SAM 3 phrases, in the order a pixel claimed by two goes to (a tree over a hedge is a tree).
KINDS = {'road': None, 'sidewalk': ['sidewalk', 'pavement'], 'building': ['house', 'building'],
         'wall': ['wall', 'fence'], 'tree': ['tree'], 'hedge': ['hedge'], 'shrub': ['bush'],
         'grass': ['grass', 'lawn']}
VEG = {'tree': 'tree', 'hedge': 'shrub', 'shrub': 'shrub'}  # band kind -> catalog group
COLOURS = {'road': (90, 90, 90), 'sidewalk': (200, 200, 200), 'building': (230, 120, 60), 'wall': (180, 150, 110),
           'tree': (30, 140, 40), 'hedge': (120, 200, 60), 'shrub': (200, 230, 80), 'grass': (150, 255, 150)}
MAX_M = 60  # MoGe's points farther than this are too noisy to measure a band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--catalog', required=True, help='veg_catalog.py --dir')
    ap.add_argument('--threshold', type=float, default=0.4)
    ap.add_argument('--model', default=os.environ.get('SIM2CHIP_REVIEW_MODEL', MODEL))
    ap.add_argument('--votes', type=int, default=3)
    args = ap.parse_args()
    d = args.dir
    img = Image.open(os.path.join(d, 'background.png')).convert('RGB')
    photo = np.asarray(img)
    h, w = photo.shape[:2]
    scene = json.load(open(os.path.join(d, 'isaac', 'scene.json')))
    up = np.asarray(scene['up'], float)
    up /= np.linalg.norm(up)
    n, dd = np.asarray(scene['floor_plane'][0], float), float(scene['floor_plane'][1])
    dp = np.load(os.path.join(d, 'depth_background.npz'))
    P = dp['points']
    ok = (dp['mask'] > 0) & np.isfinite(P).all(-1) & (np.linalg.norm(np.nan_to_num(P), axis=-1) < MAX_M)
    above = (P @ n + dd) / (n @ up)  # height above the floor along up

    # Masks per kind: the road from the flow's own ground masks, the rest from SAM 3; instances kept for trees.
    g = np.load(os.path.join(d, 'ground.npz'))
    road = np.zeros((h, w), bool)
    for m, p in zip(g['masks'], g['phrases']):
        if any(x in str(p) for x in ROAD_WORDS):
            road |= m
    model = Sam3Model.from_pretrained('facebook/sam3').cuda().eval()
    proc = Sam3Processor.from_pretrained('facebook/sam3')
    masks, instances = {'road': road}, {}
    for kind, phrases in KINDS.items():
        if phrases is None:
            continue
        inst = [m for p in phrases for s, m in segment(model, proc, img, p, args.threshold) if m.sum() > 200]
        instances[kind] = inst
        masks[kind] = np.any(inst, 0) if inst else np.zeros((h, w), bool)
    del model
    torch.cuda.empty_cache()
    claimed = np.zeros((h, w), bool)
    for kind in KINDS:  # one kind per pixel, in KINDS' order
        masks[kind] = masks[kind] & ~claimed
        claimed |= masks[kind]

    # The road's axis on the floor: the long axis of its points (as street_template.py), a pointing away from the
    # camera, b to its right; u across from the road's centre line, v along.
    R = P[road & ok]
    flat = lambda X: X - np.outer(X @ up, up)  # noqa: E731
    F = flat(R)
    c = np.median(F, 0)
    ev, evec = np.linalg.eigh(np.cov((F - c).T))
    a = evec[:, np.argmax(ev)]
    a -= (a @ up) * up
    a /= np.linalg.norm(a)
    if a @ c < 0:
        a = -a
    b = np.cross(a, up)
    if b[0] < 0:  # the camera's right (+x in its frame)
        b = -b
    u_road = (F - c) @ b
    mid = (np.percentile(u_road, 3) + np.percentile(u_road, 97)) / 2
    c = c + mid * b
    u_road = u_road - mid
    half = [float(-np.percentile(u_road, 3)), float(np.percentile(u_road, 97))]  # left, right
    v_road = (F - c) @ a
    visible = [float(np.percentile(v_road, 2)), float(np.percentile(v_road, 98))]
    edge = {-1: half[0], 1: half[1]}

    bands = []
    for kind in KINDS:
        if kind == 'road':
            continue
        sel = masks[kind] & ok
        if sel.sum() < 300:
            continue
        X = P[sel]
        u, v = (flat(X) - c) @ b, (flat(X) - c) @ a
        hh = above[sel]
        col = np.median(photo[sel], 0).round().astype(int).tolist()
        for side in (-1, 1):
            on = side * u > edge[side] - 0.5
            if on.sum() < 200:
                continue
            dist = side * u[on] - edge[side]
            bins = np.arange(visible[0], visible[1] + 1, 1.0)
            seen = np.histogram(v[on], bins)[0] > 5
            band = {'side': side, 'kind': kind, 'from_m': round(float(np.percentile(dist, 10)), 2),
                    'to_m': round(float(np.percentile(dist, 90)), 2),
                    # how deep the band goes: its farthest points (a crown seen from the road shows its front)
                    'depth_m': round(float(np.percentile(dist, 98)), 2),
                    'height_m': round(float(np.percentile(hh[on], 95)), 2),
                    'cover': round(float(seen.mean()), 2), 'colour': col, 'pixels': int(on.sum())}
            if kind == 'tree':  # stems along the street: each instance's median point
                spots = []
                for m in instances.get('tree', []):
                    q = m & ok & masks['tree']
                    if q.sum() < 200:
                        continue
                    Y = flat(P[q])
                    if side * float(np.median((Y - c) @ b)) > edge[side]:
                        spots.append(float(np.median((Y - c) @ a)))
                spots.sort()
                gaps = np.diff(spots)
                band['spacing_m'] = round(float(np.median(gaps)), 1) if len(gaps) else None
                band['trees'] = len(spots)
            bands.append(band)

    # The NVIDIA asset of each vegetation band: the VLM on its largest piece (flow_vegetation.py's question).
    cat = json.load(open(os.path.join(args.catalog, 'catalog.json')))
    client = Client(usage_log=ROOT / 'runs' / 'llm_usage.jsonl')
    sheets = {}
    os.makedirs(os.path.join(d, 'section'), exist_ok=True)
    for band in bands:
        if band['kind'] not in VEG:
            continue
        group = VEG[band['kind']]
        items = [x for x in cat if x['group'] == group]
        if group not in sheets:
            sheets[group] = sheet(args.catalog, items, os.path.join(d, 'section', f'catalog_{group}.jpg'))
        side = band['side']
        best, area = None, 0
        for m in instances.get(band['kind'], []):
            q = m & masks[band['kind']] & ok
            if q.sum() > area and side * float(np.median((flat(P[q]) - c) @ b)) > edge[side] - 0.5:
                best, area = m & masks[band['kind']], int(q.sum())
        if best is None:
            continue
        pic = os.path.join(d, 'section', f"{band['kind']}_{'left' if side < 0 else 'right'}.jpg")
        picture(img, best, pic)

        def ask(t, pic=pic, kind=band['kind'], group=group, items=items):
            msg = [{'role': 'user', 'content': [{'type': 'text', 'text': PROMPT.format(p=kind, group=group)},
                                                image_part(pic), image_part(sheets[group])]}]
            okk = lambda x: isinstance(x.get('pick'), int) and 1 <= x['pick'] <= len(items)  # noqa: E731
            return ask_json(client, msg, args.model, purpose='scene_section_pick', check=okk, temperature=t,
                            max_tokens=8192)
        got = [x for x in map_within(ask, [0.0] + [0.7] * (args.votes - 1), seconds=600) if x]
        if got:
            votes = Counter(x['pick'] for x in got) + Counter(x['second'] for x in got if isinstance(
                x.get('second'), int) and 1 <= x['second'] <= len(items))
            ranked = [k for k, _ in votes.most_common()]
            band['pick'] = items[ranked[0] - 1]['name']
            band['picks'] = [items[k - 1]['name'] for k in ranked[:2]]  # two species mixed in a zone

    out = {'road': {'centre': c.tolist(), 'along': a.tolist(), 'across': b.tolist(), 'half': half,
                    'visible': visible, 'colour': np.median(photo[road], 0).round().astype(int).tolist()},
           # how much of the photo each kind of plant covers: the target the filled zones are rendered against
           'image_cover': {'tree': round(float(masks['tree'].mean()), 3),
                           'shrub': round(float((masks['hedge'] | masks['shrub']).mean()), 3)},
           'bands': bands}
    with open(os.path.join(d, 'section.json'), 'w') as f:
        json.dump(out, f, indent=1)

    # Sheet: the photo tinted by band kind, and the bands from above (u across, v along).
    tint = photo.astype(np.float32) * 0.45
    for kind, m in masks.items():
        tint[m] = tint[m] + np.array(COLOURS[kind]) * 0.55
    left = Image.fromarray(tint.clip(0, 255).astype(np.uint8)).resize((800, round(800 * h / w)))
    top = Image.new('RGB', (500, left.height), 'white')
    dr = ImageDraw.Draw(top)
    span = max(visible[1], 10.0)
    sx, sy = 500 / 40.0, left.height / (span + 5)
    cx0 = 250
    dr.rectangle([cx0 - half[0] * sx, 0, cx0 + half[1] * sx, top.height], fill=COLOURS['road'])
    for band in bands:
        x0 = edge[band['side']] + band['from_m']
        x1 = edge[band['side']] + max(band['to_m'], band['from_m'] + 0.2)
        xa, xb = sorted([cx0 + band['side'] * x0 * sx, cx0 + band['side'] * x1 * sx])
        dr.rectangle([xa, top.height * (1 - band['cover']), xb, top.height], fill=COLOURS[band['kind']],
                     outline='black')
        dr.text((xa + 2, top.height * (1 - band['cover']) + 2), band['kind'][:5], fill='black')
    dr.text((5, 5), f"road {half[0] + half[1]:.1f} m, seen {visible[0]:.0f}-{visible[1]:.0f} m", fill='black')
    sheet_img = Image.new('RGB', (left.width + top.width + 10, left.height), 'white')
    sheet_img.paste(left, (0, 0))
    sheet_img.paste(top, (left.width + 10, 0))
    sheet_img.save(os.path.join(d, 'section.jpg'), quality=90)
    print(json.dumps({'road_m': round(half[0] + half[1], 1), 'visible': [round(x) for x in visible],
                      'bands': [(b2['side'], b2['kind'], b2['from_m'], b2['to_m'], b2['height_m'], b2['cover'],
                                 b2.get('spacing_m'), b2.get('pick')) for b2 in bands]}))


if __name__ == '__main__':
    main()
