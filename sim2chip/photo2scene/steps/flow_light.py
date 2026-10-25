"""Step 2.3b of the scene flow: the light of the photo, picked by measurement. piece_isaac.py --sweep renders the
photo's camera under a fixed set of lights (8 sun directions at 3 heights, full or soft, and sky only); each render
and the photo are compared on the same pixel grid as flow_photometry.py, after taking each image to the photo's mean
luminance, so what is compared is where the light falls (which facades are lit, how the ground stands against
them), not how much. The light with the smallest colour difference (each piece's median, averaged over the pieces
and objects) goes into the place's scene.json, and
view 0 is rendered with it from then on, every light scaled so the place is as bright as the photo before any
exposure gain (its "intensity"; home_office's room came out 57 times too dark for the exposure to lift). A full sun only where the photo's ground shows hard shadows
(HARD_SUN_STOPS); otherwise a soft sun or the sky alone. Without it a random sun lit the main house from behind and the ground
came out white (basque_street, 2026-10-07): per-object colour gains would have baked that shadow into the
colours. Run in the `trellis2` env:

    PYOPENGL_PLATFORM=egl python sim2chip/photo2scene/steps/flow_light.py --dir data/street_test/flow/basque_street

Writes <dir>/light.json (every light's score) and the chosen light into place/isaac/scene.json.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

from flow_gates import render
from flow_photometry import LUMA, STEP, items_of, lab, to_linear


MIN_PART = 50  # sampled pixels a piece or object needs to count in the score
# Hard sunlight is allowed only where the photo's ground shows it: its luminance from the darkest tenth to the
# brightest spanning at least this many stops (cast shadows on lit asphalt). On the 2026-10-15 baseline the two
# photos with sun shadows on the ground gave 4.1 (church_corner) and 5.0 (facade_car), the overcast, shaded or
# dusk ones 0.6 to 2.3; by the score alone every one of them took a full sun, which burnt photo 68's shaded wall
# from the photo's camera and drew black shadows in the new views (user, 2026-10-15).
HARD_SUN_STOPS = 3.0
MAX_INTENSITY = 64.0  # the lights' intensity (piece_isaac.light_scale) is kept within 1/64 to 64


def ground_contrast(d):
    """Stops between the 10th and 90th percentile of the photo's ground luminance (erased pixels left out), or
    None without ground masks."""
    gp = os.path.join(d, 'ground.npz')
    if not os.path.exists(gp):
        return None
    g = np.load(gp)['masks'].any(0)
    size = g.shape[::-1]
    img = np.asarray(Image.open(os.path.join(d, 'background.png')).convert('RGB').resize(size), np.float32)
    ep = os.path.join(d, 'erased.png')
    if os.path.exists(ep):
        g &= ~(np.asarray(Image.open(ep).resize(size)) > 0)
    y = to_linear(img[g]) @ LUMA
    y = y[y > 1e-4]
    if len(y) < 1000:
        return None
    lo, hi = np.percentile(np.log2(y), [10, 90])
    return round(float(hi - lo), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    args = ap.parse_args()
    d = args.dir
    src = os.path.join(d, 'place', 'isaac')
    scene = json.load(open(os.path.join(src, 'scene.json')))
    arr = np.load(os.path.join(src, 'meshes.npz'))
    w, h = scene['size']
    items, labels = items_of(scene, arr)
    _, ids = render(scene, arr, items, w, h)
    # All the vegetation is one part of the score: by object, the 24 NVIDIA trees of a leafy avenue outvoted its
    # road, and their dark materials chose a strong sun that burnt the road white (roads photo 29, 2026-10-14).
    veg = {e['index'] for e in scene.get('env', []) if e.get('vegetation')}
    group = np.arange(len(items) + 1)
    group[[k + 1 for k, (kind, i, _) in enumerate(labels) if kind == 'object' and i in veg]] = len(items) + 1
    ids = np.where(ids > 0, group[np.clip(ids, 0, len(items))], 0)
    erased = np.asarray(Image.open(os.path.join(d, 'erased.png')).resize((w, h))) > 0
    grid = np.zeros((h, w), bool)
    grid[STEP // 2::STEP, STEP // 2::STEP] = True
    sel = grid & (ids > 0) & ~erased
    photo = to_linear(np.asarray(Image.open(os.path.join(d, 'background.png')).convert('RGB').resize((w, h)))
                      .astype(float)[sel])
    target = lab(photo)
    who = ids[sel]
    parts = [k for k in np.unique(who) if (who == k).sum() >= MIN_PART]
    lights = json.load(open(os.path.join(src, 'sweep', 'lights.json')))
    scores = []
    for k, L in enumerate(lights):
        r = to_linear(np.asarray(Image.open(os.path.join(src, 'sweep', f'sweep_{k:02d}.png')).convert('RGB'))
                      .astype(float)[sel])
        # The render before its exposure gain, against the photo: how much stronger every light must be for the
        # place to come out as bright as the photo with no gain (the median, past clipped highlights).
        raw = np.median(r @ LUMA) / max(float(L.get('gain', 1.0)), 1e-6)
        intensity = float(np.clip(np.median(photo @ LUMA) / max(raw, 1e-9), 1 / MAX_INTENSITY, MAX_INTENSITY))
        r = r * (photo @ LUMA).mean() / max((r @ LUMA).mean(), 1e-6)  # same mean luminance: compare the pattern
        err = np.linalg.norm(lab(np.clip(r, 0, 1)) - target, axis=1)
        # Each piece and object counts the same, by its own median: by pixels the facades filling a street photo
        # chose a bright sky whose painted sun burnt the ground white (the pots' photo 27, 2026-10-13).
        de = float(np.mean([np.median(err[who == k]) for k in parts]))
        scores.append({**{k2: v for k2, v in L.items() if k2 != 'gain'}, 'intensity': round(intensity, 3),
                       'median_de': round(de, 2)})
    contrast = ground_contrast(d)
    soft = contrast is not None and contrast < HARD_SUN_STOPS
    allowed = [s for s in scores if not (soft and s.get('sun') is True)] or scores
    best = min(allowed, key=lambda s: s['median_de'])
    for name in ('scene.json', 'orig_scene.json'):
        p = os.path.join(src, name)
        if os.path.exists(p):
            sc = json.load(open(p))
            sc['light'] = {k: v for k, v in best.items() if k != 'median_de'}  # sun, or indoor intensities
            json.dump(sc, open(p, 'w'))
    out = {'best': best, 'worst_de': max(s['median_de'] for s in scores), 'ground_contrast_stops': contrast,
           'hard_sun_allowed': not soft, 'lights': scores}
    json.dump(out, open(os.path.join(d, 'light.json'), 'w'), indent=1)
    print(json.dumps({'best': best, 'spread': [min(s['median_de'] for s in scores), out['worst_de']]}))


if __name__ == '__main__':
    main()
