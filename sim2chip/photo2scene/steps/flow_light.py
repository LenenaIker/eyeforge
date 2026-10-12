"""Step 2.3b of the scene flow: the light of the photo, picked by measurement. piece_isaac.py --sweep renders the
photo's camera under a fixed set of lights (8 sun directions at 3 heights, and sky only); each render and the
photo are compared on the same pixel grid as flow_photometry.py, after taking each image to the photo's mean
luminance, so what is compared is where the light falls (which facades are lit, how the ground stands against
them), not how much. The light with the smallest median colour difference goes into the place's scene.json, and
view 0 is rendered with it from then on. Without it a random sun lit the main house from behind and the ground
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    args = ap.parse_args()
    d = args.dir
    src = os.path.join(d, 'place', 'isaac')
    scene = json.load(open(os.path.join(src, 'scene.json')))
    arr = np.load(os.path.join(src, 'meshes.npz'))
    w, h = scene['size']
    _, ids = render(scene, arr, items_of(scene, arr)[0], w, h)
    erased = np.asarray(Image.open(os.path.join(d, 'erased.png')).resize((w, h))) > 0
    grid = np.zeros((h, w), bool)
    grid[STEP // 2::STEP, STEP // 2::STEP] = True
    sel = grid & (ids > 0) & ~erased
    photo = to_linear(np.asarray(Image.open(os.path.join(d, 'background.png')).convert('RGB').resize((w, h)))
                      .astype(float)[sel])
    target = lab(photo)
    lights = json.load(open(os.path.join(src, 'sweep', 'lights.json')))
    scores = []
    for k, L in enumerate(lights):
        r = to_linear(np.asarray(Image.open(os.path.join(src, 'sweep', f'sweep_{k:02d}.png')).convert('RGB'))
                      .astype(float)[sel])
        r = r * (photo @ LUMA).mean() / max((r @ LUMA).mean(), 1e-6)  # same mean luminance: compare the pattern
        de = float(np.median(np.linalg.norm(lab(np.clip(r, 0, 1)) - target, axis=1)))
        scores.append({**L, 'median_de': round(de, 2)})
    best = min(scores, key=lambda s: s['median_de'])
    for name in ('scene.json', 'orig_scene.json'):
        p = os.path.join(src, name)
        if os.path.exists(p):
            sc = json.load(open(p))
            sc['light'] = {k: v for k, v in best.items() if k != 'median_de'}  # sun, or indoor intensities
            json.dump(sc, open(p, 'w'))
    out = {'best': best, 'worst_de': max(s['median_de'] for s in scores), 'lights': scores}
    json.dump(out, open(os.path.join(d, 'light.json'), 'w'), indent=1)
    print(json.dumps({'best': best, 'spread': [min(s['median_de'] for s in scores), out['worst_de']]}))


if __name__ == '__main__':
    main()
