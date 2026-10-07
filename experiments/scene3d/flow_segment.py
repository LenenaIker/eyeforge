"""Step 1.3 of the scene flow, surroundings: SAM 3 instances of each surroundings phrase from the observation
(flow_observe.py), on the photo with the movable objects erased (bg_prepare.py background.png), so a car parked
in front of a house does not cut it. Cheap gates in code first (catalog entries 4 and 7): too small, a sliver
cut by the border, or mostly inside another instance. Each candidate that passes gets a picture for the VLM
check (flow_check_masks.py). Run in the `teacher-sam3` env:

    python experiments/scene3d/flow_segment.py --dir data/street_test/flow/basque_street

Writes <dir>/env_candidates.npz (masks, phrases, scores), <dir>/env_candidates.json (per candidate: phrase,
score, area, borders touched, gate verdict and why) and <dir>/env_check/cand_NN.jpg.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from transformers import Sam3Model, Sam3Processor

from bg_prepare import segment


def borders(m):
    return [name for name, edge in (('top', m[0]), ('bottom', m[-1]), ('left', m[:, 0]), ('right', m[:, -1]))
            if edge.mean() > 0.02]


def picture(img, m, path):
    """The whole photo with the mask tinted and outlined, next to a crop around it: what the VLM judges."""
    a = np.asarray(img).astype(np.float32)
    tint = a.copy()
    tint[m] = tint[m] * 0.45 + np.array([255, 40, 200]) * 0.55
    tint[~m] *= 0.55
    full = Image.fromarray(tint.astype(np.uint8))
    ys, xs = np.nonzero(m)
    pad = int(0.08 * max(m.shape))
    box = (max(0, xs.min() - pad), max(0, ys.min() - pad), min(m.shape[1], xs.max() + pad),
           min(m.shape[0], ys.max() + pad))
    crop = Image.fromarray(a.astype(np.uint8)).crop(box)
    edge = np.zeros_like(m)
    edge[1:-1, 1:-1] = m[1:-1, 1:-1] & ~(m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:])
    cm = np.asarray(crop).copy()
    e = edge[box[1]:box[3], box[0]:box[2]]
    for dy in (-1, 0, 1):  # a 3 px outline
        cm[np.roll(e, dy, 0)] = [255, 40, 200]
    crop = Image.fromarray(cm)
    h = 512
    full = full.resize((round(full.width * h / full.height), h))
    crop = crop.resize((max(1, round(crop.width * h / crop.height)), h))
    sheet = Image.new('RGB', (full.width + crop.width + 10, h), 'white')
    sheet.paste(full, (0, 0))
    sheet.paste(crop, (full.width + 10, 0))
    sheet.save(path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--min-area', type=float, default=0.003, help='smallest instance, share of the image')
    ap.add_argument('--min-area-cut', type=float, default=0.02,
                    help='smallest instance that touches the photo border (slivers out)')
    ap.add_argument('--max', type=int, default=12, help='most candidates kept, largest first (2 min of SAM 3D each)')
    args = ap.parse_args()
    obs = json.load(open(os.path.join(args.dir, 'observe.json')))['final']
    phrases = [e['phrase'] if isinstance(e, dict) else e for e in obs['surroundings']]
    img = Image.open(os.path.join(args.dir, 'background.png')).convert('RGB')
    model = Sam3Model.from_pretrained('facebook/sam3').cuda().eval()
    proc = Sam3Processor.from_pretrained('facebook/sam3')
    cands = []
    for phrase in phrases:
        for score, m in segment(model, proc, img, phrase, args.threshold):
            cands.append({'phrase': phrase, 'score': round(score, 3), 'mask': m})
    cands.sort(key=lambda c: -c['mask'].sum())
    kept = []
    for c in cands:
        m = c['mask']
        c['area'] = round(float(m.mean()), 4)
        c['borders'] = borders(m)
        why = []
        if c['area'] < args.min_area:
            why.append('tiny')
        elif c['borders'] and c['area'] < args.min_area_cut:
            why.append('sliver cut by the border')
        for k in kept:
            inside = (m & k['mask']).sum() / max(m.sum(), 1)
            if inside > 0.6:
                why.append(f"{inside:.0%} inside a {k['phrase']} already kept")
                break
        if not why and len(kept) >= args.max:
            why.append('over the limit')
        c['gate'] = not why
        c['why'] = why
        if c['gate']:
            kept.append(c)
    os.makedirs(os.path.join(args.dir, 'env_check'), exist_ok=True)
    for i, c in enumerate(kept):
        picture(img, c['mask'], os.path.join(args.dir, 'env_check', f'cand_{i:02d}.jpg'))
    np.savez_compressed(os.path.join(args.dir, 'env_candidates.npz'),
                        masks=np.array([c['mask'] for c in kept], bool).reshape(-1, img.height, img.width),
                        phrases=np.array([c['phrase'] for c in kept]),
                        scores=np.array([c['score'] for c in kept]))
    report = [{k: v for k, v in c.items() if k != 'mask'} for c in cands]
    with open(os.path.join(args.dir, 'env_candidates.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps({'candidates': len(cands), 'kept': [c['phrase'] for c in kept],
                      'out': [(c['phrase'], c['why']) for c in cands if not c['gate']]}))


if __name__ == '__main__':
    main()
