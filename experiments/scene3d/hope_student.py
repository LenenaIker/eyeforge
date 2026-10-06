"""Does our photo-to-scene pipeline help the student? PicoDet-S 320 counting "product" on HOPE (class-agnostic),
trained as the product trains it (sim2chip/detector.py), scored on the 188 test photos of data/hope/split.json.

    python experiments/scene3d/hope_student.py ARM [--seeds 3]     # in the env with TensorFlow (edge-sdg)

Arms:
    A  real only: the 50 train photos (a fixed 20 % of them for val)
    C  A + our synthetic views (render_NN.png of data/scene3d_test/ours/hope*/isaac, boxes from view_labels.py)
    Q  C with only the scenes that pass a visual review (QC_REJECTED out), what a good coordinator would keep
    D  our synthetic views only (val: a fixed 20 % of them)
    DQ D with only the reviewed scenes
    R  A + the scenes the VLM review kept (bg_review.py, <scene>/review.json "keep")
    (B, real + the current Isaac HDRI + Objaverse synthetic, comes later.)
Truth boxes: every instance's mask_visib. The threshold is picked on val, then F1 and count error on test with
the product's point matching (sim2chip/counting.py). Results in runs/hope_student/<arm>/seed<k>/ and
runs/hope_student/<arm>.json.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim2chip import counting, detector  # noqa: E402

HOPE = ROOT / 'data/hope'
OURS = ROOT / 'data/scene3d_test/ours'
OUT = ROOT / 'runs/hope_student'
SIZE = 320
LABELS = ['background', 'product']
GRID = counting.grid_size(224)
# Reviewed by eye on 2026-10-06 (contact sheets): wrong replica and overexposed (floor with bags), the open
# cardboard box taken for a product, meshes stretched across two shelves.
QC_REJECTED = {'hope_test_000001', 'hope_test_000011', 'hope_test_000026'}
THRESHOLDS = [round(t, 2) for t in np.arange(0.1, 0.91, 0.05)]


def hope_boxes():
    """{photo path: [[x0, y0, x1, y1] in 0..1, ...]} for every HOPE photo, from mask_visib (cached)."""
    cache = OUT / 'hope_boxes.json'
    if cache.exists():
        return json.loads(cache.read_text())
    out = {}
    for rgb in sorted(glob.glob(str(HOPE / '*/*/rgb/*.png'))):
        scene, im = Path(rgb).parents[1], Path(rgb).stem
        boxes = []
        for m in sorted(glob.glob(str(scene / 'mask_visib' / f'{im}_*.png'))):
            a = np.asarray(Image.open(m)) > 0
            ys, xs = np.nonzero(a)
            if len(xs):
                h, w = a.shape
                boxes.append([xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h])
        out[rgb] = boxes
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def synthetic(skip=()):
    """Our rendered views with their boxes, every HOPE train scene but skip (view 0, the replica, included)."""
    items = []
    for lab in sorted(glob.glob(str(OURS / 'hope*/isaac/labels.json'))):
        d = Path(lab).parent
        if d.parent.name in skip or d.parent.name.endswith('_plan'):  # _plan: the same scene planned by the VLM
            continue
        w, h = json.loads((d / 'scene.json').read_text())['size']
        for name, boxes in json.loads(Path(lab).read_text()).items():
            if (d / name).exists() and boxes:
                items.append({'path': str(d / name), 'boxes': [[b[0] / w, b[1] / h, b[2] / w, b[3] / h, 0]
                                                               for b in boxes]})
    return items


def split_val(items, share=0.2):
    idx = set(np.random.default_rng(0).permutation(len(items))[:round(len(items) * share)].tolist())
    return [x for i, x in enumerate(items) if i not in idx], [x for i, x in enumerate(items) if i in idx]


def points(boxes):
    return [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 'product', b[2] - b[0], b[3] - b[1]) for b in boxes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arm', choices=['A', 'C', 'Q', 'D', 'DQ', 'R'])
    ap.add_argument('--seeds', type=int, default=3, help='repeated runs (PaddlePaddle is not seeded)')
    args = ap.parse_args()
    split = json.loads((HOPE / 'split.json').read_text())
    truth = hope_boxes()

    def photos(scenes):
        return [p for s in scenes for p in sorted(glob.glob(str(HOPE / s / 'rgb/*.png')))]

    real = [{'path': p, 'boxes': [b + [0] for b in truth[p]]} for p in photos(split['train'])]
    test = photos(split['test'])
    real_train, real_val = split_val(real)
    synth = synthetic()
    if args.arm == 'A':
        train, val = real_train, real_val
    elif args.arm == 'C':
        train, val = real_train + synth, real_val
    elif args.arm == 'Q':
        train, val = real_train + synthetic(QC_REJECTED), real_val
    elif args.arm == 'R':
        rejected = {p.parent.name for p in OURS.glob('hope*/review.json') if not json.loads(p.read_text())['keep']}
        unreviewed = {p.parent.parent.name for p in OURS.glob('hope*/isaac/labels.json')
                      if not (p.parent.parent / 'review.json').exists()}
        print('VLM rejected:', sorted(rejected), 'unreviewed (left out):', sorted(unreviewed), flush=True)
        train, val = real_train + synthetic(rejected | unreviewed), real_val
    elif args.arm == 'D':
        train, val = split_val(synth)
    else:
        train, val = split_val(synthetic(QC_REJECTED))
    print(f'arm {args.arm}: {len(train)} train, {len(val)} val, {len(test)} test', flush=True)

    results = []
    for seed in range(args.seeds):
        run = OUT / args.arm / f'seed{seed}'
        run.mkdir(parents=True, exist_ok=True)
        onnx = run / 'model.onnx'
        if not onnx.exists():
            onnx = detector.train(train, val, LABELS, SIZE, run)
        blob = detector.to_tflite(onnx, SIZE)
        out = {}
        for side, paths, tr in (('val', [v['path'] for v in val], [points(v['boxes']) for v in val]),
                                ('test', test, [points(truth[p]) for p in test])):
            raw = detector.predict(blob, detector.load([Path(p) for p in paths], SIZE))
            out[side] = (raw, tr)
        f1 = {t: counting.score([detector.decode(r, LABELS, t) for r in out['val'][0]], out['val'][1], GRID)['f1']
              for t in THRESHOLDS}
        th = detector.pick_threshold(f1)
        res = {'seed': seed, 'threshold': th,
               **counting.score([detector.decode(r, LABELS, th) for r in out['test'][0]], out['test'][1], GRID)}
        print(json.dumps(res), flush=True)
        (run / 'score.json').write_text(json.dumps(res))
        results.append(res)
    summary = {'arm': args.arm, 'train': len(train), 'val': len(val), 'seeds': results,
               'f1_mean': round(float(np.mean([r['f1'] for r in results])), 4),
               'count_mae_mean': round(float(np.mean([r['count_mae'] for r in results])), 3)}
    (OUT / f'{args.arm}.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'seeds'}))


if __name__ == '__main__':
    main()
