"""Which Token Factory VLM can replace the hand-written inputs of the photo-to-scene pipeline
(experiments/scene3d): the object phrases and the support phrase for SAM 3, the typical size of every object
(for the absolute scale that MoGe gets ~10 % wrong) and which objects are closed and roughly convex (--solid).

One call per photo and model: the clean photo plus the same photo with every SAM 3 instance outlined and numbered
(from bg_prepare's instances.npz). Run in the `teacher-sam3` env from the repo root:

    python experiments/vlm_bench/vlm_scene.py marks                       # numbered images, no API
    python experiments/vlm_bench/vlm_scene.py ask --model moonshotai/Kimi-K3 [--model ...]
    python experiments/vlm_bench/vlm_scene.py score                       # SAM 3 on the phrases + size errors

Answers in runs/vlm_bench/<model>/<case>.json, table in runs/vlm_bench/scores.json.
Truth: HOPE val scenes from the exact CAD models (instance matched to mask_visib by IoU), bottles from the
user's tape, cafe has no size truth (phrases only, against the hand-written ones).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
OURS = os.path.join(ROOT, 'data/scene3d_test/ours')
OUT = os.path.join(ROOT, 'runs/vlm_bench')
HOPE = os.path.join(ROOT, 'data/hope')

# case -> (directory under OURS, HOPE scene with full ground truth or None, hand-written phrases, hand support)
CASES = {
    'bottles': ('bottles', None, 'bottle,bottle cap,moka pot', 'table'),
    'cafe': ('005_cafe_patio', None, 'bistro chair,cafe table,patio umbrella,umbrella base,bicycle', 'sidewalk'),
    'hope021': ('hope021', None, 'can,box,bottle,carton', 'table'),
    'hope_val1': ('hope_val_000001', 'val/000001', 'can,box,bottle,carton', 'sofa'),
    'hope_val6': ('hope_val_000006', 'val/000006', 'can,box,bottle,carton', 'table'),
}
# Bottles measured with tape by the user (cm, without caps), sorted dimensions per number in marks_bottles.png.
# 1 and 3 differ by half a centimetre, so which of them is 22 cm barely matters.
BOTTLES_TRUTH = {1: [7.5, 7.5, 22.0], 2: [7.0, 7.0, 15.5], 3: [7.4, 7.4, 21.5]}

PROMPT = """You help build a 3D replica of the scene in the photo. The first image is the photo. The second is the \
same photo with some objects outlined and numbered.

Answer with one JSON object, nothing else:
{
  "support": "<the surface the movable objects rest on, 1-3 words, e.g. table, floor, shelf, lawn>",
  "phrases": ["<short noun phrase for one kind of movable object, as an open-vocabulary segmenter expects>", ...],
  "objects": [
    {"id": <number in the second image>, "name": "<what it is>",
     "height_cm": <typical real height>, "width_cm": <typical real width>, "depth_cm": <typical real depth>,
     "closed_convex": <true if it is a closed, roughly convex solid like a can, box or bottle; false for chairs, \
bikes, mugs, open containers>}
  ]
}
Rules: "phrases" must cover every movable object in the photo (not only the numbered ones), one phrase per kind, \
generic names (e.g. "can", not the brand). Sizes are what such an object measures in the real world, from your \
knowledge of the object; use the photo only to tell what it is. Give every numbered object once."""


def outline(mask, px=2):
    m = mask.astype(bool)
    edge = np.zeros_like(m)
    for dy in range(-px, px + 1):
        for dx in range(-px, px + 1):
            edge |= np.roll(np.roll(m, dy, 0), dx, 1) != m
    return edge


def marks(case):
    d = os.path.join(OURS, CASES[case][0])
    photo = Image.open(os.path.join(d, 'photo.png')).convert('RGB')
    inst = np.load(os.path.join(d, 'instances.npz'))
    arr = np.asarray(photo).copy()
    rng = np.random.default_rng(0)
    cols = rng.integers(60, 255, (len(inst['masks']), 3))
    for m, c in zip(inst['masks'], cols):
        arr[outline(m)] = c
    im = Image.fromarray(arr)
    dr = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=max(16, photo.width // 60))
    for i, m in enumerate(inst['masks']):
        ys, xs = np.nonzero(m)
        x, y = int(np.median(xs)), int(np.median(ys))
        dr.text((x, y), str(i + 1), fill=(255, 255, 255), font=font, anchor='mm', stroke_width=3, stroke_fill=(0, 0, 0))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'marks_{case}.png')
    im.save(path)
    return path


def ask(models, cases, max_tokens):
    from sim2chip.llm import Client, image_part, parse_json
    from pathlib import Path
    client = Client(usage_log=Path(OUT) / 'usage.jsonl')
    for model in models:
        for case in cases:
            dst = os.path.join(OUT, model.replace('/', '__'), f'{case}.json')
            if os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            d = os.path.join(OURS, CASES[case][0])
            msg = [{'role': 'user', 'content': [
                {'type': 'text', 'text': PROMPT},
                image_part(os.path.join(d, 'photo.png')),
                image_part(os.path.join(OUT, f'marks_{case}.png'))]}]
            t0 = time.time()
            text, answer = '', None
            try:
                text = client.chat(msg, model, purpose='vlm_bench', temperature=0.0, max_tokens=max_tokens)
                answer = parse_json(text)
            except Exception as e:  # keep going: one bad model must not stop the bench
                text = f'{type(e).__name__}: {e}\n---\n{text}'
            with open(dst, 'w') as f:
                json.dump({'model': model, 'case': case, 'seconds': round(time.time() - t0, 1),
                           'answer': answer, 'raw': text}, f, indent=1)
            print(model, case, 'ok' if answer else 'FAILED', flush=True)


def hope_truth(case):
    """HOPE instance number (1-based, as in the marks) -> sorted CAD sizes in cm, by mask IoU with mask_visib."""
    d, scene = os.path.join(OURS, CASES[case][0]), CASES[case][1]
    inst = np.load(os.path.join(d, 'instances.npz'))['masks']
    info = json.load(open(os.path.join(HOPE, 'models/models_info.json')))
    gt = json.load(open(os.path.join(HOPE, scene, 'scene_gt.json')))['0']
    h, w = inst.shape[1:]
    out = {}
    gts = []
    for n, g in enumerate(gt):
        m = Image.open(os.path.join(HOPE, scene, 'mask_visib', f'000000_{n:06d}.png')).resize((w, h), Image.NEAREST)
        gts.append((np.asarray(m) > 0, g['obj_id']))
    for i, m in enumerate(inst):
        ious = [(m & g).sum() / max(1, (m | g).sum()) for g, _ in gts]
        j = int(np.argmax(ious))
        if ious[j] >= 0.5:
            s = info[str(gts[j][1])]
            out[i + 1] = sorted([s['size_x'] / 10, s['size_y'] / 10, s['size_z'] / 10])
    return out


def size_errors(answer, truth):
    """Per matched object, |log(pred / true)| of the sorted dimensions (orientation does not matter)."""
    errs = {}
    for o in answer.get('objects', []):
        try:
            k = int(o['id'])
            pred = sorted(float(o[f]) for f in ('height_cm', 'width_cm', 'depth_cm'))
        except (KeyError, TypeError, ValueError):
            continue
        if k in truth and min(pred) > 0:
            errs[k] = [abs(math.log(p / t)) for p, t in zip(pred, truth[k])]
    return errs


def score(cases, threshold, sam=True):
    """sam=False scores only the sizes and closed_convex (no GPU)."""
    if sam:
        from transformers import Sam3Model, Sam3Processor
        sys.path.insert(0, os.path.join(ROOT, 'sim2chip/photo2scene/steps'))
        from bg_prepare import segment
        model = Sam3Model.from_pretrained('facebook/sam3').to('cuda').eval()
        proc = Sam3Processor.from_pretrained('facebook/sam3')
    cache = {}

    def union(case, phrases):
        image = Image.open(os.path.join(OURS, CASES[case][0], 'photo.png')).convert('RGB')
        masks = []
        for p in phrases:
            key = (case, p.strip().lower())
            if key not in cache:
                cache[key] = [m for s, m in segment(model, proc, image, key[1], threshold)]
            for m in cache[key]:  # one instance per object, as bg_prepare dedups
                if not any((m & k).sum() / max(1, (m | k).sum()) > 0.7 for k in masks):
                    masks.append(m)
        u = np.zeros(image.size[::-1], bool)
        for m in masks:
            u |= m
        return u, len(masks)

    def iou(a, b):
        return float((a & b).sum() / max(1, (a | b).sum()))

    rows = []
    for mdir in sorted(os.listdir(OUT)):
        if not os.path.isdir(os.path.join(OUT, mdir)):
            continue
        for case in cases:
            f = os.path.join(OUT, mdir, f'{case}.json')
            if not os.path.exists(f):
                continue
            r = json.load(open(f))
            a = r['answer'] or {}
            d = os.path.join(OURS, CASES[case][0])
            hand = np.asarray(Image.open(os.path.join(d, 'objects.png'))) > 0
            hand_n = len(np.load(os.path.join(d, 'instances.npz'))['phrases'])
            phrases = [p for p in a.get('phrases', []) if isinstance(p, str)]
            row = {'model': r['model'], 'case': case, 'seconds': r['seconds'], 'parsed': bool(a),
                   'phrases': phrases, 'support': a.get('support')}
            if sam:
                u, n = union(case, phrases) if phrases else (np.zeros_like(hand), 0)
                row.update({'objects_iou_vs_hand': round(iou(u, hand), 3), 'instances': n, 'instances_hand': hand_n})
            sup = a.get('support')
            if sam and isinstance(sup, str) and sup:
                image = Image.open(os.path.join(d, 'photo.png')).convert('RGB')
                sm = segment(model, proc, image, sup, threshold)
                hs = np.asarray(Image.open(os.path.join(d, 'support_raw.png'))) > 0
                row['support_iou_vs_hand'] = round(iou(sm[0][1], hs), 3) if sm else 0.0
            truth = hope_truth(case) if CASES[case][1] else ({int(k): v for k, v in BOTTLES_TRUTH.items()} if case == 'bottles' else {})
            if truth:
                errs = size_errors(a, truth)
                flat = [e for v in errs.values() for e in v]
                row['size_objects'] = f'{len(errs)}/{len(truth)}'
                row['size_err_median'] = round(float(np.median(flat)), 3) if flat else None
                # The scale the pipeline would take from these sizes: median ratio of the largest dimension.
                ratios = [math.log(sorted(float(o[k]) for k in ('height_cm', 'width_cm', 'depth_cm'))[2] / truth[int(o['id'])][2])
                          for o in a.get('objects', []) if str(o.get('id')).isdigit() and int(o['id']) in errs]
                row['scale_bias'] = round(math.exp(float(np.median(ratios))), 3) if ratios else None
            if CASES[case][1] or case.startswith('hope') or case == 'bottles':
                objs = a.get('objects', [])
                row['closed_convex_true'] = f"{sum(1 for o in objs if o.get('closed_convex') is True)}/{len(objs)}"
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'phrases'}), flush=True)
    with open(os.path.join(OUT, 'scores.json'), 'w') as f:
        json.dump(rows, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['marks', 'ask', 'score'])
    ap.add_argument('--model', action='append', default=[])
    ap.add_argument('--case', action='append', default=[])
    ap.add_argument('--max-tokens', type=int, default=8192, help='reasoning models think before answering')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--no-sam', action='store_true', help='score: sizes and closed_convex only, no GPU')
    args = ap.parse_args()
    cases = args.case or [c for c in CASES if os.path.exists(os.path.join(OURS, CASES[c][0], 'instances.npz'))]
    if args.cmd == 'marks':
        for c in cases:
            print(marks(c))
    elif args.cmd == 'ask':
        ask(args.model, cases, args.max_tokens)
    else:
        score(cases, args.threshold, sam=not args.no_sam)


if __name__ == '__main__':
    main()
