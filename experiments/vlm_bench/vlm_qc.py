"""Can a Token Factory VLM coordinate the photo-to-scene pipeline as Claude did by hand on 2026-10-06? Two checks
on the HOPE train scenes (data/scene3d_test/ours/hope_*):

  precheck  the photo only: does a single flat support hold the objects (one table top, one shelf board)? Bags,
            soft cushions, boxes the objects sit in, or two levels break our one-plane scene.
  review    a sheet with the photo (top left) and 5 generated views with their label boxes: good, fair or bad,
            and the defects.
  rate      the same sheet scored 0-10 as training data, told that a synthetic look is normal (the binary review
            called nearly everything bad); scored by how well it ranks usable scenes above bad ones.

Truth is the review written by eye in experiments/NOTES.md (to be checked by the user). Run from the repo root in
any env with numpy and Pillow:

    python experiments/vlm_bench/vlm_qc.py sheets
    python experiments/vlm_bench/vlm_qc.py ask --model moonshotai/Kimi-K3 [--model ...]
    python experiments/vlm_bench/vlm_qc.py rate --model ...
    python experiments/vlm_bench/vlm_qc.py score

Answers in runs/vlm_bench/qc/<model>/<scene>.json, table in runs/vlm_bench/qc/scores.json.
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
OURS = os.path.join(ROOT, 'data/scene3d_test/ours')
OUT = os.path.join(ROOT, 'runs/vlm_bench/qc')
VIEWS = ['render_00.png', 'render_04.png', 'render_08.png', 'render_12.png', 'render_16.png']

# By eye, 2026-10-06. precheck: True if one flat support fits; review: good / fair / bad.
TRUTH = {
    'hope_val_000001': (False, 'fair'),   # sofa: seat and backrest, objects float in front of the backrest
    'hope_val_000006': (True, 'good'),
    'hope_test_000001': (False, 'bad'),   # objects inside plastic bags on the floor; wrong replica, overexposed
    'hope_test_000006': (True, 'good'),
    'hope_test_000011': (False, 'bad'),   # objects inside an open cardboard box, which became a "product"
    'hope_test_000016': (True, 'fair'),   # cartons and bottles far too large in close views
    'hope_test_000026': (False, 'bad'),   # two shelves: stretched meshes
    'hope_test_000031': (True, 'good'),
    'hope_test_000036': (True, 'fair'),   # windowsill, some objects large
}

PRECHECK = """We rebuild a scene from this single photo: the movable objects are cut out, and copies of them are \
placed again on ONE flat support plane (a table top, a floor, one shelf board) to render new training images.

Does one flat support plane hold all the movable objects in this photo? It does not if the objects sit in bags, \
in an open box, on soft or uneven cushions, or on two different levels (e.g. two shelves).
Answer with one JSON object, nothing else:
{"one_flat_support": true or false, "support": "<the surface, 1-3 words>", "reason": "<one sentence>"}"""

REVIEW = """We generate synthetic training images for a small object detector from one real photo. The sheet shows \
the real photo (top left) and 5 generated images (the others). Red boxes are the labels the detector will learn \
from; every movable product must have one box and nothing else should.

Judge whether the generated images are good training data compared with the real photo: objects resting \
plausibly on the support (not floating, not sunk), sizes consistent with the real photo, believable lighting and \
colours, correct and tight boxes, no broken or stretched meshes, nothing that is part of the background (a box the \
products sit in, furniture) labelled as a product.
Answer with one JSON object, nothing else:
{"verdict": "good" | "fair" | "bad", "defects": ["<short phrase>", ...]}
good: usable as is. fair: usable with visible flaws. bad: would teach the detector wrong things; discard."""

RATE = """We generate synthetic training images for a small object detector from one real photo. The sheet shows \
the real photo (top left) and 5 generated images (the others). Red boxes are the labels the detector will learn \
from.

Synthetic images never look fully real: flat lighting, slightly odd textures and missing contact shadows are \
normal and cost little. Judge only what would teach the detector wrong things: products with broken, stretched or \
hollow shapes, sizes far from the real photo, objects floating clearly in the air or sunk in the support, \
background parts (a box the products sit in, furniture) labelled as products, products without a box, boxes far \
larger than their object.

Rate the generated images as training data from 0 (useless, would hurt the detector) to 10 (as useful as more real \
photos). Answer with one JSON object, nothing else:
{"score": <0-10>, "worst": ["<the defects that cost most, short phrases>", ...]}"""


def sheets():
    os.makedirs(OUT, exist_ok=True)
    for scene in TRUTH:
        d = os.path.join(OURS, scene, 'isaac')
        labels = json.load(open(os.path.join(d, 'labels.json')))
        tiles = [Image.open(os.path.join(OURS, scene, 'photo.png')).convert('RGB')]
        for k in VIEWS:
            im = Image.open(os.path.join(d, k)).convert('RGB')
            dr = ImageDraw.Draw(im)
            for b in labels[k]:
                dr.rectangle(b, outline=(255, 0, 0), width=4)
            tiles.append(im)
        w, h = 672, 378
        sheet = Image.new('RGB', (w * 3, h * 2), 'white')
        for i, t in enumerate(tiles):
            sheet.paste(t.resize((w, h)), ((i % 3) * w, (i // 3) * h))
        sheet.save(os.path.join(OUT, f'sheet_{scene}.jpg'), quality=90)
    print(f'{len(TRUTH)} sheets in {OUT}')


def ask(models, max_tokens, keys=('precheck', 'review')):
    from sim2chip.llm import Client, image_part, parse_json
    client = Client(usage_log=Path(OUT) / 'usage.jsonl')
    for model in models:
        for scene in TRUTH:
            dst = os.path.join(OUT, model.replace('/', '__'), f'{scene}.json')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            res = json.load(open(dst)) if os.path.exists(dst) else {'model': model, 'scene': scene}
            todo = [k for k in keys if k not in res]
            if not todo:
                continue
            calls = {'precheck': (PRECHECK, os.path.join(OURS, scene, 'photo.png')),
                     'review': (REVIEW, os.path.join(OUT, f'sheet_{scene}.jpg')),
                     'rate': (RATE, os.path.join(OUT, f'sheet_{scene}.jpg'))}
            for key in todo:
                prompt, image = calls[key]
                msg = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}, image_part(image)]}]
                t0, text, answer = time.time(), '', None
                try:
                    text = client.chat(msg, model, purpose='vlm_qc', temperature=0.0, max_tokens=max_tokens)
                    answer = parse_json(text)
                except Exception as e:  # keep going: one bad model must not stop the bench
                    text = f'{type(e).__name__}: {e}\n---\n{text}'
                res[key] = {'seconds': round(time.time() - t0, 1), 'answer': answer, 'raw': text}
            with open(dst, 'w') as f:
                json.dump(res, f, indent=1)
            print(model, scene, {k: res[k]['answer'] is not None for k in todo}, flush=True)


def score():
    rank = {'good': 0, 'fair': 1, 'bad': 2}
    rows = []
    for mdir in sorted(glob.glob(os.path.join(OUT, '*/'))):
        files = sorted(glob.glob(os.path.join(mdir, '*.json')))
        if not files:
            continue
        pre_ok = rev_exact = rev_bad_caught = rev_good_kept = n = 0
        bad = sum(1 for t in TRUTH.values() if t[1] == 'bad')
        not_bad = len(TRUTH) - bad
        detail = {}
        for f in files:
            r = json.load(open(f))
            s = r['scene']
            p = (r['precheck']['answer'] or {}).get('one_flat_support')
            v = str((r['review']['answer'] or {}).get('verdict', '')).lower()
            tp, tv = TRUTH[s]
            n += 1
            pre_ok += p is tp
            rev_exact += v == tv
            rev_bad_caught += tv == 'bad' and v == 'bad'
            rev_good_kept += tv != 'bad' and v in ('good', 'fair')
            detail[s] = {'precheck': p, 'review': v}
            rate = (r.get('rate') or {}).get('answer') or {}
            if isinstance(rate.get('score'), (int, float)):
                detail[s]['score'] = rate['score']
        scored = [(d['score'], TRUTH[s][1]) for s, d in detail.items() if 'score' in d]
        # Rank check: share of (bad, usable) pairs where the usable scene scores higher (ties count half).
        pairs = [(a, b) for a, ta in scored if ta == 'bad' for b, tb in scored if tb != 'bad']
        auc = sum(1.0 if b > a else 0.5 if b == a else 0.0 for a, b in pairs) / len(pairs) if pairs else None
        rows.append({'model': json.load(open(files[0]))['model'], 'scenes': n,
                     'rate_auc': None if auc is None else round(auc, 2),
                     'scores': {s[5:]: d.get('score') for s, d in detail.items()},
                     'precheck_right': f'{pre_ok}/{n}', 'review_exact': f'{rev_exact}/{n}',
                     'bad_caught': f'{rev_bad_caught}/{bad}', 'usable_kept': f'{rev_good_kept}/{not_bad}',
                     'detail': detail})
        print(json.dumps({k: v for k, v in rows[-1].items() if k != 'detail'}))
    with open(os.path.join(OUT, 'scores.json'), 'w') as f:
        json.dump(rows, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['sheets', 'ask', 'rate', 'score'])
    ap.add_argument('--model', action='append', default=[])
    ap.add_argument('--max-tokens', type=int, default=16384)
    args = ap.parse_args()
    {'sheets': sheets, 'ask': lambda: ask(args.model, args.max_tokens),
     'rate': lambda: ask(args.model, args.max_tokens, ('rate',)), 'score': score}[args.cmd]()


if __name__ == '__main__':
    main()
