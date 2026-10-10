"""Step 0 of our scene from one photo: a Token Factory VLM plans it, as Claude did by hand until 2026-10-06. It says
whether one flat support holds the movable objects (bags, open boxes, cushions or two levels break our one-plane
scene: on HOPE, scenes like that hurt the student more than the good ones helped), and gives the support phrase,
the object phrases for SAM 3, which of them are closed, roughly convex solids (bg_build --solid) and which are
furniture that stays in place (bg_build --fixed). Any env:

    python sim2chip/photo2scene/steps/bg_plan.py --image data/hope/val/000006/rgb/000000.png --out <dir> [--model ...]

Writes <dir>/plan.json: {"one_flat_support", "support", "objects": [phrase, ...], "solid": [...], "fixed": [...], "reason",
"model"}. Exit code 3 when the scene does not fit (the driver skips it), 1 when the model gave no usable plan.
Model: --model, else SIM2CHIP_PLAN_MODEL, else Kimi-K2.6 (K3 had the best phrases in experiments/vlm_bench but no longer takes images).
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sim2chip.llm import Client, image_part, parse_json  # noqa: E402

MODEL = 'moonshotai/Kimi-K2.6'  # K3 stopped taking images on Token Factory (2026-10-11)
PROMPT = """We rebuild a scene from this single photo to render new training images for a small object detector: \
the movable objects are cut out with an open-vocabulary segmenter, and copies of them are placed again on ONE flat \
support plane (a table top, a floor, one shelf board).

Answer with one JSON object, nothing else:
{
  "one_flat_support": <true if one flat support plane holds all the movable objects; false if they sit in bags, \
in an open box, on soft or uneven cushions, or on two different levels such as two shelves>,
  "support": "<the surface they rest on, 1-3 words, e.g. table, floor, shelf, lawn>",
  "objects": ["<short generic noun phrase for one kind of movable object, as the segmenter expects>", ...],
  "solid": ["<the phrases from objects that are closed, roughly convex solids like cans, boxes, bottles; not \
chairs, bikes, mugs or open containers>", ...],
  "fixed": ["<the phrases from objects that are large furniture or fixtures standing where they belong, like a \
desk, bookshelf, wardrobe or counter: they stay in place while the others are moved>", ...],
  "reason": "<one sentence>"
}
"objects" covers every movable object kind in the photo, one generic phrase per kind (e.g. "can", not a brand); \
nothing that is part of the place (walls, furniture the objects rest on, the support itself). Furniture \
standing on the floor counts as objects on the floor support when the photo is about the furniture (a room) \
rather than about small things on it."""


def valid(v):
    return (isinstance(v, dict) and isinstance(v.get('one_flat_support'), bool) and isinstance(v.get('support'), str)
            and v['support'].strip() and isinstance(v.get('objects'), list) and v['objects']
            and all(isinstance(p, str) and p.strip() for p in v['objects']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default=os.environ.get('SIM2CHIP_PLAN_MODEL', MODEL))
    args = ap.parse_args()
    client = Client(usage_log=Path(os.environ.get('SIM2CHIP_LLM_USAGE', ROOT / 'runs' / 'llm_usage.jsonl')))
    msg = [{'role': 'user', 'content': [{'type': 'text', 'text': PROMPT}, image_part(args.image)]}]
    plan = None
    for _ in range(2):  # one retry on a bad or cut answer
        try:
            v = parse_json(client.chat(msg, args.model, purpose='scene_plan', temperature=0.0, max_tokens=16384))
        except ValueError:
            continue
        if valid(v):
            plan = v
            break
    if plan is None:
        print('no usable plan from', args.model)
        sys.exit(1)
    objects = list(dict.fromkeys(p.strip().lower() for p in plan['objects']))
    solid = [p for p in (s.strip().lower() for s in plan.get('solid') or [] if isinstance(s, str)) if p in objects]
    fixed = [p for p in (s.strip().lower() for s in plan.get('fixed') or [] if isinstance(s, str)) if p in objects]
    out = {'one_flat_support': plan['one_flat_support'], 'support': plan['support'].strip().lower(),
           'objects': objects, 'solid': solid, 'fixed': fixed, 'reason': str(plan.get('reason', '')), 'model': args.model}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'plan.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out))
    sys.exit(0 if out['one_flat_support'] else 3)


if __name__ == '__main__':
    main()
