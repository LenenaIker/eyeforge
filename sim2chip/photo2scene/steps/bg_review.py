"""Last step of our scene from one photo: a Token Factory VLM rates the generated views as training data, as
Claude did by eye on 2026-10-06. A sheet with the photo and 5 views with their label boxes is scored 0-10; scenes
under --min-score are kept out of training (on HOPE, unreviewed scenes made the student worse than real photos
alone; experiments/vlm_bench/vlm_qc.py: Qwen3.8-27B ranked all 9 scenes right, bad ones 1-2, usable ones 3-4).
Any env with Pillow:

    python sim2chip/photo2scene/steps/bg_review.py --dir data/scene3d_test/ours/hope_val_000006 [--model ...]

Reads <dir>/photo.png and <dir>/isaac/{render_NN.png,labels.json}; writes <dir>/review.jpg (the sheet) and
<dir>/review.json ({"score", "worst", "keep", "model"}). Exit code 3 when the scene is rejected.
Model: --model, else SIM2CHIP_REVIEW_MODEL, else Cosmos Reason 2 on this machine (llm.LOCAL_VLM; Qwen3.8-27B before 2026-10-15).
"""
import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sim2chip.llm import LOCAL_VLM, Client, image_part, parse_json  # noqa: E402

MODEL = LOCAL_VLM  # Cosmos Reason 2 on this machine's GPU (user, 2026-10-15)
PROMPT = """We generate synthetic training images for a small object detector from one real photo. The sheet shows \
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


def sheet(d):
    """The photo and 5 views spread over the generated ones, label boxes in red, 3 x 2 tiles."""
    labels = json.load(open(os.path.join(d, 'isaac', 'labels.json')))
    names = sorted(labels)
    picks = [names[round(i * (len(names) - 1) / 4)] for i in range(5)] if len(names) >= 5 else names
    tiles = [Image.open(os.path.join(d, 'photo.png')).convert('RGB')]
    for k in picks:
        im = Image.open(os.path.join(d, 'isaac', k)).convert('RGB')
        dr = ImageDraw.Draw(im)
        for b in labels[k]:
            if isinstance(b, dict):  # with roles: targets in red, distractors in blue
                dr.rectangle(b['box'], outline=(255, 0, 0) if b['role'] == 'target' else (0, 90, 255), width=4)
            else:
                dr.rectangle(b, outline=(255, 0, 0), width=4)
        tiles.append(im)
    # Tiles keep the photo's shape: a portrait photo squeezed into 16:9 tiles made the VLM see stretched,
    # oversized objects that were not (bottles, 2026-10-08).
    pw, ph = tiles[0].size
    w, h = (672, round(672 * ph / pw)) if pw >= ph else (round(672 * pw / ph), 672)
    out = Image.new('RGB', (w * 3, h * 2), 'white')
    for i, t in enumerate(tiles):
        out.paste(t.resize((w, h)), ((i % 3) * w, (i // 3) * h))
    path = os.path.join(d, 'review.jpg')
    out.save(path, quality=90)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--model', default=os.environ.get('SIM2CHIP_REVIEW_MODEL', MODEL))
    ap.add_argument('--min-score', type=float, default=3)
    args = ap.parse_args()
    path = sheet(args.dir)
    client = Client(usage_log=Path(os.environ.get('SIM2CHIP_LLM_USAGE', ROOT / 'runs' / 'llm_usage.jsonl')))
    msg = [{'role': 'user', 'content': [{'type': 'text', 'text': PROMPT}, image_part(path)]}]
    answer = None
    for _ in range(2):  # one retry on a bad or cut answer
        try:
            v = parse_json(client.chat(msg, args.model, purpose='scene_review', temperature=0.0, max_tokens=16384))
        except ValueError:
            continue
        if isinstance(v, dict) and isinstance(v.get('score'), (int, float)):
            answer = v
            break
    if answer is None:  # no verdict: keep the scene, say so
        out = {'score': None, 'worst': [], 'keep': True, 'model': args.model, 'note': 'no usable answer'}
    else:
        out = {'score': answer['score'], 'worst': [str(x) for x in answer.get('worst') or []],
               'keep': answer['score'] >= args.min_score, 'model': args.model}
    with open(os.path.join(args.dir, 'review.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out))
    sys.exit(0 if out['keep'] else 3)


if __name__ == '__main__':
    main()
