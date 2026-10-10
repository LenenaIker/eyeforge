"""Step 1.3a of the scene flow for a room: every thing in the photo as a whole object, so what is scene and what
is erased can be decided by measured size (scene_split.py) instead of by which words SAM 3 happens to find. On
user_desk (2026-10-08) SAM 3 never found the shelf by name and the big desk's mesh was lost, while the VLM listed
both. The VLM lists each thing with a box and picks the scale (a desk with its legs, not its top); SAM 3's box
prompt (Sam3TrackerModel) turns each box into a mask. The support phrases from the observation are segmented
too (ground.npz), for the floor plane. Run in the `teacher-sam3` env:

    python experiments/scene3d/flow_things.py --image data/scene3d_test/photos/user_desk.jpg \
        --dir data/scene3d_test/flow/user_desk [--targets "dog,cat"]

Writes <dir>/photo.png (long side 1344, as bg_prepare.py), things.json ({"things": [{"name", "box", "on",
"target"}], "model"}), things.npz (masks, names) and ground.npz.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'sim2chip', 'photo2scene', 'steps'))  # noqa: E402
from bg_prepare import segment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim2chip.llm import Client, ask_json, image_part  # noqa: E402

MODEL = 'Qwen/Qwen3.8-27B'
PROMPT = """List every distinct physical thing in this photo, each as a whole object (a desk with its legs is one \
desk, not its top; a bed with its drawers is one bed; a radiator is one radiator, not its fins). Include furniture, \
fixtures, appliances and small items; leave out the floor, the walls and the ceiling.{targets} Answer with ONE \
JSON object:
{{"things": [{{"name": "<1-3 words>", "box": [x0, y0, x1, y1], "on": "floor|wall|furniture|ceiling"{target_field}}}]}}
Box coordinates are normalised to 0-1000 (0,0 top left)."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--targets', default='', help="comma-separated classes the user's model detects: always erased")
    ap.add_argument('--long-side', type=int, default=1344)
    ap.add_argument('--model', default=os.environ.get('SIM2CHIP_REVIEW_MODEL', MODEL))
    args = ap.parse_args()
    d = args.dir
    os.makedirs(d, exist_ok=True)
    photo = ImageOps.exif_transpose(Image.open(args.image)).convert('RGB')
    s = args.long_side / max(photo.size)
    photo = photo.resize((round(photo.width * s), round(photo.height * s)), Image.Resampling.LANCZOS)
    photo.save(os.path.join(d, 'photo.png'))
    w, h = photo.size

    targets = [t.strip() for t in args.targets.split(',') if t.strip()]
    text = PROMPT.format(
        targets=f' The user\'s classes are: {", ".join(targets)}.' if targets else '',
        target_field=', "target": <true if it is one of the user\'s classes>' if targets else '')
    client = Client(usage_log=ROOT / 'runs' / 'llm_usage.jsonl')
    msg = [{'role': 'user', 'content': [{'type': 'text', 'text': text}, image_part(os.path.join(d, 'photo.png'))]}]

    def ok(v):
        return isinstance(v.get('things'), list) and all(
            isinstance(t.get('box'), list) and len(t['box']) == 4 for t in v['things'])
    a = ask_json(client, msg, args.model, purpose='scene_things', check=ok, temperature=0.0, max_tokens=8192)
    if a is None:
        sys.exit('the VLM gave no usable list of things')
    things = []
    for t in a['things']:
        x0, y0, x1, y1 = (float(v) for v in t['box'])
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        things.append({'name': str(t.get('name', 'thing')), 'box': [x0, y0, x1, y1], 'on': t.get('on'),
                       'target': bool(t.get('target', False))})

    obs = json.load(open(os.path.join(d, 'observe.json')))['final']
    support = [e['phrase'] if isinstance(e, dict) else e for e in obs['ground']]
    model = Sam3Model.from_pretrained('facebook/sam3').cuda().eval()
    proc = Sam3Processor.from_pretrained('facebook/sam3')
    ground = {}
    for phrase in support:
        ground[phrase] = np.zeros((h, w), bool)
        for _, m in segment(model, proc, photo, phrase, 0.5):
            ground[phrase] |= m
    del model
    torch.cuda.empty_cache()

    tracker = Sam3TrackerModel.from_pretrained('facebook/sam3').cuda().eval()
    tproc = Sam3TrackerProcessor.from_pretrained('facebook/sam3')
    boxes = [[t['box'][0] * w / 1000, t['box'][1] * h / 1000, t['box'][2] * w / 1000, t['box'][3] * h / 1000]
             for t in things]
    masks = np.zeros((0, h, w), bool)
    if boxes:
        p = tproc(images=photo, input_boxes=[boxes], return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = tracker(**p, multimask_output=False)
        masks = tproc.post_process_masks(out.pred_masks.cpu(), p['original_sizes'].cpu())[0][:, 0].numpy()
    np.savez_compressed(os.path.join(d, 'things.npz'), masks=masks, names=np.array([t['name'] for t in things]))
    np.savez_compressed(os.path.join(d, 'ground.npz'), masks=np.array(list(ground.values())),
                        phrases=np.array(list(ground)))
    json.dump({'things': things, 'targets': targets, 'model': args.model},
              open(os.path.join(d, 'things.json'), 'w'), indent=1)
    print(json.dumps({'things': len(things), 'names': [t['name'] for t in things]}))


if __name__ == '__main__':
    main()
