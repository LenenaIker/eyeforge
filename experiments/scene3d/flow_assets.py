"""After the erase: every thing taken out of the photo becomes an asset for the generator, which places targets
and distractors again at random (user, 2026-10-08: a distractor that is erased must not be lost). The erased
things are the instances that were asked for, plus what the edit model removed on its own (Lightning took the
bottles' moka pot, which SAM 3 never found): large regions that changed between the photo and the background
outside the asked masks. Each such region is cut out by SAM 3's box prompt, and the VLM, shown it before and
after, names the thing that went or says it was only a change of the background. Sizes come from MoGe's metric
points of the photo and the support plane. Run in the `teacher-sam3` env after the erase and bg_depth.py:

    python experiments/scene3d/flow_assets.py --dir data/scene3d_test/flow/bottles --masks instances.npz \
        [--targets "bottle"]

Writes <dir>/assets.npz (masks, phrases: the asked instances first, in their order, then the found ones; what
flow_sam3d.py meshes) and <dir>/assets.json ({"assets": [{"index", "name", "role", "found", "height_m",
"width_m", "mesh"}]}), the catalogue the generator reads.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from transformers import Sam3TrackerModel, Sam3TrackerProcessor

from scene_split import fit_floor, measure

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim2chip.llm import Client, ask_json, image_part, map_within  # noqa: E402

MODEL = 'Qwen/Qwen3.8-27B'
PROMPT = """The LEFT picture is a crop of a photo, the RIGHT one the same crop after an object was erased from it. \
Is a physical object present on the left and gone on the right (not just a change of light, texture or a \
stain)? Answer with ONE JSON object: {"gone": <true|false>, "name": "<1-3 words naming the object, empty if \
none>"}"""


def components(mask):
    """Label image of the 8-connected regions of mask (max-pool propagation of pixel ids, on the GPU)."""
    m = torch.from_numpy(mask).cuda()
    cur = (torch.arange(1, mask.size + 1, device='cuda').reshape(mask.shape) * m).float()
    while True:
        nxt = F.max_pool2d(cur[None, None], 3, 1, 1)[0, 0] * m
        if torch.equal(nxt, cur):
            return cur.long().cpu().numpy()
        cur = nxt


def pair(photo, background, box, path):
    x0, y0, x1, y1 = box
    pad = int(0.5 * max(x1 - x0, y1 - y0))
    b = (max(0, x0 - pad), max(0, y0 - pad), min(photo.width, x1 + pad), min(photo.height, y1 + pad))
    left, right = photo.crop(b), background.crop(b)
    s = 384 / max(left.size)
    size = (max(1, round(left.width * s)), max(1, round(left.height * s)))
    out = Image.new('RGB', (size[0] * 2 + 12, size[1]), 'white')
    out.paste(left.resize(size), (0, 0))
    out.paste(right.resize(size), (size[0] + 12, 0))
    out.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--masks', default='instances.npz', help='npz (masks, phrases) of what was asked to be erased')
    ap.add_argument('--targets', default='', help="comma-separated user's classes: role target, else distractor")
    ap.add_argument('--change', type=float, default=30.0, help='blurred mean change that counts, 0-255')
    ap.add_argument('--min-area', type=float, default=0.002, help='smallest region or asset, share of the photo')
    ap.add_argument('--model', default=os.environ.get('SIM2CHIP_REVIEW_MODEL', MODEL))
    args = ap.parse_args()
    d = args.dir
    photo = Image.open(os.path.join(d, 'photo.png')).convert('RGB')
    background = Image.open(os.path.join(d, 'background.png')).convert('RGB')
    w, h = photo.size
    z = np.load(os.path.join(d, args.masks))
    masks = [m.astype(bool) for m in z['masks']]
    names = [str(p) for p in z['phrases']]
    asked = len(masks)

    # Regions the edit changed a lot outside the asked masks (grown a little: the edit's own seams).
    a, b = np.asarray(photo, np.float32), np.asarray(background, np.float32)
    change = np.asarray(Image.fromarray(np.abs(a - b).mean(2).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4)))
    known = np.any(masks, 0) if masks else np.zeros((h, w), bool)
    known = np.asarray(Image.fromarray(known.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(15))) > 0
    lab = components((change > args.change) & ~known)
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    boxes = []
    for i in ids[counts > args.min_area * w * h]:
        ys, xs = np.nonzero(lab == i)
        boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])

    found = []
    if boxes:
        os.makedirs(os.path.join(d, 'assets_check'), exist_ok=True)
        client = Client(usage_log=ROOT / 'runs' / 'llm_usage.jsonl')

        def ask(k):
            path = os.path.join(d, 'assets_check', f'region_{k:02d}.jpg')
            pair(photo, background, boxes[k], path)
            msg = [{'role': 'user', 'content': [{'type': 'text', 'text': PROMPT}, image_part(path)]}]
            return ask_json(client, msg, args.model, purpose='scene_assets',
                            check=lambda v: isinstance(v.get('gone'), bool), temperature=0.0, max_tokens=4096)
        answers = map_within(ask, list(range(len(boxes))), seconds=600)
        gone = [(bx, ans['name'].strip() or 'object') for bx, ans in zip(boxes, answers)
                if ans and ans['gone'] and isinstance(ans.get('name'), str)]
        if gone:
            model = Sam3TrackerModel.from_pretrained('facebook/sam3').cuda().eval()
            proc = Sam3TrackerProcessor.from_pretrained('facebook/sam3')
            p = proc(images=photo, input_boxes=[[g[0] for g in gone]], return_tensors='pt').to('cuda')
            with torch.no_grad():
                out = model(**p, multimask_output=False)
            ms = proc.post_process_masks(out.pred_masks.cpu(), p['original_sizes'].cpu())[0][:, 0].numpy()
            for (bx, name), m in zip(gone, ms):
                m = m & ~known
                if m.mean() > args.min_area:
                    found.append((name, m))
    masks += [m for _, m in found]
    names += [n for n, _ in found]

    dp = np.load(os.path.join(d, 'depth_photo.npz'))
    pts = dp['points']
    g = pts[np.load(os.path.join(d, 'ground.npz'))['masks'].any(0) & (dp['mask'] > 0)]
    g = g[np.isfinite(g).all(1)]
    g = g[np.random.default_rng(0).choice(len(g), min(len(g), 60000), replace=False)]
    n, dd = fit_floor(g)
    targets = {t.strip().lower() for t in args.targets.split(',') if t.strip()}
    rows = []
    for k, (m, name) in enumerate(zip(masks, names)):
        s = measure(pts, m, n, dd)
        # Height of the thing itself: a cup on a desk counts from its own bottom, not from the floor.
        rows.append({'index': k, 'name': name, 'found': k >= asked,
                     'role': 'target' if name.lower() in targets else 'distractor',
                     'height_m': round(s['top'] - s['bottom'], 3) if s else None,
                     'width_m': round(s['width'], 3) if s else None,
                     'area': round(float(m.mean()), 4), 'mesh': f'assets/sam3d/{k:03d}_canon.glb'})
    np.savez_compressed(os.path.join(d, 'assets.npz'), masks=np.array(masks, bool).reshape(-1, h, w),
                        phrases=np.array(names))
    json.dump({'assets': rows, 'regions_checked': len(boxes)}, open(os.path.join(d, 'assets.json'), 'w'), indent=1)
    print(json.dumps({'asked': asked, 'found': [r['name'] for r in rows if r['found']],
                      'regions_checked': len(boxes)}))


if __name__ == '__main__':
    main()
