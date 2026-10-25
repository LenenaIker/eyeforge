"""Step 1.5v of the scene flow: the trees and shrubs of the photo become NVIDIA vegetation assets (veg_catalog.py)
instead of SAM 3D meshes, which came out as green blocks the likeness filter threw away (roads, C.6; the user's
choice, 2026-10-13). For each kept surroundings mask (flow_check_masks.py) whose phrase is a tree or a shrub, the
VLM sees the photo with the region and a numbered sheet of the catalog's thumbnails, and picks the closest asset
(crown shape, conifer or broadleaf, leaf colour); three samples, the majority pick. Those masks leave
env_masks.npz, so SAM 3D does not spend its minutes on them. Run in the `teacher-sam3` env:

    python sim2chip/photo2scene/steps/flow_vegetation.py --dir <flow dir> --catalog <cache>/vegetation

Writes <dir>/vegetation.json ([{"index", "phrase", "group", "name", "url", "votes"}], index into
vegetation.npz), <dir>/vegetation.npz (masks, phrases) and <dir>/veg_sheet_<group>.jpg; rewrites
<dir>/env_masks.npz without them.
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sim2chip.llm import Client, ask_json, image_part, map_within  # noqa: E402

# Qwen, not Cosmos: on 2026-10-15 Cosmos left most street bands without a species (photo68 lost its 21 plants)
# and picked hibiscus and oleander for basque_street's hedges. Back until a local pick is measured as good.
MODEL = 'Qwen/Qwen3.8-27B'
# Phrases of the observation that are vegetation (hedges stay with SAM 3D: a trimmed hedge is a green box, and
# SAM 3D makes it well, 2026-10-06).
TREE = re.compile(r'\b(trees?|palms?|conifers?|firs?|pines?|cypress(es)?|oaks?|birch(es)?)\b')
SHRUB = re.compile(r'\b(bush(es)?|shrubs?|plants?|foliage|vegetation)\b')
PROMPT = """The first picture is a street photo with one region tinted magenta (left) and a close crop of it \
(right): a segmenter labelled it "{p}". The second picture is a catalog of 3D {group}s, each numbered. We will \
put the catalog {group} that looks most like the one in the photo where it stands.

Compare the overall shape of the crown (round, wide, columnar, conical, drooping), conifer or broadleaf, how dense \
the leaves are and their colour. Ignore the size and the background. Answer only with one JSON object:
{{"pick": <the closest number>, "second": <the next closest number>, "why": "<one short sentence>"}}"""


def group_of(phrase):
    p = phrase.lower()
    if TREE.search(p):
        return 'tree'
    if SHRUB.search(p) and 'hedge' not in p:
        return 'shrub'
    return None


def sheet(cat_dir, items, path, cols=8, tw=138, th=108):
    """The thumbnails numbered from 1, in a grid."""
    rows = math.ceil(len(items) / cols)
    out = Image.new('RGB', (cols * tw, rows * th), 'white')
    for k, it in enumerate(items):
        im = Image.open(os.path.join(cat_dir, it['thumb'])).convert('RGB').resize((tw, th))
        x, y = (k % cols) * tw, (k // cols) * th
        out.paste(im, (x, y))
        d = ImageDraw.Draw(out)
        d.rectangle([x, y, x + 26, y + 16], fill='black')
        d.text((x + 3, y + 2), str(k + 1), fill='yellow')
    out.save(path, quality=90)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--catalog', required=True, help='veg_catalog.py --dir')
    ap.add_argument('--model', default=os.environ.get('SIM2CHIP_REVIEW_MODEL', MODEL))
    ap.add_argument('--votes', type=int, default=3)
    args = ap.parse_args()
    d = args.dir
    z = dict(np.load(os.path.join(d, 'env_masks.npz')))
    phrases = [str(p) for p in z['phrases']]
    veg = [(i, g) for i, p in enumerate(phrases) if (g := group_of(p))]
    cat = json.load(open(os.path.join(args.catalog, 'catalog.json')))
    client = Client(usage_log=ROOT / 'runs' / 'llm_usage.jsonl')
    sheets, by_group = {}, {}
    for _, g in veg:
        if g not in sheets:
            by_group[g] = [c for c in cat if c['group'] == g]
            sheets[g] = sheet(args.catalog, by_group[g], os.path.join(d, f'veg_sheet_{g}.jpg'))

    def ask(i, g, temperature):
        n = len(by_group[g])
        msg = [{'role': 'user', 'content': [{'type': 'text', 'text': PROMPT.format(p=phrases[i], group=g)},
                                            image_part(str(z['pictures'][i])), image_part(sheets[g])]}]
        ok = lambda v: isinstance(v.get('pick'), int) and 1 <= v['pick'] <= n  # noqa: E731
        return ask_json(client, msg, args.model, purpose='scene_vegetation_pick', check=ok, temperature=temperature,
                        max_tokens=8192)

    out = []
    for i, g in veg:
        got = [a for a in map_within(lambda t: ask(i, g, t), [0.0] + [0.7] * (args.votes - 1), seconds=600) if a]
        if not got:  # no answer: it stays with SAM 3D
            continue
        votes = Counter(a['pick'] for a in got)
        pick = votes.most_common(1)[0][0]
        it = by_group[g][pick - 1]
        out.append({'index': len(out), 'mask': i, 'phrase': phrases[i], 'group': g, 'name': it['name'],
                    'url': it['url'], 'votes': dict(votes), 'why': got[0].get('why', '')})
    taken = {o['mask'] for o in out}
    keep = [i for i in range(len(phrases)) if i not in taken]
    np.savez_compressed(os.path.join(d, 'vegetation.npz'), masks=z['masks'][sorted(taken)].reshape(
        -1, *z['masks'].shape[1:]), phrases=np.array([phrases[i] for i in sorted(taken)]))
    for o in out:  # index into vegetation.npz, in mask order
        o['index'] = sorted(taken).index(o.pop('mask'))
    np.savez_compressed(os.path.join(d, 'env_masks.npz'), **{k: v[keep] for k, v in z.items()})
    with open(os.path.join(d, 'vegetation.json'), 'w') as f:
        json.dump({'assets': out, 'model': args.model}, f, indent=1)
    print(json.dumps([(o['phrase'], o['name'], o['votes']) for o in out]))


if __name__ == '__main__':
    main()
