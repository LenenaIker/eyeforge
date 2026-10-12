"""Step 1.5c of photo2scene: the VLM checks each SAM 3D surroundings mesh (flow_sam3d.py, posed by flow_refine.py),
next to the photo crop it came from. Closed yes/no questions, three samples voted per
question (one answer changed between runs on the mask check): is it recognisable as the thing, does it carry
pieces of something else, is it a solid shape rather than a sheet. A mesh that fails is left out of the scene
(this replaces the hand-picked `crossroad.py --drop`). Run in the `trellis2` env:

    PYOPENGL_PLATFORM=egl python sim2chip/photo2scene/steps/flow_check_meshes.py --dir data/street_test/flow/corner_timber

Writes <dir>/mesh_check/mesh_NNN.jpg and <dir>/meshes.json ({"meshes": [{"node", "phrase", "answers", "keep",
"why"}], "model"}).
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sim2chip.llm import Client, ask_json, image_part, map_within  # noqa: E402
from env_add import env_meshes, turntable  # noqa: E402

MODEL = 'Qwen/Qwen3.8-27B'
QUESTIONS = {
    'looks_like': ('Do the views of the 3D model look like a {p}?', 'yes'),
    'foreign': ('Does the 3D model carry large pieces of something that is not a {p} (a slab of ground, sky, '
                'another building, a flat backdrop)?', 'no'),
    'solid': ('Is the 3D model a solid shape from the side views, not a thin flat sheet or a shell with big '
              'holes?', 'yes'),
}
PROMPT = """Top: a crop of a street photo with one {p} outlined in magenta. Bottom row: a 3D model of that {p}, \
made from the photo alone, seen from the photo's side (0) and turned 45, 90, 180 and 270 degrees. The back \
was never seen, so it may be invented; that is fine. We use the model as scenery for rendering new street views.

Answer each question with "yes" or "no" only, as one JSON object, nothing else:
{questions}"""


def yn(v):
    return str(v).strip().lower()[:3].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--model', default=os.environ.get('SIM2CHIP_REVIEW_MODEL', MODEL))
    ap.add_argument('--votes', type=int, default=3)
    args = ap.parse_args()
    d = args.dir
    case = os.path.join(d, 'meshes')
    meshes = env_meshes(case, 20000)
    graph = json.load(open(os.path.join(case, 'scene_graph.json')))
    nodes = {n['id']: n for n in graph['nodes'] if n['kind'] == 'object'}
    # Mesh k is the k-th kept mask of flow_check_masks.py; its picture (repaired or not) is stored there.
    pictures = list(np.load(os.path.join(d, 'env_masks.npz'))['pictures'])
    out_dir = os.path.join(d, 'mesh_check')
    os.makedirs(out_dir, exist_ok=True)
    items = []
    for node, m in sorted(meshes.items()):
        oid = next((k for k in nodes if k in node), None)
        if oid is None:
            continue
        n = nodes[oid]
        k = n['mask_index']
        turntable({n['name']: m}, os.path.join(out_dir, f'turn_{k:03d}.jpg'))
        # The photo crop above the turntable row.
        crop = Image.open(pictures[k])
        turn = Image.open(os.path.join(out_dir, f'turn_{k:03d}.jpg'))
        crop.thumbnail((turn.width, 420))
        sheet = Image.new('RGB', (turn.width, crop.height + turn.height + 8), 'white')
        sheet.paste(crop, (0, 0))
        sheet.paste(turn, (0, crop.height + 8))
        path = os.path.join(out_dir, f'mesh_{k:03d}.jpg')
        sheet.save(path, quality=88)
        items.append({'node': node, 'index': k, 'phrase': n['name'], 'picture': path,
                      'faces': int(len(m.faces)), 'extent_m': np.ptp(m.vertices, 0).round(2).tolist()})
    client = Client(usage_log=ROOT / 'runs' / 'llm_usage.jsonl')

    def ask_once(it, t):
        p = it['phrase']
        qs = '{\n' + ',\n'.join(f'  "{k}": "<yes|no>  ({q.format(p=p)})"' for k, (q, _) in QUESTIONS.items()) + '\n}'
        msg = [{'role': 'user', 'content': [{'type': 'text', 'text': PROMPT.format(p=p, questions=qs)},
                                            image_part(it['picture'])]}]
        ok = lambda v: all(yn(v.get(k, '')) in ('yes', 'no') for k in QUESTIONS)  # noqa: E731
        a = ask_json(client, msg, args.model, purpose='scene_mesh_check', check=ok, temperature=t, max_tokens=8192)
        return {k: yn(a[k]) for k in QUESTIONS} if a else None

    def ask(it):
        got = [a for a in map_within(lambda t: ask_once(it, t), [0.0] + [0.7] * (args.votes - 1), seconds=600) if a]
        if not got:
            return None
        return {k: Counter(g[k] for g in got).most_common(1)[0][0] for k in QUESTIONS} | {'votes': len(got)}

    answers = map_within(ask, items, seconds=1800, workers=4)
    for it, a in zip(items, answers):
        it['answers'] = a
        it['why'] = [k for k, (_, want) in QUESTIONS.items() if not a or a[k] != want] if a else ['no answer']
        it['keep'] = not it['why']
    with open(os.path.join(d, 'meshes.json'), 'w') as f:
        json.dump({'meshes': items, 'model': args.model}, f, indent=1)
    print(json.dumps([(it['index'], it['phrase'], it['keep'], it['why']) for it in items]))


if __name__ == '__main__':
    main()
