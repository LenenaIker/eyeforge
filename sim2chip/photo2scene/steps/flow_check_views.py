"""Step 3.4 of the scene flow: each view of the extended place (crossroad.py or street_template.py, rendered by
piece_isaac.py) is checked on its own. Code first (catalog entry 5: a view almost all one tone is a camera in a
wall or a dull exposure), then closed yes/no questions to the VLM, one picture each. A failing view is left out
of the views handed to Replicator; the place passes when most views pass. Any env:

    python sim2chip/photo2scene/steps/flow_check_views.py --dir data/street_test/flow/corner_timber/extended

Reads <dir>/isaac/render_NN.png (view 0, the photo's camera, is skipped: phase 2 judged it); writes
<dir>/views.json ({"views": [{"view", "stats", "answers", "keep", "why"}], "kept", "share", "ok"}).
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sim2chip.llm import Client, ask_json, image_part, map_within  # noqa: E402

MODEL = 'Qwen/Qwen3.8-27B'
QUESTIONS = {
    'street_view': ('Does this look like a street or lane seen by a person standing or driving on it?', 'yes'),
    'sky_up': ('Is the sky at the top and the ground at the bottom of the picture?', 'yes'),
    'inside': ('Is most of the picture one flat texture or the inside of something (the camera inside a wall, a '
               'hedge or a house)?', 'no'),
    'floating': ('Does any building, wall or hedge float in the air above the ground?', 'no'),
}
QUESTIONS_INDOOR = {
    'room_view': ('Does this look like the inside of a room seen by a person or a small robot in it?', 'yes'),
    'floor_down': ('Is the floor at the bottom of the picture and are the walls upright?', 'yes'),
    'inside': ('Is most of the picture one flat texture or the inside of something (the camera inside a wall or '
               'a piece of furniture)?', 'no'),
    'floating': ('Does any piece of furniture float in the air above the floor?', 'no'),
}
PROMPT = """This is a rendered view of a 3D {place} rebuilt from one photo; it will be a background for training \
an object detector. Simple textures and plain light are expected and do not matter.

Answer each question with "yes" or "no" only, as one JSON object, nothing else:
{questions}"""


def yn(v):
    return str(v).strip().lower()[:3].strip()


def stats(path):
    a = np.asarray(Image.open(path).convert('RGB')).astype(np.float32) / 255
    lum = a @ np.array([0.2126, 0.7152, 0.0722])
    return {'median': round(float(np.median(lum)), 3), 'p97': round(float(np.percentile(lum, 97)), 3),
            'spread': round(float(np.percentile(lum, 90) - np.percentile(lum, 10)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--model', default=os.environ.get('SIM2CHIP_REVIEW_MODEL', MODEL))
    ap.add_argument('--min-share', type=float, default=0.6)
    ap.add_argument('--indoor', action='store_true', help='a room (room_template.py), not a street')
    args = ap.parse_args()
    renders = sorted(glob.glob(os.path.join(args.dir, 'isaac', 'render_*.png')))[1:]
    client = Client(usage_log=ROOT / 'runs' / 'llm_usage.jsonl')
    questions = QUESTIONS_INDOOR if args.indoor else QUESTIONS
    place = 'room' if args.indoor else 'street'
    qs = '{\n' + ',\n'.join(f'  "{k}": "<yes|no>  ({q})"' for k, (q, _) in questions.items()) + '\n}'

    def ask(path):
        msg = [{'role': 'user', 'content': [{'type': 'text', 'text': PROMPT.format(questions=qs, place=place)},
                                            image_part(path)]}]
        ok = lambda v: all(yn(v.get(k, '')) in ('yes', 'no') for k in questions)  # noqa: E731
        a = ask_json(client, msg, args.model, purpose='scene_view_check', check=ok, temperature=0.0, max_tokens=8192)
        return {k: yn(a[k]) for k in questions} if a else None

    answers = map_within(ask, renders, seconds=1200, workers=6)
    views = []
    for path, a in zip(renders, answers):
        st = stats(path)
        why = []
        if st['spread'] < 0.08:
            why.append('one tone (camera in a surface or flat light)')
        why += [k for k, (_, want) in questions.items() if a and a[k] != want] if a else ['no answer']
        views.append({'view': int(os.path.basename(path)[7:9]), 'stats': st, 'answers': a, 'keep': not why,
                      'why': why})
    kept = [v['view'] for v in views if v['keep']]
    share = len(kept) / max(len(views), 1)
    out = {'views': views, 'kept': kept, 'share': round(share, 3), 'ok': share >= args.min_share, 'model': args.model}
    with open(os.path.join(args.dir, 'views.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({'kept': kept, 'share': out['share'], 'dropped': [(v['view'], v['why']) for v in views
                                                                       if not v['keep']]}))


if __name__ == '__main__':
    main()
