"""Step 2.4 of the scene flow: the VLM compares the place rebuilt from the photo, rendered in Isaac from the
photo's camera (view 0), with the photo whose movable objects were erased (background.png). A fixed list of
closed questions (docs/escena_agente_literatura.md), each asked with the pictures in both orders against
position bias; an answer counts only when both orders agree, otherwise it is "unsure". The geometric gates
(flow_gates.py) ran before; this catches what code does not see (a sky on its side, a wall where the street
should go on, a desert sky behind a town). Any env:

    python experiments/scene3d/flow_compare.py --dir data/street_test/flow/corner_timber [--model ...]

Reads <dir>/background.png and <dir>/place/isaac/render_00.png; writes <dir>/compare_{ab,ba}.jpg and
<dir>/compare.json ({"answers": {q: yes|no|unsure}, "raw", "failed", "ok", "model"}).
"""
import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim2chip.llm import Client, ask_json, image_part, map_within  # noqa: E402

MODEL = 'Qwen/Qwen3.8-27B'
# (question, wanted answer, blocking): a blocking "no" sends the photo back; the others are only logged. Until the
# questions are calibrated on hand-labelled sheets, only the plain ones block: on 2026-10-07 "block on street"
# failed a good church_corner replica.
QUESTIONS = {
    'sky_up': ('In the RENDER, is the sky above the ground and the horizon roughly level (not sideways or upside '
               'down)?', 'yes', True),
    'sky_fits': ('Does the sky of the RENDER fit a town street in daylight (not a desert, rocks, sea or an '
                 'indoor room)?', 'yes', True),
    'horizon': ('Is the horizon, where the street meets the distance, at about the same height in both '
                'pictures?', 'yes', True),
    'ground_same': ('Does the street ground cover about the same part of both pictures?', 'yes', True),
    'buildings_same': ('Are the main buildings in about the same places and of about the same size in both '
                       'pictures?', 'yes', False),
    'block_on_street': ('Does the RENDER show a wall, block or large shape where the PHOTO shows open street or '
                        'sky?', 'no', False),
    'floating': ('In the RENDER, does any building, wall or hedge float above the ground, with sky or a far '
                 'landscape showing under it?', 'no', True),
    'missing': ('Is a large building, wall or hedge of the PHOTO missing from the RENDER?', 'no', False),
}
PROMPT = """Two pictures side by side, labelled PHOTO and RENDER. The PHOTO is a real street (cars and people \
were erased from it). The RENDER is a 3D rebuild of that street from the same camera, made from the photo alone: \
simpler textures, other light and invented backs of buildings are expected and do not matter. Judge only the \
layout.

Answer each question with "yes" or "no" only, as one JSON object, nothing else:
{questions}"""


def side_by_side(left, right, names, path):
    h = 480
    a = left.resize((round(left.width * h / left.height), h))
    b = right.resize((round(right.width * h / right.height), h))
    sheet = Image.new('RGB', (a.width + b.width + 16, h + 34), 'white')
    sheet.paste(a, (0, 34))
    sheet.paste(b, (a.width + 16, 34))
    dr = ImageDraw.Draw(sheet)
    dr.text((8, 8), names[0], fill=(0, 0, 0))
    dr.text((a.width + 24, 8), names[1], fill=(0, 0, 0))
    sheet.save(path, quality=90)


def yn(v):
    return str(v).strip().lower()[:3].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--model', default=os.environ.get('SIM2CHIP_REVIEW_MODEL', MODEL))
    args = ap.parse_args()
    d = args.dir
    photo = Image.open(os.path.join(d, 'background.png')).convert('RGB')
    render = Image.open(os.path.join(d, 'place', 'isaac', 'render_00.png')).convert('RGB')
    side_by_side(photo, render, ('PHOTO', 'RENDER'), os.path.join(d, 'compare_ab.jpg'))
    side_by_side(render, photo, ('RENDER', 'PHOTO'), os.path.join(d, 'compare_ba.jpg'))
    client = Client(usage_log=ROOT / 'runs' / 'llm_usage.jsonl')
    qs = '{\n' + ',\n'.join(f'  "{k}": "<yes|no>  ({q})"' for k, (q, _, _) in QUESTIONS.items()) + '\n}'

    def ask(pic):
        msg = [{'role': 'user', 'content': [{'type': 'text', 'text': PROMPT.format(questions=qs)},
                                            image_part(os.path.join(d, pic))]}]
        ok = lambda v: all(yn(v.get(k, '')) in ('yes', 'no') for k in QUESTIONS)  # noqa: E731
        a = ask_json(client, msg, args.model, purpose='scene_compare', check=ok, temperature=0.0, max_tokens=8192)
        return {k: yn(a[k]) for k in QUESTIONS} if a else None

    ab, ba = map_within(ask, ['compare_ab.jpg', 'compare_ba.jpg'], seconds=900)
    ab = ab or ask('compare_ab.jpg')  # one more try for an order that gave no usable answer
    ba = ba or ask('compare_ba.jpg')
    answers = {}
    for k in QUESTIONS:
        got = {x[k] for x in (ab, ba) if x}
        answers[k] = got.pop() if len(got) == 1 else 'unsure'
    failed = [k for k, (_, want, block) in QUESTIONS.items() if block and answers[k] not in (want, 'unsure')]
    logged = [k for k, (_, want, block) in QUESTIONS.items() if answers[k] != want]
    out = {'answers': answers, 'raw': {'photo_left': ab, 'render_left': ba}, 'failed': failed, 'not_wanted': logged,
           'ok': not failed and (ab is not None or ba is not None), 'model': args.model}
    with open(os.path.join(d, 'compare.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
