"""Step 1.3c of the scene flow: the VLM checks each surroundings candidate (flow_segment.py) before it costs two
minutes of SAM 3D. Closed yes/no questions, no score and no long reasoning (docs/escena_agente_literatura.md):
is the region one thing of its kind, does it take in something else, is enough of it seen to rebuild it.

Two retries, both local to the candidate:
- repair: when the region takes in something else, the VLM names it, SAM 3 segments that phrase, it is cut out
  of the region and the questions are asked again (a house mask that took in the garden gate in front of it);
- second opinion: a large candidate (the main buildings) is only dropped when a second model agrees; on the
  street test Qwen once dropped a clean main house.
Run in the `teacher-sam3` env (the repair needs SAM 3):

    python experiments/scene3d/flow_check_masks.py --dir data/street_test/flow/basque_street [--model ...]

Reads <dir>/env_candidates.npz and <dir>/background.png; writes <dir>/env_masks.json ({"candidates": [{"index",
"phrase", "answers", "keep", "why", "repaired", "second"}], "model"}), <dir>/env_masks.npz (the final masks of
the kept ones) and <dir>/env_check/cand_NN[_fix].jpg.
Model: --model, else SIM2CHIP_REVIEW_MODEL, else Qwen3.8-27B; --second (Kimi-K3).
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim2chip.llm import Client, ask_json, image_part, map_within  # noqa: E402
from flow_segment import picture  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'sim2chip', 'photo2scene', 'steps'))  # noqa: E402

MODEL = 'Qwen/Qwen3.8-27B'
SECOND = 'moonshotai/Kimi-K3'
QUESTIONS = {
    'one_thing': ('Does the highlighted region cover one single {p} (a row of attached {p}s that form one block '
                  'also counts)?', 'yes'),
    'takes_other': ('Does the highlighted region also take in a large part of something that is not a {p} (sky, '
                    'road, a different building, a hedge, a tree, a car, a gate)?', 'no'),
    'enough_seen': ('Is enough of the {p} visible to rebuild its 3D shape (not only a thin sliver or a small '
                    'corner)?', 'yes'),
}
PROMPT = """The picture shows a street photo twice: on the left the whole photo with one region tinted magenta \
(the rest darkened), on the right a close crop with the same region outlined in magenta. A segmenter labelled the \
region "{p}". We will turn the region into a 3D mesh, so it must hold one {p} and nothing else.

Answer each question with "yes" or "no" only, and name the other thing if there is one, as one JSON object, \
nothing else:
{questions}"""


def yn(v):
    return str(v).strip().lower()[:3].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--model', default=os.environ.get('SIM2CHIP_REVIEW_MODEL', MODEL))
    ap.add_argument('--second', default=SECOND)
    ap.add_argument('--large', type=float, default=0.04, help='share of the image from which a drop needs '
                    'the second model to agree')
    ap.add_argument('--votes', type=int, default=3)
    ap.add_argument('--out', default='env_masks.json')
    args = ap.parse_args()
    d = args.dir
    cand = np.load(os.path.join(d, 'env_candidates.npz'))
    phrases, masks = list(cand['phrases']), list(cand['masks'])
    img = Image.open(os.path.join(d, 'background.png')).convert('RGB')
    client = Client(usage_log=ROOT / 'runs' / 'llm_usage.jsonl')

    def ask_once(p, pic, model, temperature):
        qs = '{\n' + ',\n'.join(f'  "{k}": "<yes|no>  ({q.format(p=p)})"' for k, (q, _) in QUESTIONS.items()) + \
             ',\n  "other": "<if takes_other is yes: the other thing in 1-3 words, as a segmenter would name it; ' \
             'else empty>"\n}'
        msg = [{'role': 'user', 'content': [{'type': 'text', 'text': PROMPT.format(p=p, questions=qs)},
                                            image_part(pic)]}]
        ok = lambda v: all(yn(v.get(k, '')) in ('yes', 'no') for k in QUESTIONS)  # noqa: E731
        a = ask_json(client, msg, model, purpose='scene_mask_check', check=ok, temperature=temperature,
                     max_tokens=8192)
        if a is None:
            return None
        return {**{k: yn(a[k]) for k in QUESTIONS}, 'other': str(a.get('other') or '').strip().lower()}

    def ask(p, pic, model):
        """Majority per question over --votes samples: one answer at temperature 0 changed between runs."""
        got = [a for a in map_within(lambda t: ask_once(p, pic, model, t), [0.0] + [0.7] * (args.votes - 1),
                                     seconds=600) if a]
        if not got:
            return None
        a = {k: Counter(g[k] for g in got).most_common(1)[0][0] for k in QUESTIONS}
        others = [g['other'] for g in got if g['other']]
        a['other'] = Counter(others).most_common(1)[0][0] if others and a['takes_other'] == 'yes' else ''
        a['votes'] = len(got)
        return a

    def verdict(a):
        return [k for k, (_, want) in QUESTIONS.items() if a[k] != want] if a else ['no answer']

    pics = [os.path.join(d, 'env_check', f'cand_{i:02d}.jpg') for i in range(len(phrases))]
    first = map_within(lambda i: ask(phrases[i], pics[i], args.model), list(range(len(phrases))), seconds=1800,
                       workers=4)
    out = [{'index': i, 'phrase': p, 'answers': a, 'why': verdict(a), 'repaired': None, 'second': None}
           for i, (p, a) in enumerate(zip(phrases, first))]

    # Repair: cut out what the VLM names, then ask again.
    fix = [c for c in out if 'takes_other' in c['why'] and c['answers'] and c['answers']['other']]
    if fix:
        import torch
        from transformers import Sam3Model, Sam3Processor
        from bg_prepare import dilate, segment
        model = Sam3Model.from_pretrained('facebook/sam3').cuda().eval()
        proc = Sam3Processor.from_pretrained('facebook/sam3')
        for c in fix:
            i, cur, a = c['index'], masks[c['index']], c['answers']
            c['repaired'] = []
            for r in range(2):  # a house can hide behind a gate and a hedge: two cuts at most
                cut = np.zeros_like(cur)
                for _, m in segment(model, proc, img, a['other'], 0.4):
                    cut |= m
                new = cur & ~dilate(cut, 3)
                share = float(new.sum() / max(masks[i].sum(), 1))
                step = {'other': a['other'], 'kept_share': round(share, 3)}
                c['repaired'].append(step)
                # The cut must leave most of the thing: if it eats it, the phrase named the thing itself.
                if not 0.3 <= share <= 0.97 * float(cur.sum() / max(masks[i].sum(), 1)):
                    break
                pic = os.path.join(d, 'env_check', f'cand_{i:02d}_fix{r}.jpg')
                picture(img, new, pic)
                a = ask(c['phrase'], pic, args.model)
                step['answers'] = a
                cur = new
                if a is None:
                    break
                if not verdict(a):
                    masks[i], c['answers'], c['why'], pics[i] = new, a, [], pic
                    break
                if not (a['takes_other'] == 'yes' and a['other']):
                    break
        del model
        torch.cuda.empty_cache()

    # Second opinion before dropping a large candidate.
    big = [c for c in out if c['why'] and masks[c['index']].mean() >= args.large]
    second = map_within(lambda c: ask(c['phrase'], pics[c['index']], args.second), big, seconds=900)
    for c, a in zip(big, second):
        c['second'] = {'model': args.second, 'answers': a}
        if a and a['votes'] >= 2 and not verdict(a):
            c['why'] = []
    # A large instance whose only fault is taking in a little of something else goes on with a flag: the mesh
    # check ("carries pieces of something else") and the silhouette gate judge it later. Losing the main house
    # of a street (basque_street, 2026-10-07) costs more than two minutes of SAM 3D.
    for c in out:
        if c['why'] == ['takes_other'] and masks[c['index']].mean() >= args.large:
            c['why'], c['lenient'] = [], True
    for c in out:
        c['keep'] = not c['why']
    np.savez_compressed(os.path.join(d, 'env_masks.npz'),
                        masks=np.array([masks[c['index']] for c in out if c['keep']], bool).reshape(
                            -1, img.height, img.width),
                        phrases=np.array([c['phrase'] for c in out if c['keep']]),
                        pictures=np.array([pics[c['index']] for c in out if c['keep']]))
    with open(os.path.join(d, args.out), 'w') as f:
        json.dump({'candidates': out, 'model': args.model}, f, indent=1)
    print(json.dumps([(c['phrase'], c['keep'], c['why'], [r['kept_share'] for r in c['repaired'] or []])
                      for c in out]))


if __name__ == '__main__':
    main()
