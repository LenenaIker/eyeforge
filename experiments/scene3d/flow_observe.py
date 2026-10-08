"""Step 1.1 of the scene flow (docs/escena_agente_literatura.md): a Token Factory VLM looks at the photo and
describes it in closed fields, observation first and decisions after, with the builder's capability sheet
(flow_capabilities.md) and the failure catalog (flow_failures.md) in the prompt. Three answers are sampled and
voted: enums by majority, phrase lists from the answer that agrees most with the other two. When the three
disagree on indoor/outdoor or the time of day, the photo is marked unsure and the route sends it to a person.
Any env:

    python experiments/scene3d/flow_observe.py --image data/street_test/images/basque_street.jpg --out <dir>

Writes <dir>/observe.json: {"final": {...}, "votes": {...}, "samples": [...], "unsure": [...], "model"}.
Exit code 1 when fewer than two samples gave a usable answer.
Model: --model, else SIM2CHIP_PLAN_MODEL, else Kimi-K3.
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from sim2chip.llm import Client, ask_json, image_part, map_within  # noqa: E402

MODEL = 'moonshotai/Kimi-K3'
ENUMS = {
    'outdoor': [True, False],
    'time_of_day': ['day', 'dusk_or_dawn', 'night'],
    'weather': ['sunny', 'overcast', 'rain', 'snow', 'fog'],
    'camera_height': ['ground', 'pedestrian', 'raised', 'aerial'],
    'ground_flat': ['flat', 'gentle_slope', 'steep_or_stairs'],
    'place': ['village_street', 'city_street', 'square', 'road_outside_town', 'garden_or_yard', 'landscape',
              'room', 'work_surface', 'ground_close', 'other'],
    'layout': ['corner', 'straight_street', 'facade_only', 'square', 'room', 'support_only', 'other'],
    'route': ['village_street_day', 'support', 'indoor_floor', 'out_of_scope'],
    'template': ['crossroads', 'straight_street', 'square', 'room', 'none'],
    'hdri_group': ['urban', 'countryside', 'indoor'],
}
LISTS = ['ground', 'surroundings', 'objects', 'far']
PROMPT = """{capabilities}

{failures}

You are planning the scene for the photo below. First look, then decide. Answer with ONE JSON object, nothing else:
{{
  "observation": "<4-6 sentences: what the photo shows, from where, the light, what stands on the ground, what is \
near and far. Describe only, decide nothing yet.>",
  "outdoor": <true|false>,
  "time_of_day": "day|dusk_or_dawn|night",
  "weather": "sunny|overcast|rain|snow|fog",
  "camera_height": "ground|pedestrian|raised|aerial",
  "ground_flat": "flat|gentle_slope|steep_or_stairs",
  "place": "village_street|city_street|square|road_outside_town|garden_or_yard|landscape|room|work_surface|\
ground_close|other",
  "layout": "corner|straight_street|facade_only|square|room|support_only|other",
  "horizon_y": <where the horizon line would cross the image, 0 = top edge, 1 = bottom edge; may be <0 or >1>,
  "camera_height_m": <your estimate of the camera height above the ground, metres>,
  "ground": [{{"phrase": "<1-3 words a segmenter understands, e.g. sidewalk, asphalt road, cobblestone street>", \
"material": "<what it is made of>"}}],
  "surroundings": [{{"phrase": "<a kind of large fixed thing near the street, e.g. house, hedge, low stone wall, \
fence, tree>", "count": <how many in the photo>, "note": "<is any instance cut by the border, tiny, or \
merged with something behind it?>"}}],
  "objects": [{{"phrase": "<a kind of movable thing to erase, e.g. car, bicycle, bench, trash bin, person, \
sign>", "count": <n>}}],
  "far": ["<things farther than ~25 m: mountains, church tower, trees, buildings at the end of the street>"],
  "route": "village_street_day|support|indoor_floor|out_of_scope",
  "template": "crossroads|straight_street|square|room|none",
  "hdri_group": "urban|countryside|indoor",
  "reason": "<one sentence for the route and template>"
}}
Use generic phrases (no brands, no colours unless needed to tell two kinds apart). Every phrase names one kind of \
thing; one entry per kind. Phrases go to an open-vocabulary segmenter, so prefer common words."""


def valid(v):
    if not isinstance(v, dict) or not isinstance(v.get('observation'), str):
        return False
    for k, opts in ENUMS.items():
        if v.get(k) not in opts:
            return False
    return all(isinstance(v.get(k), list) for k in LISTS)


def phrases(sample, key):
    out = []
    for e in sample.get(key) or []:
        p = e.get('phrase') if isinstance(e, dict) else e
        if isinstance(p, str) and p.strip():
            out.append(p.strip().lower())
    return out


def vote(samples):
    """Enums by majority (None when no value has two votes), numbers by median, lists from the medoid sample."""
    votes, final, unsure = {}, {}, []
    for k in ENUMS:
        c = Counter(s[k] for s in samples)
        votes[k] = {str(a): n for a, n in c.items()}
        top, n = c.most_common(1)[0]
        final[k] = top if n >= 2 or len(samples) == 1 else None
        if final[k] is None:
            unsure.append(k)
    for k in ('horizon_y', 'camera_height_m'):
        vals = sorted(float(s[k]) for s in samples if isinstance(s.get(k), (int, float)))
        final[k] = vals[len(vals) // 2] if vals else None
    # The medoid: the sample whose phrases overlap most with the others' (Jaccard summed over the lists).
    def jac(a, b):
        return len(a & b) / max(1, len(a | b))
    sets = [{k: set(phrases(s, k)) for k in LISTS} for s in samples]
    score = [sum(jac(sets[i][k], sets[j][k]) for j in range(len(samples)) if j != i for k in LISTS)
             for i in range(len(samples))]
    best = samples[max(range(len(samples)), key=lambda i: score[i])]
    for k in LISTS:
        final[k] = best[k]
        # How many samples name each phrase: a phrase only one sample saw is kept but flagged.
        support = Counter(p for s in sets for p in s[k])
        final[k + '_support'] = {p: support[p] for p in phrases(best, k)}
    final['observation'] = best['observation']
    final['reason'] = best.get('reason', '')
    return final, votes, unsure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default=os.environ.get('SIM2CHIP_PLAN_MODEL', MODEL))
    ap.add_argument('--samples', type=int, default=3)
    ap.add_argument('--temperature', type=float, default=0.6)
    args = ap.parse_args()
    client = Client(usage_log=ROOT / 'runs' / 'llm_usage.jsonl')
    text = PROMPT.format(capabilities=(HERE / 'flow_capabilities.md').read_text(),
                         failures=(HERE / 'flow_failures.md').read_text())
    msg = [{'role': 'user', 'content': [{'type': 'text', 'text': text}, image_part(args.image)]}]

    def one(_):
        return ask_json(client, msg, args.model, purpose='scene_observe', check=valid,
                        temperature=args.temperature, max_tokens=16384)
    samples = [s for s in map_within(one, list(range(args.samples)), seconds=900) if s]
    if len(samples) < 2:
        print(f'only {len(samples)} usable answers from {args.model}')
        sys.exit(1)
    final, votes, unsure = vote(samples)
    os.makedirs(args.out, exist_ok=True)
    out = {'image': str(Path(args.image).resolve()), 'final': final, 'votes': votes, 'unsure': unsure,
           'samples': samples, 'model': args.model}
    with open(os.path.join(args.out, 'observe.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: final[k] for k in ENUMS}), 'unsure:', unsure)


if __name__ == '__main__':
    main()
