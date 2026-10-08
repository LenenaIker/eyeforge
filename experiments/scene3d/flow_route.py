"""Step 1.2 of the scene flow: code, not the VLM, picks the route and the template from the observation
(flow_observe.py). The VLM's own route and template are only compared: when they disagree with the rules the
photo goes to a person instead of being forced into a scene. Any env:

    python experiments/scene3d/flow_route.py --dir data/street_test/flow/basque_street

Writes <dir>/route.json: {"route", "template", "hdri_group", "checks": {name: bool}, "why": [...]}.
Exit code 3 when the photo is out of scope or unsure (the driver stops there).
"""
import argparse
import json
import os
import sys

STREET_PLACES = {'village_street', 'city_street', 'square', 'road_outside_town', 'garden_or_yard'}
TEMPLATE = {'corner': 'crossroads', 'straight_street': 'straight_street', 'facade_only': 'straight_street',
            'square': 'square', 'room': 'room', 'support_only': 'none'}
BUILT = {'crossroads', 'straight_street', 'room'}  # templates crossroad.py etc. can build today; the others route but stop before phase 3


def route(obs):
    f = obs['final']
    why = []
    if obs.get('unsure'):
        why.append('the three answers disagree on ' + ', '.join(obs['unsure']))
    street = {
        'outdoor': f['outdoor'] is True,
        'daylight': f['time_of_day'] == 'day',
        'dry ground': f['weather'] in ('sunny', 'overcast', 'fog'),
        'pedestrian camera': f['camera_height'] in ('ground', 'pedestrian'),
        'flat ground': f['ground_flat'] in ('flat', 'gentle_slope'),
        'a street place': f['place'] in STREET_PLACES,
        'ground phrase': bool(f['ground']),
        'surroundings phrase': bool(f['surroundings']),
    }
    if f['outdoor'] is False:
        r = 'table' if f['place'] == 'table_or_shelf' else 'indoor_floor' if f['place'] == 'room' else 'out_of_scope'
        checks = {'indoor place known': r != 'out_of_scope'}
    elif all(street.values()):
        r, checks = 'village_street_day', street
    else:
        r, checks = 'out_of_scope', street
        why += ['fails: ' + k for k, ok in street.items() if not ok]
    if r == 'village_street_day' and f.get('route') != r:
        why.append(f"the VLM said {f.get('route')}")
        r = 'out_of_scope'
    if why and r != 'out_of_scope':
        r = 'out_of_scope'
    template = TEMPLATE.get(f['layout'], 'none') if r != 'out_of_scope' else 'none'
    if r != 'out_of_scope' and f.get('template') != template:
        why.append(f"template by rule {template}, the VLM said {f.get('template')}")
    # Village streets and squares open onto fields and hills: the urban skies put tower blocks and tram wires
    # behind a village church (church_corner, 2026-10-07).
    hdri = 'urban' if r == 'village_street_day' and f['place'] == 'city_street' else \
        'countryside' if r == 'village_street_day' else 'indoor'
    return {'route': r, 'template': template, 'template_built': template in BUILT, 'hdri_group': hdri,
            'checks': checks, 'why': why}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    args = ap.parse_args()
    obs = json.load(open(os.path.join(args.dir, 'observe.json')))
    out = route(obs)
    with open(os.path.join(args.dir, 'route.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out))
    sys.exit(3 if out['route'] == 'out_of_scope' else 0)


if __name__ == '__main__':
    main()
