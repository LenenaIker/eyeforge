"""Step 1.2 of the scene flow: code, not the VLM, picks the route and the template from the observation
(flow_observe.py). The VLM's own route and template are only compared: when they disagree with the rules the
photo goes to a person instead of being forced into a scene. Any env:

    python sim2chip/photo2scene/steps/flow_route.py --dir data/street_test/flow/basque_street

Daylight is measured on the photo, not voted (user, 2026-10-15: roads photos 27 and 29, an overcast afternoon,
were stopped because the VLM said dusk three times out of three). The VLM's time of day is kept beside it.

Writes <dir>/route.json: {"route", "template", "hdri_group", "checks": {name: bool}, "light": {...}, "why": [...]}.
Exit code 3 when the photo is out of scope or unsure (the driver stops there).
"""
import argparse
import json
import os
import sys

SUPPORT_PLACES = {'work_surface', 'ground_close'}
STREET_PLACES = {'village_street', 'city_street', 'square', 'road_outside_town', 'garden_or_yard'}
TEMPLATE = {'corner': 'crossroads', 'straight_street': 'straight_street', 'facade_only': 'straight_street',
            'square': 'square', 'room': 'room', 'support_only': 'none'}
BUILT = {'crossroads', 'straight_street', 'room'}  # templates crossroad.py etc. can build today; the others route but stop before phase 3
# Daylight by measure, CIE L* (0-100) of the photo at 512 px. Phones expose night shots up to a mean L* of 30-35,
# so the brightness alone is not enough: street lamps and lit windows also cast the whole photo orange (red over
# blue in linear light 2.6 and 3.2 on night_market and night_street; 0.5 to 1.6 on the 17 day photos of the
# 2026-10-15 baseline). Day photos: mean L* 41 to 60 (roads_cars' 74 photos from 36.7), the brightest tenth from
# L* 65; the two night photos 30 and 35, 59 and 59.
DAY_MEAN_L = 38.0
DAY_P90_L = 62.0
NIGHT_RED_BLUE = 1.8


def measure_light(photo) -> dict:
    """The photo's brightness and colour cast, and whether that reads as daylight (DAY_* above)."""
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(photo) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        im.thumbnail((512, 512))
        a = np.asarray(im, np.float32) / 255
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    lum = 116 * np.cbrt(lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)) - 16
    mean_l, p90_l = float(lum.mean()), float(np.percentile(lum, 90))
    red_blue = float(lin[..., 0].mean() / max(lin[..., 2].mean(), 1e-6))
    day = (mean_l >= DAY_MEAN_L or p90_l >= DAY_P90_L) and red_blue < NIGHT_RED_BLUE
    return {'mean_L': round(mean_l, 1), 'p90_L': round(p90_l, 1), 'red_over_blue': round(red_blue, 2), 'day': day}


def route(obs, light=None):
    """light: measure_light() of the photo; without it the VLM's time of day decides (older runs)."""
    f = obs['final']
    why = []
    unsure = obs.get('unsure') or []
    vlm_day = f['time_of_day'] == 'day'
    day = light['day'] if light else vlm_day
    overruled = day != vlm_day  # the VLM's own route and template followed from its time of day: not compared
    if light:
        unsure = [k for k in unsure if k != 'time_of_day']
    if f['place'] in SUPPORT_PLACES:  # light and weather do not matter on the support route, the place does
        unsure = [k for k in unsure if k in ('place', 'route')]
    if unsure:
        why.append('the three answers disagree on ' + ', '.join(unsure))
    street = {
        'outdoor': f['outdoor'] is True,
        'daylight': day,
        'dry ground': f['weather'] in ('sunny', 'overcast', 'fog'),
        'pedestrian camera': f['camera_height'] in ('ground', 'pedestrian'),
        'flat ground': f['ground_flat'] in ('flat', 'gentle_slope'),
        'a street place': f['place'] in STREET_PLACES,
        'ground phrase': bool(f['ground']),
        'surroundings phrase': bool(f['surroundings']),
    }
    if f['place'] in SUPPORT_PLACES:
        # One working surface seen by the camera (bottles on a table, a conveyor belt, a lawn from above): the
        # cheap route, indoors or outdoors, any light; the photo stays the background.
        r = 'support'
        checks = {'a working surface': True, 'ground phrase': bool(f['ground'])}
        if not f['ground']:
            r = 'out_of_scope'
            why.append('fails: ground phrase')
    elif f['outdoor'] is False:
        r = 'indoor_floor' if f['place'] == 'room' else 'out_of_scope'
        checks = {'indoor place known': r != 'out_of_scope'}
    elif all(street.values()):
        r, checks = 'village_street_day', street
    else:
        r, checks = 'out_of_scope', street
        why += ['fails: ' + k for k, ok in street.items() if not ok]
    if r == 'village_street_day' and f.get('route') != r and not overruled:
        why.append(f"the VLM said {f.get('route')}")
        r = 'out_of_scope'
    if why and r != 'out_of_scope':
        r = 'out_of_scope'
    template = 'none' if r in ('out_of_scope', 'support') else TEMPLATE.get(f['layout'], 'none')
    if r != 'out_of_scope' and f.get('template') != template and not overruled:
        why.append(f"template by rule {template}, the VLM said {f.get('template')}")
    # Village streets and squares open onto fields and hills: the urban skies put tower blocks and tram wires
    # behind a village church (church_corner, 2026-10-07).
    hdri = 'urban' if r == 'village_street_day' and f['place'] == 'city_street' else \
        'countryside' if r == 'village_street_day' or (r == 'support' and f['outdoor']) else 'indoor'
    out = {'route': r, 'template': template, 'template_built': template in BUILT, 'hdri_group': hdri,
           'checks': checks, 'why': why}
    if light:
        out['light'] = {**light, 'vlm_time_of_day': f['time_of_day']}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--force', choices=['support'], help="the spec already says so (a fixed camera on a work "
                    "surface): the route the photo takes whatever the VLM saw")
    args = ap.parse_args()
    obs = json.load(open(os.path.join(args.dir, 'observe.json')))
    if args.force:
        obs['final']['place'] = 'work_surface'
        obs['unsure'] = []
    photo = os.path.join(args.dir, 'photo.png')
    out = route(obs, measure_light(photo) if os.path.exists(photo) else None)
    with open(os.path.join(args.dir, 'route.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out))
    sys.exit(3 if out['route'] == 'out_of_scope' else 0)


if __name__ == '__main__':
    main()
