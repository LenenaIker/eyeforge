"""Step 2.4a of the scene flow (user's idea, 2026-10-07): deterministic photometry of the imitated place against
the photo, before any VLM looks at it, and per-object colour fixes.

1. A regular grid of pixels (one in three along each axis, about 10 %) is taken from the photo with its movable
   objects erased (background.png) and from the render from the photo's camera (place/isaac/render_00.png). Sky
   and the pixels LaMa painted are left out.
2. Global, as a camera's grey-world white balance: the mean linear RGB of both; exposure is the luminance ratio,
   the cast is each channel's gain over it.
3. Per object: every sampled pixel knows which piece it shows (floor, each facade plane, each surroundings mesh;
   the id map is drawn with pyrender from the same camera). Per object: median colour difference (CIE76 delta E
   in Lab), luminance ratio, per-channel gain (median photo / render in linear light), and its share of the total
   error, so the worst object is named. A coarse map of where in the image the error sits is saved too.
4. Proportions (user's idea, 2026-10-07): each object's photo / render ratio is compared with the others'. If the
   place were only darker or tinted as a whole, every object would share one ratio; an object whose ratio stands
   apart from the median of all (by more than --max-dev) is the one out of proportion, and is named.
5. Conditions: exposure within 0.8-1.25, cast under 10 %, no object out of proportion, each object's median
   delta E under --max-de.

Run in the `trellis2` env:

    PYOPENGL_PLATFORM=egl python experiments/scene3d/flow_photometry.py --dir data/street_test/flow/basque_street

Writes <dir>/photometry.json and <dir>/photometry.jpg (photo | render | error map with the worst objects).
With --apply (default --mode relative), first the objects out of proportion are brought back to the others'
ratio, then the grey-world white balance is computed on the render as it will be after that fix and applied to
everything. The gains are written into place/isaac (piece "gain" fields, mesh vertex colours) from a kept original, so repeated rounds do
not compound.
"""
import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image, ImageDraw

from flow_gates import render

STEP = 3  # one pixel in three along each axis: about 11 % of them


def to_linear(srgb):
    c = srgb / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lab(lin):
    """Linear sRGB (D65) to CIE Lab."""
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


LUMA = np.array([0.2126, 0.7152, 0.0722])


def items_of(scene, arr):
    """Everything drawn from the photo's camera with a label: pieces first, then the surroundings meshes."""
    items, labels = [], []
    for i, p in enumerate(scene['pieces']):
        items.append((np.asarray(p['points']), np.array([[0, 1, 2], [0, 2, 3]])))
        tex = os.path.basename(p['texture'] or '')
        labels.append(('piece', i, 'floor' if i == 0 else tex.replace('tex_', '').replace('.png', '') or 'plane'))
    names = {e['index']: e['name'] for e in scene.get('env', [])}
    for i in range(scene['objects']):
        items.append((arr[f'obj{i}_points'], arr[f'obj{i}_faces']))
        labels.append(('object', i, names.get(i, f'object {i}')))
    return items, labels


def measure(d, args):
    src = os.path.join(d, 'place', 'isaac')
    scene = json.load(open(os.path.join(src, 'scene.json')))
    arr = np.load(os.path.join(src, 'meshes.npz'))
    w, h = scene['size']
    items, labels = items_of(scene, arr)
    _, ids = render(scene, arr, items, w, h)  # 0 nothing (sky), k + 1 item k
    photo = np.asarray(Image.open(os.path.join(d, 'background.png')).convert('RGB').resize((w, h))).astype(float)
    rend = np.asarray(Image.open(os.path.join(src, 'render_00.png')).convert('RGB').resize((w, h))).astype(float)
    erased = np.asarray(Image.open(os.path.join(d, 'erased.png')).resize((w, h))) > 0
    grid = np.zeros((h, w), bool)
    grid[STEP // 2::STEP, STEP // 2::STEP] = True
    sel = grid & (ids > 0) & ~erased
    lp, lr = to_linear(photo[sel]), to_linear(rend[sel])
    de = np.linalg.norm(lab(lp) - lab(lr), axis=1)
    who = ids[sel] - 1

    mp, mr = lp.mean(0), lr.mean(0)
    exposure = float((mp @ LUMA) / max(mr @ LUMA, 1e-6))
    cast = (mp / np.maximum(mr, 1e-6)) / exposure  # per channel, 1 = same white balance
    report = {'samples': int(sel.sum()), 'share_of_image': round(float(sel.sum() / grid.sum()), 3),
              'exposure': round(exposure, 3), 'cast': cast.round(3).tolist(),
              'mean_de': round(float(de.mean()), 2), 'objects': []}
    total = float(de.sum())
    for k in np.unique(who):
        m = who == k
        if m.sum() < args.min_samples:
            continue
        gain = np.median(lp[m], 0) / np.maximum(np.median(lr[m], 0), 1e-4)
        kind, idx, name = labels[k]
        o = {'kind': kind, 'index': idx, 'name': name, 'samples': int(m.sum()),
             'median_de': round(float(np.median(de[m])), 2),
             'luminance_ratio': round(float(np.median(lp[m] @ LUMA) / max(np.median(lr[m] @ LUMA), 1e-4)), 3),
             'gain': np.clip(gain, 1 / args.max_gain, args.max_gain).round(3).tolist(),
             'error_share': round(float(de[m].sum() / max(total, 1e-6)), 3),
             'ratio': np.clip(np.median(lp[m], 0), 1e-4, None) / np.clip(np.median(lr[m], 0), 1e-4, None)}
        o['ok'] = o['median_de'] <= args.max_de
        # Three times darker or brighter than the photo is light (a facade in shadow), not colour: no gain for it.
        o['lighting'] = not 1 / 3 <= o['luminance_ratio'] <= 3
        o['ratio'] = o['ratio'].round(4).tolist()
        report['objects'].append(o)
    # Proportions: each object's ratio against the median ratio of all (weighted by samples), in log space.
    objs = report['objects']
    if len(objs) >= 3:
        logs = np.log(np.array([o['ratio'] for o in objs]))
        wts = np.array([o['samples'] for o in objs], float)
        # The floor and facade planes are textured with the photo's own pixels: their ratio is light and exposure
        # only, the colour of the place as it should be. SAM 3D meshes are the suspects (a stone church came out
        # white and, as the largest object, made a right-coloured house look wrong, church_corner 2026-10-07).
        ref_of = np.array([o['kind'] == 'piece' for o in objs]) if args.reference == 'pieces' else np.ones(len(objs), bool)
        if not ref_of.any():
            ref_of[:] = True
        ref = np.array([weighted_median(logs[ref_of, c], wts[ref_of]) for c in range(3)])
        report['reference'] = [o['name'] for o, r in zip(objs, ref_of) if r]
        for o, l in zip(objs, logs):
            dev = l - ref
            o['relative'] = np.exp(dev).round(3).tolist()
            o['out_of_proportion'] = bool(np.abs(dev).max() > np.log(args.max_dev))
            # Three times brighter or darker than the rest is light (a facade in shadow), not colour: named, no gain.
            o['lighting'] = not 1 / 3 <= float(np.exp(dev) @ LUMA) <= 3
    else:
        for o in objs:
            o['relative'], o['out_of_proportion'] = [1.0, 1.0, 1.0], False
    report['objects'].sort(key=lambda o: -o['error_share'])
    report['checks'] = {
        'exposure within 0.8-1.25': 0.8 <= exposure <= 1.25,
        'cast under 10 %': bool(np.abs(cast - 1).max() < 0.10),
        'no object out of proportion': not any(o['out_of_proportion'] and not o['lighting']
                                               for o in report['objects']),
        f'every object under delta E {args.max_de:g}': all(o['ok'] for o in report['objects']),
    }
    report['ok'] = all(report['checks'].values())
    report['worst'] = [o['name'] for o in report['objects'][:3]]
    report['out_of_proportion'] = [o['name'] for o in report['objects'] if o['out_of_proportion']]

    # Where the error sits: delta E on the grid, averaged in 8 x 6 cells; worst objects outlined.
    err = np.full((h, w), np.nan)
    err[sel] = de
    cells = np.zeros((6, 8))
    for r in range(6):
        for c in range(8):
            v = err[r * h // 6:(r + 1) * h // 6, c * w // 8:(c + 1) * w // 8]
            cells[r, c] = np.nanmean(v) if np.isfinite(v).any() else 0
    heat = np.kron(cells / max(cells.max(), 1e-6), np.ones((h // 6 + 1, w // 8 + 1)))[:h, :w]
    over = (photo * 0.5 + np.stack([heat * 255, np.zeros_like(heat), (1 - heat) * 80], -1) * 0.5)
    for o in report['objects'][:3]:
        k = labels.index(next(l for l in labels if l[0] == o['kind'] and l[1] == o['index']))
        m = ids == k + 1
        edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        over[edge] = [255, 255, 0]
    tiles = [Image.fromarray(photo.astype(np.uint8)), Image.fromarray(rend.astype(np.uint8)),
             Image.fromarray(over.clip(0, 255).astype(np.uint8))]
    sheet = Image.new('RGB', (w * 3 // 2, h // 2 + 30), 'white')
    for j, t in enumerate(tiles):
        sheet.paste(t.resize((w // 2, h // 2)), (j * w // 2, 30))
    ImageDraw.Draw(sheet).text((6, 8), f"exposure {exposure:.2f}  cast {np.round(cast, 2).tolist()}  mean dE "
                               f"{report['mean_de']}  worst: {', '.join(report['worst'])}", fill=(0, 0, 0))
    samples = {'lp': lp, 'lr': lr, 'who': who, 'labels': labels}
    return report, sheet, samples


def weighted_median(v, w):
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, c[-1] / 2)])


def relative_gains(report, samples, args):
    """Objects out of proportion first, each to the median ratio of the others; then the grey-world cast of the
    render as it will be after that, applied to everything."""
    lp, lr, who, labels = samples['lp'], samples['lr'], samples['who'], samples['labels']
    own = {}
    for o in report['objects']:
        if o['out_of_proportion'] and not o['lighting']:
            own[(o['kind'], o['index'])] = np.clip(np.array(o['relative']), 1 / args.max_gain, args.max_gain)
    per_sample = np.ones_like(lr)
    for k, lb in enumerate(labels):
        if (lb[0], lb[1]) in own:
            per_sample[who == k] = own[(lb[0], lb[1])]
    after = lr * per_sample
    mp, ma = lp.mean(0), after.mean(0)
    exposure = (mp @ LUMA) / max(ma @ LUMA, 1e-6)
    cast = (mp / np.maximum(ma, 1e-6)) / exposure
    glob = cast.clip(1 / args.max_gain, args.max_gain) if np.abs(cast - 1).max() >= 0.10 else np.ones(3)
    return {(o['kind'], o['index']): own.get((o['kind'], o['index']), np.ones(3)) * glob
            for o in report['objects']}


def apply(d, report, args, samples):
    """Gains into the place, always from the kept original: a failing object gets its own gain, the rest the
    global one when exposure or cast fail."""
    src = os.path.join(d, 'place', 'isaac')
    for f in ('scene.json', 'meshes.npz'):
        if not os.path.exists(os.path.join(src, f'orig_{f}')):
            shutil.copy(os.path.join(src, f), os.path.join(src, f'orig_{f}'))
    scene = json.load(open(os.path.join(src, 'orig_scene.json')))
    arr = dict(np.load(os.path.join(src, 'orig_meshes.npz')))
    # Manual exposure from here on: the first render's auto exposure, so a brighter house does not darken the rest.
    if 'exposure_0' not in scene and os.path.exists(os.path.join(src, 'exposure_0.json')):
        scene['exposure_0'] = json.load(open(os.path.join(src, 'exposure_0.json')))['gain']
        json.dump(scene, open(os.path.join(src, 'orig_scene.json'), 'w'))
    prev = report.get('applied', {})
    # Global: the white balance only (the grey-world cast); exposure stays with the camera.
    # cast is photo over render per channel: the render's colours times it match the photo's balance.
    glob = np.array(report['cast']).clip(1 / args.max_gain, args.max_gain) \
        if not report['checks']['cast under 10 %'] else np.ones(3)
    applied = {}
    rel = relative_gains(report, samples, args) if args.mode == 'relative' else {}
    for o in report['objects']:
        key = f"{o['kind']} {o['index']}"
        if args.mode == 'relative':
            g = rel[(o['kind'], o['index'])]
        else:
            g = glob if args.mode == 'global' or o['ok'] or o['lighting'] else np.array(o['gain'])
        g = g * np.array(prev.get(key, [1, 1, 1]))  # rounds build on what was applied before
        g = g.clip(1 / args.max_gain, args.max_gain)
        applied[key] = g.round(3).tolist()
        if o['kind'] == 'piece':
            scene['pieces'][o['index']]['gain'] = applied[key]
        else:
            arr[f"obj{o['index']}_colors"] = np.clip(arr[f"obj{o['index']}_colors"] * g, 0, 1).astype(np.float32)
            if f"obj{o['index']}_photo_faces" in arr:  # its photo-textured faces take the gain in the material
                arr[f"obj{o['index']}_gain"] = np.asarray(g, np.float32)
    json.dump(scene, open(os.path.join(src, 'scene.json'), 'w'))
    np.savez_compressed(os.path.join(src, 'meshes.npz'), **arr)
    return applied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--max-de', type=float, default=20.0, help='largest median delta E per object')
    ap.add_argument('--max-gain', type=float, default=2.0)
    ap.add_argument('--min-samples', type=int, default=200, help='objects with fewer sampled pixels are skipped')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--max-dev', type=float, default=1.10,
                    help='an object whose photo / render ratio differs from the median by more than this factor '
                         '(any channel) is out of proportion')
    ap.add_argument('--reference', choices=['pieces', 'median'], default='pieces',
                    help='pieces: the ratio of the photo-textured floor and planes; median: of every object')
    ap.add_argument('--mode', choices=['relative', 'global', 'object'], default='relative',
                    help='relative: objects out of proportion to the median ratio, then grey world; global: the '
                         'white balance only; object: each failing object\'s absolute gain (made two of two street '
                         'places worse, 2026-10-07)')
    ap.add_argument('--out', default='photometry')
    args = ap.parse_args()
    report, sheet, samples = measure(args.dir, args)
    old = os.path.join(args.dir, 'photometry.json')
    if args.apply and os.path.exists(old):
        report['applied'] = json.load(open(old)).get('applied', {})
    if args.apply and not report['ok']:
        report['applied'] = apply(args.dir, report, args, samples)
    sheet.save(os.path.join(args.dir, f'{args.out}.jpg'), quality=88)
    with open(os.path.join(args.dir, f'{args.out}.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ('exposure', 'cast', 'mean_de', 'worst', 'out_of_proportion', 'ok')} |
                     {'objects': [(o['name'], o['median_de'], o['relative'], o['error_share'])
                                  for o in report['objects'][:5]]}))


if __name__ == '__main__':
    main()
