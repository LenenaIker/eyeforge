"""photo2scene: a place for Isaac Sim from one photo (docs/escena_agente_literatura.md, three phases): each step is one script with its
outputs on disk and its own check, run in its own env; a step whose outputs exist is skipped, so a run resumes
where it stopped. Every step writes its part of <dir>/flow.json (status, seconds, check, why). No loop crosses
phases: a failed check stops the photo there, with the reason in the log.

Everything it runs lives here or in its envs (SAM 3, SAM 3D Objects, MoGe-2, LaMa, Isaac Sim); no outside
pipeline is called.

    python experiments/scene3d/photo2scene.py --image data/street_test/images/basque_street.jpg \
        --dir data/street_test/flow/basque_street [--until 1.5] [--redo 1.3b]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HOME = Path.home()
PY = {
    'sam3': HOME / 'miniforge3/envs/teacher-sam3/bin/python',
    'trellis': HOME / 'miniforge3/envs/trellis2/bin/python',
    'mesh': Path('/media/lenena-iker/Crucial/envs/eyeforge-sam3d/bin/python'),
    'isaac': Path('/opt/isaacsim/isaac-sim-5.1.0-20260318-clean/python.sh'),
}
ENV = {'PYOPENGL_PLATFORM': 'egl', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}


def gpu_temp():
    try:
        t = int(subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
                               capture_output=True, text=True).stdout.strip())
    except ValueError:
        return
    if t >= 87:
        sys.exit(f'GPU at {t} C, stopping')
    while t >= 84:  # let it cool before the next GPU step
        time.sleep(20)
        t = int(subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
                               capture_output=True, text=True).stdout.strip())


def jload(d, name):
    return json.load(open(d / name))


class Flow:
    def __init__(self, image, d):
        self.image, self.d = Path(image).resolve(), Path(d)
        self.d.mkdir(parents=True, exist_ok=True)
        self.log_path = self.d / 'flow.json'
        self.log = json.load(open(self.log_path)) if self.log_path.exists() else {'image': str(self.image), 'steps': {}}

    def save(self):
        json.dump(self.log, open(self.log_path, 'w'), indent=1)

    def run(self, step, env, args, outputs, cwd=None, ok_codes=(0,)):
        """Runs one script unless its outputs exist; returns its exit code (0 when skipped)."""
        rec = self.log['steps'].setdefault(step, {})
        if outputs and all((self.d / o).exists() for o in outputs) and rec.get('status') in ('ok', 'stop'):
            return rec.get('code', 0)
        if env in ('mesh', 'sam3', 'isaac', 'trellis'):
            gpu_temp()
        t0 = time.time()
        logf = self.d / 'logs' / f'{step}.log'
        logf.parent.mkdir(exist_ok=True)
        with open(logf, 'w') as fh:
            code = subprocess.run([str(PY[env])] + [str(a) for a in args], cwd=cwd or ROOT, stdout=fh,
                                  stderr=subprocess.STDOUT, env={**os.environ, **ENV}).returncode
        rec.update(status='ok' if code in ok_codes else 'error', code=code, seconds=round(time.time() - t0, 1),
                   log=str(logf.relative_to(self.d)))
        self.save()
        if code not in ok_codes:
            tail = open(logf).read()[-1500:]
            sys.exit(f'{step} failed (exit {code}), log {logf}:\n{tail}')
        return code

    def check(self, step, ok, details, why=''):
        rec = self.log['steps'].setdefault(step, {})
        rec['check'] = details
        if not ok:
            rec['status'] = 'stop'
            rec['why'] = why
        self.save()
        print(f'{step}: {"ok" if ok else "STOP " + why} {json.dumps(details)[:300]}', flush=True)
        return ok


def phase1(f, until):
    d, s3 = f.d, HERE
    # 1.1 Observe: closed fields, 3 samples voted.
    f.run('1.1_observe', 'sam3', [s3 / 'flow_observe.py', '--image', f.image, '--out', d], ['observe.json'])
    obs = jload(d, 'observe.json')
    f.check('1.1_observe', True, {'unsure': obs['unsure'], 'route_vlm': obs['final']['route']})
    # 1.2 Route: rules in code.
    code = f.run('1.2_route', 'sam3', [s3 / 'flow_route.py', '--dir', d], ['route.json'], ok_codes=(0, 3))
    r = jload(d, 'route.json')
    if not f.check('1.2_route', code == 0, {k: r[k] for k in ('route', 'template', 'hdri_group')},
                   '; '.join(r['why']) or 'out of scope'):
        return False
    if until == '1.2':
        return True
    fin = obs['final']
    ground = ','.join(e['phrase'] if isinstance(e, dict) else e for e in fin['ground'])
    objects = ','.join(e['phrase'] if isinstance(e, dict) else e for e in fin['objects']) or 'car'
    # 1.3a Movable objects erased, ground masks (one per phrase, the union is the support).
    f.run('1.3a_prepare', 'sam3', [s3 / 'bg_prepare.py', '--image', f.image, '--objects', objects, '--support', ground,
                                   '--out', d] + (['--shadows'] if fin['weather'] == 'sunny' else []),
          ['background.png', 'ground.npz'])
    prep = jload(d, 'prepare.json')
    if not f.check('1.3a_prepare', prep['support_frac'] > 0.03,
                   {'instances': len(prep['instances']), 'support_frac': prep['support_frac']},
                   'the segmenter found almost no ground'):
        return False
    # 1.4 Depth and camera: MoGe-2 on the photo and on the erased photo.
    f.run('1.4_depth', 'mesh', [s3 / 'bg_depth.py', '--dir', d], ['depth_photo.npz', 'depth_background.npz'])
    f.run('1.4_camera', 'trellis', [s3 / 'flow_camera.py', '--dir', d], ['camera.json'], ok_codes=(0, 3))
    cam = jload(d, 'camera.json')
    if not f.check('1.4_camera', cam['ok'], cam['checks'], '; '.join(cam['why'])):
        return False
    if until == '1.4':
        return True
    # 1.3b Surroundings: SAM 3 candidates on the erased photo, gates in code, then the VLM per candidate.
    f.run('1.3b_segment', 'sam3', [s3 / 'flow_segment.py', '--dir', d], ['env_candidates.npz'])
    f.run('1.3c_check_masks', 'sam3', [s3 / 'flow_check_masks.py', '--dir', d], ['env_masks.json'])
    em = jload(d, 'env_masks.json')
    kept = [c for c in em['candidates'] if c['keep']]
    if not f.check('1.3c_check_masks', bool(kept), {'kept': [c['phrase'] for c in kept],
                                                  'dropped': [(c['phrase'], c['why']) for c in em['candidates']
                                                              if not c['keep']]},
                   'no surroundings instance passed'):
        return False
    if until == '1.3':
        return True
    # 1.5 Meshes: SAM 3D Objects whole on each kept surroundings mask, with our depth as its point map.
    # With --voxels ccm|hybrid, Mira-CCM's pixel-aligned voxels go into SAM 3D's stage 2 (flow_ccm.py).
    for attempt in range(2):
        if f.voxels != 'sam3d':
            f.run('1.5a_ccm', 'mesh', [s3 / 'flow_ccm.py', '--dir', d, '--fresh'], [])
        f.run('1.5a_sam3d', 'mesh', [s3 / 'flow_sam3d.py', '--dir', d, '--fresh', '--voxels', f.voxels],
              ['meshes/scene_graph.json'])
        # A mesh can fail on GPU memory after a large one; the meshes made are kept, so a fresh process fills the
        # gap. An instance that fails twice is too big for the GPU (a whole row of houses, straight_road
        # 2026-10-07): it leaves the kept masks and the meshes are made again without it (indices follow masks).
        n_masks = len(np.load(d / 'env_masks.npz')['phrases'])
        have = {int(p.stem) for p in (d / 'meshes/sam3d').glob('[0-9][0-9][0-9].glb')}
        if len(have) < n_masks:
            f.run('1.5a_sam3d_retry', 'mesh', [s3 / 'flow_sam3d.py', '--dir', d, '--voxels', f.voxels], [])
            have = {int(p.stem) for p in (d / 'meshes/sam3d').glob('[0-9][0-9][0-9].glb')}
        missing = sorted(set(range(n_masks)) - have)
        if not missing:
            break
        z = dict(np.load(d / 'env_masks.npz'))
        keep = [i for i in range(len(z['phrases'])) if i not in missing]
        np.savez_compressed(d / 'env_masks.npz', **{k: v[keep] for k, v in z.items()})
        f.check('1.5a_sam3d', True, {'dropped_no_mesh': [str(z['phrases'][i]) for i in missing]})
        for k, rec in f.log['steps'].items():  # every step from the meshes on
            if k.startswith(('1.5', '2', '3')):
                rec['status'] = 'redo'
    else:
        return f.check('1.5a_sam3d', False, {'missing': missing}, 'meshes still missing after dropping instances')
    f.run('1.5b_refine', 'trellis', [s3 / 'flow_refine.py', '--dir', d] + (['--axis-scale'] if f.axis_scale else []),
          ['refine.json'])
    f.run('1.5c_check_meshes', 'trellis', [s3 / 'flow_check_meshes.py', '--dir', d], ['meshes.json'])
    ms = jload(d, 'meshes.json')['meshes']
    kept = [x for x in ms if x['keep']]
    if not f.check('1.5c_check_meshes', bool(kept), {'kept': [(x['index'], x['phrase']) for x in kept],
                                                   'dropped': [(x['index'], x['phrase'], x['why'])
                                                               for x in ms if not x['keep']]},
                   'no surroundings mesh passed'):
        return False
    if until == '1.5':
        return True
    return True


def phase2(f, until):
    """Imitation: the place from the photo's camera, without its movable objects, checked against the photo."""
    d, s3 = f.d, HERE
    r = jload(d, 'route.json')
    f.run('2.1a_surfaces', 'sam3', [s3 / 'piece_surfaces.py', '--dir', d], ['surfaces.npz'])
    f.run('2.1b_pieces', 'trellis', [s3 / 'piece_build.py', '--dir', d, '--outdoor', '--views', '0', '--depth',
                                     'background'], ['pieces.json', 'isaac/scene.json'])
    keep = [x['index'] for x in jload(d, 'meshes.json')['meshes'] if x['keep']]
    # 2.2 Surroundings meshes into the place, then the geometric gates; a mesh that fails is dropped once.
    for attempt in range(2):
        f.run(f'2.2a_env_{attempt}', 'trellis', [s3 / 'env_add.py', '--env', d / 'meshes', '--scene', d,
                                                 '--out', d / 'place', '--drop-covered-walls', '--fill-ground', '--photo-texture',
                                                 '--keep',
                                                 ','.join(map(str, keep))], ['place/isaac/scene.json'])
        f.run(f'2.2b_gates_{attempt}', 'trellis', [s3 / 'flow_gates.py', '--dir', d], ['gates.json'])
        g = jload(d, 'gates.json')
        drop = [i for i in g['drop'] if i in keep]
        if not drop or attempt == 1:
            break
        keep = [i for i in keep if i not in drop]
        for k in ('place/isaac/scene.json', 'gates.json'):
            (d / k).unlink()
    if not f.check('2.2b_gates', g['ok'] and bool(keep) and not drop,
                   {'ground_depth_err': g['ground_depth_err'], 'ground_cover': g.get('ground_cover'), 'kept': keep,
                    'meshes': [(m['name'], m['iou'], m['base_gap_m'], m['depth_err']) for m in g['meshes']]},
                   'ground off the photo (depth or cover)' if not g['ok'] else 'meshes still failing: ' + str(drop)):
        return False
    # 2.3 Light and sky: the route's HDRI group.
    hdri = ROOT / 'data' / 'hdri' / ('urban' if r['hdri_group'] == 'urban' else '')
    f.run('2.3_render', 'isaac', [s3 / 'piece_isaac.py', '--dir', d / 'place', '--hdri-dir', hdri],
          ['place/isaac/render_00.png'])
    # 2.3b The photo's light, by measurement: view 0 under a fixed set of lights, the closest one kept.
    f.run('2.3b_sweep', 'isaac', [s3 / 'piece_isaac.py', '--dir', d / 'place', '--hdri-dir', hdri, '--sweep'],
          ['place/isaac/sweep/lights.json'])
    f.run('2.3b_light', 'trellis', [s3 / 'flow_light.py', '--dir', d], ['light.json'])
    lt = jload(d, 'light.json')
    f.check('2.3b_light', True, {'best': lt['best'], 'worst_de': lt['worst_de']})
    f.run('2.3c_render', 'isaac', [s3 / 'piece_isaac.py', '--dir', d / 'place', '--hdri-dir', hdri], [])
    # 2.4a Photometry against the photo: objects out of proportion to the photo-textured floor first, then grey
    # world (flow_photometry.py --mode relative); two rounds,
    # a round kept only if the colour difference goes down.
    first = None
    if (d / 'photometry.json').exists():
        (d / 'photometry.json').unlink()  # its gains were for the place as built before
    for r in range(2):
        f.run(f'2.4a_photometry_{r}', 'trellis', [s3 / 'flow_photometry.py', '--dir', d, '--apply'], [])
        ph = jload(d, 'photometry.json')
        first = first or ph
        if ph['ok'] or not ph.get('applied'):
            break
        f.run(f'2.4a_render_{r}', 'isaac', [s3 / 'piece_isaac.py', '--dir', d / 'place', '--hdri-dir', hdri], [])
    f.run('2.4a_photometry_final', 'trellis', [s3 / 'flow_photometry.py', '--dir', d, '--out', 'photometry_final'],
          [])
    fin = jload(d, 'photometry_final.json')
    if fin['mean_de'] > first['mean_de'] and (d / 'place/isaac/orig_scene.json').exists():
        for name in ('scene.json', 'meshes.npz'):  # the gains made it worse: back to the original colours
            (d / 'place/isaac' / name).write_bytes((d / 'place/isaac' / f'orig_{name}').read_bytes())
        f.run('2.4a_render_revert', 'isaac', [s3 / 'piece_isaac.py', '--dir', d / 'place', '--hdri-dir', hdri], [])
        fin = first
    if not fin['checks']['exposure within 0.8-1.25']:
        # The place's views will be exposed by this much more or less (piece_isaac.py "exposure_scale").
        for name in ('scene.json', 'orig_scene.json'):
            p = d / 'place/isaac' / name
            if p.exists():
                sc = json.load(open(p))
                sc['exposure_scale'] = round(min(1.5, max(0.5, fin['exposure'])), 3)
                json.dump(sc, open(p, 'w'))
        # View 0 itself keeps a fixed exposure from its first render: with a light that shades the facade the
        # photo shows lit, it stayed 4.5 times darker than the photo (terrace_row, 2026-10-07). Corrected and
        # measured again.
        src = d / 'place/isaac'
        sc = json.load(open(src / 'scene.json'))
        gain0 = sc.get('exposure_0') or json.load(open(src / 'exposure_0.json'))['gain']
        for name in ('scene.json', 'orig_scene.json'):
            p = src / name
            if p.exists():
                sc = json.load(open(p))
                sc['exposure_0'] = float(gain0 * min(4.0, max(0.25, fin['exposure'])))
                json.dump(sc, open(p, 'w'))
        f.run('2.4a_render_exposure', 'isaac', [s3 / 'piece_isaac.py', '--dir', d / 'place', '--hdri-dir', hdri], [])
        f.run('2.4a_photometry_exposed', 'trellis', [s3 / 'flow_photometry.py', '--dir', d, '--out',
                                                     'photometry_final'], [])
        fin = jload(d, 'photometry_final.json')
    if not f.check('2.4a_photometry', fin['mean_de'] <= 35,
                   {'mean_de': [first['mean_de'], fin['mean_de']], 'exposure': fin['exposure'], 'cast': fin['cast'],
                    'worst': fin['worst'], 'checks': fin['checks']},
                   f"colours far from the photo (mean delta E {fin['mean_de']})"):
        return False
    # 2.4 Side by side with the photo, closed questions in both orders.
    f.run('2.4_compare', 'sam3', [s3 / 'flow_compare.py', '--dir', d], ['compare.json'])
    c = jload(d, 'compare.json')
    return f.check('2.4_compare', c['ok'], c['answers'], 'failed: ' + ', '.join(c['failed']))


def phase3(f, until):
    """Creation of the place: the template the route chose extends the imitated place so any camera can see it."""
    d, s3 = f.d, HERE
    r = jload(d, 'route.json')
    t = r['template']
    if t == 'crossroads':
        f.run('3.2_extend', 'trellis', [s3 / 'crossroad.py', '--scene', d / 'place', '--photo-scene', d, '--out',
                                        d / 'extended', '--no-objects', '--views', '16'], ['extended/isaac/scene.json'])
    elif t == 'straight_street':
        f.run('3.2_extend', 'trellis', [s3 / 'street_template.py', '--scene', d / 'place', '--photo-scene', d,
                                        '--out', d / 'extended', '--views', '16'], ['extended/isaac/scene.json'])
    else:
        return f.check('3.1_template', False, {'template': t}, f'template {t} not built yet')
    f.check('3.1_template', True, {'template': t})
    hdri = ROOT / 'data' / 'hdri' / ('urban' if r['hdri_group'] == 'urban' else '')
    f.run('3.4a_render', 'isaac', [s3 / 'piece_isaac.py', '--dir', d / 'extended', '--hdri-dir', hdri],
          ['extended/sheet_isaac.jpg'])
    f.run('3.4b_views', 'sam3', [s3 / 'flow_check_views.py', '--dir', d / 'extended'], ['extended/views.json'])
    v = jload(d, 'extended/views.json')
    return f.check('3.4b_views', v['ok'], {'kept': v['kept'], 'share': v['share'],
                                          'dropped': [(x['view'], x['why']) for x in v['views'] if not x['keep']]},
                   f"only {v['share']:.0%} of the views pass")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--until', default='', help='stop after this step number (1.2, 1.4, 1.3, ...)')
    ap.add_argument('--redo', default='', help='a step number (1.3c) to run again with all the steps after it')
    ap.add_argument('--voxels', choices=['sam3d', 'ccm', 'hybrid'], default='sam3d',
                    help="SAM 3D stage-2 voxels (flow_sam3d.py --voxels)")
    ap.add_argument('--axis-scale', action='store_true', help='flow_refine.py --axis-scale')
    args = ap.parse_args()
    f = Flow(args.image, args.dir)
    f.voxels = args.voxels
    f.axis_scale = args.axis_scale
    if args.redo:
        # The step and every step after it in the flow's order: later outputs depend on earlier ones.
        order = ['1.1', '1.2', '1.3a', '1.4', '1.3b', '1.3c', '1.5a', '1.5b', '1.5c', '2', '3']
        start = min(i for i, o in enumerate(order) if any(o.startswith(r) or r.startswith(o)
                                                           for r in args.redo.split(',')))
        for k, rec in f.log['steps'].items():
            if any(k.startswith(o) for o in order[start:]):
                rec['status'] = 'redo'
    if not phase1(f, args.until):
        print('stopped in phase 1')
        return
    if args.until.startswith('1'):
        return
    if not phase2(f, args.until):
        print('stopped in phase 2')
        return
    if args.until.startswith('2'):
        return
    if not phase3(f, args.until):
        print('stopped in phase 3')
        return
    print('place ready:', f.d / 'extended')


if __name__ == '__main__':
    main()
