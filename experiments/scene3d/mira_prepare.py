"""Mira-Scene cases from our own photos, with SAM 3 masks from a hand-written object list (run in the
`teacher-sam3` conda env).

Mira-Scene's own segmentation stage asks a VLM (Gemini or Qwen through a paid API) which objects the photo holds
and what each one rests on. For this test the object list and the supports are written by hand in a JSON file;
SAM 3 then segments every phrase and the case is laid out as Mira-Scene's later stages expect:

    <out>/<case>/case.json
    <out>/<case>/scene_graph.json
    <out>/<case>/input/{source,scene,scene_fg,floor_mask,mask_000,...}.png, preprocessing.json

cases.json:
    {"street": {"image": "data/real/maceta/x.jpg", "floor": "road",
                "objects": [{"phrase": "motorcycle", "on": "floor"}, {"phrase": "traffic sign", "on": "wall"}]}}
"on" is "floor", "wall" (hangs from the world) or the phrase of another object (rests on its first instance).

Usage:
    python mira_prepare.py --cases cases.json --out <dir> [--threshold 0.5]
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import Sam3Model, Sam3Processor

SIZE = 518  # Mira-Scene's square model input


def segment(model, proc, image, phrase, threshold):
    inp = proc(images=image, text=phrase, return_tensors='pt').to('cuda')
    with torch.no_grad():
        res = model(**inp)
    r = proc.post_process_instance_segmentation(
        res, threshold=threshold, mask_threshold=0.5, target_sizes=inp.get('original_sizes').tolist())[0]
    order = r['scores'].argsort(descending=True)
    return [(float(r['scores'][i]), r['masks'][i].cpu().numpy().astype(bool)) for i in order]


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def save_mask(mask, path):
    Image.fromarray(mask.astype(np.uint8) * 255, 'L').save(path)


def prepare(model, proc, name, case, out, threshold, root, min_area, max_objects):
    src = os.path.join(root, case['image'])
    original = ImageOps.exif_transpose(Image.open(src)).convert('RGB')
    w, h = original.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    scene = original.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.Resampling.LANCZOS)

    case_dir = os.path.join(out, name)
    inp = os.path.join(case_dir, 'input')
    os.makedirs(inp, exist_ok=True)
    os.makedirs(os.path.join(case_dir, 'logs'), exist_ok=True)
    original.save(os.path.join(inp, 'source.png'))
    scene.save(os.path.join(inp, 'scene.png'))

    masks, nodes, edges, first = [], [], [], {}
    for obj in case['objects']:
        for score, m in segment(model, proc, scene, obj['phrase'], threshold):
            if m.mean() < min_area or len(masks) >= max_objects or any(iou(m, k) > 0.7 for k in masks):
                continue
            k = len(masks)
            masks.append(m)
            first.setdefault(obj['phrase'], k)
            oid = f'object_{k:03d}'
            nodes.append({'id': oid, 'kind': 'object', 'mask_index': k, 'mask_file': f'input/mask_{k:03d}.png',
                          'name': obj['phrase'], 'short_name': obj['phrase'], 'caption': obj['phrase'],
                          'caption_source': 'human', 'name_locked': True, 'caption_locked': True,
                          'motion': 'fixed', 'fixed': True, 'dynamic': False,
                          'upright_mode': 'force' if obj['on'] == 'wall' else 'auto', 'confidence': round(score, 3)})
            on = obj['on']
            parent = {'floor': 'floor', 'wall': 'world_anchor'}.get(on) or f'object_{first.get(on, k):03d}'
            if parent != oid:
                edges.append({'child': oid, 'parent': parent,
                              'relation': 'hangs_from' if on == 'wall' else 'rests_on',
                              'confidence': round(score, 3), 'operational': True})
            save_mask(m, os.path.join(inp, f'mask_{k:03d}.png'))

    floor = np.zeros((SIZE, SIZE), bool)
    for _, m in segment(model, proc, scene, case['floor'], threshold):
        floor |= m
    floor &= ~np.any(masks, axis=0) if masks else True
    save_mask(floor, os.path.join(inp, 'floor_mask.png'))
    fg = np.asarray(scene).copy()
    fg[~np.any(masks, axis=0)] = 0
    Image.fromarray(fg).save(os.path.join(inp, 'scene_fg.png'))

    digest = hashlib.sha256(open(src, 'rb').read()).hexdigest()
    with open(os.path.join(inp, 'preprocessing.json'), 'w') as f:
        json.dump({'schema': 'mira_preprocessing_v1', 'source': src, 'source_filename': os.path.basename(src),
                   'source_sha256': digest, 'original_size': [w, h], 'crop_xywh': [left, top, side, side],
                   'output_size': [SIZE, SIZE], 'exif_transposed': True}, f, indent=2)
    with open(os.path.join(case_dir, 'case.json'), 'w') as f:
        json.dump({'schema': 'mira_case_v1', 'case_id': name, 'source': src,
                   'source_filename': os.path.basename(src), 'source_sha256': digest}, f, indent=2)
    graph = {'schema': 'mira_scene_graph_v1', 'case': name, 'edge_direction': 'child_to_direct_supporter',
             'nodes': [{'id': 'floor', 'kind': 'static_environment', 'name': 'floor', 'motion': 'fixed',
                        'fixed': True, 'dynamic': False},
                       {'id': 'world_anchor', 'kind': 'virtual_static_anchor', 'name': 'wall_or_ceiling',
                        'motion': 'fixed', 'fixed': True, 'dynamic': False}] + nodes,
             'edges': edges}
    with open(os.path.join(case_dir, 'scene_graph.json'), 'w') as f:
        json.dump(graph, f, indent=2)
    return {'case': name, 'objects': len(masks), 'floor_frac': round(float(floor.mean()), 3),
            'names': [n['name'] for n in nodes]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--root', default=os.path.join(os.path.dirname(__file__), '..', '..'))
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--min-area', type=float, default=0.002, help='smallest mask, as a fraction of the image')
    ap.add_argument('--max-objects', type=int, default=12, help='each object costs a SAM 3D run')
    args = ap.parse_args()

    model = Sam3Model.from_pretrained('facebook/sam3').cuda().eval()
    proc = Sam3Processor.from_pretrained('facebook/sam3')
    with open(args.cases) as f:
        cases = json.load(f)
    for name, case in cases.items():
        print(prepare(model, proc, name, case, args.out, args.threshold, os.path.abspath(args.root),
                      args.min_area, args.max_objects), flush=True)


if __name__ == '__main__':
    main()
