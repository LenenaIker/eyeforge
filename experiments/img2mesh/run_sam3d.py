"""Image-to-3D with SAM 3D Objects on the SAM 3 masks (run in the `eyeforge-sam3d` env).

Unlike TRELLIS it takes the whole photo plus the mask (masks/<stem>/mask.png from sam3_masks.py), not a cutout:
the scene around the object is context, and it also returns the object's pose in the camera frame.

The mesh is exported on the licence-clean path: the pytorch3d rasteriser (as Meta's own notebook does), gsplat
(Apache) instead of the Inria Gaussian rasteriser (non-commercial) and no hole filling, which goes through
nvdiffrast (non-commercial). --texture decimates the mesh by 95% (pyvista) and bakes a 1024 texture; without
it the GLB carries the full mesh with vertex colours, Meta's default.

Usage:
    python run_sam3d.py --root ~/work/eyeforge/data/img2mesh_test [--only wrench_1] [--texture]
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np
import torch
from PIL import Image, ImageOps

from run_trellis import render_glb

SAM3D_DIR = os.environ['SIM2CHIP_SAM3D_DIR']  # the sam-3d-objects checkout, checkpoints in checkpoints/hf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--only', default=None, help='comma-separated input stems')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--texture', action='store_true', help='bake a texture instead of vertex colours')
    ap.add_argument('--full-mask', action='store_true', help='no SAM 3: the mask is the whole photo')
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    out_root = os.path.join(root, ('out_sam3d_tex' if args.texture else 'out_sam3d') + ('_fullmask' if args.full_mask else ''))
    os.makedirs(out_root, exist_ok=True)

    sys.path.insert(0, os.path.join(SAM3D_DIR, 'notebook'))
    from inference import Inference
    from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils, render_utils

    render_frames = render_utils.render_frames

    def render_frames_gsplat(sample, extrinsics, intrinsics, options={}, **kwargs):
        return render_frames(sample, extrinsics, intrinsics, {**options, 'backend': 'gsplat'}, **kwargs)

    postprocess_mesh = postprocessing_utils.postprocess_mesh

    def postprocess_mesh_no_holes(*args, **kwargs):
        return postprocess_mesh(*args, **{**kwargs, 'fill_holes': False})

    render_utils.render_frames = render_frames_gsplat
    postprocessing_utils.postprocess_mesh = postprocess_mesh_no_holes

    inference = Inference(os.path.join(SAM3D_DIR, 'checkpoints', 'hf', 'pipeline.yaml'), compile=False)

    stems = sorted(d for d in os.listdir(os.path.join(root, 'masks')) if os.path.isdir(os.path.join(root, 'masks', d)))
    if args.only:
        stems = [s for s in stems if s in args.only.split(',')]
    report = []
    for stem in stems:
        out = os.path.join(out_root, stem)
        os.makedirs(out, exist_ok=True)
        image = np.asarray(ImageOps.exif_transpose(Image.open(os.path.join(root, 'inputs', stem + '.jpg'))).convert('RGB'))
        mask = np.asarray(Image.open(os.path.join(root, 'masks', stem, 'mask.png'))) > 127
        if args.full_mask:
            mask = np.ones_like(mask)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            rgba = inference.merge_mask_to_rgba(image, mask)
            output = inference._pipeline.run(
                rgba, None, args.seed, stage1_only=False, with_mesh_postprocess=args.texture,
                with_texture_baking=args.texture, with_layout_postprocess=False, use_vertex_color=not args.texture,
                stage1_inference_steps=None, pointmap=None)
        except torch.OutOfMemoryError as e:
            row = {'stem': stem, 'error': 'OOM', 'peak_gib': round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            print(row, flush=True)
            report.append(row)
            torch.cuda.empty_cache()
            continue
        t_gen = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30

        glb_path = os.path.join(out, 'model.glb')
        output['glb'].export(glb_path)
        output['gs'].save_ply(os.path.join(out, 'splat.ply'))
        Image.fromarray(render_glb(glb_path)).save(os.path.join(out, 'views.jpg'), quality=90)
        pose = {k: output[k].detach().cpu().numpy().round(4).tolist() for k in ('rotation', 'translation', 'scale')
                if k in output}

        row = {'stem': stem, 'gen_s': round(t_gen, 1), 'peak_gib': round(peak, 1),
               'faces': int(len(output['glb'].faces)), 'glb_mb': round(os.path.getsize(glb_path) / 2**20, 2),
               'pose': pose}
        print(row, flush=True)
        report.append(row)
        del output
        torch.cuda.empty_cache()

    with open(os.path.join(out_root, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
