"""Image-to-3D with TRELLIS.2-4B on the SAM 3 cutouts (run in the `trellis2` conda env).

Same inputs and outputs as run_trellis.py --cutout sam3 (masks/<stem>/rgba.png from sam3_masks.py), so the
render sheets compare side by side. The GLB is exported light, as Isaac Sim would use it.

Usage:
    python run_trellis2.py --root ~/work/eyeforge/data/img2mesh_test [--only wrench_1] [--type 512]
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import torch
from PIL import Image

from run_trellis import render_glb

TRELLIS2_DIR = os.environ['SIM2CHIP_TRELLIS2_DIR']  # the TRELLIS.2 checkout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--only', default=None, help='comma-separated input stems')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--type', default='1024_cascade', help="pipeline type: '512', '1024', '1024_cascade'")
    ap.add_argument('--faces', type=int, default=20000, help='GLB decimation target')
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    out_root = os.path.join(root, f'out_sam3_trellis2_{args.type}')
    os.makedirs(out_root, exist_ok=True)

    sys.path.insert(0, TRELLIS2_DIR)
    import o_voxel
    from trellis2.pipelines import Trellis2ImageTo3DPipeline, rembg

    class NoRembg:
        """Our cutouts already carry alpha, so the background remover never runs. The default one, RMBG-2.0,
        is gated and CC BY-NC: never load it."""

        def __init__(self, *args, **kwargs):
            pass

        def to(self, *args, **kwargs):
            return self

        cpu = cuda = to

    for name in dir(rembg):
        if isinstance(getattr(rembg, name), type):
            setattr(rembg, name, NoRembg)

    # transformers 5 moved the DINOv3 blocks from model.layer to model.model.layer.
    import torch.nn.functional as F
    from trellis2.modules import image_feature_extractor

    def extract_features(self, image):
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)
        for layer_module in self.model.model.layer:
            hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)
        return F.layer_norm(hidden_states, hidden_states.shape[-1:])

    image_feature_extractor.DinoV3FeatureExtractor.extract_features = extract_features

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    pipeline.cuda()

    stems = sorted(d for d in os.listdir(os.path.join(root, 'masks')) if os.path.isdir(os.path.join(root, 'masks', d)))
    if args.only:
        stems = [s for s in stems if s in args.only.split(',')]
    report = []
    for stem in stems:
        out = os.path.join(out_root, stem)
        os.makedirs(out, exist_ok=True)
        rgba = Image.open(os.path.join(root, 'masks', stem, 'rgba.png'))

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        prep = pipeline.preprocess_image(rgba)
        prep.save(os.path.join(out, 'preprocessed.png'))
        mesh = pipeline.run(prep, seed=args.seed, preprocess_image=False, pipeline_type=args.type)[0]
        t_gen = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30

        t1 = time.time()
        mesh.simplify(16777216)  # nvdiffrast limit
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
            attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=args.faces, texture_size=1024, remesh=True, remesh_band=1, remesh_project=0,
            verbose=False)
        glb_path = os.path.join(out, 'model.glb')
        glb.export(glb_path)
        t_glb = time.time() - t1
        peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
        Image.fromarray(render_glb(glb_path)).save(os.path.join(out, 'views.jpg'), quality=90)

        row = {'stem': stem, 'type': args.type, 'gen_s': round(t_gen, 1), 'glb_s': round(t_glb, 1),
               'peak_gib': round(peak, 1), 'faces': int(len(glb.faces)),
               'glb_mb': round(os.path.getsize(glb_path) / 2**20, 2)}
        print(row, flush=True)
        report.append(row)
        del mesh, glb
        torch.cuda.empty_cache()

    with open(os.path.join(out_root, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
