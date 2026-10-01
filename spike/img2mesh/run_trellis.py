"""Image-to-3D feasibility test with TRELLIS v1 (run in the `trellis2` conda env).

TRELLIS.2 was dropped: it needs DINOv3, which is gated on Hugging Face.
TRELLIS v1 (MIT) uses DINOv2, which is not.

For each input photo: an open-vocabulary detector (OWLv2) finds the object named
by its text query and crops it, BiRefNet (MIT) cuts it out, TRELLIS turns it into
a textured mesh, and we save a light GLB (the size Isaac Sim would use) plus a
render sheet of that GLB from 4 sides.

Usage:
    python run_trellis.py --root ~/work/nvidia/data/img2mesh_test [--only wrench_1,bee_1]
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('ATTN_BACKEND', 'flash-attn')
os.environ.setdefault('SPCONV_ALGO', 'native')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from transformers import AutoModelForImageSegmentation, Owlv2ForObjectDetection, Owlv2Processor

TRELLIS_DIR = os.environ['SIM2CHIP_TRELLIS_DIR']  # the TRELLIS checkout

# Input file prefix -> detector query.
QUERIES = {
    'wrench': 'an adjustable wrench',
    'bee': 'a bee',
    'cam': 'a security camera',
}


def detect_and_crop(image, query, processor, model, margin=0.15):
    """Crop the highest-scoring box for `query`, square, with some margin."""
    inputs = processor(text=[[query]], images=image, return_tensors='pt').to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)
    # OWLv2 pads to a square: boxes are relative to the padded side.
    side = max(image.size)
    target = torch.tensor([[side, side]], device='cuda')
    res = processor.post_process_grounded_object_detection(outputs, threshold=0.05, target_sizes=target)[0]
    if len(res['scores']) == 0:
        return image, None, 0.0
    i = int(res['scores'].argmax())
    x0, y0, x1, y1 = res['boxes'][i].tolist()
    score = float(res['scores'][i])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * (1 + 2 * margin) / 2
    box = [max(0, cx - half), max(0, cy - half), min(image.width, cx + half), min(image.height, cy + half)]
    return image.crop([int(v) for v in box]), [x0, y0, x1, y1], score


class Cutout:
    """BiRefNet foreground mask -> RGBA image."""

    def __init__(self):
        self.model = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True).float().eval()
        self.tf = transforms.Compose([
            transforms.Resize((1024, 1024)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __call__(self, image):
        self.model.cuda()
        with torch.no_grad():
            pred = self.model(self.tf(image).unsqueeze(0).cuda())[-1].sigmoid().cpu()[0, 0]
        self.model.cpu()
        mask = transforms.ToPILImage()(pred).resize(image.size)
        out = image.copy()
        out.putalpha(mask)
        return out


def render_glb(path, size=384):
    """Render a GLB from 4 yaw angles with pyrender (what the mesh really looks like)."""
    import pyrender
    import trimesh
    scene_mesh = trimesh.load(path, force='scene')
    bounds = scene_mesh.bounds
    center = bounds.mean(axis=0)
    radius = np.linalg.norm(bounds[1] - bounds[0]) / 2
    r = pyrender.OffscreenRenderer(size, size)
    views = []
    for yaw in np.radians([30, 120, 210, 300]):
        scene = pyrender.Scene.from_trimesh_scene(scene_mesh, ambient_light=[0.35] * 3)
        scene.bg_color = np.array([1.0, 1.0, 1.0, 1.0])
        pitch = np.radians(20)
        d = 3.4 * radius
        eye = center + d * np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
        z = (eye - center) / np.linalg.norm(eye - center)
        x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        pose = np.eye(4); pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, y, z, eye
        scene.add(pyrender.PerspectiveCamera(yfov=np.radians(40)), pose=pose)
        scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
        color, _ = r.render(scene)
        views.append(color)
    r.delete()
    return np.concatenate(views, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--only', default=None, help='comma-separated input stems')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    out_root = os.path.join(root, 'out_trellis1')
    os.makedirs(out_root, exist_ok=True)

    sys.path.insert(0, TRELLIS_DIR)
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    processor = Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble')
    detector = Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble').eval()
    cutout = Cutout()
    pipeline = TrellisImageTo3DPipeline.from_pretrained('microsoft/TRELLIS-image-large')
    pipeline.cuda()

    stems = sorted(f[:-4] for f in os.listdir(os.path.join(root, 'inputs')) if f.endswith('.jpg'))
    if args.only:
        stems = [s for s in stems if s in args.only.split(',')]
    report = []
    for stem in stems:
        query = QUERIES[stem.split('_')[0]]
        out = os.path.join(out_root, stem)
        os.makedirs(out, exist_ok=True)
        image = Image.open(os.path.join(root, 'inputs', stem + '.jpg')).convert('RGB')

        detector.cuda()
        crop, box, score = detect_and_crop(image, query, processor, detector)
        detector.cpu()
        vis = image.copy()
        if box:
            ImageDraw.Draw(vis).rectangle(box, outline='red', width=6)
        vis.save(os.path.join(out, 'detection.jpg'))
        rgba = cutout(crop)
        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        prep = pipeline.preprocess_image(rgba)
        prep.save(os.path.join(out, 'preprocessed.png'))
        outputs = pipeline.run(prep, seed=args.seed, formats=['mesh', 'gaussian'], preprocess_image=False)
        t_gen = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30

        t1 = time.time()
        glb = postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0],
                                          simplify=0.95, texture_size=1024, verbose=False)
        glb_path = os.path.join(out, 'model.glb')
        glb.export(glb_path)
        t_glb = time.time() - t1
        Image.fromarray(render_glb(glb_path)).save(os.path.join(out, 'views.jpg'), quality=90)

        row = {'stem': stem, 'query': query, 'det_score': round(score, 3), 'gen_s': round(t_gen, 1),
               'glb_s': round(t_glb, 1), 'peak_gib': round(peak, 1), 'faces': int(len(glb.faces)),
               'glb_mb': round(os.path.getsize(glb_path) / 2**20, 2)}
        print(row, flush=True)
        report.append(row)
        del outputs, glb
        torch.cuda.empty_cache()

    with open(os.path.join(out_root, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
