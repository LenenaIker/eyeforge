"""Optional step before 1.5a of photo2scene: Mira-CCM (MIT, VAST) canonical coordinates for each surroundings
mask. For every photo pixel of the object the model predicts where that point sits in the object's own normalised
frame (z-up cube, -0.5 to 0.5), and from that the occupied 64^3 voxels. They are pixel aligned, so the visible
shape follows the photo more closely than SAM 3D's own stage 1, but the hidden back is missing; flow_sam3d.py
(`--voxels ccm|hybrid`) feeds them to SAM 3D's stage 2. Runs in the `eyeforge-sam3d` env with the newer diffusers,
transformers and peft that CCM needs kept apart in tools/ccm_deps (SAM 3D keeps its own):

    python experiments/scene3d/flow_ccm.py --dir data/street_test/flow/basque_street

Writes <dir>/meshes/ccm/NNN.npz: `ccm` (H, W, 3) float16 in photo pixels, zero off the object, and `voxels`
(K, 3) int indices on the 64^3 grid.
"""
import argparse
import json
import os
import shutil
import sys
import time

TOOLS = os.environ.get('SIM2CHIP_TOOLS', '/media/lenena-iker/Crucial/tools')
MIRA = os.path.join(TOOLS, 'mira_src')
for p in (os.path.join(MIRA, 'UniDataset', 'src'), os.path.join(MIRA, 'Mira-CCM', 'src'),
          os.path.join(TOOLS, 'ccm_deps')):
    sys.path.insert(0, p)
CKPT = os.environ.get('CCM_CKPT', os.path.join(TOOLS, 'ccm_ckpt', 'pipeline'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidance', type=float, default=3.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fresh', action='store_true', help='start from no CCM maps (the masks changed)')
    args = ap.parse_args()
    from miraccm.pipelines.shape_synthesis.pipeline_ccm_voxel import CCMVoxelPipeline
    from miraccm.systems.shape_synthesis.data_processor.ccm_voxel import DataProcessor

    d = args.dir
    out_dir = os.path.join(d, 'meshes', 'ccm')
    if args.fresh:
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    em = np.load(os.path.join(d, 'env_masks.npz'))
    image = np.asarray(Image.open(os.path.join(d, 'background.png')).convert('RGB')).astype(np.float32) / 255
    h, w = image.shape[:2]
    # CCM centre-crops to a square; pad instead so the street's sides stay in.
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    canvas = np.zeros((s, s, 3), np.float32)
    canvas[top:top + h, left:left + w] = image

    processor = DataProcessor(use_cropped_condition=True)
    pipe = CCMVoxelPipeline.from_pretrained(CKPT, torch_dtype=torch.bfloat16, safety_checker=None,
                                            requires_safety_checker=False).to('cuda')
    layout_res = pipe.transformer.config.latent_config['layout']['pos_embedder']['resolution']
    done = []
    for k in range(len(em['phrases'])):
        path = os.path.join(out_dir, f'{k:03d}.npz')
        if os.path.exists(path):
            continue
        mask = em['masks'][k].astype(bool)
        if mask.shape != (h, w):
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 0
        m = np.zeros((s, s), np.float32)
        m[top:top + h, left:left + w] = mask
        t0 = time.time()
        inp = processor.prepare_inference_input(canvas, [m], device='cuda')
        mi = {key: v.to(torch.bfloat16) if isinstance(v, torch.Tensor) else v for key, v in inp.items()}
        mask_src = inp['mask_cropped_1ch']
        layout_mask = F.interpolate(mask_src.float(), size=(layout_res, layout_res), mode='nearest').to(torch.bfloat16)
        layout_image = F.interpolate(mi['image_cropped'].float(), size=(layout_res, layout_res), mode='bilinear',
                                     align_corners=False).to(torch.bfloat16)
        with torch.no_grad():
            out = pipe(image=mi['image'], mask=mi['mask'], image_cropped=mi['image_cropped'],
                       mask_cropped=mi['mask_cropped'], num_inference_steps=args.steps, resolution=64,
                       guidance_scale=args.guidance, keep_layout_condition_in_uncond=True,
                       layout_mask_for_concat=layout_mask, layout_image_for_conv=layout_image,
                       use_cropped_condition=True, layout_pred_mode='velocity', layout_t_eps=1e-5,
                       generator=torch.Generator(device='cuda').manual_seed(args.seed + k))
        # As Mira's 2_inference_CCM.py: mask, upsample to the crop, paste back on the 518 canvas.
        ccm = out.latent_voxel_cam_pts
        ccm = ccm * (F.interpolate(mask_src.float(), size=ccm.shape[-2:], mode='nearest') > 0.5).to(ccm.dtype)
        H, W = inp['ori_image'].shape[-2:]
        up = F.interpolate(ccm.float(), size=(H, W), mode='bilinear', align_corners=False).clamp(-0.5, 0.5)
        up = up * (F.interpolate(mask_src.float(), size=(H, W), mode='nearest') > 0.5).float()
        full = processor.restore_canonical_coord_map(canonical_coord_map_cropped=up, crop_params=inp['crop_params'])
        full = F.interpolate(full.float(), size=(s, s), mode='nearest')[0].permute(1, 2, 0).cpu().numpy()
        full = full[top:top + h, left:left + w] * mask[..., None]
        vox = out.coords[0]
        vox = (vox.cpu().numpy() if torch.is_tensor(vox) else np.asarray(vox)).astype(np.int32)
        np.savez_compressed(path, ccm=full.astype(np.float16), voxels=vox)
        done.append(k)
        print(f'{k:03d} {em["phrases"][k]}: {len(vox)} voxels, {time.time() - t0:.1f} s', flush=True)
        torch.cuda.empty_cache()
    print(json.dumps({'ccm': done}))


if __name__ == '__main__':
    main()
