"""Object removal with Qwen-Image-Edit-2511 (Apache 2.0) and the Object-Remover LoRA (prithivMLmods, Apache 2.0),
to compare with LaMa in bg_prepare.py. LaMa only continues textures, so a large mask (a chair in front of a
bookshelf) comes out as a blur; an edit model can invent what stands behind. The objects to remove are named in
the prompt (bg_prepare.py's instances). Its whole picture is kept, since it also clears the shadows LaMa's mask
left out, unless it moved off the photo's pixels; then the edit is pasted back only inside bg_prepare.py's erase
mask, feathered, so the rest of the photo stays exact. Fits the 3090 with both big models in 4 bits and CPU
offload. Run in /media/lenena-iker/Crucial/envs/eyeforge-qwenedit with HF_HOME on Crucial:

    python experiments/scene3d/erase_qwen.py --dir data/scene3d_test/flow/home_office [--dir ...]

Qwen does not always remove every object it is told to (bottles 2026-10-08: of three bottles and three caps it
took one bottle). Each instance whose pixels barely changed (mean difference under --left, 0-255) is still there:
a second pass names only those, and what is left after it takes LaMa's fill inside its mask.

Writes <dir>/background_qwen.png (the model's whole picture, or pasted when it moved), background_qwen_raw.png
and erase_qwen.json (shift, which one was used, what each pass left). With
--replace (photo2scene.py step 1.3a_erase) LaMa's background.png is kept as background_lama.png and the Qwen one
takes its place, so every later step reads it.
"""
import argparse
import filecmp
import json
import os
import shutil
import time

import numpy as np
import torch
from diffusers import BitsAndBytesConfig as DiffusersBnb
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from PIL import Image, ImageFilter
from transformers import BitsAndBytesConfig as TransformersBnb
from transformers import Qwen2_5_VLForConditionalGeneration

BASE = 'Qwen/Qwen-Image-Edit-2511'
LORA = 'prithivMLmods/Qwen-Image-Edit-2511-Object-Remover'
PROMPT = ('Remove the {objects} from the image while preserving the background and remaining elements, '
          'maintaining realism and original details.')


def shift(a, b, keep):
    """Sub-pixel translation of b against a (phase correlation on grey levels, only where keep), in pixels."""
    a = np.asarray(a.convert('L'), np.float32) * keep
    b = np.asarray(b.convert('L'), np.float32) * keep
    r = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    c = np.real(np.fft.ifft2(r / (np.abs(r) + 1e-9)))
    y, x = np.unravel_index(np.argmax(c), c.shape)
    y = y - c.shape[0] if y > c.shape[0] // 2 else y
    x = x - c.shape[1] if x > c.shape[1] // 2 else x
    return float(np.hypot(x, y))


def left_behind(photo, out, masks, thr):
    """Indices of the instances whose pixels the edit barely changed: the object is still there."""
    a = np.asarray(photo, np.float32)
    b = np.asarray(out, np.float32)
    return [i for i, m in enumerate(masks) if m.any() and np.abs(a[m] - b[m]).mean() < thr]


def prompt_for(names):
    names = list(dict.fromkeys(names))
    return PROMPT.format(objects=', '.join(names[:-1]) + (' and ' if len(names) > 1 else '') + names[-1])


def load(bits):
    if bits == 16:
        pipe = QwenImageEditPlusPipeline.from_pretrained(BASE, torch_dtype=torch.bfloat16)
    else:
        nf4 = dict(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
        # In 4 bits the first block's modulation and the in/out projections turn the whole picture into
        # speckle (home_office, 2026-10-08): they stay in bf16.
        skip = ['transformer_blocks.0.img_mod', 'img_in', 'txt_in', 'proj_out', 'norm_out', 'time_text_embed']
        transformer = QwenImageTransformer2DModel.from_pretrained(
            BASE, subfolder='transformer', quantization_config=DiffusersBnb(**nf4, llm_int8_skip_modules=skip),
            torch_dtype=torch.bfloat16)
        encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE, subfolder='text_encoder', quantization_config=TransformersBnb(**nf4), torch_dtype=torch.bfloat16)
        pipe = QwenImageEditPlusPipeline.from_pretrained(BASE, transformer=transformer, text_encoder=encoder,
                                                         torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(LORA)
    pipe.enable_model_cpu_offload()
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', action='append', required=True)
    ap.add_argument('--bits', type=int, choices=[4, 16], default=4, help='4 on the 3090, 16 on an L40S')
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--cfg', type=float, default=4.0)
    ap.add_argument('--feather', type=int, default=6, help='blur of the paste edge, pixels')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--paste', choices=['auto', 'raw', 'mask'], default='auto',
                    help="auto: the model's whole picture unless it moved half a pixel or more, then mask")
    ap.add_argument('--replace', action='store_true', help='background.png becomes the Qwen one (LaMa kept aside)')
    ap.add_argument('--passes', type=int, default=2, help='Qwen passes; later ones name only what is left')
    ap.add_argument('--left', type=float, default=12.0, help='mean change under which an object is still there '
                    '(left ones measured 4-10, erased ones 15 and up on 5 photos)')
    args = ap.parse_args()
    pipe = load(args.bits)
    for d in args.dir:
        photo = Image.open(os.path.join(d, 'photo.png')).convert('RGB')
        names = list(dict.fromkeys(json.load(open(os.path.join(d, 'prepare.json')))['instances']))
        if not names:
            continue
        z = np.load(os.path.join(d, 'instances.npz'))
        masks, phrases = z['masks'].astype(bool), [str(p) for p in z['phrases']]
        t = time.time()
        out, todo, passes = photo, list(range(len(masks))), []
        for k in range(args.passes):
            out = pipe(image=[out], prompt=prompt_for(phrases[i] for i in todo) if k else prompt_for(names),
                       negative_prompt=' ', true_cfg_scale=args.cfg, num_inference_steps=args.steps,
                       generator=torch.Generator('cpu').manual_seed(args.seed + k)).images[0]
            out = out.resize(photo.size, Image.Resampling.LANCZOS)
            todo = [i for i in left_behind(photo, out, masks, args.left)]
            passes.append([phrases[i] for i in todo])
            if not todo:
                break
        bg, qw = os.path.join(d, 'background.png'), os.path.join(d, 'background_qwen.png')
        # On a redo of this step alone background.png is already the last Qwen one: LaMa's copy must stay.
        fresh_lama = not (os.path.exists(qw) and filecmp.cmp(bg, qw, shallow=False))
        lama = Image.open(bg if fresh_lama else os.path.join(d, 'background_lama.png')).convert('RGB')
        if todo:
            # Still there after every pass: LaMa's fill inside those masks (grown, feathered).
            m = Image.fromarray((np.any(masks[todo], 0) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(15))
            out = Image.composite(lama, out, m.filter(ImageFilter.GaussianBlur(args.feather)))
        out.save(os.path.join(d, 'background_qwen_raw.png'))
        erased = Image.open(os.path.join(d, 'erased.png')).convert('L')
        mask = erased.filter(ImageFilter.GaussianBlur(args.feather))
        # The model's whole picture also clears the shadows left outside LaMa's mask (user, 2026-10-08), but it
        # must stay on the photo's pixels: depth, masks and the photo texture come from the photo. A shift, as
        # on basque_street (1.8 px), falls back to the paste inside the mask.
        far = np.asarray(erased.filter(ImageFilter.MaxFilter(41))) == 0
        moved = shift(photo, out, far)
        use_raw = args.paste == 'auto' and moved < 0.5 or args.paste == 'raw'
        (out if use_raw else Image.composite(out, photo, mask)).save(qw)
        if args.replace:
            if fresh_lama:
                shutil.copy(bg, os.path.join(d, 'background_lama.png'))
            shutil.copy(qw, bg)
        info = {'objects': names, 'left_after_pass': passes, 'lama_for': [phrases[i] for i in todo],
                'shift_px': round(moved, 2), 'used': 'raw' if use_raw else 'mask',
                'seconds': round(time.time() - t, 1)}
        json.dump(info, open(os.path.join(d, 'erase_qwen.json'), 'w'), indent=1)
        print(json.dumps({'dir': d, **info}), flush=True)


if __name__ == '__main__':
    main()
