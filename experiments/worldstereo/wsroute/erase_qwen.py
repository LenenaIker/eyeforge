"""Object removal with Qwen-Image-Edit-2511 (Apache 2.0) and the Object-Remover LoRA (prithivMLmods, Apache 2.0),
to compare with LaMa in bg_prepare.py. LaMa only continues textures, so a large mask (a chair in front of a
bookshelf) comes out as a blur; an edit model can invent what stands behind. The objects to remove are named in
the prompt (bg_prepare.py's instances). Its whole picture is kept, since it also clears the shadows LaMa's mask
left out, unless it moved off the photo's pixels; then the edit is pasted back only inside bg_prepare.py's erase
mask, feathered, so the rest of the photo stays exact. Fits the 3090 with both big models in 4 bits and CPU
offload. By default with the Lightning 4-step LoRA (lightx2v, Apache 2.0) and up to five passes, under a minute
and a half per photo; --full for the 40-step model. Run in /media/lenena-iker/Crucial/envs/eyeforge-qwenedit with HF_HOME on Crucial:

    python experiments/worldstereo/wsroute/erase_qwen.py --dir <folder> [--dir ...]

Qwen does not always remove every object it is told to (bottles 2026-10-08: of three bottles and three caps it
took one bottle). Each instance whose pixels barely changed (mean difference under --left, 0-255) is still there:
the next pass names only those. What is still there after the last pass stays (no LaMa blur, user 2026-10-08)
and is listed under "left" in erase_qwen.json.

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


LIGHTNING = ('lightx2v/Qwen-Image-Edit-2511-Lightning', 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors')


def local(repo):
    """The repo's folder in the Hugging Face cache when it is there, else its id. Offline (a cloud job, its weights
    pulled from the cache bucket), diffusers still asks the Hub about a sharded model given by id."""
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception:
        return repo


def base_pipe(bits, encoder_bits=None):
    """Qwen-Image-Edit-2511 without LoRAs, on the CPU. bits: the transformer's, encoder_bits the text encoder's
    (default the same). The panorama step (pano.py) loads it too, with its own LoRAs."""
    encoder_bits = encoder_bits or bits
    base = local(BASE)
    nf4 = dict(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
    saved = os.environ.get('SIM2CHIP_QWEN_NF4')  # the pipeline already in 4 bits (--save-nf4): a third of the
    # bytes to read, nothing to quantize (on a Nebius disk the bf16 load took 7-8 min, 2026-10-12)
    if bits == 4 and encoder_bits == 4 and saved and os.path.exists(os.path.join(saved, 'model_index.json')):
        pipe = QwenImageEditPlusPipeline.from_pretrained(saved, torch_dtype=torch.bfloat16)
    elif bits == 16 and encoder_bits == 16:
        pipe = QwenImageEditPlusPipeline.from_pretrained(base, torch_dtype=torch.bfloat16)
    else:
        # In 4 bits the first block's modulation and the in/out projections turn the whole picture into
        # speckle (home_office, 2026-10-08): they stay in bf16.
        skip = ['transformer_blocks.0.img_mod', 'img_in', 'txt_in', 'proj_out', 'norm_out', 'time_text_embed']
        transformer = QwenImageTransformer2DModel.from_pretrained(
            base, subfolder='transformer', torch_dtype=torch.bfloat16,
            **({} if bits == 16 else {'quantization_config': DiffusersBnb(**nf4, llm_int8_skip_modules=skip)}))
        encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base, subfolder='text_encoder', torch_dtype=torch.bfloat16,
            **({} if encoder_bits == 16 else {'quantization_config': TransformersBnb(**nf4)}))
        pipe = QwenImageEditPlusPipeline.from_pretrained(base, transformer=transformer, text_encoder=encoder,
                                                         torch_dtype=torch.bfloat16)
    return pipe


def load(bits, lightning=False, encoder_bits=None, gpu=False, save=None):
    """bits: the transformer's, encoder_bits the text encoder's (default the same). gpu: the whole pipeline stays on
    the GPU instead of moving each model there in turn from the CPU (an L40S holds both in 4 bits). save: a folder
    to write the quantized pipeline to, before its LoRAs, instead."""
    pipe = base_pipe(bits, encoder_bits)
    if save:
        lightning = False
    else:
        pipe.load_lora_weights(local(LORA), adapter_name='remover')
    if lightning:
        # 4 steps instead of 40 (lightx2v, Apache 2.0): 23 s instead of 260 on the 3090. One pass leaves more
        # behind (home_office 12 of 24 against 1), but three passes, each naming what is left, came out cleaner
        # than one full pass on user_desk, home_office and bottles in 45-90 s (erase_bench.py, 2026-10-08).
        pipe.load_lora_weights(local(LIGHTNING[0]), weight_name=LIGHTNING[1], adapter_name='lightning')
        pipe.set_adapters(['remover', 'lightning'], adapter_weights=[1.0, 1.0])
    if save:
        pipe.save_pretrained(save)
        return None
    if gpu:
        pipe.to('cuda')
    else:
        pipe.enable_model_cpu_offload()
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', action='append', default=[])
    ap.add_argument('--save-nf4', help='write the 4-bit pipeline to this folder and stop (SIM2CHIP_QWEN_NF4 reads it)')
    ap.add_argument('--bits', type=int, choices=[4, 16], default=int(os.environ.get('SIM2CHIP_ERASE_BITS', 4)),
                    help="the transformer's: 4 on the 3090")
    ap.add_argument('--encoder-bits', type=int, choices=[4, 16], help="the text encoder's (default --bits)")
    ap.add_argument('--gpu', action='store_true', default=os.environ.get('SIM2CHIP_ERASE_GPU') == '1',
                    help='keep the whole pipeline on the GPU, no CPU offload (a 48 GB GPU)')
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--cfg', type=float, default=4.0)
    ap.add_argument('--feather', type=int, default=6, help='blur of the paste edge, pixels')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--paste', choices=['auto', 'raw', 'mask'], default='auto',
                    help="auto: the model's whole picture unless it moved half a pixel or more, then mask")
    ap.add_argument('--replace', action='store_true', help='background.png becomes the Qwen one (LaMa kept aside)')
    ap.add_argument('--full', action='store_true', help='40 steps without the Lightning LoRA (the default is '
                    'Lightning, 4 steps and true_cfg 1: 3 passes cleaner than one full pass, in a fifth of the time)')
    ap.add_argument('--passes', type=int, default=5, help='most Qwen passes; later ones name only what is left, '
                    'and they stop once nothing is (Lightning needed 1-2 on 5 photos)')
    ap.add_argument('--left', type=float, default=12.0, help='mean change under which an object is still there '
                    '(left ones measured 4-10, erased ones 15 and up on 5 photos)')
    args = ap.parse_args()
    if not args.full:
        args.steps, args.cfg = 4, 1.0
    if args.save_nf4:
        load(4, save=args.save_nf4)
        return
    t_load = time.time()
    pipe = load(args.bits, lightning=not args.full, encoder_bits=args.encoder_bits, gpu=args.gpu)
    t_load = round(time.time() - t_load, 1)
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
        info = {'objects': names, 'left_after_pass': passes, 'left': [phrases[i] for i in todo],
                'shift_px': round(moved, 2), 'used': 'raw' if use_raw else 'mask',
                'seconds': round(time.time() - t, 1), 'load_seconds': t_load,
                'bits': [args.bits, args.encoder_bits or args.bits], 'gpu_resident': args.gpu,
                'peak_gpu_gb': round(torch.cuda.max_memory_allocated() / 1e9, 1)}
        json.dump(info, open(os.path.join(d, 'erase_qwen.json'), 'w'), indent=1)
        print(json.dumps({'dir': d, **info}), flush=True)


if __name__ == '__main__':
    main()
