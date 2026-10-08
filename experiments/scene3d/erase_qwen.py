"""Object removal with Qwen-Image-Edit-2511 (Apache 2.0) and the Object-Remover LoRA (prithivMLmods, Apache 2.0),
to compare with LaMa in bg_prepare.py. LaMa only continues textures, so a large mask (a chair in front of a
bookshelf) comes out as a blur; an edit model can invent what stands behind. The objects to remove are named in
the prompt (bg_prepare.py's instances); the edit is pasted back only inside bg_prepare.py's erase mask, feathered,
so the rest of the photo stays exact. Fits the 3090 with both big models in 4 bits and CPU offload. Run in
/media/lenena-iker/Crucial/envs/eyeforge-qwenedit with HF_HOME on Crucial:

    python experiments/scene3d/erase_qwen.py --dir data/scene3d_test/flow/home_office [--dir ...]

Writes <dir>/background_qwen.png (pasted) and <dir>/background_qwen_raw.png (the model's whole picture).
"""
import argparse
import json
import os
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
    args = ap.parse_args()
    pipe = load(args.bits)
    for d in args.dir:
        photo = Image.open(os.path.join(d, 'photo.png')).convert('RGB')
        names = list(dict.fromkeys(json.load(open(os.path.join(d, 'prepare.json')))['instances']))
        if not names:
            continue
        prompt = PROMPT.format(objects=', '.join(names[:-1]) + (' and ' if len(names) > 1 else '') + names[-1])
        t = time.time()
        out = pipe(image=[photo], prompt=prompt, negative_prompt=' ', true_cfg_scale=args.cfg,
                   num_inference_steps=args.steps, generator=torch.Generator('cpu').manual_seed(args.seed)).images[0]
        out = out.resize(photo.size, Image.Resampling.LANCZOS)
        out.save(os.path.join(d, 'background_qwen_raw.png'))
        mask = Image.open(os.path.join(d, 'erased.png')).convert('L').filter(ImageFilter.GaussianBlur(args.feather))
        Image.composite(out, photo, mask).save(os.path.join(d, 'background_qwen.png'))
        print(json.dumps({'dir': d, 'objects': names, 'seconds': round(time.time() - t, 1)}), flush=True)


if __name__ == '__main__':
    main()
