#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from PIL import Image
from tqdm.auto import tqdm

import diffusers

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from MAE.util import misc
import torch.nn as nn

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.flux.util import load_ae, load_flow_model
from src.flux.sampling import denoise, get_noise, get_schedule, unpack
from datas.misato_dataset import InpaintingDataset
from einops import rearrange
from accelerate import DistributedDataParallelKwargs

logger = get_logger(__name__)



def log_validation(vae, flow_transformer, val_dataloader, step, accelerator, weight_dtype, visual_condition_extractor, alignment_clip, alignment_flant, img_size):
    if accelerator.is_main_process:
        logger.info(f"Validation log in step {step}")

    seed = 42

    none_clip_fea = torch.load("data/vec_empty.pt").to(accelerator.device, dtype=weight_dtype)
    none_flant_fea = torch.load("data/txt_256.pt").to(accelerator.device, dtype=weight_dtype)


    condition_weight = args.condition_weight

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for batch in tqdm(val_dataloader, desc=f"Validation rank{accelerator.process_index}..."):
            orig_img = batch["orig_img"].to(accelerator.device, dtype=weight_dtype)
            mask = batch['mask'].to(accelerator.device, dtype=weight_dtype)
            mask_cond = batch['mask_cond'].to(accelerator.device, dtype=weight_dtype)

            height, width = img_size, img_size
            noise = get_noise(
                orig_img.shape[0],
                height,
                width,
                device=accelerator.device,
                dtype=torch.bfloat16,
                seed=seed
            )

            inp = get_flux_fill_input(orig_img, mask, mask_cond, vae, accelerator.device, weight_dtype, img_size)
            inp['img'] = rearrange(noise, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            mae, mae_mask = batch['mae'].to(accelerator.device, dtype=weight_dtype), batch['mask_mae'].to(accelerator.device, dtype=weight_dtype)
            clip_fea, flant_fea = get_visual_learned_conditioning(visual_condition_extractor, alignment_clip, alignment_flant, mae, mae_mask)
            clip_fea = clip_fea.to(dtype=weight_dtype)
            flant_fea = flant_fea.to(dtype=weight_dtype)

            bs = orig_img.shape[0]
            clip_fea = none_clip_fea + condition_weight * (clip_fea - none_clip_fea.repeat(bs, 1))
            flant_fea = none_flant_fea + condition_weight * (flant_fea - none_flant_fea.repeat(bs, 1, 1))


            inp['vec'] = clip_fea
            inp['txt'] = flant_fea
            inp['txt_ids'] = torch.zeros(flant_fea.shape[0], flant_fea.shape[1], 3).to(accelerator.device).to(dtype=weight_dtype)
            timesteps = get_schedule(50, inp["img"].shape[1], shift=True)

            denoised_x = denoise(flow_transformer, **inp, timesteps=timesteps, guidance=30.)
            denoised_x = unpack(denoised_x, height, width)
            preds = vae.my_decode(denoised_x, orig_img, mask)
            preds =  (preds.permute(0, 2, 3, 1) + 1) / 2.
            preds = preds.float().cpu().numpy()

            files = batch['file_name']
            for i in range(preds.shape[0]):
                pred = np.clip(preds[i] * 255, 0, 255).astype(np.uint8)
                file = Path(files[i]).name
                Image.fromarray(pred).save(os.path.join(args.result_dir, file))


    torch.cuda.empty_cache()

    return 0






def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of an evaluation.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=5, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument(
        "--full_val",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--decoder_ckpt_path",
        type=str,
        default=None,
        help="decoder path.",
    )
    parser.add_argument(
        "--condition_weight",
        type=float,
        default=None,
        help="condition_weight.",
    )




    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()


    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")



    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def prepare_model(chkpt_dir, arch='mae_vit_base_patch16', random_mask=False, finetune=False, mae_mask_concat=False):
    # build model
    model = misc.get_mae_model(arch, random_mask=random_mask, finetune=finetune, mae_mask_concat=mae_mask_concat)
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    print('Load MAE')
    return model

def get_visual_learned_conditioning(visual_condition_extractor, alignment_clip, alignment_flant, x, mask):
    with torch.no_grad():
        x = visual_condition_extractor.forward_return_feature(x, mask, decoder_layer=6).detach() # 8 * 256 * 512
        if torch.any(torch.isnan(x)):
            print('nan found in mae feature')
            x = torch.zeros_like(x)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        clip_fea = alignment_clip(x)  # 8 * 256 * 768
        flant_fea = alignment_flant(x)
    clip_fea = torch.mean(clip_fea, dim=1)
    return clip_fea, flant_fea


from src.flux.modules.layers import SingleStreamBlockAsuka
def get_single_stram_blk_asuka(dim=768):
    head_num = dim // 64
    alignment = nn.Sequential(
                nn.Linear(512, dim),
                SingleStreamBlockAsuka(dim, head_num),
                SingleStreamBlockAsuka(dim, head_num),
                SingleStreamBlockAsuka(dim, head_num),
                SingleStreamBlockAsuka(dim, head_num),
                nn.LayerNorm(dim),
            )
    return alignment



def get_alignment_clip():
    return get_single_stram_blk_asuka(768)


def get_alignment_flant():
    return get_single_stram_blk_asuka(4096)


def get_flux_fill_input(orig_img, mask, mask_cond, ae, device, weight_dtype, img_size=512):
    img_cond = orig_img * (1 - mask)
    with torch.no_grad():
        img_cond = ae.encode(img_cond)
    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    img_cond = torch.cat((img_cond, mask_cond), dim=-1)

    bs = img_cond.shape[0]
    txt = torch.load("./data/txt.pt").repeat((bs, 1, 1))
    img_ids = torch.load(f"./data/img_ids_{img_size}.pt").repeat((bs, 1, 1))

    ret = dict(img_ids=img_ids, txt=txt, img_cond=img_cond)

    for k, v in ret.items():
        ret[k] = v.to(device, dtype=weight_dtype)

    return ret


from torch import Tensor

def one_step_denoise(model,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: Tensor,
    guidance: float = 4.0,
    # extra img tokens
    img_cond: Tensor | None = None,
):
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    pred = model(
        img=torch.cat((img, img_cond), dim=-1) if img_cond is not None else img,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        y=vec,
        timesteps=timesteps,
        guidance=guidance_vec,
    )

    return pred



def get_resume_path(resume_from_checkpoint, output_dir):
     if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            import sys; sys.exit(-1)
        else:
            return os.path.join(output_dir, path)


def main(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs]
    )


    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
        os.makedirs(args.result_dir, exist_ok=True)
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()





    # Load Model
    name = "flux-dev-fill"
    flow_transformer = load_flow_model(name, device="cpu")
    vae = load_ae(name, device="cpu")
    alignment_clip = get_alignment_clip()
    alignment_flant = get_alignment_flant()


    from ldm.modules.diffusionmodules.asuka_decoder import Decoder
    ddconfig = OmegaConf.load("./configs/condition_decoder.yaml")
    decoder = Decoder(**ddconfig)
    state_dict = torch.load(args.decoder_ckpt_path, map_location=torch.device('cpu'))['state_dict']
    flux_ae = {}
    for k, v in state_dict.items():
        if k.startswith("encoder.") or k.startswith("decoder."):
            flux_ae[k] = v

    vae.decoder = decoder
    vae.load_state_dict(flux_ae, strict=False)



    resume_path = get_resume_path(args.resume_from_checkpoint, args.resume_path)
    accelerator.print(f"Resuming from checkpoint {resume_path}")

    load_model = torch.load(os.path.join(resume_path, "alignment1.pt"))
    alignment_clip.load_state_dict(load_model)
    load_model = torch.load(os.path.join(resume_path, "alignment2.pt"))
    alignment_flant.load_state_dict(load_model)





    logger.info("all models loaded successfully")

    vae.requires_grad_(False)
    flow_transformer.requires_grad_(False)
    visual_condition_extractor = prepare_model('ckpt/mae_300.pth', random_mask=False, finetune=True, mae_mask_concat=False)
    visual_condition_extractor.requires_grad_(False)



    weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    visual_condition_extractor.to(accelerator.device, dtype=weight_dtype)
    flow_transformer.to(accelerator.device, dtype=weight_dtype)
    alignment_clip.to(accelerator.device, dtype=weight_dtype)
    alignment_flant.to(accelerator.device, dtype=weight_dtype)


    # ====== Start Dataset ======
    val_dataset = InpaintingDataset(mode='val', img_size=args.resolution, rank=accelerator.process_index, full_eval=args.full_val)


    val_sampler = DistributedSampler(val_dataset, shuffle=False, rank=accelerator.process_index, num_replicas=accelerator.num_processes)

    # DataLoaders creation:
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        sampler=val_sampler,
        batch_size=args.val_batch_size,
        num_workers=8,
    )

    # ====== End Dataset ======

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

        result_dir = args.result_dir
        result_dir = Path(result_dir).parent
        import json
        with open(result_dir/'args.json', 'w') as f:
            json.dump(vars(args), f, indent=4)



    global_step = 0
    first_epoch = 0




    res = log_validation(
        vae, flow_transformer, val_dataloader, global_step, accelerator, weight_dtype, visual_condition_extractor, alignment_clip, alignment_flant, args.resolution
    )


    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    main(args)