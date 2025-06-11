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
import copy
import os
import random
from pathlib import Path

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from PIL import Image
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_torch_npu_available

from torch.utils.data import DataLoader, DistributedSampler
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.flux.util import load_ae, load_clip, load_flow_model, load_t5
from src.flux.sampling import denoise, get_noise, get_schedule, unpack

from datas.misato_dataset import InpaintingDataset
from einops import rearrange

logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False

def log_validation(model, vae, val_dataloader, step, accelerator, weight_dtype, img_size):
    if accelerator.is_main_process:
        logger.info(f"Validation log in step {step}")
    seed = 42

    clip_fea = torch.load("ckpt/vec.pt").to(accelerator.device, dtype=weight_dtype)
    flant_fea = torch.load("ckpt/txt.pt").to(accelerator.device, dtype=weight_dtype)

    with torch.no_grad():
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

            inp['vec'] = clip_fea.repeat(orig_img.shape[0], 1)
            inp['txt'] = flant_fea.repeat(orig_img.shape[0], 1, 1)
            inp['txt_ids'] = torch.zeros(inp['txt'].shape[0], inp['txt'].shape[1], 3).to(accelerator.device).to(dtype=weight_dtype)
            timesteps = get_schedule(50, inp["img"].shape[1], shift=True)

            denoised_x = denoise(model, **inp, timesteps=timesteps, guidance=30.)
            denoised_x = unpack(denoised_x, height, width)
            preds = vae.decode(denoised_x)
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
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
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


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()




    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def get_flux_fill_input(orig_img, mask, mask_cond, ae, device, weight_dtype, img_size=512):
    img_cond = orig_img * (1 - mask)
    with torch.no_grad():
        img_cond = ae.encode(img_cond)
    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    img_cond = torch.cat((img_cond, mask_cond), dim=-1)

    bs = img_cond.shape[0]
    txt = torch.load("./ckpt/txt.pt").repeat((bs, 1, 1))
    img_ids = torch.load(f"./ckpt/img_ids_{img_size}.pt").repeat((bs, 1, 1))

    ret = dict(img_ids=img_ids, txt=txt, img_cond=img_cond)

    for k, v in ret.items():
        ret[k] = v.to(device, dtype=weight_dtype)

    return ret

def get_models(name: str, device: torch.device, offload: bool):
    t5 = load_t5(device, max_length=128)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu" if offload else device)
    ae = load_ae(name, device="cpu" if offload else device)
    return model, ae, t5, clip


def main(args):

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
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
    model, vae, t5, clip = get_models(
        name,
        device="cpu",
        offload=False,
    )

    del t5
    del clip


    logger.info("all models loaded successfully")

    vae.requires_grad_(False)
    model.requires_grad_(False)


    weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    model.to(accelerator.device, dtype=weight_dtype)

    val_dataset = InpaintingDataset(img_size=args.resolution)
    val_sampler = DistributedSampler(val_dataset, shuffle=False, rank=accelerator.process_index, num_replicas=accelerator.num_processes)
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        sampler=val_sampler,
        batch_size=args.val_batch_size,
        num_workers=4,
    )
    global_step = 0

    res = log_validation(
        model, vae, val_dataloader, global_step, accelerator, weight_dtype, args.resolution
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