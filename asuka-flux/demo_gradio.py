import gradio as gr
import numpy as np
import cv2
from PIL import Image


import random
import datetime
import time
import os
import time

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from transformers import pipeline

from src.flux.sampling import denoise, get_noise, get_schedule, prepare_fill_empty_prompt, unpack
from src.flux.util import embed_watermark, load_ae, load_flow_model
from MAE.util import misc
import torch.nn as nn
from cprint import ccprint
from einops import rearrange, repeat
import cv2

from ldm.modules.diffusionmodules.asuka_decoder import Decoder
from omegaconf import OmegaConf

NSFW_THRESHOLD = 0.85


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


def add_border_and_mask(image, zoom_all=1.0, zoom_left=0, zoom_right=0, zoom_up=0, zoom_down=0, overlap=0):
    """Adds a black border around the image with individual side control and mask overlap"""
    orig_width, orig_height = image.size

    # Calculate padding for each side (in pixels)
    left_pad = int(orig_width * zoom_left)
    right_pad = int(orig_width * zoom_right)
    top_pad = int(orig_height * zoom_up)
    bottom_pad = int(orig_height * zoom_down)

    # Calculate overlap in pixels
    overlap_left = int(orig_width * overlap)
    overlap_right = int(orig_width * overlap)
    overlap_top = int(orig_height * overlap)
    overlap_bottom = int(orig_height * overlap)

    # If using the all-sides zoom, add it to each side
    if zoom_all > 1.0:
        extra_each_side = (zoom_all - 1.0) / 2
        left_pad += int(orig_width * extra_each_side)
        right_pad += int(orig_width * extra_each_side)
        top_pad += int(orig_height * extra_each_side)
        bottom_pad += int(orig_height * extra_each_side)

    # Calculate new dimensions (ensure they're multiples of 32)
    new_width = 32 * round((orig_width + left_pad + right_pad) / 32)
    new_height = 32 * round((orig_height + top_pad + bottom_pad) / 32)

    # Create new image with black border
    bordered_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
    # Paste original image in position
    paste_x = left_pad
    paste_y = top_pad
    bordered_image.paste(image, (paste_x, paste_y))

    # Create mask (white where the border is, black where the original image was)
    mask = Image.new("L", (new_width, new_height), 255)  # White background
    # Paste black rectangle with overlap adjustment
    mask.paste(
        0,
        (
            paste_x + overlap_left,  # Left edge moves right
            paste_y + overlap_top,  # Top edge moves down
            paste_x + orig_width - overlap_right,  # Right edge moves left
            paste_y + orig_height - overlap_bottom,  # Bottom edge moves up
        ),
    )

    return bordered_image, mask


def get_models(name: str, device: torch.device, offload: bool):
    model = load_flow_model(name, device="cpu" if offload else device)
    ae = load_ae(name, device="cpu" if offload else device)
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)
    return model, ae, nsfw_classifier

def get_models_asuka(name: str, device: torch.device):
    visual_condition_extractor = prepare_model('ckpt/mae_300.pth', random_mask=False, finetune=True, mae_mask_concat=False)
    visual_condition_extractor.requires_grad_(False)
    alignment_clip, alignment_flant = get_alignment_clip(), get_alignment_flant()

    resume_path = "./ckpt"
    load_model = torch.load(os.path.join(resume_path, "asuka_alignment_clip.pt"), map_location=torch.device('cpu'))
    alignment_clip.load_state_dict(load_model)
    load_model = torch.load(os.path.join(resume_path, "asuka_alignment_t5.pt"), map_location=torch.device('cpu'))
    alignment_flant.load_state_dict(load_model)

    visual_condition_extractor.to(device, dtype=torch.bfloat16)
    alignment_clip.to(device, dtype=torch.bfloat16)
    alignment_flant.to(device, dtype=torch.bfloat16)

    ae = load_ae(name, device=device)

    ddconfig = OmegaConf.load("./configs/condition_decoder.yaml")
    decoder = Decoder(**ddconfig)
    ae.decoder = decoder

    state_dict = torch.load("ckpt/asuka_decoder.ckpt")
    asuka_decoder = {}
    for k, v in state_dict['state_dict'].items():
        if k.startswith("decoder."):
            asuka_decoder[k] = v

    ae.load_state_dict(asuka_decoder, strict=False)

    return visual_condition_extractor, alignment_clip, alignment_flant, ae


def resize(img: Image.Image, min_mp: float = 0.5, max_mp: float = 2.0) -> Image.Image:
    width, height = img.size
    mp = (width * height) / 1_000_000  # Current megapixels

    if min_mp <= mp <= max_mp:
        # Even if MP is in range, ensure dimensions are multiples of 32
        new_width = int(32 * round(width / 32))
        new_height = int(32 * round(height / 32))
        if new_width != width or new_height != height:
            return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return img

    # Calculate scaling factor
    if mp < min_mp:
        scale = (min_mp / mp) ** 0.5
    else:  # mp > max_mp
        scale = (max_mp / mp) ** 0.5

    new_width = int(32 * round(width * scale / 32))
    new_height = int(32 * round(height * scale / 32))

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)




def downscale_image(img: Image.Image, scale_factor: float) -> Image.Image:
    """Downscale image by a given factor while maintaining 32-pixel multiple dimensions"""
    if scale_factor >= 1.0:
        return img

    width, height = img.size
    new_width = int(32 * round(width * scale_factor / 32))
    new_height = int(32 * round(height * scale_factor / 32))

    # Ensure minimum dimensions
    new_width = max(64, new_width)  # minimum 64 pixels
    new_height = max(64, new_height)  # minimum 64 pixels

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)



@torch.no_grad
def get_flux_fill_res(tmp_img, tmp_mask, prompt, height, width, num_steps, guidance, model, ae, torch_device, seed, offload):
    x = get_noise(
        1,
        height,
        width,
        device=torch_device,
        dtype=torch.bfloat16,
        seed=seed,
    )

    if offload:
        ae = ae.to(torch_device)

    inp = prepare_fill_empty_prompt(
        x,
        prompt=prompt,
        ae=ae,
        img_cond_path=tmp_img,
        mask_path=tmp_mask,
    )

    timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=True)

    if offload:
        ae = ae.cpu()
        torch.cuda.empty_cache()
        model = model.to(torch_device)

    x = denoise(model, **inp, timesteps=timesteps, guidance=guidance)

    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.decoder.to(x.device)

    x = unpack(x.float(), height, width)
    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
        x = ae.decode(x)

    return x


def get_flux_fill_input(orig_img, mask, mask_cond, vae, torch_device, img_size=512):
    img_cond = orig_img * (1 - mask)
    with torch.no_grad():
        img_cond = vae.encode(img_cond)
    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    img_cond = torch.cat((img_cond, mask_cond), dim=-1)

    bs = img_cond.shape[0]

    bs, c, h, w = orig_img.shape

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)


    return {
        'img_ids': img_ids.to(torch_device),
        'img_cond': img_cond
    }


def get_visual_learned_conditioning(visual_condition_extractor, alignment_clip, alignment_flant, x, mask):
    with torch.no_grad():
        x = visual_condition_extractor.forward_return_feature(x, mask, decoder_layer=6).detach()
        if torch.any(torch.isnan(x)):
            print('nan found in mae feature')
            x = torch.zeros_like(x)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        clip_fea = alignment_clip(x)  # 8 * 256 * 768
        flant_fea = alignment_flant(x)
    clip_fea = torch.mean(clip_fea, dim=1)
    return clip_fea, flant_fea


def prepare_model(chkpt_dir, arch='mae_vit_base_patch16', random_mask=False, finetune=False, mae_mask_concat=False):
    # build model
    model = misc.get_mae_model(arch, random_mask=random_mask, finetune=finetune, mae_mask_concat=mae_mask_concat)
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    print('Load MAE')
    return model

def prepare_data(img, mask, null_mask=False, imagenet_mean=np.array([0.485, 0.456, 0.406]), imagenet_std=np.array([0.229, 0.224, 0.225])):
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = np.array(img) / 255.
    img = img - imagenet_mean
    img = img / imagenet_std

    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask = np.array(mask)

    x = torch.tensor(img)
    mask = torch.tensor(mask)

    # make it a batch-like
    x = torch.einsum('hwc->chw', x)
    mask = mask.reshape(1, mask.shape[0], mask.shape[1])
    unmasked_img = x * (mask<0.5)
    return x.float(), mask.float(), unmasked_img.float()




@torch.no_grad
def get_flux_asuka_res(tmp_img, tmp_mask, height, width, num_steps, guidance, torch_device, seed, offload, model, ae, visual_condition_extractor, alignment_clip, alignment_flant):
    x = get_noise(
        1,
        height,
        width,
        device=torch_device,
        dtype=torch.bfloat16,
        seed=seed,
    )

    if offload:
        ae = ae.to(torch_device)

    ae = ae.float()
    inp = prepare_fill_empty_prompt(
        x,
        prompt="",
        ae=ae,
        img_cond_path=tmp_img,
        mask_path=tmp_mask
    )



    weight_dtype = torch.bfloat16


    # mae input
    img_path = tmp_img
    img = np.array(Image.open(img_path).convert('RGB'))
    mask_path = tmp_mask
    mask = np.array(Image.open(mask_path).convert('L')) / 255.
    mask = mask.astype(np.float32)
    _, mask_mae, unmasked_img_mae = prepare_data(img, mask)
    unmasked_img_mae, mask_mae = unmasked_img_mae.to(torch_device, dtype=weight_dtype), mask_mae.to(torch_device, dtype=weight_dtype)
    unmasked_img_mae = unmasked_img_mae.unsqueeze(0)
    mask_mae = mask_mae.unsqueeze(0)

    visual_condition_extractor = visual_condition_extractor.to(dtype=weight_dtype)
    alignment_clip = alignment_clip.to(dtype=weight_dtype)
    alignment_flant = alignment_flant.to(dtype=weight_dtype)



    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        clip_fea, flant_fea = get_visual_learned_conditioning(visual_condition_extractor, alignment_clip, alignment_flant, unmasked_img_mae, mask_mae)
    clip_fea = clip_fea.to(dtype=weight_dtype)
    flant_fea = flant_fea.to(dtype=weight_dtype)

    none_clip_fea = torch.load("./ckpt/vec.pt").to(torch_device, dtype=weight_dtype)
    none_flant_fea = torch.load("./ckpt/txt.pt").to(torch_device, dtype=weight_dtype)

    condition_weight = 0.5
    clip_fea = none_clip_fea + condition_weight * (clip_fea - none_clip_fea.repeat(1, 1))
    flant_fea = none_flant_fea + condition_weight * (flant_fea - none_flant_fea.repeat(1, 1, 1))


    inp['vec'] = clip_fea
    inp['txt'] = flant_fea
    inp['txt_ids'] = torch.zeros(flant_fea.shape[0], flant_fea.shape[1], 3).to(torch_device).to(dtype=weight_dtype)



    timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=True)




    if offload:
        ae = ae.cpu()
        torch.cuda.empty_cache()
        model = model.to(torch_device)

    x = denoise(model, **inp, timesteps=timesteps, guidance=guidance)

    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.decoder.to(x.device)

    x = unpack(x.float(), height, width)

    orig_img = torch.from_numpy(img).float() / 127.5 - 1.0
    orig_img = rearrange(orig_img, "h w c -> 1 c h w")

    mask = torch.from_numpy(mask)
    mask = rearrange(mask, "h w -> 1 1 h w")

    ae.to(torch_device, dtype=torch.bfloat16)
    orig_img = orig_img.to(torch_device, dtype=torch.bfloat16)
    mask = mask.to(torch_device, dtype=torch.bfloat16)
    x = x.to(torch_device, dtype=torch.bfloat16)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        x = ae.my_decode(x, orig_img, mask)

    x = x.to(torch.float32)
    return x


def main(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    offload: bool = False,
    output_dir: str = "output",
):
    torch_device = torch.device(device)

    # Model selection and loading
    name = "flux-dev-fill"

    model, ae, nsfw_classifier = get_models(
        name,
        device=torch_device,
        offload=offload,
    )

    visual_condition_extractor, alignment_clip, alignment_flant, asuka_ae = get_models_asuka(name, torch_device)

    ccprint('All models loaded successfully!', 'green')


    def get_res(input_image):
        # resize image with validation
        image = input_image['image']
        mask = input_image['mask']
        image = resize(image)
        mask = resize(mask)

        width, height = image.size
        assert (width, height) == mask.size, "image and mask must have the same size"


        output_dir = "./tmp"
        os.makedirs(output_dir, exist_ok=True)
        current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        tag = f"{current_time}_{random.randint(10000, 99999)}"
        tmp_img = os.path.join(output_dir, f"{tag}_image.png")
        tmp_mask = os.path.join(output_dir, f"{tag}_mask.png")

        image.save(tmp_img)
        mask.save(tmp_mask)

        seed = 42

        prompt = ""
        num_steps = 50
        guidance = 30.0
        print(f"Generating with seed {seed}:\n{prompt}")


        # Inpainting
        t0 = time.perf_counter()
        x = get_flux_fill_res(tmp_img, tmp_mask, prompt, height, width, num_steps, guidance, model, ae, torch_device, seed, offload)
        t1 = time.perf_counter()

        print(f"Done in {t1 - t0:.1f}s")


        t0 = time.perf_counter()
        # import ipdb; ipdb.set_trace()
        asuka_result = get_flux_asuka_res(tmp_img, tmp_mask, height, width, num_steps, guidance, torch_device, seed, offload, model, asuka_ae, visual_condition_extractor, alignment_clip, alignment_flant)
        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s")


        # Process and display result
        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())


        asuka_result = asuka_result.clamp(-1, 1)
        asuka_result = embed_watermark(asuka_result.float())
        asuka_result = rearrange(asuka_result[0], "c h w -> h w c")
        asuka_img = Image.fromarray((127.5 * (asuka_result + 1.0)).cpu().byte().numpy())

        return img, asuka_img

    with gr.Blocks() as demo:
        with gr.Column():
            input_image = gr.Image(source='upload', type="pil", tool="sketch", mask_opacity=0.7, brush_color='#FFFFFF')
            run_button = gr.Button(label="Run")

            with gr.Row():
                flux_res = gr.Image(label="Flux")
                asuka_res = gr.Image(label="Asuka")

            run_button.click(fn=get_res, inputs=input_image, outputs=[flux_res, asuka_res])


        demo.launch()



if __name__ == "__main__":
    main()