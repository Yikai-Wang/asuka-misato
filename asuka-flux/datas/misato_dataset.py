import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from diffusers.image_processor import VaeImageProcessor
from einops import rearrange

def prepare_data(img, mask, imagenet_mean=np.array([0.485, 0.456, 0.406]), imagenet_std=np.array([0.229, 0.224, 0.225])):
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
    return mask.float(), unmasked_img.float()

class InpaintingDataset(Dataset):
    def __init__(self, img_size=512):
        self.image_processor = VaeImageProcessor(vae_scale_factor=16, do_resize=True, do_convert_rgb=True, do_normalize=True)
        self.img_size = img_size
        dir_path = os.path.join('./data', str(img_size))
        names = [f for f in os.listdir(os.path.join(dir_path, 'image')) if f.endswith('.png')]
        names.sort()

        self.images = [os.path.join(dir_path, 'image', f) for f in names]
        if img_size == 512:
            self.masks = [os.path.join(dir_path, 'mask', f) for f in names]
        else:
            self.masks = [os.path.join(dir_path, 'mask', "00"+f) for f in names]

        self.captions = ["background"] * len(self.images)
        assert len(self.images) == len(self.masks)
        self._length = len(self.images)


    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = np.array(Image.open(self.images[idx]).convert('RGB'))
        img_path = self.images[idx]
        mask = np.array(Image.open(self.masks[idx]).convert('L').resize((self.img_size, self.img_size), resample=Image.Resampling.NEAREST)) / 255.
        mask = mask.astype(np.float32)

        mask_mae, unmasked_img_mae = prepare_data(img, mask)

        mask = mask[None]
        mask_cond = mask.copy()
        mask_cond = rearrange(
            mask_cond,
            "b (h ph) (w pw) -> b (ph pw) h w",
            ph=8,
            pw=8,
        )
        mask_cond = rearrange(mask_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)[0]

        orig_img = torch.from_numpy(img).float() / 127.5 - 1.0
        orig_img = rearrange(orig_img, "h w c -> c h w")

        meta = {
            "mae": unmasked_img_mae,
            "mask_mae": mask_mae,
            "orig_img": orig_img,
            "mask_cond": mask_cond,
            "mask": mask,
            "file_name": img_path
        }
        return meta