import argparse
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import clip

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


device = "cuda:0"
clip_model = "./ckpt/ViT-L-14.pt"
model, preprocess = clip.load(clip_model, device=device)


class CustomImageDataset(Dataset):
    def __init__(self, gts, results, masks, transform=None, resolution=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """

        self.gts = gts
        self.results = results
        self.masks = masks
        self.resolution = resolution
    def __len__(self):
        return len(self.gts)

    def __getitem__(self, i):
        # Load the image
        gt = Image.open(self.gts[i]).convert('RGB')
        result = Image.open(self.results[i]).convert("RGB")
        mask = Image.open(self.masks[i]).convert("L")

        gt, result, mask = np.array(gt), np.array(result), np.array(mask) // 255

        result = Image.fromarray(result * mask[:, :, None])
        masked_image = Image.fromarray(gt * mask[:, :, None])

        return preprocess(result), preprocess(masked_image)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Config')
    args.add_argument('--generated_dir', type=str, help='config file')
    args.add_argument('--gt_dir', default='./data/512', type=str, help='experiment name')
    args.add_argument('--resolution', default=512, type=int, help='config file')
    args = args.parse_args()

    resolution = args.resolution

    names = os.listdir(os.path.join(args.gt_dir, 'image'))
    names.sort()

    gts = [os.path.join(args.gt_dir, 'image', n) for n in names]
    if int(args.gt_dir.split('/')[-1]) == 1024:
        masks = [os.path.join(args.gt_dir, 'mask', "00"+n).replace("jpg", "png") for n in names]
    elif int(args.gt_dir.split('/')[-1]) == 512:
        masks = [os.path.join(args.gt_dir, 'mask', n) for n in names]
    else:
        assert False, "gt_dir is not available"

    if os.path.exists(os.path.join(args.generated_dir,  names[0])):
        results = [os.path.join(args.generated_dir,  n) for n in names]
    else:
        results = [os.path.join(args.generated_dir,  n).replace('jpg', 'png') for n in names]

    bs = 128
    places_dataset = CustomImageDataset(gts, results, masks, resolution=resolution)
    places_loader = DataLoader(places_dataset, batch_size=bs, shuffle=False, num_workers=64)


    sim_all = []

    for batch in tqdm(places_loader):
        pred_gts = torch.cat(batch, dim=0)
        with torch.no_grad():
            pred_gts_feats = model.encode_image(pred_gts.cuda())
            pred_gts_feats = pred_gts_feats / pred_gts_feats.norm(dim=-1, keepdim=True)
        pred_feats = pred_gts_feats[:bs]
        pred_feats = torch.mean(pred_feats, axis=1)
        gts_feats = pred_gts_feats[bs:]
        gts_feats = torch.mean(gts_feats, axis=1)

        sims = (pred_feats @ gts_feats.T).diagonal()
        sim_all.append(sims)

    print("CLIP Score: ", torch.mean(torch.cat(sim_all)))
