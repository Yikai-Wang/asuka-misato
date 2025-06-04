import argparse
import os
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing as mp


torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

def process_batch(batch_data, device='cuda'):
    """Process a batch of images and calculate metrics"""
    gt_batch, result_batch, mask_batch = batch_data
    mask_batch = mask_batch.to(device)
    mask_batch = (mask_batch >= 0.5).float()

    pred_batch = result_batch * mask_batch + gt_batch * (1 - mask_batch)

    # Calculate PSNR
    mse_batch = torch.mean((pred_batch - gt_batch) ** 2, dim=[1,2,3])
    psnr_batch = -10 * torch.log10(mse_batch + 1e-7)

    # Calculate LPIPS
    pred_norm = pred_batch * 2 - 1  # Scale to [-1, 1]
    gt_norm = gt_batch * 2 - 1
    lpips_batch = loss_fn_alex(pred_norm, gt_norm)

    return psnr_batch, lpips_batch

def calculate_ssim(args):
    """Calculate SSIM for a single image pair"""
    pred_np, origin_np = args
    return structural_similarity(pred_np, origin_np, data_range=1.0)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Config')
    args.add_argument('--generated_dir', default=None, type=str)
    args.add_argument('--gt_dir', default='./data/512', type=str)
    args.add_argument('--resolution', default=None, type=int)

    args = args.parse_args()

    resolution = args.resolution
    print(f"++++ {torch.cuda.is_available()} +++")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn_alex = lpips.LPIPS(net='alex', model_path="ckpt/alex.pth").to(device)
    if torch.cuda.device_count() > 1:
        loss_fn_alex = torch.nn.DataParallel(loss_fn_alex)

    names = os.listdir(os.path.join(args.gt_dir, 'image'))
    names.sort()

    gts = [os.path.join(args.gt_dir, 'image', n) for n in names]
    if int(args.gt_dir.split('/')[-1]) == 1024:
        masks = [os.path.join(args.gt_dir, 'mask', "00"+n) for n in names]
    elif int(args.gt_dir.split('/')[-1]) == 512:
        masks = [os.path.join(args.gt_dir, 'mask', n) for n in names]
    else:
        assert False, "gt_dir is not available"

    if os.path.exists(os.path.join(args.generated_dir,  names[0])):
        results = [os.path.join(args.generated_dir,  n) for n in names]
    else:
        results = [os.path.join(args.generated_dir,  n).replace('jpg', 'png') for n in names]

    batch_size = 32  # Adjust based on your GPU memory

    # Create dataset and dataloader
    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, gt_paths, result_paths, mask_paths):
            self.gt_paths = gt_paths
            self.result_paths = result_paths
            self.mask_paths = mask_paths

        def __len__(self):
            return len(self.gt_paths)

        def __getitem__(self, idx):
            # Load and preprocess GT image
            gt = np.array(Image.open(self.gt_paths[idx]).convert('RGB'))
            mask = np.array(Image.open(self.mask_paths[idx]).convert('L'))
            result = np.array(Image.open(self.result_paths[idx]).convert("RGB"))

            return TF.to_tensor(gt), TF.to_tensor(result), TF.to_tensor(mask)

    dataset = ImageDataset(gts, results, masks)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=batch_size, pin_memory=True)

    psnr_list = []
    lpips_list = []
    ssim_list = []

    # Process batches
    for batch in tqdm(dataloader, desc="Processing batches"):
        gt_batch, result_batch, mask_batch = [b.to(device) for b in batch]

        with torch.no_grad():
            psnr_batch, lpips_batch = process_batch((gt_batch, result_batch, mask_batch), device)

        psnr_list.extend(psnr_batch.cpu().numpy())
        lpips_list.extend(lpips_batch.cpu().numpy())

        # Prepare data for SSIM calculation
        pred_batch = result_batch * (mask_batch >= 0.5).float() + gt_batch * (mask_batch < 0.5).float()
        pred_gray = TF.rgb_to_grayscale(pred_batch)
        gt_gray = TF.rgb_to_grayscale(gt_batch)

        # Calculate SSIM in parallel
        with mp.Pool() as pool:
            ssim_batch = pool.map(
                calculate_ssim,
                [(p[0].cpu().numpy(), g[0].cpu().numpy()) for p, g in zip(pred_gray, gt_gray)]
            )
        ssim_list.extend(ssim_batch)

    # Print results
    print('asuka')
    print('PSNR:', np.mean(psnr_list))
    print('SSIM:', np.mean(ssim_list))
    print('LPIPS:', np.mean(lpips_list))
    results = '{:.3f} {:.3f} {:.3f}'.format(
        np.mean(psnr_list),
        np.mean(ssim_list),
        np.mean(lpips_list)
    )
    print(results)
