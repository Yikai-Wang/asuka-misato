# Large Scale Image Completion via Co-Modulated Generative Adversarial Networks
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://openreview.net/pdf?id=sSjqmfsk95O

"""Paired/Unpaired Inception Discriminative Score (P-IDS/U-IDS)."""
from torchvision.models import inception_v3, Inception_V3_Weights
import torch
import torch.nn as nn
import scipy
import sklearn
import numpy as np
import argparse
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
prepross = Inception_V3_Weights.DEFAULT.transforms()
model.fc = nn.Identity()
model.eval()

if torch.cuda.is_available():
    model = model.cuda()

class dataset(Dataset):

    def __init__(self, images, gts=None, masks=None, reso=None):
        super().__init__()
        self.images = images
        if gts is not None:
            self.gts = gts
            self.masks = masks
        else:
            self.gts = None
        self.prepross = prepross
        self.reso = reso

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        if self.gts is not None:
            gt = Image.open(self.gts[index]).convert("RGB")
            mask = Image.open(self.masks[index]).convert('L')

            mask = np.array(mask)[:,:,None] / 255
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            img = np.array(img) * (mask) + np.array(gt) * (1 - mask)
            img = Image.fromarray(img.astype(np.uint8))
        return self.prepross(img)

def cal(real_activations, fake_activations, dim=2048):
    # Calculate FID conviniently.
    mu_real = np.mean(real_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)
    mu_fake = np.mean(fake_activations, axis=0)
    sigma_fake = np.cov(fake_activations, rowvar=False)
    m = np.square(mu_fake - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False)
    dist = m + np.trace(sigma_fake + sigma_real - 2*s)
    fid = np.real(dist)

    svm = sklearn.svm.LinearSVC(dual=False)
    svm_inputs = np.concatenate([real_activations, fake_activations])
    if dim != 2048:
        svm_inputs = PCA(dim).fit_transform(svm_inputs)
    svm_targets = np.array([1] * real_activations.shape[0] + [0] * fake_activations.shape[0])
    svm.fit(svm_inputs, svm_targets)
    u = 1 - svm.score(svm_inputs, svm_targets)
    real_outputs = svm.decision_function(svm_inputs[:len(real_activations)])
    fake_outputs = svm.decision_function(svm_inputs[len(real_activations):])
    p = np.mean(fake_outputs > real_outputs)
    return fid, u, p

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Config')
    args.add_argument('--generated_dir', default=None, type=str, help='config file')
    args.add_argument('--gt_dir', default='./data/512', type=str, help='experiment name')
    args.add_argument('--resolution', default=None, type=int, help='config file')

    args = args.parse_args()
    resolution = args.resolution

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

    gt_set = dataset(gts, reso=resolution)
    ge_set = dataset(results, gts, masks, reso=resolution)
    gt_loader = DataLoader(gt_set, batch_size=128, num_workers=16, shuffle=False)
    ge_loader = DataLoader(ge_set, batch_size=128, num_workers=16, shuffle=False)

    real_activations_full = []
    for data in gt_loader:
        outputs = model(data.cuda()).detach().cpu().numpy()
        real_activations_full.append(outputs)
    real_activations_full = np.concatenate(real_activations_full)

    fake_activations_full = []
    for data in tqdm(ge_loader):
        outputs = model(data.cuda()).detach().cpu().numpy()
        fake_activations_full.append(outputs)
    fake_activations_full = np.concatenate(fake_activations_full)

    fid_, u_, p_ = cal(real_activations_full, fake_activations_full, resolution)
    print('{}: {} {} {} '.format('full',fid_,u_,p_))
