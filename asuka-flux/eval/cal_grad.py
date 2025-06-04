import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


def calculate_gradient_gt_mean(image, gt, mask):
    # Convert image to grayscale if it is not
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    if len(gt.shape) == 3:
        gray_image_gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    else:
        gray_image_gt = gt


    # Detect edges in the mask to focus on the boundary
    edges = cv2.Canny(mask, 100, 200)
    # Compute the gradients using Sobel operator
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    grad_x_gt = cv2.Sobel(gray_image_gt, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_gt = cv2.Sobel(gray_image_gt, cv2.CV_64F, 0, 1, ksize=3)


    grad_x = grad_x - grad_x_gt
    grad_y = grad_y - grad_y_gt


    # Compute the gradient magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Focus on the gradients at the mask edges
    edge_gradients = grad_magnitude[edges > 0]

    # Compute the mean of the gradients along the mask edges
    gradient_mean = edge_gradients.mean() if edge_gradients.size > 0 else 0

    return gradient_mean

def get_result_image(pred_path, mask_path, gt_path, res):
    pred = cv2.resize(cv2.imread(pred_path), (res, res), interpolation=cv2.INTER_LANCZOS4)
    mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (res, res), interpolation=cv2.INTER_NEAREST)
    gt = cv2.resize(cv2.imread(gt_path), (res, res), interpolation=cv2.INTER_LANCZOS4)

    _, mask_binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    unmask_region = cv2.bitwise_and(gt, gt, mask=cv2.bitwise_not(mask_binary))

    result = cv2.bitwise_and(pred, pred, mask=mask_binary)
    result += unmask_region
    return result

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Config')
    args.add_argument('--generated_dir', type=str)
    args.add_argument('--gt_dir', default='./data/512', type=str)
    args.add_argument('--resolution', default=None, type=int)
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

    import concurrent.futures
    def process_image(idx, gts, results, masks, resolution):
        gt = np.array(Image.open(gts[idx]).convert("RGB"))
        mask = np.array(Image.open(masks[idx]).convert("L"))
        result = np.array(Image.open(results[idx]).convert("RGB"))

        result = result * (mask[:, :, None] / 255.)  + gt * (1 - mask[:, :, None] / 255.)
        grad_mask = mask

        grad_means = []
        for channel in range(3):
            grad_mean = calculate_gradient_gt_mean(result[:, :, channel], gt[:, :, channel], grad_mask)
            grad_means.append(grad_mean)
        return grad_means

    def parallel_processing():
        grads = [[], [], []]
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            rets = list(tqdm(executor.map(process_image, range(len(results)), [gts]*len(results), [results]*len(results), [masks]*len(results), [resolution]*len(results)), total=len(results), ncols=0))

        for grad_means in rets:
            for i in range(3):
                grads[i].append(grad_means[i])

        grads = [np.mean(grad) for grad in grads]
        print(grads, np.mean(grads))

    parallel_processing()
