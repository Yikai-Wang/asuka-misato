<p align="center">
  <h1 align="center">Towards Enhanced Image Inpainting:<br>
Mitigating Unwanted Object Insertion and Preserving Color Consistency</h1>
<center>Yikai Wang*, Chenjie Cao*, Junqiu Yu*, Ke Fan, Xiangyang Xue, Yanwei Fu†.<br>
Fudan University<br>
<b>CVPR 2025 <font color="#ed7748">(Highlight)</font></b>
</center>
  <p align="center">
    <a href="https://arxiv.org/abs/2312.04831"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2312.04831-b31b1b.svg"></a>
    <a href="https://yikai-wang.github.io/asuka/"><img alt='page' src="https://img.shields.io/badge/Project-Website-orange"></a>

  </p>
  <br>
</p>

## Overview
This repo contains the proposed ASUKA algorithm and the evaluation dataset MISATO in our paper "[Towards Enhanced Image Inpainting: Mitigating Unwanted Object Insertion and Preserving Color Consistency](https://arxiv.org/abs/2312.04831)".

> ASUKA solves two issues existed in current diffusion and rectified flow inpainting models:
<b>Unwanted object insertion</b>, where randomly elements that are not aligned with the unmasked region are generated;
<b>Color-inconsistenc</b>y, the color shift of the generated masked region, causing smear-like traces.
ASUKA proposes a post-training procedure for these models, significantly mitigates object hallucination and improves color consistency of inpainted results.

> While unwanted object insertion is a specific problem in general image inpainting, <b>color inconsistency affects all text-to-image editing models</b>. Our proposed decoder can consistently improve performance by addressing this issue.

We released ASUKA for [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev), denoted as ASUKA(FLUX.1-Fill).
We also release the MISATO dataset at resolutions 512 and 1024.
We are actively working to improve both our model and evaluation dataset.
If you encounter failure cases with ASUKA (FLUX.1-Fill) or have challenging examples in image inpainting, we would love to hear from you. Please email them to yi-kai.wang@outlook.com.
We truly appreciate your contributions!

## Modifications
Modifications to FLUX:
- The text conditional input of CLIP and T5 is replaced by the MAE condition to mitigate object hallucination.
- The decoder is replaced by our conditional decoder to enhance color consistency.

Modifications to CLIP score:

The original CLIP score in our paper was calculated using an internal model that supports mask-region CLIP similarity. As we cannot release that model, we report the original global CLIP similairty in this repo.

For the alignment module in ASUKA (FLUX.1-Fill), we use a CFG scale of 0.5. In our original paper, we did not apply CFG to ASUKA-SD or ASUKA-FLUX.

## Known Issues

- The decoder doesn't handle resized masks very well, especially when the mask is scaled down from a larger size. This can lead to jagged or unrealistic edges. For better results, it's recommended to first resize the image to the target size, and then apply the mask.
- Both the alignment and decoder may not work properly if the image resolution during testing is very different from the resolution used during training.

## Usage

### Installation

```
conda env create -f environment.yml
```
### Chekpoints

You can download the models from [Huggingface](https://huggingface.co/yikaiwang/ASUKA-FLUX.1-Fill). We assume all checkpoints are saved in the ckpt folder.

### Inference

See commands.sh.

### Gradio demo
```
python demo_gradio.py
```


## Results
We present the results of ASUKA (FLUX.1-Fill) and compare them with the standard FLUX.1-Fill-dev. Our alignment module and decoder were trained on 512 and 256 images, respectively. We plan to explore whether training at higher resolution will further improve our model.

| Model | FID($\downarrow$) | U-IDS($\uparrow$) | P-IDS($\uparrow$) | PSNR($\uparrow$) | SSIM($\uparrow$) | LPIPS($\downarrow$) |  Grad($\downarrow$) | CLIP($\uparrow$) |
| --- | --- | --- | --- | --- | --- | --- |--- |--- |
<i>MISATO@512</i>
| FLUX.1-Fill-dev | 12.170 | 0.353 | 0.194 | 22.194 | 0.798 | 0.156 | 69.428 |  0.936|
| ASUKA (FLUX.1-Fill) | <b>11.011</b> | <b>0.405</b> | <b>0.260</b> | <b>22.329</b> | <b>0.805</b> | <b>0.141</b> | <b>52.925</b> | <b>0.943</b>|
<i>MISATO@1K</i>
| FLUX.1-Fill-dev | 15.106 | 0.245 | 0.128 | 22.583 | 0.823 | 0.165 | 48.871 | <b>0.944</b> |
| ASUKA (FLUX.1-Fill) | <b>13.503</b> | <b>0.311</b> | <b>0.185</b> | <b>22.602</b> | <b>0.826</b> | <b>0.149</b> | <b>34.390</b> | 0.942 |

Global CLIP scores of Table 1 in our paper.
| Model |Co-Mod |MAT |LaMa |MAE-FAR |SD-Repaint |SD |SD-text |SD-token |SD-IP |SD-T2I |SD-CAEv2 |SD-LaMa |ASUKA-SD|
| --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |
|CLIP($\uparrow$) on MISATO | 0.878 | 0.882| 0.912|0.896|0.875|0.923|0.924|0.925|0.926|0.917|0.866|0.926|<b>0.931</b>|
| CLIP($\uparrow$) on Places 2| 0.917 |- | 0.927 | -| 0.907 | 0.928|0.929|0.929|0.917|0.912|0.865|0.917|<b>0.931</b>|

The generation results of MAT and MAE-FAR on Places2 were affected by a hardware issue. We plan to re-run the experiments and update the results later.

## Acknowledgements
We're grateful for the outstanding open-sourced works by many researchers, especially:

[SD 1.5 (unofficial link)](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)

[FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)

[MAE](https://github.com/facebookresearch/mae)

[MAE-FAR](https://github.com/ewrfcas/MAE-FAR)

[Asymmetric VQGAN](https://github.com/buxiangzhiren/Asymmetric_VQGAN)

[CLIP](https://github.com/openai/CLIP)

## BibTeX
If you find our repo helpful, please consider cite our paper :)
```bibtex
@inproceedings{wang2025towards,
  title={Towards Enhanced Image Inpainting: Mitigating Unwanted Object Insertion and Preserving Color Consistency.},
  author={Wang, Yikai and Cao, Chenjie and Yu, Junqiu and Fan, Ke and Xue, Xiangyang and Fu, Yanwei},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  year={2025}
}
```

