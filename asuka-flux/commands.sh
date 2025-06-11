# Test Orig Flux 512
RES_DIR=logs/flux_orig_512/
GEN_DIR=logs/flux_orig_512/imgs
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch test_flux.py \
    --result_dir="logs/flux_orig_512/imgs" \
    --mixed_precision="bf16" \
    --resolution=512 \
    --val_batch_size=4 \
    --full_val
CUDA_VISIBLE_DEVICES=1 python eval/cal_psnr_lpips_ssim.py --generated_dir $GEN_DIR > ${RES_DIR}/psnr_lpips_ssim.txt
CUDA_VISIBLE_DEVICES=1 python eval/cal_ids.py --generated_dir $GEN_DIR --resolution 512 > ${RES_DIR}/ids.txt
CUDA_VISIBLE_DEVICES=1 python eval/cal_grad.py --generated_dir $GEN_DIR --resolution 512 > ${RES_DIR}/grad.txt
CUDA_VISIBLE_DEVICES=1 python eval/cal_clip_score.py --generated_dir $GEN_DIR > ${RES_DIR}/clip.txt


# Test ASUKA FLux 512
RES_DIR=logs/asuka_512/
GEN_DIR=logs/asuka_512/imgs
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch test_asuka_flux.py \
    --decoder_ckpt_path=ckpt/asuka_decoder.ckpt \
    --result_dir=$GEN_DIR \
    --mixed_precision="bf16" \
    --resolution=512 \
    --val_batch_size=4 \
    --full_val
CUDA_VISIBLE_DEVICES=1 python eval/cal_psnr_lpips_ssim.py --generated_dir $GEN_DIR > ${RES_DIR}/psnr_lpips_ssim.txt
CUDA_VISIBLE_DEVICES=1 python eval/cal_ids.py --generated_dir $GEN_DIR --resolution 512 > ${RES_DIR}/ids.txt
CUDA_VISIBLE_DEVICES=1 python eval/cal_grad.py --generated_dir $GEN_DIR --resolution 512 > ${RES_DIR}/grad.txt
CUDA_VISIBLE_DEVICES=1 python eval/cal_clip_score.py --generated_dir $GEN_DIR > ${RES_DIR}/clip.txt

# Test Orig Flux 1024
RES_DIR=logs/flux_orig_1024/
GEN_DIR=logs/flux_orig_1024/imgs
CUDA_VISIBLE_DEVICES=0 accelerate launch test_flux.py \
    --result_dir="logs/flux_orig_1024/imgs" \
    --mixed_precision="bf16" \
    --resolution=1024 \
    --val_batch_size=4 \
    --full_val
CUDA_VISIBLE_DEVICES=0 python eval/cal_psnr_lpips_ssim.py --generated_dir $GEN_DIR  --gt_dir ./data/1024 --resolution 1024 > ${RES_DIR}/psnr_lpips_ssim.txt
CUDA_VISIBLE_DEVICES=0 python eval/cal_ids.py --generated_dir $GEN_DIR --gt_dir ./data/1024 --resolution 1024 > ${RES_DIR}/ids.txt
CUDA_VISIBLE_DEVICES=0 python eval/cal_grad.py --generated_dir $GEN_DIR --gt_dir ./data/1024 --resolution 1024 > ${RES_DIR}/grad.txt
CUDA_VISIBLE_DEVICES=0 python eval/cal_clip_score.py --generated_dir $GEN_DIR --gt_dir ./data/1024 --resolution 1024 > ${RES_DIR}/clip.txt

# Test Asuka Flux 1024
RES_DIR=logs/ours_1024/
GEN_DIR=logs/ours_1024/imgs
CUDA_VISIBLE_DEVICES=0 accelerate launch test_asuka_flux.py \
    --decoder_ckpt_path=ckpt/asuka_decoder.ckpt \
    --result_dir=$GEN_DIR \
    --mixed_precision="bf16" \
    --resolution=1024 \
    --val_batch_size=4 \
    --full_val
CUDA_VISIBLE_DEVICES=0 python eval/cal_psnr_lpips_ssim.py --generated_dir $GEN_DIR  --gt_dir ./data/1024 --resolution 1024 > ${RES_DIR}/psnr_lpips_ssim.txt
CUDA_VISIBLE_DEVICES=0 python eval/cal_ids.py --generated_dir $GEN_DIR --gt_dir ./data/1024 --resolution 1024 > ${RES_DIR}/ids.txt
CUDA_VISIBLE_DEVICES=0 python eval/cal_grad.py --generated_dir $GEN_DIR --gt_dir ./data/1024 --resolution 1024 > ${RES_DIR}/grad.txt
CUDA_VISIBLE_DEVICES=0 python eval/cal_clip_score.py --generated_dir $GEN_DIR --gt_dir ./data/1024 --resolution 1024 > ${RES_DIR}/clip.txt



