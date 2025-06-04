import os
import re
from termcolor import cprint

def find_res_dirs(root_dir):
    result = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if 'imgs' in dirnames:
            result.append(dirpath)
    return result




def gen_res(res_dirs):
    for res_dir in res_dirs:
        with open(os.path.join(res_dir, "ids.txt")) as f:
            ids = f.readlines()[0].split()[1:]
        ids = [round(float(idd), 3) for idd in ids]
        
        with open(os.path.join(res_dir, "psnr_lpips_ssim.txt")) as f:
            psnr_lpips_ssim = f.readlines()[-1].strip().split()
            psnr_lpips_ssim = [float(v) for v in psnr_lpips_ssim]
            
        with open(os.path.join(res_dir, "grad.txt")) as f:
            grad = [round(float(f.readlines()[0].strip().split()[-1]), 3)]
            
        with open(os.path.join(res_dir, "clip.txt")) as f:
            match = re.search(r"CLIP Score:\s*tensor\(([\d\.]+),", f.readlines()[-1])
            clip_score = [round(float(match.group(1)), 3)]
        
        cprint(res_dir+": ", "blue", end='')
        print(f"{' '.join([str(val) for val in ids + psnr_lpips_ssim + grad + clip_score])}")
        

def main():
    root_dirs = ["logs/flux_orig_512", "logs/asuka_512"]
    for root_dir in root_dirs:
        res_dirs = find_res_dirs(root_dir)
        gen_res(res_dirs)
    
if __name__ == "__main__":
    main()