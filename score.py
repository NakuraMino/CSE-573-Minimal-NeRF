"""Calculate score metrics for NeRF Models.

Usage:
    python score.py -c CKPT_PATH -r 4096 -b BASE_DIR
"""
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import nerf_model
import nerf_helpers 
from pathlib import Path
from dataloader import SyntheticDataset
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculate_scores(ckpt, base_dir, rays):
    model = nerf_model.NeRFNetwork.load_from_checkpoint(ckpt).to(device)
    test_dl = SyntheticDataset(base_dir, 'test', rays, cropping=False)

    ssim_sum = 0
    psnr_sum = 0
    for batch in tqdm(test_dl):
        all_o_rays = batch['all_origin'].to(device)
        all_d_rays = batch['all_direc'].to(device)
        gt_im = (batch['image'].numpy() * 255).clip(0, 255).astype(np.uint8)
        recon = nerf_helpers.view_reconstruction(model, all_o_rays, all_d_rays, N=rays)

        # ssim
        ssim = structural_similarity(gt_im, recon, multichannel=True)
        ssim_sum += ssim
        # psnr
        psnr = peak_signal_noise_ratio(gt_im, recon)
        psnr_sum += psnr

    print("==============Calculate Scores==============")
    print(f"average psnr score: {psnr_sum / len(test_dl)}")
    print(f"average ssim score: {ssim_sum / len(test_dl)}")


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Calculate score metrics for NeRF Models.')
    parser.add_argument('-c', '--ckpt', type=str, required=True, help='ckpt path for model')
    parser.add_argument('-r', '--rays', type=int, default=4096, help='number of rays per batch')
    parser.add_argument('-b', '--base_dir', type=Path, default='/content/CSEP573-NeRF/data/nerf_synthetic/lego/', 
                        help='where to save the resulting gif')
    args = parser.parse_args()
    
    calculate_scores(args.ckpt, args.base_dir, args.rays)