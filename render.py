"""Renders novel views (360 view) from a NeRF model.

Usage:
    python render.py -c CKPT_PATH -r 4096 -p 40 -s SAVE_DIR
"""
import torch
import argparse
import nerf_model
import nerf_helpers 
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def render(ckpt, save_dir, rays, num_poses):
    epoch_idx = ckpt.find('epoch=')
    epoch = ckpt[epoch_idx: epoch_idx+ckpt[epoch_idx:].find('-')]
    model = nerf_model.NeRFNetwork.load_from_checkpoint(ckpt).to(device)
    nerf_helpers.generate_360_view_synthesis(model, save_dir, epoch, N=rays, num_poses=num_poses)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Render a 360 view from a NeRF Model')
    parser.add_argument('-c', '--ckpt', type=str, required=True, help='ckpt path for model')
    parser.add_argument('-r', '--rays', type=int, default=4096, help='number of rays per batch')
    parser.add_argument('-p', '--num_poses', type=int, default=40, help='number of images in gif.')
    parser.add_argument('-s', '--save_dir', type=Path, default='./recons/', help='where to save the resulting gif')
    args = parser.parse_args()
    
    render(args.ckpt, args.save_dir, args.rays, args.num_poses)