import torch
import argparse
import nerf_model
import nerf_helpers 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def render(ckpt, save_path, rays):
    model = nerf_model.NeRFNetwork.load_from_checkpoint(ckpt).to(device)
    nerf_helpers.generate_360_view_synthesis(model, save_path, N=rays)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Render a 360 view from a NeRF Model')
    parser.add_argument('-c', '--ckpt', type=str, required=True, help='ckpt path for model')
    parser.add_argument('-r', '--rays', type=int, default=4096, help='number of rays per batch')
    parser.add_argument('-s', '--save_path', type=str, default='./recons/360.gif', help='where to save the resulting gif')
    args = parser.parse_args()
    
    render(args.ckpt, args.save_path, args.rays)