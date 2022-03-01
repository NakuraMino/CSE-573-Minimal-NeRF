"""Script to train NeRF models.

Usage Examples:
    python train_nerf.py -n test -s 10 simple: train simple/toy nerf with experiment name test for 10 steps.
    python train_nerf.py -n test --gpu full: trains full nerf model on a gpu with experiment name test.

My personal laptop use: 
    python train_nerf.py -n laptop -s 10 -rd ./ full -b ./data/lego/


"""
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import dataloader 
import nerf_model 
import argparse

def train_full_nerf(root_dir, base_dir, logger_name, steps, pos_enc, direc_enc, use_gpu,
                    num_rays, coarse_samples, fine_samples, near, far, args):
    """Train full NeRF model (coarse+fine network f(x,y,z,theta,rho)->rgb+sigma""" 
    wandb_logger = WandbLogger(name=f"{logger_name}", project="NeRF")
    wandb_logger.log_hyperparams(args)
    trainer = Trainer(gpus=int(use_gpu), default_root_dir=root_dir, 
                      max_steps=steps, logger=wandb_logger, check_val_every_n_epoch=10, track_grad_norm=2)
    train_dl = dataloader.getSyntheticDataloader(base_dir, 'train', num_rays, num_workers=8, shuffle=True)
    val_dl = dataloader.getSyntheticDataloader(base_dir, 'val', num_rays, num_workers=8, shuffle=False)
    model = nerf_model.NeRFNetwork(position_dim=pos_enc, direction_dim=direc_enc, 
                                   coarse_samples=coarse_samples, fine_samples=fine_samples,
                                   near=near, far=far)
    trainer.fit(model, train_dl, val_dl)

def train_simple_image(root_dir, im_path, logger_name, steps, pos_enc, use_gpu, args): 
    """Train a toy nerf model (e.g. f(x,y) -> rgb)."""
    wandb_logger = WandbLogger(name=f"{logger_name}", project="NeRF")
    wandb_logger.log_hyperparams(args)
    trainer = Trainer(gpus=int(use_gpu), default_root_dir=root_dir, 
                      max_steps=steps, logger=wandb_logger, check_val_every_n_epoch=10)
    train_dl = dataloader.getPhotoDataloader(im_path, batch_size=4096, num_workers=8, shuffle=True)
    val_dl = dataloader.getValDataloader(im_path, batch_size=1, num_workers=8, shuffle=False)
    model = nerf_model.ImageNeRFModel(position_dim=pos_enc)
    trainer.fit(model, train_dl, val_dl)

def train_single_nerf(root_dir, base_dir, logger_name, steps, pos_enc, direc_enc, use_gpu, num_rays, samples, args):
    wandb_logger = WandbLogger(name=f"{logger_name}", project="NeRF")
    wandb_logger.log_hyperparams(args)
    trainer = Trainer(gpus=int(use_gpu), default_root_dir=root_dir, 
                      max_steps=steps, logger=wandb_logger, check_val_every_n_epoch=10, track_grad_norm=2)
    train_dl = dataloader.getSyntheticDataloader(base_dir, 'train', num_rays, num_workers=8, shuffle=True)
    val_dl = dataloader.getSyntheticDataloader(base_dir, 'val', num_rays, num_workers=8, shuffle=False)
    model = nerf_model.SingleNeRF(position_dim=pos_enc, direction_dim=direc_enc, num_samples=samples)
    trainer.fit(model, train_dl, val_dl)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Train a NeRF model')
    subparsers = parser.add_subparsers(dest='type', help='Training different NeRF Versions')
    parser.add_argument('-n', '--name', type=str, help='name of the model experiment', required=True)
    parser.add_argument('-s', '--steps', type=int, default=100000, help='max number of steps')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('-p', '--position_encoding', type=int, default=10, help='position encoding length')
    parser.add_argument('-rd', '--root_dir', type=str, default="/home/nakuram/CSEP573-NeRF/experiments/", help='directory to save models')

    simple_parser = subparsers.add_parser("simple")
    full_parser = subparsers.add_parser("full")
    single_parser = subparsers.add_parser("single")

    simple_parser.add_argument('-i', '--im_path', type=str, default='./tests/test_data/grad_lounge.png',
        help='The image path to use as data')

    full_parser.add_argument('-d', '--direction_encoding', type=int, default=4, help='direction encoding length')
    full_parser.add_argument('-b', '--base_dir', type=str, default='/home/nakuram/CSEP573-NeRF/data/nerf_synthetic/lego/', help='directory for dataset')
    full_parser.add_argument('-r', '--rays', type=int, default=4096, help='number of rays per batch')
    full_parser.add_argument('-c', '--coarse', type=int, default=64, help='number of coarse samples')
    full_parser.add_argument('-f', '--fine', type=int, default=128, help='number of fine samples')
    full_parser.add_argument('-nr', '--near', type=float, default=2.0, help='near bound for dataset')
    full_parser.add_argument('-fr', '--far', type=int, default=6.0, help='far bound of dataset')

    single_parser.add_argument('-d', '--direction_encoding', type=int, default=4, help='direction encoding length')
    single_parser.add_argument('-b', '--base_dir', type=str, default='./dev_data/', help='directory for dataset')
    single_parser.add_argument('-r', '--rays', type=int, default=4096, help='number of rays per batch')
    single_parser.add_argument('-c', '--samples', type=int, default=128, help='number of samples')

    args = parser.parse_args()

    if args.type == 'simple': 
        train_simple_image(args.root_dir, args.im_path, args.name, args.steps, args.position_encoding, args.gpu, args)
    elif args.type == 'full':
        train_full_nerf(args.root_dir, args.base_dir, args.name, args.steps, args.position_encoding, 
                        args.direction_encoding, args.gpu, args.rays, args.coarse, args.fine, args.near,
                        args.far, args)
    elif args.type == 'single': 
        train_single_nerf(args.root_dir, args.base_dir, args.name, args.steps, args.position_encoding,
                          args.direction_encoding, args.gpu, args.rays, args.samples, args)