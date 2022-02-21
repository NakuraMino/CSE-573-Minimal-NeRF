from asyncio.log import logger
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import dataloader 
import nerf_model 
import argparse

def train_simple_image(im_path, logger_name, steps): 
    wandb_logger = WandbLogger(name=f"{logger_name}", project="NeRF")
    trainer = Trainer(gpus=1, default_root_dir="/home/nakuram/CSEP573-NeRF/experiments/", max_steps=steps, logger=wandb_logger)
    train_dl = dataloader.getPhotoDataloader(im_path, batch_size=4096, num_workers=8, shuffle=True)
    val_dl = dataloader.getValDataloader(im_path, batch_size=1, num_workers=1, shuffle=False)
    model = nerf_model.ImageNeRFModel(position_dim=10)
    trainer.fit(model, train_dl, val_dl)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Train a NeRF model')
    parser.add_argument('-n', '--name', type=str, help='name of the model experiment', required=True)
    parser.add_argument('-s', '--steps', type=int, default=100000, help='max number of steps')
    parser.add_argument('-i', '--im_path', type=str, default='./tests/test_data/grad_lounge.png',
        help='The image path to use as data')
    
    args = parser.parse_args()
    train_simple_image(args.im_path, args.name, args.steps)
