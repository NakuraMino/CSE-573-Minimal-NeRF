from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import dataloader 
import nerf_model 

def train_simple_image(im_path): 
    wandb_logger = WandbLogger(name="grad_lounge_image", project="NeRF")
    trainer = Trainer(gpus=1, default_root_dir="/home/nakuram/CSEP573-NeRF/experiments/", max_steps=100000, logger=wandb_logger)
    train_dl = dataloader.getPhotoDataloader(im_path, batch_size=4096, num_workers=8, shuffle=True)
    val_dl = dataloader.getValDataloader(im_path, batch_size=1, num_workers=1, shuffle=False)
    model = nerf_model.ImageNeRFModel(position_dim=10)
    trainer.fit(model, train_dl, val_dl)

if __name__ == '__main__': 
    im_path = './tests/test_data/grad_lounge.png'
    train_simple_image(im_path)
