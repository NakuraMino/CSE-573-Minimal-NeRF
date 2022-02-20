from pytorch_lightning import Trainer
import dataloader 
import nerf_model 

def train_simple_image(im_path): 
    trainer = Trainer(max_steps=10) #0000)
    train_dl = dataloader.getPhotoDataloader(im_path, batch_size=4096, num_workers=4, shuffle=True)
    val_dl = dataloader.getPhotoDataloader(im_path, batch_size=32, num_workers=1, shuffle=True)
    model = nerf_model.ImageNeRFModel(position_dim=10)
    trainer.fit(model, train_dl, val_dl)

if __name__ == '__main__': 
    im_path = './tests/test_data/grad_lounge.png'
    train_simple_image(im_path)