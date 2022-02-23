import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_lightning import LightningModule
from nerf_to_recon import photo_nerf_to_image, torch_to_numpy
import nerf_helpers
from PIL import Image


def positional_encoding(x, dim=10):
    """project input to higher dimensional space as a positional encoding.
    """ 
    positional_encoding = []    
    for i in range(dim):
        positional_encoding.append(torch.cos(2**i * torch.pi * x))
        positional_encoding.append(torch.sin(2**i * torch.pi * x))
    positional_encoding = torch.cat(positional_encoding, dim=1)
    return positional_encoding

class NeRFNetwork(LightningModule):
    def __init__(self, position_dim=10, density_dim=4, coarse_samples=64,
        fine_samples=128):
        super(NeRFNetwork, self).__init__()
        self.position_dim = position_dim
        self.density_dim = density_dim
        self.coarse_samples = coarse_samples
        self.fine_samples = fine_samples
        self.coarse_network = NeRFModel(position_dim, density_dim)
        self.fine_network = NeRFModel(position_dim, density_dim)

    def forward(self, o_rays, d_rays):
        coarse_samples, ts = nerf_helpers.generate_coarse_samples(o_rays, d_rays, self.coarse_samples)
        coarse_density, coarse_rgb =self.coarse_network(coarse_samples, d_rays)
        deltas = nerf_helpers.generate_deltas(ts)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        o_rays = train_batch['origin'] 
        d_rays = train_batch['direc']
        rgba =  train_batch['rgba']
        pred_rgb = self.forward(o_rays, d_rays)
        loss = F.mse_loss(pred_rgb, rgba)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        pass


class NeRFModel(nn.Module):

    def __init__(self, position_dim=10, density_dim=4): 
        super(NeRFModel, self).__init__()
        self.position_dim = position_dim
        self.density_dim = density_dim
        # first MLP is a simple multi-layer perceptron 
        self.mlp = nn.Sequential(
            nn.Linear(60, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.density_fn = nn.Sequential(
            nn.Linear(256 + 60, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256 + 1),  # density is a separate value
            # no activation function for density
        )

        self.rgb_fn = nn.Sequential(
            nn.Linear(256 + 24, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, x, d): 
        # positional encodings
        pos_enc_x = positional_encoding(x, dim=self.position_dim)
        pos_enc_d = positional_encoding(d, dim=self.density_dim)
        # feed forward network
        x_features = self.mlp(pos_enc_x)
        x_features = torch.cat((x_features, pos_enc_x), dim=1)
        # concatenate positional encodings again
        density_and_features = self.density_fn(x_features)
        density = density_and_features[:, 0].unsqueeze(axis=-1)
        # final rgb predictor
        features = density_and_features[:, 1:]
        features = torch.cat((features, pos_enc_d), dim=1)
        rgb = self.rgb_fn(features)

        return density, rgb


class ImageNeRFModel(LightningModule):
    def __init__(self, position_dim=10): 
        super(ImageNeRFModel, self).__init__()
        self.position_dim = position_dim
        # first MLP is a simple multi-layer perceptron 
        self.input_size = 2*2*position_dim if position_dim > 0 else 2
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )
    
    def forward(self, x): 
        # positional encodings
        if self.position_dim > 0:
            x = positional_encoding(x, dim=self.position_dim)
        # feed forward network
        rgb = self.mlp(x)
        return rgb

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch 
        pred_rgb = self.forward(x)
        loss = F.mse_loss(pred_rgb, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        im_h, im_w = val_batch
        im = photo_nerf_to_image(self, im_h, im_w)
        im = torch_to_numpy(im, is_normalized_image=True)
        im = Image.fromarray(im.astype(np.uint8))
        self.logger.log_image(key='recon', images=[im])
        return 0
