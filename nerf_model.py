import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_lightning import LightningModule
from nerf_to_recon import photo_nerf_to_image, torch_to_numpy
from PIL import Image

def naive_positional_encoding(x, dim=10):
    """project input to higher dimensional space as a positional encoding.
    """ 
    n, d = x.shape
    pos_enc_size = d * 2 * dim
    positional_encoding = torch.zeros((n, pos_enc_size))
    for i in range(dim):
        for j in range(d):
            orig_feature = x[:, j]
            positional_encoding[:,j*2*dim+2*i] = torch.sin(2**i * torch.pi * orig_feature)
            positional_encoding[:,j*2*dim+2*i+1] = torch.cos(2**i * torch.pi * orig_feature)
    return positional_encoding


def positional_encoding(x, dim=10):
    """project input to higher dimensional space as a positional encoding.
    """ 
    n, d = x.shape
    positional_encoding = []    
    for i in range(dim):
        positional_encoding.append(torch.cos(2**i * torch.pi * x))
        positional_encoding.append(torch.sin(2**i * torch.pi * x))
    positional_encoding = torch.cat(positional_encoding, dim=1)
    return positional_encoding


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
        if self.input_size > 2:
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
        im = torch_to_numpy(im, is_standardized_image=True)
        im = Image.fromarray(im.astype(np.uint8))
        self.logger.log_image(key='recon', images=[im])
        return 0

    # def validation_step(self, val_batch, batch_idx): 
    #     coords, rgb = val_batch 
    #     pred_rgb = self.forward(coords)
    #     loss = F.mse_loss(pred_rgb, rgb)
    #     self.log('val_loss', loss)
    #     return loss
    
