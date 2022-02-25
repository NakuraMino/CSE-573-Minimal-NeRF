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

    Args:
        x: [N x ... x C] Tensor of input floats. 
    Returns: 
        positional_encoding: [N x ... x 2*dim*C] Tensor of higher 
                             dimensional representation of inputs.
    """
    positional_encoding = []    
    for i in range(dim):
        positional_encoding.append(torch.cos(2**i * torch.pi * x))
        positional_encoding.append(torch.sin(2**i * torch.pi * x))
    positional_encoding = torch.cat(positional_encoding, dim=-1)
    return positional_encoding

class NeRFNetwork(LightningModule):
    """A full NeRF Network.

    Pytorch-Lightning Wrapper to train both the coarse and the fine network
    in one model. 
    """
    def __init__(self, position_dim=10, direction_dim=4, coarse_samples=64,
                 fine_samples=128):
        """NeRF Constructor.

        Args:
            position_dim: the size of the position encoding. Resulting size will be 
                input_size*2*position_dim.
            direction_dim: the size of the direction encoding. Resulting size will be 
                input_size*2*direction_dim.
            coarse_samples: number of samples for the coarse network.
            fine_samples: number of additional samples for the fine network. (i.e. fine network 
                gets coarse+fine samples)
        """
        super(NeRFNetwork, self).__init__()
        self.position_dim = position_dim
        self.direction_dim = direction_dim
        self.coarse_samples = coarse_samples
        self.fine_samples = fine_samples
        self.coarse_network = NeRFModel(position_dim, direction_dim)
        self.fine_network = NeRFModel(position_dim, direction_dim)

    def forward(self, o_rays, d_rays):
        """Single forward pass on both coarse and fine network.

        Args:
            o_rays: [N x 3] coordinates of the ray origin.
            d_rays: [N x 3] directions of the ray.
        Returns:
            dictionary with keys:
                'pred_rgbs': [N x 3] tensor of predicted rgb value of each ray.
                'all_rgb': [N x coarse*2+fine_samples x 3] rgb predictions at each location.
                'all_density': [N x coarse*2+fine_samples x 1] density predictions.
                'all_ts': [N x coarse*2+fine_samples x 1] time values along a ray direction.
        """
        # calculating coarse
        coarse_samples, coarse_ts = nerf_helpers.generate_coarse_samples(o_rays, d_rays, self.coarse_samples)
        coarse_density, coarse_rgb =self.coarse_network(coarse_samples, d_rays)
        coarse_deltas = nerf_helpers.generate_deltas(coarse_ts)

        weights = nerf_helpers.calculate_unnormalized_weights(coarse_density, coarse_deltas)
        fine_samples, fine_ts = nerf_helpers.inverse_transform_sampling(o_rays, d_rays, weights, 
                                                                        coarse_ts, self.fine_samples)
        # fine_deltas = nerf_helpers.generate(fine_ts)
        fine_samples = torch.cat([fine_samples, coarse_samples], axis=1)
        fine_ts = torch.cat([fine_ts, coarse_ts], axis=1)
        fine_density, fine_rgb = self.fine_network(fine_samples, d_rays)

        # sort ts to be sequential (in order to calculate deltas correctly) and sort density and rgb to align.
        all_ts = torch.cat([coarse_ts, fine_ts], dim=1)
        all_ts, idxs = torch.sort(all_ts, dim=1)        
        all_density = torch.gather(torch.cat([coarse_density, fine_density], dim=1), 1, idxs)
        idxs = torch.broadcast_to(idxs, list(all_density.shape[:-1]) + [3])
        all_rgb = torch.gather(torch.cat([coarse_rgb, fine_rgb], dim=1), 1, idxs)

        # calculate ray color.
        all_deltas = nerf_helpers.generate_deltas(all_ts)
        all_weights = nerf_helpers.calculate_unnormalized_weights(all_density, all_deltas)
        pred_rgbs = nerf_helpers.estimate_ray_color(all_weights, all_rgb)
        return {'pred_rgbs': pred_rgbs, 'all_rgb': all_rgb, 'all_density': all_density, 'all_ts': all_ts}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        nerf_helpers.fix_batchify(train_batch)
        o_rays = train_batch['origin'] 
        d_rays = train_batch['direc']
        rgba =  train_batch['rgba']
        pred_dict = self.forward(o_rays, d_rays)
        pred_rgb = pred_dict['pred_rgbs']
        loss = F.mse_loss(pred_rgb, rgba)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        nerf_helpers.fix_batchify(val_batch)
        o_rays = val_batch['origin'] 
        d_rays = val_batch['direc']
        rgba =  val_batch['rgba']
        pred_dict = self.forward(o_rays, d_rays)
        pred_rgb = pred_dict['pred_rgbs']
        loss = F.mse_loss(pred_rgb, rgba)
        self.log('val_loss', loss)
        return loss


class NeRFModel(nn.Module):
    """A single NeRF model.

    A single NeRF model (used for both coarse and fine networks) is made up of an
    8-layer multi-layer perceptron with ReLU activation functions. The input is a
    position and direction (each of which are 3 values), while the output is a
    scalar density and rgb value. The final layer predicting the rgb uses a sigmoid
    activation.
    """

    def __init__(self, position_dim=10, direction_dim=4):
        """NeRF Constructor.

        Args:
            position_dim: the size of the position encoding. Resulting size will be 
                input_size*2*position_dim.
            direction_dim: the size of the direction encoding. Resulting size will be 
                input_size*2*direction_dim.
        """
        super(NeRFModel, self).__init__()
        self.position_dim = position_dim
        self.direction_dim = direction_dim
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

        self.feature_fn = nn.Sequential(
            nn.Linear(256 + 60, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.density_fn = nn.Sequential(
            nn.Linear(256, 1),
            nn.ReLU()  # rectified to ensure nonnegative density
        )

        self.rgb_fn = nn.Sequential(
            nn.Linear(256 + 24, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, samples, direc): 
        """Forward pass through a NeRF Model (MLP).

        Args:
            samples: [N x samples x 3] coordinate locations to query the network.
            direc: [N x 3] directions for each ray.
        Returns: 
            density: [N x samples x 1] density predictions.
            rgb: [N x samples x 3] color/rgb predictions.
        """
        # direction needs to be broadcasted since it hasn't been sampled
        direc = torch.broadcast_to(direc[:, None, :], samples.shape)
        # positional encodings
        pos_enc_samples = positional_encoding(samples, dim=self.position_dim)
        pos_enc_direc = positional_encoding(direc, dim=self.direction_dim)
        # feed forward network
        x_features = self.mlp(pos_enc_samples)
        # concatenate positional encodings again
        x_features = torch.cat((x_features, pos_enc_samples), dim=-1)
        x_features = self.feature_fn(x_features)
        density = self.density_fn(x_features)
        # final rgb predictor
        dim_features = torch.cat((x_features, pos_enc_direc), dim=-1)
        rgb = self.rgb_fn(dim_features)
        return density, rgb


class ImageNeRFModel(LightningModule):
    """Toy NeRF model for 2D reconstruction.

    Instead of reconstructing/learning a full 3D model, we only learn a single 
    image. Input is a single pixel coordinate (x, y), and the output is an rgb
    color.
    """
    def __init__(self, position_dim=10):
        """Image NeRF Constructor.

        Args:
            position_dim: the size of the position encoding. Resulting size will be 
                input_size*2*position_dim. if position_dim <= 0, then no encoding is
                used.
        """
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
        """Forward pass.

        Args: 
            x: [N x 2] Tensor of coordinate locations.
        Returns:
            rgb: [N x 3] Tensor of colors.
        """
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
        """Reconstruct an image.

        Unlike a traditional validation step, we query the network at
        every pixel coordinate to get a reconstructed image and log it.
        This just qualitatively helps us see whether the network is 
        learning or not.
        """
        im_h, im_w = val_batch
        im = photo_nerf_to_image(self, im_h, im_w)
        im = torch_to_numpy(im, is_normalized_image=True)
        im = Image.fromarray(im.astype(np.uint8))
        self.logger.log_image(key='recon', images=[im])
        return 0
