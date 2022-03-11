"""NeRF models.

Contains the various models and sub-models used to train a Neural Radiance Field (NeRF).
"""
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer as timer
from pytorch_lightning import LightningModule

import nerf_helpers


ACT_FN = nn.ReLU()  # nn.LeakyReLU(0.1)


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

def normalize_coordinates(x, bound=math.pi): 
    """Normalize coordinates to be within [-1,1].

    Coordinates have to be within [-1,1] so they are not
    affected by the periodicity of the positional encodings.

    Why is math.pi the default value you ask? Good question. It's because
    I saw empirically that all coordinates are within [-3,3], but 3 feels
    like a random number. https://github.com/bmild/nerf/issues/12 gives a
    better justification for why we use pi :)

    Args:
        x: [N x num_samples x 3] tensor of coordinates. All coordinates
          should be within [-bound, bound] so that they can be normalized
          within [-1,1].
        bound: the maximum value that x can have. Cannot be bound=0.
    Returns:
        normalized coordinates (i.e. x / bound)
    """
    return x / bound

class NeRFNetwork(LightningModule):
    """A full NeRF Network.

    Pytorch-Lightning Wrapper to train both the coarse and the fine network
    in one model. 
    """

    def __init__(self, position_dim=10, direction_dim=4, coarse_samples=64,
                 fine_samples=128, near=2.0, far=6.0):
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
        self.near = near
        self.far = far
        self.coarse_network = NeRFModel(position_dim, direction_dim)
        self.fine_network = NeRFModel(position_dim, direction_dim)
        self.im_idx = 0
        self.max_idx = 1
        self.timer = timer()

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
        # calculating coarse network.
        coarse_samples, coarse_ts = nerf_helpers.generate_coarse_samples(o_rays, d_rays, self.coarse_samples, self.near, self.far)
        coarse_density, coarse_rgb =self.coarse_network(coarse_samples, d_rays)
        self.log('coarse_density_norms', torch.linalg.norm(coarse_density), batch_size=1)
        self.log('coarse_density_non_zeros', (coarse_density != 0).sum().float(), batch_size=1)

        # calculate coarse ray color.        
        coarse_deltas = nerf_helpers.generate_deltas(coarse_ts)
        coarse_weights = nerf_helpers.calculate_unnormalized_weights(coarse_density, coarse_deltas)
        coarse_rgb_ray = nerf_helpers.estimate_ray_color(coarse_weights, coarse_rgb)

        # sample points for fine samples.
        fine_samples, fine_ts = nerf_helpers.inverse_transform_sampling(o_rays, d_rays, coarse_weights, 
                                                                        coarse_ts, self.fine_samples)
        fine_samples = torch.cat([fine_samples, coarse_samples], axis=1)
        fine_ts = torch.cat([fine_ts, coarse_ts], axis=1)
        fine_ts, idxs = torch.sort(fine_ts, dim=1)
        idxs = torch.broadcast_to(idxs, fine_samples.shape)
        fine_samples = torch.gather(fine_samples, 1, idxs)

        # calculating fine network.
        fine_density, fine_rgb = self.fine_network(fine_samples, d_rays)
        self.log('fine_density_norms', torch.linalg.norm(fine_density), batch_size=1)
        self.log('fine_density_non_zeros', (fine_density != 0).sum().float(), batch_size=1)

        # calculate fine ray color.
        fine_deltas = nerf_helpers.generate_deltas(fine_ts)
        fine_weights = nerf_helpers.calculate_unnormalized_weights(fine_density, fine_deltas)
        fine_rgb_ray = nerf_helpers.estimate_ray_color(fine_weights, fine_rgb)
        
        return {'fine_rgb_rays': fine_rgb_ray, 'coarse_rgb_rays': coarse_rgb_ray}
        
    def configure_optimizers(self):
        # end_lr = start_lr * gamma^epochs
        start_lr = 5e-4
        end_lr = 5e-5
        num_epochs = 1200
        gamma = (end_lr / start_lr) ** (1/num_epochs)
        optimizer = torch.optim.Adam(self.parameters(), lr=start_lr)
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/960
        lr_decay_optimizer = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
        return {'optimizer': optimizer, 'lr_scheduler': lr_decay_optimizer}

    def training_step(self, train_batch, batch_idx):
        nerf_helpers.fix_batchify(train_batch)
        # inputs
        o_rays = train_batch['origin'] 
        d_rays = train_batch['direc']
        rgb =  train_batch['rgb']

        # forward pass
        pred_dict = self.forward(o_rays, d_rays)
        fine_rgb = pred_dict['fine_rgb_rays']
        coarse_rgb = pred_dict['coarse_rgb_rays']
        
        # loss
        N, _ = fine_rgb.shape
        coarse_loss = F.mse_loss(coarse_rgb, rgb)
        fine_loss = F.mse_loss(fine_rgb, rgb)
        loss = coarse_loss + fine_loss

        # logging
        self.log('train_loss', loss, batch_size=N)
        self.log('train_fine_loss', fine_loss, batch_size=N)
        self.log('train_coarse_loss', coarse_loss, batch_size=N)
        self.log('train iteration speed', timer() - self.timer, batch_size=N)
        self.timer = timer()
        return loss

    def validation_step(self, val_batch, batch_idx):
        self.max_idx = max(self.max_idx, batch_idx)
        if batch_idx == 0:
            self.im_idx = random.randint(0, self.max_idx)
        nerf_helpers.fix_batchify(val_batch)
        # Regular Validation Step

        # inputs
        o_rays = val_batch['origin'] 
        d_rays = val_batch['direc']
        rgb =  val_batch['rgb']
        
        # forward pass
        pred_dict = self.forward(o_rays, d_rays)
        fine_rgb = pred_dict['fine_rgb_rays']
        coarse_rgb = pred_dict['coarse_rgb_rays']
        
        # loss
        N, _ = fine_rgb.shape
        coarse_loss = F.mse_loss(coarse_rgb, rgb)
        fine_loss = F.mse_loss(fine_rgb, rgb)
        loss = coarse_loss + fine_loss

        # logging
        self.log('val_loss', loss, batch_size=N)
        self.log('val_fine_loss', fine_loss, batch_size=N)
        self.log('val_coarse_loss', coarse_loss, batch_size=N)

        if batch_idx == self.im_idx:
            all_o_rays = val_batch['all_origin']
            all_d_rays = val_batch['all_direc']
            im = nerf_helpers.view_reconstruction(self, all_o_rays, all_d_rays, N=N)
            self.logger.log_image(key='recon', images=[im], caption=[f'val/{self.im_idx}.png'])
            del im
        return loss


class SingleNeRF(LightningModule):
    """A Single NeRF Network.

    Pytorch-Lightning Wrapper to train a single NeRF Model. Mostly for debugging purposes.
    """

    def __init__(self, position_dim=10, direction_dim=4, num_samples=128, near=2.0, far=6.0):
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
        super(SingleNeRF, self).__init__()
        self.position_dim = position_dim
        self.direction_dim = direction_dim
        self.num_samples = num_samples
        self.near = near
        self.far = far
        self.network = NeRFModel(position_dim, direction_dim)

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
        samples, ts = nerf_helpers.generate_coarse_samples(o_rays, d_rays, self.num_samples, self.near, self.far)
        density, rgb =self.network(samples, d_rays)
        deltas = nerf_helpers.generate_deltas(ts)

        # calculate ray color.
        weights = nerf_helpers.calculate_unnormalized_weights(density, deltas)
        pred_rgbs = nerf_helpers.estimate_ray_color(weights, rgb)
        return {'pred_rgbs': pred_rgbs, 'density': density, 'ts': ts, 
                'samples': samples, 'deltas': deltas}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        start = timer()
        nerf_helpers.fix_batchify(train_batch)
        # inputs
        o_rays = train_batch['origin'] 
        d_rays = train_batch['direc']
        rgb =  train_batch['rgb']
        
        # forward pass
        pred_dict = self.forward(o_rays, d_rays)
        pred_rgb = pred_dict['pred_rgbs']
        
        # loss
        N, _ = pred_rgb.shape
        loss = F.mse_loss(pred_rgb, rgb)
        end = timer()
        self.log('train_loss', loss, batch_size=N)
        self.log('val iteration speed', end - start, batch_size=N)
        return loss

    def validation_step(self, val_batch, batch_idx):
        nerf_helpers.fix_batchify(val_batch)
        # Regular Validation Step

        # inputs
        o_rays = val_batch['origin'] 
        d_rays = val_batch['direc']
        rgb =  val_batch['rgb']
        
        # forward pass
        pred_dict = self.forward(o_rays, d_rays)
        pred_rgbs = pred_dict['pred_rgbs']
        
        # loss
        N, _ = pred_rgbs.shape
        loss = F.mse_loss(pred_rgbs, rgb)
        self.log('val_loss', loss, batch_size=N)

        all_o_rays = val_batch['all_origin']
        all_d_rays = val_batch['all_direc']
        im = nerf_helpers.view_reconstruction(self, all_o_rays, all_d_rays, N=N)
        self.logger.log_image(key='recon', images=[im], caption=[f'val/0.png'])
        del im
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
            nn.Linear(self.position_dim*2*3, 256),
            ACT_FN,
            nn.Linear(256, 256),
            ACT_FN,
            nn.Linear(256, 256),
            ACT_FN,
            nn.Linear(256, 256),
            ACT_FN
        )

        self.feature_fn = nn.Sequential(
            nn.Linear(256 + self.position_dim*2*3, 256),
            ACT_FN,
            nn.Linear(256, 256),
            ACT_FN,
            nn.Linear(256, 256),
        )

        self.density_fn = nn.Sequential(
            nn.Linear(256, 1),
            nn.Softplus()  # nn.ReLU() # rectified to ensure nonnegative density
        )

        self.rgb_fn = nn.Sequential(
            nn.Linear(256 + self.direction_dim*2*3, 128),
            ACT_FN,
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
        direc = direc / torch.linalg.norm(direc, dim=1, keepdim=True)  # unit direction
        direc = torch.broadcast_to(direc[:, None, :], samples.shape)

        # positional encodings
        samples = normalize_coordinates(samples)
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
        im = nerf_helpers.photo_nerf_to_image(self, im_h, im_w)
        im = nerf_helpers.torch_to_numpy(im, is_normalized_image=True)
        im = Image.fromarray(im.astype(np.uint8))
        self.logger.log_image(key='recon', images=[im])
        return 0
