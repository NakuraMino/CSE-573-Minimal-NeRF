import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_lightning import LightningModule
from nerf_to_recon import photo_nerf_to_image, torch_to_numpy
import nerf_helpers
from PIL import Image
import random
from timeit import default_timer as timer


ACT_FN = nn.LReLU(0.1)  # nn.ReLU()


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
        # calculating coarse
        coarse_samples, coarse_ts = nerf_helpers.generate_coarse_samples(o_rays, d_rays, self.coarse_samples, self.near, self.far)
        coarse_density, coarse_rgb =self.coarse_network(coarse_samples, d_rays)
        self.log('coarse_density_norms', torch.linalg.norm(coarse_density), batch_size=1)
        self.log('coarse_density_non_zeros', (coarse_density != 0).sum().float(), batch_size=1)
        coarse_deltas = nerf_helpers.generate_deltas(coarse_ts)

        weights = nerf_helpers.calculate_unnormalized_weights(coarse_density, coarse_deltas)
        fine_samples, fine_ts = nerf_helpers.inverse_transform_sampling(o_rays, d_rays, weights, 
                                                                        coarse_ts, self.fine_samples)
        # fine_deltas = nerf_helpers.generate(fine_ts)
        fine_samples = torch.cat([fine_samples, coarse_samples], axis=1)
        fine_ts = torch.cat([fine_ts, coarse_ts], axis=1)
        fine_density, fine_rgb = self.fine_network(fine_samples, d_rays)
        self.log('fine_density_norms', torch.linalg.norm(fine_density), batch_size=1)
        self.log('fine_density_non_zeros', (fine_density != 0).sum().float(), batch_size=1)

        # sort ts to be sequential (in order to calculate deltas correctly) and sort density and rgb to align.
        all_ts = torch.cat([coarse_ts, fine_ts], dim=1)
        all_ts, idxs = torch.sort(all_ts, dim=1)
        all_density = torch.gather(torch.cat([coarse_density, fine_density], dim=1), 1, idxs)
        idxs = torch.broadcast_to(idxs, list(all_density.shape[:-1]) + [3])
        all_rgb = torch.gather(torch.cat([coarse_rgb, fine_rgb], dim=1), 1, idxs)
        all_samples = torch.gather(torch.cat([coarse_samples, fine_samples], dim=1), 1, idxs)

        # calculate ray color.
        all_deltas = nerf_helpers.generate_deltas(all_ts)
        all_weights = nerf_helpers.calculate_unnormalized_weights(all_density, all_deltas)
        pred_rgbs = nerf_helpers.estimate_ray_color(all_weights, all_rgb)
        return {'pred_rgbs': pred_rgbs, 'all_rgb': all_rgb, 'all_density': all_density, 'all_ts': all_ts, 
                'all_samples': all_samples, 'all_deltas': all_deltas}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
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
        self.log('train_loss', loss, batch_size=N)
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
        pred_rgbs = pred_dict['pred_rgbs']
        
        # loss
        N, _ = pred_rgbs.shape
        loss = F.mse_loss(pred_rgbs, rgb)
        self.log('val_loss', loss, batch_size=N)

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
        self.idx = 0
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
            # nn.ReLU(),
        )

        self.density_fn = nn.Sequential(
            nn.Linear(256, 1),
            nn.ReLU() # rectified to ensure nonnegative density
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
        # self._change_density_activation()
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

    # def _change_density_activation(self): 
    #     self.idx += 1
    #     if self.idx == 1000:
    #         use_relu = list(self.density_fn.children())[:-1] + [nn.ReLU()]
    #         self.density_fn = nn.Sequential(*use_relu)


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
