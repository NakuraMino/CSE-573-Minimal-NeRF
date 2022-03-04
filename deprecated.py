import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import dataloader 
import nerf_model 

def sample_random_coordinates(N, height, width, alpha=None, prop=0.0): 
    """Two [N,] torch tensors representing random coordinates.

    Args:
        N: int representing number of coordinates to sample
        height: the maximum height value (exclusive)
        width: maximum width value (exclusive)
        alpha: alpha channel of an image, used to weight coordinates to sample.
               No weighting if None.
        prop: minimum proportion of coordinates that have to 
                    have alpha values > 0.
    Returns:
        xs: [N,] torch tensor of random ints [0,width)
        ys: [N,] torch tensor of random ints [0,height)
    """
    if alpha == None or prop == 0:
        xs = torch.randint(0, width, size=(N,))
        ys = torch.randint(0, height, size=(N,))
    else: 
        num_in_alpha = int(N * prop)
        num_random = N - num_in_alpha
        # pseudo science to pick an initial number of samples
        num_samples = int(N * (1 / (1 - prop)))
        xs = torch.randint(0, width, size=(num_samples,))
        ys = torch.randint(0, height, size=(num_samples,))
        # keep the ones that have alpha > 0.
        valid = alpha[ys, xs]
        xs = xs[valid]; ys = ys[valid]
        if xs.shape[0] > num_in_alpha:
            xs = xs[:num_in_alpha]; ys = ys[:num_in_alpha]
        xs = torch.cat((xs, torch.randint(0, width, size=(num_random,))))
        ys = torch.cat((ys, torch.randint(0, height, size=(num_random,))))
    return xs, ys

def convert_to_ndc_rays(o_rays, d_rays, focal, width, height, near=1.0): 
    """This is only for FRONT-FACING scenes.

    Args:
        o_rays: [H x W x 3] representing ray origin.
        d_rays: [H x W x 3] representing ray direction.
        focal: focal length.
        width: the maximum width 
        height: the maximum height
        near: the near depth bound (i.e. 1)
    """
    # shift o to the ray’s intersection with the near plane at z = −n
    t_near  = - (near + o_rays[:,:,2] ) / d_rays[:,:,2] 
    o_rays = o_rays + t_near[...,None] * d_rays

    ox, oy, oz = o_rays[:,:,0], o_rays[:,:,1], o_rays[:,:,2] 
    dx, dy, dz = d_rays[:,:,0], d_rays[:,:,1], d_rays[:,:,2] 
    
    ox_new =  -1.0 * focal / (width / 2) * (ox / oz)
    oy_new =  -1.0 * focal / (height / 2) * (oy / oz)
    oz_new = 1.0 + (2 * near) / oz
    dx_new =  -1.0 * focal / (width / 2) * ((dx / dz) - (ox / oz))
    dy_new =  -1.0 * focal / (height / 2) * ((dy / dz) - (oy / oz))
    dz_new = (- 2 * near) / oz
    
    o_ndc_rays = torch.stack([ox_new, oy_new, oz_new], axis=-1)
    d_ndc_rays = torch.stack([dx_new, dy_new, dz_new], axis=-1)
    d_ndc_rays = d_ndc_rays / torch.linalg.norm(d_ndc_rays, dim=-1,keepdim=True)
    return o_ndc_rays, d_ndc_rays

def train_full_nerf(root_dir, base_dir, logger_name, steps, pos_enc, direc_enc, use_gpu,
                    num_rays, coarse_samples, fine_samples, near, far, cropping_epochs, ckpt, args):
    """Train full NeRF model (coarse+fine network f(x,y,z,theta,rho)->rgb+sigma""" 
    wandb_logger = WandbLogger(name=logger_name, project="NeRF")
    wandb_logger.log_hyperparams(args)
    trainer = Trainer(gpus=int(use_gpu), default_root_dir=root_dir, max_steps=steps, 
                      resume_from_checkpoint=ckpt, logger=wandb_logger,
                      check_val_every_n_epoch=10, track_grad_norm=2)
    train_dl = dataloader.getSyntheticDataloader(base_dir, 'train', num_rays, cropping_epochs=cropping_epochs, num_workers=2, shuffle=True)
    val_dl = dataloader.getSyntheticDataloader(base_dir, 'val', num_rays, cropping_epochs=cropping_epochs, num_workers=2, shuffle=False)
    model = nerf_model.NeRFNetwork(position_dim=pos_enc, direction_dim=direc_enc, 
                                   coarse_samples=coarse_samples, fine_samples=fine_samples,
                                   near=near, far=far)
    trainer.fit(model, train_dl, val_dl)

def _change_density_activation(self): 
    self.idx += 1
    if self.idx == 1000:
        use_relu = list(self.density_fn.children())[:-1] + [nn.ReLU()]
        self.density_fn = nn.Sequential(*use_relu)