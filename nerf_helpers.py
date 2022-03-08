import torch
import numpy as np
import itertools
from PIL import Image
import dataloader
import imageio
from tqdm import tqdm
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fix_batchify(batch): 
    """Squeezes torch Tensors so first dimension is the true batch dimension.

    One batch is technically [Nx3] rays and [Nx3] rgba colors, 
    but due to the way dataloaders works,             
    I have to unbatch the 0th dimension (e.g. [1xNx3] to [Nx3])
    """
    for key, value in batch.items():
        batch[key] = value.squeeze(0)  # in-place operation

def generate_coarse_samples(o_rays: torch.Tensor, d_rays: torch.Tensor, 
                            num_samples: int, near=2.0, far=6.0): 
    """Generates [N x num_samples x 3] coordinate samples.

    For each of the N rays, it samples num_samples coordinate locations
    uniformly. I think we actually sample slightly beyong far due to the
    fact that linspace is inclusive, but that should be fine.

    Args: 
        o_rays: [N x 3] coordinates of the ray origin.
        d_rays: [N x 3] directions of the ray.
        num_samples: The number of coordinates to sample from the ray.
        near: the near bound of the samples.
        far: the far bound of the samples
    Returns:
        Samples: [N x num_samples x 3] tensor of coordinates. 
        ts: [N x num_samples x 1] tensor of the time increment for each sample. 
    """
    N, _ = o_rays.shape
    o_rays = o_rays.unsqueeze(1)
    d_rays = d_rays.unsqueeze(1)

    step_size = (far - near) / num_samples
    ts = torch.broadcast_to(torch.arange(near, far, step_size, device=device)[None, ...], (N, num_samples))
    rand = torch.rand(ts.shape, device=device) * step_size
    ts = ts + rand
    ts = ts.unsqueeze(-1)
    samples = d_rays * ts + o_rays
    return samples, ts

def generate_deltas(ts: torch.Tensor, far=6.0):
    """Calculates the difference between each 'time' in ray samples.

    Args:
        ts: [N x num_samples x 1] tensor of times. The values are increasing from [near,far] along
            the num_samples dimension.
    Returns:
        deltas: [N x num_samples x 1]  where delta_i = t_i+1 - t_i. t_num_samples=far
    """
    N, _, _ = ts.shape
    upper_bound = torch.cat([ts[:,1:,:], torch.full((N, 1, 1), far, device=device)], dim=1)
    deltas = upper_bound - ts
    return deltas

def calculate_unnormalized_weights(density: torch.Tensor, deltas: torch.Tensor):
    """Calculate unnormalized weights for the ray color.

    Args:
        density: [N x num_samples x 1] of nonnegative values represnting density at each point.
        deltas: [N x num_samples x 1] of time deltas between previous sample and current sample.
    Returns: 
        weights: [N x num_samples x 1] tensor of weights calculated as 
                 w = T(1 - exp(- density * delta)).
    """
    N, s, _ = density.shape
    neg_delta_density = - 1 * density * deltas
    shifted_neg_delta_density = torch.cat((torch.zeros((N,1,1), device=device), 
                                          neg_delta_density[:,:-1,:]), axis=1)
    transmittance =  torch.exp(torch.cumsum(shifted_neg_delta_density, dim=1))
    weights = (1 - torch.exp(neg_delta_density)) * transmittance
    return weights

def estimate_ray_color(weights, rgb):
    """Estimates the color of a ray as a weighted average of the weights and colors.
    
    Args:
        weights: [N x num_samples x 1] tensor of weights for each rgb color. 
                 Weights do not need to and should not be normalized.
        rgb: [N x num_samples x 3] tensor of colors at each location.
    Returns:
        ray_color: [N x 3] tensor of the color of each ray.
    """
    ray_color = torch.sum(weights * rgb, dim=1)
    return ray_color

def inverse_transform_sampling(o_rays: torch.Tensor, d_rays: torch.Tensor, weights, ts, num_samples):
    """Performs inverse transform sampling according to the weights.

    Samples from ts according to the weights (i.e. ts with higher weights are 
    more likely to be sampled).
    
    Probably not the best implementation, since the official NeRF implementation 
    does something different. This is probably good enough though? Good thing
    I don't have to be rigorous. 

    Args:
        o_rays: [N x 3] coordinates of the ray origin.
        d_rays: [N x 3] directions of the ray.
        weights: [N x C x 1] tensor of weights calculated as 
                 w = T(1 - exp(- density * delta)). N is the batch size, and C 
                 is the number of coarse samples.
        ts: [N x C x 1] is the increment between each sample. N is the batch 
            size, and C is the number of coarse samples. 
        num_samples: number of samples to return per ray.
    Returns:
        fine_samples: [N x num_samples x 3] tensor sampled according to weights.
                      Instead of using the same values as in ts, we pertube it by 
                      adding random noise (sampled from U(0, 1/num_samples)).
        fine_ts: [N x num_samples x 1] tensor of the time increment for each sample. 
    """
    N, C, _ = ts.shape
    o_rays = o_rays.unsqueeze(1)
    d_rays = d_rays.unsqueeze(1)

    cdf = torch.cumsum(weights, axis=1)  # [N x C x 1]
    cdf = cdf / cdf[:, -1, None]
    eps = torch.rand((N, 1), device=device) / num_samples  # low variance sampling
    samples = torch.arange(0, 1, 1 / num_samples, device = device)
    samples = torch.broadcast_to(samples, (N, num_samples))
    samples = samples + eps

    cdf = torch.squeeze(cdf, -1)  # make dimensions match, [N x C]
    idxs = torch.searchsorted(cdf, samples).unsqueeze(-1)  # [N x C x 1]
    idxs[idxs >= C] = C - 1
    bins = torch.gather(ts, 1, idxs)
    
    fine_ts = bins + torch.rand((N, num_samples, 1), device=device) / num_samples
    fine_samples = o_rays + fine_ts * d_rays
    return fine_samples, fine_ts

"""""""""""""""
View / Image reconstruction utilities
"""""""""""""""

def generate_360_view_synthesis(model, save_dir: Path, epoch, height=800, width=800,
                                radius=4.0, cam_angle_x=0.6911112070083618, N=4096):
    """Generates a 360 view of a NeRF model.

    Saves a 360 degree view of the NeRF model at SAVE_DIR/EPOCH-360.gif
    
    Args:
        model: a nerf_model.NeRFNetwork object
        save_dir: path to a save directory.
        epoch: the ckpt epoch, used in naming the resulting gif.
        height/width: height of the images that NeRF was trained on.
        radius: The 360 view from a radius.
        cam_angle_x: x-axis field of view in angles.
    """
    assert save_dir.exists() and save_dir.is_dir()
    poses = [pose_spherical(angle, -30, radius) for angle in np.linspace(-180,180,40+1)[:-1]]
    focal = 0.5 * width / np.tan(0.5 * cam_angle_x)
    views = []
    for pose in tqdm(poses):
        o_rays, d_rays = dataloader.get_rays(height, width, focal, pose)
        im = view_reconstruction(model, o_rays.to(device), d_rays.to(device), N=N)
        views.append(im)
        del o_rays, d_rays, im
    imageio.mimwrite(Path(save_dir, f'{epoch}-360.gif'), views)

def view_reconstruction(model, all_o_rays, all_d_rays, N=4096):
    """Queries the model at every ray direction to generate an image from a view.
    
    Args:
        model: a nerf_model.ImageNeRFModel object 
        all_o_rays: [H x W x 3] vector of 3D origins. (should all be identical)
        all_d_rays: [H x W x 3] vector of directions.
        N: batch size to pass through model.
    Returns:
        an [im_h x im_w x 3] numpy array representing an image.
    """
    H, W, C = all_o_rays.shape
    all_o_rays = all_o_rays.view((H*W, C))
    all_d_rays = all_d_rays.view((H*W, C))
    im = []
    for i in range(0, H*W, N): 
        recon_preds = model.forward(all_o_rays[i:min(H*W,i+N),:], all_d_rays[i:min(H*W,i+N),:])
        im.append(recon_preds['fine_rgb_rays'].cpu().clone().detach().numpy())
    im = np.concatenate(im, axis=0).reshape((H, W, C))
    im *= 255
    im = np.clip(im, 0, 255)
    return im.astype(np.uint8)

def photo_nerf_to_image(model, im_h, im_w): 
    """Queries the model at every idx to generate an image 
    
    Args:
        model: a nerf_model.ImageNeRFModel object 
        im_h: the height of the image 
        im_w: the width of the image 
    Returns:
        an [im_h x im_w x 3] tensor representing an image.
    """
    if type(im_h) != int:
        im_h = int(im_h[0])
        im_w = int(im_w[0])
    idxs = [(i,j) for i,j in itertools.product(np.arange(0,im_h), np.arange(0,im_w))]
    idxs = torch.FloatTensor(idxs).to(model.device)
    idxs[:,0] /= (im_h-1)
    idxs[:,1] /= (im_w-1)
    N, _ = idxs.shape
    recon = []
    step = 4096
    for i in range(0, N, step):
        # break up whole tensor into sizeable chunks
        batch = idxs[i:i+step,:]
        rgb = model(batch)
        recon.append(rgb)
    recon = torch.cat(recon, axis=0).reshape((im_h, im_w, 3))
    return recon

def torch_to_numpy(torch_tensor, is_normalized_image = False):
    """ Converts torch tensor (...CHW) to numpy tensor (...HWC) for plotting
    
        If it's a normalized image, it puts it back in [0,255] range
    """
    np_tensor = torch_tensor.cpu().clone().detach().numpy()
    if np_tensor.ndim >= 4: # ...CHW -> ...HWC
        np_tensor = np.moveaxis(np_tensor, [-3,-2,-1], [-1,-3,-2])
    if is_normalized_image:
        np_tensor *= 255
        np_tensor = np.clip(np_tensor, 0, 255)
    return np_tensor

def save_torch_as_image(torch_tensor, file_path, is_normalized_image=False):
    """ saves torch tensor as a sequence of images

        @param torch_tensor: [N x C x H x W] tensor or a single [H x W x C] tensor
        @param file_path: place to save file to
        @param is_normalized_image: whether the image is normalized or not
    """
    im = torch_to_numpy(torch_tensor, is_normalized_image=is_normalized_image)
    img = Image.fromarray(im.astype(np.uint8)) 
    img = img.save(f"{file_path}.png")


"""""""""""""""
Code from original NeRF repository
"""""""""""""""

trans_t = lambda t : torch.tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=torch.float32)

rot_phi = lambda phi : torch.tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1],
], dtype=torch.float32)

rot_theta = lambda th : torch.tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1],
], dtype=torch.float32)

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.tensor([[-1.0,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w