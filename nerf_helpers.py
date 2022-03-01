import torch
import numpy as np
from nerf_to_recon import torch_to_numpy

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
    del rand
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
    del upper_bound
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
    neg_delta_density = - 1 * density * deltas
    transmittance =  torch.exp(torch.cumsum(- 1 * neg_delta_density, dim=1))
    weights = (1 - torch.exp(neg_delta_density)) * transmittance
    del transmittance
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
    del cdf, idxs, bins, samples, eps
    return fine_samples, fine_ts

def view_reconstruction(model, all_o_rays, all_d_rays, N=4096):
    H, W, C = all_o_rays.shape
    all_o_rays = all_o_rays.view((H*W, C))
    all_d_rays = all_d_rays.view((H*W, C))
    im = []
    for i in range(0, H*W, N): 
        recon_preds = model.forward(all_o_rays[i:min(H*W,i+N),:], all_d_rays[i:min(H*W,i+N),:])
        im.append(recon_preds['pred_rgbs'])
    im = torch.cat(im, dim=0).view((H,W, C))
    im = torch_to_numpy(im, is_normalized_image=True)
    return im