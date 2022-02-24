import torch
import numpy as np

def fix_batchify(batch): 
    """Squeezes torch Tensors so first dimension is the true batch dimension.

    One batch is technically [Nx3] rays and [Nx4] rgba colors, 
    but due to the way pytorch-lightning and dataloaders works,             
    I have to unbatch the 0th dimension (e.g. [1xNx3] to [Nx3])
    """
    for key, value in batch.items():
        batch[key] = value.squeeze(0) # in-place operation

def generate_coarse_samples(o_rays: torch.Tensor, d_rays: torch.Tensor, num_samples: int): 
    """Generates [N x num_samples x 3] coordinate samples.

    Args: 
        o_rays: [N x 3] coordinates of the ray origin.
        d_rays: [N x 3] directions of the ray.
        num_samples: The number of coordinates to sample from the ray.
    Returns:
        Samples: [N x num_samples x 3] tensor of coordinates. 
        ts: [N x num_samples x 1] is the increment between each sample. 
    """
    N, _ = o_rays.shape
    o_rays = o_rays.unsqueeze(1)
    d_rays = d_rays.unsqueeze(1)
    ts, _ = torch.meshgrid(torch.arange(num_samples), torch.arange(N), indexing='xy')
    del _ 
    rand = torch.rand(ts.shape)
    ts = (ts + rand) / num_samples
    ts = ts.unsqueeze(-1)
    samples = d_rays * ts + o_rays
    return samples, ts

def generate_deltas(ts: torch.Tensor):
    """Calculates the difference between each 'time' in ray samples.

    Args:
        ts: [N x num_samples x 1] tensor of times. The values are increasing from [0,1) along
            the num_samples dimension.
    Returns:
        deltas: [N x num_samples x 1]  where delta_i = t_i+1 - t_i. Last t_i+1=1.
    """
    N, _, _ = ts.shape
    upper_bound = torch.cat([ts[:,1:,:], torch.ones((N, 1, 1))], dim=1)
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
    neg_delta_density = - 1 * density * deltas
    transparency =  torch.exp(torch.cumsum(- 1 * neg_delta_density, dim=1))
    weights = (1 - torch.exp(neg_delta_density)) * transparency
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