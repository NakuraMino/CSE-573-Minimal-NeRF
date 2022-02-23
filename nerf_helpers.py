import torch
import numpy as np

def fix_batchify(batch): 
        # One batch is technically [Nx3] rays and [Nx4] rgba colors, 
        # but due to the way pytorch-lightning and dataloaders works,             
        # I have to unbatch the 0th dimension (e.g. [1xNx3] to [Nx3])
        for key, value in batch.items():
            batch[key] = value.squeeze(0) # in-place operation

def generate_coarse_samples(o_rays, d_rays, num_samples): 
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

def generate_deltas(ts): 
    N, _, _ = ts.shape
    upper_bound = torch.cat([ts[:,1:,:], torch.ones((N, 1, 1))], dim=1)
    deltas = upper_bound - ts
    return deltas