"""Dataloaders for a photo and synthetic dataset.
"""

import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import imageio
import json
from pathlib import Path

def image_to_tensor(image):
    if image.ndim == 4: # NHWC
        tensor = torch.from_numpy(image).permute(0,3,1,2).float()
    elif image.ndim == 3: # HWC
        tensor = torch.from_numpy(image).permute(2,0,1).float()
    return tensor

def sample_random_coordinates(N, height, width, cropping=False): 
    """Two [N,] torch tensors representing random coordinates.

    Args:
        N: int representing number of coordinates to sample
        height: the maximum height value (exclusive)
        width: maximum width value (exclusive)
    Returns:
        xs: [N,] torch tensor of random ints [0,width)
        ys: [N,] torch tensor of random ints [0,height)
    """
    if cropping:
        edge_width = width // 4
        edge_height = height // 4
        xs = torch.randint(edge_width, width - edge_width, size=(N,))
        ys = torch.randint(edge_height, height - edge_height, size=(N,))
    else:
        xs = torch.randint(0, width, size=(N,))
        ys = torch.randint(0, height, size=(N,))
    return xs, ys

def get_rays(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                       torch.arange(H, dtype=torch.float32), indexing='xy')
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = torch.broadcast_to(c2w[:3, -1], rays_d.shape)
    return rays_o, rays_d

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

class SyntheticDataset(Dataset):
    """Dataset for Synthetic images."""

    def __init__(self, base_dir, tvt, num_rays, cropping=0):
        """Constructor.

        Args:
            base_dir: str or path to dataset directory.
            tvt: str of either "train", "val", or "test".
            num_rays: number of rays to sample per batch / image.
            cropping: number of iterations to crop input image.
        """ 
        self._setup(base_dir, tvt, num_rays, cropping)
        file = open(Path(self.base_dir, f'transforms_{tvt}.json'))
        self.data = json.load(file)
        self.frames = self.data['frames']
        self.camera_angle = self.data['camera_angle_x']
        self.focal = 0.5 * self.W / np.tan(0.5 * self.camera_angle)
        self.cropping = cropping
        self._preprocess()
        file.close()

    def _setup(self, base_dir, tvt, num_rays, cropping): 
        # we know synthetic images are 800x800.
        self.H = 800
        self.W = 800
        self.tvt = tvt
        self.cropping = cropping
        self.num_rays = num_rays
        self.base_dir = base_dir

    def _preprocess(self):
        """Change json dicts to remove 'rotation' and make path into object."""
        for f in self.frames:
            f['file_path'] = Path(self.base_dir, f"{f['file_path']}.png")
            del f['rotation']; f.pop('rotation', None)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        # retrieve image 
        xs, ys = sample_random_coordinates(self.num_rays, self.H, self.W, cropping=self.cropping > 0) # (N,), (N,)
        image = (torch.Tensor(imageio.imread(frame['file_path'], pilmode="RGB")) / 255.0).float()  # [HxWx3]
        cam_to_world = torch.Tensor(frame['transform_matrix']) 
        o_rays, d_rays = get_rays(self.H, self.W, self.focal, cam_to_world)  # [HxWx3]
        rgb = image[xs,ys,:]
        origin = o_rays[xs, ys, :]
        direction = d_rays[xs, ys, :]
        self.cropping -= 1
        del image, cam_to_world
        if self.tvt == 'train':
            return {'origin': origin, 'direc': direction, 'rgb': rgb, 'xs': xs, 'ys': ys}
        else: 
            return {'origin': origin, 'direc': direction, 'rgb': rgb, 'xs': xs, 'ys': ys,
                    'all_origin': o_rays, 'all_direc': d_rays}

def getSyntheticDataloader(base_dir, tvt, num_rays, cropping=1000, num_workers=8, shuffle=True): 
    dataset = SyntheticDataset(base_dir, tvt, num_rays, cropping=cropping)
    return DataLoader(dataset=dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)

class PhotoDataset(Dataset):

    def __init__(self, im_path): 
        self.im_path = im_path
        im = np.array(Image.open(im_path)) / 255.0
        self.im = image_to_tensor(im) # C x H x W
        self.C, self.H, self.W = self.im.shape 
        del im

    def __len__(self):
        return self.H * self.W

    def __getitem__(self, idx):
        h = idx // self.W
        w = idx % self.W
        coords = torch.FloatTensor([h / (self.H - 1), w / (self.W - 1)])
        rgb = self.im[:, h, w]
        return coords, rgb

def getPhotoDataloader(im_path, batch_size=1024, num_workers=4, shuffle=True): 
    dataset = PhotoDataset(im_path)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

class ValDataset(Dataset):

    def __init__(self, im_path):
        self.im_path = im_path
        self.im = np.array(Image.open(im_path)) / 255.0
        self.H, self.W, self.C = self.im.shape 

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return (self.H, self.W)

def getValDataloader(im_path, batch_size=1, num_workers=1, shuffle=False): 
    dataset = ValDataset(im_path)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
