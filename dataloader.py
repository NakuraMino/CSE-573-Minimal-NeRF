"""Dataloaders for a photo and synthetic dataset.
"""
import os 
import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json
from pathlib import Path


def image_to_tensor(image):
    if image.ndim == 4: # NHWC
        tensor = torch.from_numpy(image).permute(0,3,1,2).float()
    elif image.ndim == 3: # HWC
        tensor = torch.from_numpy(image).permute(2,0,1).float()
    return tensor

class SyntheticDataset(Dataset):

    def __init__(self, base_dir, tvt): 
        self.base_dir = base_dir
        self.tvt = tvt
        file = open(f'{self.base_dir}transforms_{tvt}.json')
        self.data = json.load(file)
        self.camera_angle = self.data['camera_angle_x']
        self.frames = self.data['frames']

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        frame.pop('rotation', None)  # not used?
        im_path = Path(self.base_dir, f"{frame['file_path']}.png")
        frame['image'] = image_to_tensor(np.array(Image.open(im_path)) / 255.0)
        frame['transform_matrix'] = np.array(frame['transform_matrix'], dtype=np.float32)
        frame['camera_angle'] = self.camera_angle
        return frame

    def collate_fn(self, batch):
        im_paths = []
        images = []
        poses = []
        camera_angles = []
        for frame in batch:
            im_paths.append(frame['file_path'])
            images.append(frame['image'])
            poses.append(frame['transform_matrix'])
            camera_angles.append(frame['camera_angle']) 
        images = torch.stack(images)
        poses = torch.stack(poses)
        camera_angles = torch.stack(camera_angles)
        batch_dict = {'images': images, 'poses': poses, 'im_paths': im_paths, 'camera_angles': camera_angles}
        return batch_dict

def getSyntheticDataloader(base_dir, tvt, batch_size=16, num_workers=8, shuffle=True): 
    dataset = SyntheticDataset(base_dir, tvt)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

class PhotoDataset(Dataset):

    def __init__(self, im_path): 
        self.im_path = im_path
        im = np.array(Image.open(im_path)) / 255.0
        self.im = image_to_tensor(im) # C x H x W
        self.C, self.H, self.W = self.im.shape 

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
