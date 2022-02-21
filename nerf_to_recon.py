import torch 
import numpy as np
from PIL import Image
import itertools
import nerf_model
import cv2

def torch_to_numpy(torch_tensor, is_standardized_image = False):
    """ Converts torch tensor (...CHW) to numpy tensor (...HWC) for plotting
    
        If it's an rgb image, it puts it back in [0,255] range (and undoes ImageNet standardization)
    """
    np_tensor = torch_tensor.cpu().clone().detach().numpy()
    if np_tensor.ndim >= 4: # ...CHW -> ...HWC
        np_tensor = np.moveaxis(np_tensor, [-3,-2,-1], [-1,-3,-2])
    if is_standardized_image:
        _mean=[0.485, 0.456, 0.406]; _std=[0.229, 0.224, 0.225]
        for i in range(3):
            np_tensor[...,i] *= _std[i]
            np_tensor[...,i] += _mean[i]
        np_tensor *= 255
    np_tensor = np.clip(np_tensor, 0, 255)
    return np_tensor

def save_torch_as_image(torch_tensor, file_path, is_standardized_image=False):
    """ saves torch tensor as a sequence of images

        @param torch_tensor: [N x C x H x W] tensor or a single [H x W x C] tensor
        @param file_path: place to save file to
        @param is_standardized_image: whether the image is standardized or not
    """
    im = torch_to_numpy(torch_tensor, is_standardized_image=is_standardized_image)
    img = Image.fromarray(im.astype(np.uint8)) 
    img = img.save(f"{file_path}.png")

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

if __name__ == '__main__':
    # load image dimensions
    im_path = './tests/test_data/grad_lounge.png'
    im_h, im_w, im_c = cv2.imread(im_path, 1).shape
    # load model

    # chk_path = '/home/nakuram/CSEP573-NeRF/experiments/grad_lounge/version_None/checkpoints/epoch=323-step=17171.ckpt'
    chk_path = None
    model = nerf_model.ImageNeRFModel()
    if chk_path is not None:
        model = model.load_from_checkpoint(chk_path)
    
    # query model to reconstruct image
    recon = photo_nerf_to_image(model, im_h, im_w)
    
    # save
    save_torch_as_image(recon, './recon', is_standardized_image=True)
