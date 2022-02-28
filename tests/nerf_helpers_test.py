# TODO: this is such an ugly patch
import sys 
sys.path.append("../")

import unittest 
import torch 
import dataloader
import nerf_helpers
import torch.testing as testing
import numpy as np

class DataloaderTest(unittest.TestCase): 

    def setUp(self):
        pass
    
    def test_ndc_rays_direction(self):
        origin = torch.Tensor([1,1,1]).view((1,1,-1))
        direc = torch.Tensor([1,1,1]).view((1,1,-1))
        _, d_new = dataloader.convert_to_ndc_rays(origin, direc, 1, 4, 4)
        gt_d_new = torch.Tensor([0,0,2]).view((1,1,-1))
        np.testing.assert_array_equal(d_new.numpy(), gt_d_new.numpy())        

    def test_ndc_rays_unit_direction(self):
        origin = torch.Tensor([1,1,1]).view((1,1,-1))
        direc = torch.Tensor([1,1,1]).view((1,1,-1))
        _, d_new = dataloader.convert_to_ndc_rays(origin, direc, 1, 4, 4)
        d_norm = np.linalg.norm(d_new.numpy())
        np.testing.assert_array_equal(d_norm, 1.0)        

    def test_ndc_rays_origin(self):
        origin = torch.Tensor([1,1,1]).view((1,1,-1))
        direc = torch.Tensor([1,1,1]).view((1,1,-1))
        o_new, _ = dataloader.convert_to_ndc_rays(origin, direc, 1, 4, 4)
        gt_o_new = torch.Tensor([-0.5,-0.5,-1]).view((1,1,-1))
        np.testing.assert_almost_equal(o_new.numpy(), gt_o_new.numpy())        

if __name__ == '__main__':
    unittest.main(verbosity=2)
