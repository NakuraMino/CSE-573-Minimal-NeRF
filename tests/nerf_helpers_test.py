# TODO: this is such an ugly patch
import sys 
sys.path.append("../")

import unittest 
import torch 
import nerf_helpers
import torch.testing as testing
import numpy as np

class DataloaderTest(unittest.TestCase): 

    def setUp(self):
        pass

    def test_calculate_unnormalized_weights(self):
        # (density: torch.Tensor, deltas: torch.Tensor)
        deltas = torch.full((1,5,1), 0.2)
        density = torch.Tensor([0,50,1,0.3,1]).view(deltas.shape)
        weights = nerf_helpers.calculate_unnormalized_weights(density, deltas)
        gt_weights = torch.Tensor([0, 0.9999546001, 8.229611e-6, 2.1646e-6, 6.34545e-6]).view(deltas.shape)
        testing.assert_close(weights, gt_weights)
        pass

    def test_estimate_ray_color(self):
        # testing one ray with equal weights and equal colors
        weights = torch.full((1, 256, 1), 1/256)
        rgbs = torch.full((1, 256, 3), 1)
        # pure white ray
        ray_color = nerf_helpers.estimate_ray_color(weights, rgbs)
        gt_ray_color = torch.ones((1,3))
        testing.assert_close(ray_color, gt_ray_color)

    def test_estimate_ray_color_one_weight(self):
        # testing one ray with equal weights and equal colors
        weights = torch.zeros((1, 256, 1))
        weights[:, 200,:] = 1.0
        rgbs = torch.full((1, 256, 3), 1)
        # pure white ray
        ray_color = nerf_helpers.estimate_ray_color(weights, rgbs)
        gt_ray_color = torch.ones((1,3))
        testing.assert_close(ray_color, gt_ray_color)

    def test_generate_deltas(self):
        ts = torch.arange(2, 6, 1).view((1,-1,1))
        gt_deltas = torch.ones((1,4,1))
        deltas = nerf_helpers.generate_deltas(ts, far=6.0)
        testing.assert_close(deltas, gt_deltas)

if __name__ == '__main__':
    unittest.main(verbosity=2)
