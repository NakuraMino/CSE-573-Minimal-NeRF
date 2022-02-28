# TODO: this is such an ugly patch
import sys 
sys.path.append("../")

import torch 
import unittest 
import torch.testing as testing

# my files
import nerf_model
import dataloader 

class NerfModelTest(unittest.TestCase): 
    def setUp(self):
        # dataloader to test fwd fn
        base_dir = './test_data/'
        self.dl = dataloader.getSyntheticDataloader(base_dir, 'train', 1, num_workers=1, shuffle=True)
        self.batch = next(iter(self.dl))
        # random tensors to use for testing
        self.vector = torch.Tensor([[1.0, 1.0, 1.0]])
        self.complex_vector = torch.Tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        self.coord = torch.Tensor([[1.0, 0.0]])
        self.samples = torch.rand((4, 4, 3))
        self.direc = torch.rand((4, 3))
        # NeRF models.
        self.single_model = nerf_model.NeRFModel(position_dim=10, direction_dim=4)
        self.im_model = nerf_model.ImageNeRFModel(position_dim=-1)
        self.network = nerf_model.NeRFNetwork(position_dim=10, direction_dim=4, coarse_samples=64, fine_samples=128)

    """""""""
    Testing full NeRF model
    """""""""
    def test_nerf_network_training_step(self):
        loss = self.network.training_step(self.batch, 0)
        self.assertGreaterEqual(loss, 0)

    """""""""
    Testing positional encoding function
    """""""""

    def test_positional_encoding_shape(self): 
        enc = nerf_model.positional_encoding(self.vector, dim=1)
        self.assertEqual(enc.shape, (1, 6))

    def test_positional_encoding_values(self): 
        enc = nerf_model.positional_encoding(self.vector, dim=1)
        expected = torch.Tensor([[-1.0, -1.0, -1.0, 0.0, 0.0, 0.0]])
        testing.assert_close(enc, expected)

    def test_complex_positional_encoding_shape(self): 
        enc = nerf_model.positional_encoding(self.complex_vector, dim=1)
        self.assertEqual(enc.shape, (2, 6))

    def test_complex_positional_encoding_values(self): 
        enc = nerf_model.positional_encoding(self.complex_vector, dim=1)
        expected = torch.Tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                                 [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0]])
        testing.assert_close(enc, expected)

    def test_3D_positional_encoding_shape(self): 
        samples = torch.rand((4096, 64, 3))
        enc = nerf_model.positional_encoding(samples, dim=10)
        self.assertEqual(enc.shape, (4096, 64, 60))

    """""""""
    Testing single NeRF Model
    """""""""

    def test_single_complex_forward_prop_shape(self): 
        density, rgb = self.single_model(self.samples, self.direc)
        self.assertEqual(density.shape, (4, 4, 1))
        self.assertEqual(rgb.shape, (4,4, 3))

    """""""""
    Testing image NeRF Model
    """""""""

    def test_im_nerf_model(self):
        rgb = self.im_model(self.coord)
        self.assertEqual(rgb.shape, (1,3))

if __name__ == '__main__':
    unittest.main(verbosity=2)
