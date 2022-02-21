# TODO: this is such an ugly patch
import sys 
sys.path.append("../")

import unittest 
import torch 
import nerf_model
import torch.testing as testing

class NerfModelTest(unittest.TestCase): 
    def setUp(self):
        self.model = nerf_model.NeRFModel(position_dim=10, density_dim=4)
        self.vector = torch.Tensor([[1.0, 1.0, 1.0]])
        self.complex_vector = torch.Tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        self.im_model = nerf_model.ImageNeRFModel(position_dim=-1)
        self.coord = torch.Tensor([[1.0, 0.0]])

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

    def test_forward_prop_shape(self): 
        density, rgb = self.model(self.vector, self.vector)
        self.assertEqual(density.shape, (1, 1))
        self.assertEqual(rgb.shape, (1, 3))

    def test_complex_forward_prop_shape(self): 
        density, rgb = self.model(self.complex_vector, self.complex_vector)
        self.assertEqual(density.shape, (2, 1))
        self.assertEqual(rgb.shape, (2, 3))

    def test_im_nerf_model(self):
        rgb = self.im_model(self.coord)
        self.assertEqual(rgb.shape, (1,3))

if __name__ == '__main__':
    unittest.main(verbosity=2)
