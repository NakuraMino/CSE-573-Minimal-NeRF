# TODO: this is such an ugly patch
import sys 
sys.path.append("../")

import unittest 
import torch 
import dataloader
import torch.testing as testing

class DataloaderTest(unittest.TestCase): 

    def setUp(self):
        im_path = './test_data/grad_lounge.png'
        self.pds = dataloader.PhotoDataset(im_path)
        self.pdl = dataloader.getPhotoDataloader(im_path, batch_size=32, num_workers=1, shuffle=True)
        base_dir = './test_data/'
        self.sds = dataloader.SyntheticDataset(base_dir, 'train', 1)
        self.sdl = dataloader.getSyntheticDataloader(base_dir, 'train', 4096, num_workers=1, shuffle=True)

    def test_photo_get_0th_idx(self):
        coords, rgb = self.pds[0]
        gt = torch.zeros((2,))
        testing.assert_close(coords, gt)
        self.assertEqual(rgb.shape, (3,))

    def test_photo_get_final_idx(self):
        coords, rgb = self.pds[403*538 - 1]
        gt = torch.FloatTensor([1.0, 1.0])
        testing.assert_close(coords, gt)
        self.assertEqual(rgb.shape, (3,))

    def test_photo_get_dataloader(self):
        dl_iter = iter(self.pdl)
        batch = next(dl_iter)
        coords, rgb = batch
        self.assertEqual(coords.shape, (32, 2))
        self.assertEqual(rgb.shape, (32, 3))

    def test_synthetic_focal_length(self): 
        # 0.5 * W / tan(0.5 * cam_angle_x) = 0.5 * 800 / tan(0.5 * 0.6) = 1293.09128
        self.assertAlmostEqual(self.sds.focal, 1293.091257506331)

    def test_get_batch(self):
        batch = next(iter(self.sdl))
        self.assertTrue('origin' in batch)
        self.assertTrue('direc' in batch)
        self.assertTrue('rgb' in batch)

if __name__ == '__main__':
    unittest.main(verbosity=2)
