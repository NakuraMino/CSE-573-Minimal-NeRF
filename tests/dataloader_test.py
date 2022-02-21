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
        self.ds = dataloader.PhotoDataset(im_path)
        self.dl = dataloader.getPhotoDataloader(im_path, batch_size=32, num_workers=1, shuffle=True)

    def test_get_0th_idx(self):
        coords, rgb = self.ds[0]
        gt = torch.zeros((2,))
        testing.assert_close(coords, gt)
        self.assertEqual(rgb.shape, (3,))

    def test_get_final_idx(self):
        coords, rgb = self.ds[403*538 - 1]
        gt = torch.FloatTensor([1.0, 1.0])
        testing.assert_close(coords, gt)
        self.assertEqual(rgb.shape, (3,))

    def test_get_dataloader(self):
        dl_iter = iter(self.dl)
        batch = next(dl_iter)
        coords, rgb = batch
        self.assertEqual(coords.shape, (32, 2))
        self.assertEqual(rgb.shape, (32, 3))
        
    # def test_get_oob(self):
    #     self.assertRaises(IndexError, self.dl[403*538])


if __name__ == '__main__':
    unittest.main(verbosity=2)
