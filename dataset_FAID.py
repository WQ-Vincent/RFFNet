import os
import numpy as np
from torch.utils.data import Dataset
import torch
from pdb import set_trace as stx
import random
from utils import dstools, augmentor
from utils.augmentor import flip_and_route
import cv2

def custom_sort(s):
    a, n = s.split('_')[0], int(s.split('_')[1])
    return (a, n)

class DataLoaderTrain(Dataset):
    def __init__(self, data_dir, ps):
        super(DataLoaderTrain, self).__init__()
        files = os.listdir(os.path.join(data_dir))

        self.ambient_filenames = [os.path.join(data_dir, x)  for x in files if x.endswith('_ambient.png')]
        self.flash_filenames = [os.path.join(data_dir, x) for x in files if x.endswith('_flash.png')]
        
        self.ambient_filenames = sorted(self.ambient_filenames, key=custom_sort)
        self.flash_filenames = sorted(self.flash_filenames, key=custom_sort)
        
        self.sizex = len(self.ambient_filenames)  # get the size of target

        self.ps = (ps, ps)
        self.rng = np.random.RandomState()

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        ambient_path = self.ambient_filenames[index_]
        flash_path = self.flash_filenames[index_]

        gt_img = cv2.imread(ambient_path, 1) / 255.0
        flash = cv2.imread(flash_path, 1) / 255.0
        if gt_img.shape != flash.shape:
            assert ValueError(os.path.basename(ambient_path))

        # add noise
        sigma = self.rng.randint(10, 100) / 255.0
        # gaussian noise
        _gaussian_img = np.random.normal(loc=0, scale=sigma, size=gt_img.shape)
        #alpha = self.rng.randint(1, 10) / 255.0
        # poisson nosie
        #_poisson_img = np.random.poisson(lam=gt_img/alpha)*alpha
        inp_img = _gaussian_img + gt_img

        # crop
        hh, ww = gt_img.shape[0], gt_img.shape[1]
        ch, cw = self.ps
        rr = random.randint(0, hh-ch)
        cc = random.randint(0, ww-cw)

        # Crop patch
        inp_img = inp_img[rr:rr+ch, cc:cc+cw, :]
        gt_img = gt_img[rr:rr+ch, cc:cc+cw, :]
        flash = flash[rr:rr+ch, cc:cc+cw, :]

        inp_img = np.clip(inp_img, 0,1)

        inp_img = inp_img.transpose(2, 0, 1)
        gt_img = gt_img.transpose(2, 0, 1)
        flash = flash.transpose(2, 0, 1)
        # augmentation
        aug = random.randint(0, 8)
        inp_img = flip_and_route(inp_img, aug).copy()
        gt_img = flip_and_route(gt_img, aug).copy()
        flash = flip_and_route(flash, aug).copy()

        inp_img, gt_img, flash = inp_img.astype(np.float32), gt_img.astype(np.float32), flash.astype(np.float32)
        

        return inp_img, flash, gt_img


class DataLoaderTest(Dataset):
    def __init__(self, data_dir, sigma = 25):
        super(DataLoaderTest, self).__init__()
        files = os.listdir(os.path.join(data_dir))

        self.ambient_filenames = [os.path.join(data_dir, x)  for x in files if x.endswith('_ambient.png')]
        self.flash_filenames = [os.path.join(data_dir, x) for x in files if x.endswith('_flash.png')]

        self.ambient_filenames = sorted(self.ambient_filenames, key=custom_sort)
        self.flash_filenames = sorted(self.flash_filenames, key=custom_sort)

        self.sizex = len(self.ambient_filenames)  # get the size of target
        
        self.sigma = sigma
        self.rng = np.random.RandomState()

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        ambient_path = self.ambient_filenames[index_]
        flash_path = self.flash_filenames[index_]

        name = os.path.basename(ambient_path)

        gt_bgr = cv2.imread(ambient_path, 1) / 255.0
        flash = cv2.imread(flash_path, 1) / 255.0

        # gaussian noise
        _gaussian_img = np.random.normal(loc=0, scale=self.sigma / 255.0, size=gt_bgr.shape)
        # poisson nosie
        #_poisson_img = np.random.poisson(lam=gt_bgr/self.alpha)*self.alpha
        inp_bgr = _gaussian_img + gt_bgr
        inp_bgr = np.clip(inp_bgr, 0,1)
        
        inp_bgr = inp_bgr.transpose(2, 0, 1)
        gt_bgr = gt_bgr.transpose(2, 0, 1)
        flash = flash.transpose(2, 0, 1)

        inp_bgr = inp_bgr.copy()
        gt_bgr = gt_bgr.copy()
        flash = flash.copy()

        inp_bgr, gt_bgr, flash = inp_bgr.astype(np.float32), gt_bgr.astype(np.float32), flash.astype(np.float32)


        return inp_bgr, flash, gt_bgr, name


class DataLoaderReal(Dataset):
    def __init__(self, data_dir):
        super(DataLoaderReal, self).__init__()

        files = os.listdir(os.path.join(data_dir))

        self.ambient_filenames = [os.path.join(data_dir, x)  for x in files if x.endswith('_ambient.png')]
        self.flash_filenames = [os.path.join(data_dir, x) for x in files if x.endswith('_flash.png')]

        self.ambient_filenames = sorted(self.ambient_filenames, key=custom_sort)
        self.flash_filenames = sorted(self.flash_filenames, key=custom_sort)

        self.sizex = len(self.ambient_filenames)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        ambient_path = self.ambient_filenames[index_]
        flash_path = self.flash_filenames[index_]

        noisy = cv2.imread(ambient_path, 1)/255.0
        flash = cv2.imread(flash_path, 1)/255.0

        # add gaussian noise
        _gaussian_img = np.random.normal(loc=0, scale=10 / 255.0, size=noisy.shape)
        noisy = _gaussian_img + noisy
        noisy = np.clip(noisy, 0,1)

        noisy = noisy.transpose(2, 0, 1)
        flash = flash.transpose(2, 0, 1)

        noisy, flash = noisy.astype(np.float32), flash.astype(np.float32)

        return noisy, flash


if __name__ == "__main__":
    pass
