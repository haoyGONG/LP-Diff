import os
from glob import glob
from glog import logger
from torch.utils.data import Dataset
from data import aug
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import re

class LRHRDataset(Dataset):
    def __init__(self, opt, phase):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        height = opt['height']
        width = opt['width']

        self.files_lr = os.path.join(opt['dataroot'], phase, 'inputs')
        self.files_hr = os.path.join(opt['dataroot'], phase, 'gt')
        
        lr_folders = os.listdir(self.files_lr)
        self.lr = []
        self.hr = []
        
        for folder in lr_folders:
            self.lr.append(sorted(glob(os.path.join(self.files_lr, folder, '*.jpg')), key=self.extract_number))
            self.hr.extend(glob(os.path.join(self.files_hr, folder, '*.jpg')))
            
            
        assert len(self.lr) == len(self.hr)

        # if self.opt.mode == 'train':
        self.transform_fn1 = aug.get_transforms(size=(height, width))
        self.transform_fn2 = aug.get_transforms(size=(height, width))
        self.transform_fn3 = aug.get_transforms(size=(height, width))
        self.transform_fn = aug.get_transforms(size=(height, width))

        self.normalize_fn = aug.get_normalize()
        logger.info(f'Dataset has been created with {len(self.lr)} samples')
        
    def extract_number(self, file_path):
        match = re.search(r'img_(\d+).jpg', os.path.basename(file_path))
        if match:
            return int(match.group(1))
        else:
            print('Sort Error at: ', file_path)
            return -1


    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx):
        # print(self.lr[idx][0], self.lr[idx][1], self.lr[idx][2], self.hr[idx])
        assert len(self.lr[idx]) != 0, f'Not enough LR images for index {idx}: {self.lr[idx]} found, expected at least 1.'
        if len(self.lr[idx]) < 3:
            sample_id1, sample_id2, sample_id3  = 0, 1, 1
            if len(self.lr[idx]) < 2:
                sample_id1, sample_id2, sample_id3  = 0, 0, 0
        else:
            sample_id1, sample_id2, sample_id3  = 0, 1, 2
        lr_image_1 = Image.open(self.lr[idx][sample_id1])
        lr_image_2 = Image.open(self.lr[idx][sample_id2])
        lr_image_3 = Image.open(self.lr[idx][sample_id3])        
        hr_image = Image.open(self.hr[idx])
        lr_image_1 = np.array(lr_image_1)
        lr_image_2 = np.array(lr_image_2)
        lr_image_3 = np.array(lr_image_3)
        hr_image = np.array(hr_image)

        lr_image_1 = self.transform_fn1(lr_image_1)
        lr_image_2 = self.transform_fn2(lr_image_2)
        lr_image_3 = self.transform_fn3(lr_image_3)
        hr_image = self.transform_fn(hr_image)
        
        lr_image_1 = self.normalize_fn(lr_image_1)
        lr_image_2 = self.normalize_fn(lr_image_2)
        lr_image_3 = self.normalize_fn(lr_image_3)
        hr_image = self.normalize_fn(hr_image)

        lr_image_1 = transforms.ToTensor()(lr_image_1)
        lr_image_2 = transforms.ToTensor()(lr_image_2)
        lr_image_3 = transforms.ToTensor()(lr_image_3)
        hr_image = transforms.ToTensor()(hr_image)

        # return {'LR1': lr_image_1, 'LR2': lr_image_2, 'LR3': lr_image_3, 'HR': hr_image, 
        #         'LR1_path': self.lr[idx][0], 'LR2_path': self.lr[idx][1], 'LR3_path': self.lr[idx][2], 'HR_path': self.hr[idx]}
        return {'LR1': lr_image_1, 'LR2': lr_image_2, 'LR3': lr_image_3, 'HR': hr_image, 'path': self.hr[idx]} 

    def load_data(self):
        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=int(self.opt.num_threads))
        return dataloader


def create_dataset(opt):
    return LRHRDataset(opt).load_data()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default=r'')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()
    opt = args
    dataset = LRHRDataset(opt)
    data = dataset[0]
    print(data['LR1'].shape)


