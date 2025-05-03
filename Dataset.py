# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms as T
from torchvision import transforms
from torch.utils.data import DataLoader
import random
from util import add_noise
from configchange import opt


class Train_Data(data.Dataset):
    def __init__(self, data_root):

        self.transform = T.ToTensor()
        self.transform1 = T.ToPILImage()
        self.transform2 = transforms.RandomHorizontalFlip(p=0.7)
        self.transform3 = transforms.RandomVerticalFlip(p=0.7)

        self.data_root = data_root

    def __getitem__(self, index):
        img_index = random.randint(1, 800)
        img = Image.open(self.data_root + "/" + str(img_index)+'.png')
        img_H = img.size[0]
        img_W = img.size[1]
        H_start = random.randint(0, img_H - opt.crop_size)
        W_start = random.randint(0, img_W - opt.crop_size)
        crop_box = (W_start, H_start, W_start + opt.crop_size, H_start + opt.crop_size)
        img_crop = img.crop(crop_box)

        label = self.transform(img_crop)
        label_increase = self.transform2(label)
        label_increase = self.transform3(label_increase)
        noise = add_noise(label_increase, opt.noise_level)

        return noise, label_increase

    def __len__(self):
        return opt.num_data


if __name__ == '__main__':
    train_data = Train_Data(data_root=opt.data_root)

    train_loader = DataLoader(train_data, 1)

    for i, (data, label) in enumerate(train_data):
        print(i)
        if i == 0:
            print(data)
            print(label)
            print(data.size())
            print(label.size())
            break





