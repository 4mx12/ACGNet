
import torch
import numpy as np
import math
import cv2 as cv
import os
import numpy as np
from PIL import Image
from skimage import util

# input type:tensor ; output type:tensor


def add_noise(input_img, noise_sigma):
    '''
    Modeling multiplicative noise
    '''

    rows = input_img.size(1)
    columns = input_img.size(2)
    s = np.zeros((rows, columns))
    for k in range(0, noise_sigma):
        gamma = np.abs(np.random.randn(rows, columns) + np.random.randn(rows, columns) * 1j) ** 2 / 2
        s = s + gamma
    s_amplitude = np.sqrt(s / noise_sigma)
    img_L = input_img * s_amplitude
    img_L = 255 * (img_L > 255) + img_L * (img_L <= 255)
    noise_img = torch.clamp(torch.tensor(img_L).float(), 0.0, 1.0)

    return noise_img




def PSNR(img1, img2, color=False):


    mse1 = np.mean((img1 / 255. - img2 / 255.) ** 2) * 255 * 255
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return mse1, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    '''mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return mse * 255 * 255, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))'''






















