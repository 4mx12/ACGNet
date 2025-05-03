
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
    '''img = np.array(input_img)
    noise_gs_img = util.random_noise(img, mode='poisson')
    #noise_gs_img = util.random_noise(img, mode='speckle',var=noise_sigma)
    noise_img = torch.clamp(torch.tensor(noise_gs_img).float(), 0.0, 1.0)'''

    '''noise_sigma = noise_sigma / 255
    noise_img = torch.clamp(input_img + noise_sigma * torch.randn_like(input_img), 0.0, 1.0)
    return noise_img'''

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
    # noise_img = torch.clamp(torch.tensor(img_L), 0.0, 1.0)
    # print(noise_img.numpy().shape)
    '''plt.imshow(noise_img.numpy()),plt.title('IMG1')
    plt.show()'''
    return noise_img


def add_noise2(input_img, noise_sigma):
    # 将噪声标准差归一化到[0,1]范围 (假设输入noise_sigma是0-255范围的数值)
    noise_sigma_normalized = noise_sigma / 255.0

    # 生成与输入图像相同尺寸的高斯噪声
    gaussian_noise = torch.randn_like(input_img) * noise_sigma_normalized

    # 添加噪声并限制像素值范围
    noisy_img = input_img + gaussian_noise
    noisy_img = torch.clamp(noisy_img, 0.0, 1.0)

    return noisy_img


def add_gaussian_noise(image, mean=0, std_dev=25):


    # 生成与图像相同尺寸的高斯噪声
    gaussian_noise = np.random.normal(mean, std_dev, image.shape)

    # 将噪声添加到图片中
    noisy_img_array = image + gaussian_noise

    # 确保图片像素值在0到255之间
    noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)

    # 将数组转换回图片
    noisy_image = Image.fromarray(noisy_img_array)

    return noisy_image


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






















