
import torch
from PIL import Image
import numpy as np
from torchvision import transforms as T
# from utils import add_noise
from util import PSNR
import math
import time
from idea import Network
# from modelchange3 import DPDNN
#from model import DPDNN

import torch.nn as nn
#from cofigchange2 import opt
from configtest import opt
import os
from skimage.metrics import structural_similarity as compare_ssim
np.set_printoptions(suppress=True)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def ENL(img):
    mean=np.average(img)
    std=np.std(img)
    ENL1=(mean*mean)/(std*std)
    return ENL1

def EPD_ROA_HD(im_before,im_after): #水平
    sum1=0
    sum2=0
    for i in range(img_H):
        for j in range(img_W-1):
            a=abs(im_before[i,j]/im_before[i,j+1])
            b=abs(im_after[i,j]/im_after[i,j+1])
            if a<=1000:sum1=sum1+a
            if b<=1000:sum2=sum2+b
    e=sum2/sum1
    return e

def EPD_ROA_VD(im_before,im_after): #垂直
    sum1=0
    sum2=0
    for i in range(img_H-1):
        for j in range(img_W):
            a=abs(im_before[i,j]/im_before[i+1,j])
            b=abs(im_after[i,j]/im_after[i+1,j])
            if a<=1000:sum1=sum1+a
            if b<=1000:sum2=sum2+b
    e=sum2/sum1
    return e

def MOR(im_before,im_after):
    ratio_image=np.ones((img_H,img_W))
    for i in range(img_H):
        for j in range(img_W):
            if abs(im_before[i,j]-im_after[i,j])<=15: ratio_image[i,j]=(im_after[i,j]+0.00001)/(im_before[i,j]+0.00001)
    mean_MOR = np.average(ratio_image)
    return mean_MOR

# Here is the path of your test image, 'i' means the ith image, you only need to provide the ground truth image
# Then we add Gaussian noise to the gt image
i = 2
#label_img = '/scratch/184/hzf/DPDNN_PyTorch-master/DENOISE/Set12/%.2d.png'%i
#label_img = r'/home/lincong/QCH/DPDNN_PyTorch-master/Set12/%.2d.png'%i
noise_img='./SAR_TEST/%.2d.bmp'%i

transform1 = T.ToTensor()
transform2 = T.ToPILImage()


with torch.no_grad():
    net = Network()
    net = nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load('/home/hainandx/mx/learn/SAR_denoise/checkpoints/base_L8_no256_no_dega.pth'))
    img = Image.open(noise_img)
    # img.show()
    label = np.array(img).astype(np.float32)   # label:0~255
    img_H = img.size[0]
    img_W = img.size[1]
    img = transform1(img)

    img_noise = img.resize_(1, 1, img_H, img_W)
    start_time = time.time()
    output = net(img_noise)
    end_time = time.time()
    output = output.cpu()
    output = output.resize_(img_H, img_W)
    output = torch.clamp(output, min=0, max=1)
    output = transform2(output)
    # To save the output(denoised) image, you must create a new folder. Here is my path.
    output.save('/home/hainandx/mx/learn/CBDNet-pytorch-master/SAR_output/L2/ACGNet/no_dega.bmp')

    img_noise = transform2(img_noise.resize_(img_H, img_W))
    #img_noise.show()
    # img_noise.save('./SAR_output/L%d/mse_%d_noise.png'%(opt.noise_level, i))

    # show output image
    #output.show()
    output = np.array(output)   # output:0~255
    img_noise=np.array(img_noise)
    enl = ENL(output)
    epd_hd=EPD_ROA_HD(img_noise,output)
    epd_vd=EPD_ROA_VD(img_noise,output)
    mor=MOR(img_noise,output)
    print('ENL of image ' + str(i) + ' is ', enl, 'EPD_ROA_HD:',epd_hd,'EPD_ROA_VD:',epd_vd,'MOR',mor)

    print("程序运行时间：", end_time - start_time, "秒")










