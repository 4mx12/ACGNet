
import torch
from PIL import Image
import numpy as np
from torchvision import transforms as T
from util import add_noise
from util import PSNR
import math

from idea import Network
#from model import DPDNN

import torch.nn as nn
#from cofigchange2 import opt
from configchange import opt
import os
from skimage.metrics import structural_similarity as compare_ssim




os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# Here is the path of your test image, 'i' means the ith image, you only need to provide the ground truth image
# Then we add Gaussian noise to the g2t image
n = 12
tpsnr=0
tssim=0
for i in range(1,n+1):
#label_img = '/scratch/184/hzf/DPDNN_PyTorch-master/DENOISE/Set12/%.2d.png'%i
    label_img = r'/home/hainandx/mx/learn/SAR_denoise/Set12/%.2d.png'%i


    transform1 = T.ToTensor()
    transform2 = T.ToPILImage()

    with torch.no_grad():
        net = Network()
        net = nn.DataParallel(net).cuda()
        net.load_state_dict(torch.load('/home/hainandx/mx/learn/SAR_denoise/checkpoints/losepro_L2.pth'))

        img = Image.open(label_img).resize((512,512))
        # img.show()
        label = np.array(img).astype(np.float32)   # label:0~255
        img_H = img.size[0]
        img_W = img.size[1]
        img = transform1(img)

        img_noise = add_noise(img, opt.noise_level).resize_(1, 1, img_H, img_W)

        output = net(img_noise.cuda())
        output = output.cpu()
        output = output.resize_(img_H, img_W)
        output = torch.clamp(output, min=0, max=1)
        output = transform2(output)
        # To save the output(denoised) image, you must create a new folder. Here is my path.
        # output.save('/home/hainandx/mx/learn/SAR_denoise/result_visual/denoise_%d_%d.png'%(opt.noise_level,i))

        img_noise = transform2(img_noise.resize_(img_H, img_W))
        #img_noise.show()
        # img_noise.save('/home/hainandx/mx/learn/SAR_denoise/result_visual/noise_%d_%d.png'%(opt.noise_level, i))

        # show output image
        #output.show()
        output = np.array(output)   # output:0~255

        mse, psnr = PSNR(output, label)
        ssim = compare_ssim(output, label, data_range=255)
        #print(ssim)
        # Because of the randomness of Gaussian noise, the output results are different each time.
        print(i, 'MSE loss:%f, PSNR:%f, SSIM:%.3f'%(mse, psnr,ssim))
        tpsnr=tpsnr+psnr
        tssim=tssim+ssim
print(tpsnr/n,tssim/n)








