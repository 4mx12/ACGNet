import numpy as np
import torch
import torch.nn as nn
from torch import optim
from idea import Network
# from network_swinir import SwinIR
from Dataset import Train_Data
from configchange import opt
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms as T
from util import PSNR
import os, time, datetime
from util import add_noise
from skimage.metrics import structural_similarity as compare_ssim
import cv2 as cv
import torch.nn.functional as F
from loss_improve import SARLoss
# torch.manual_seed(123)


torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def train(use_gpu=True):

    train_data = Train_Data(opt.data_root)
    train_loader = DataLoader(train_data, opt.batch_size, shuffle=True)

    net = Network()
    # criterion = nn.MSELoss()
    # 0.0005 0.0005\ 0.05 0.001
    criterion = SARLoss(
    lambda_edge=0.1,
    lambda_tv=0.001,
    print_interval=10000).cuda()


    if use_gpu:
        net = net.cuda()
        net = nn.DataParallel(net)
        criterion = criterion.cuda()
        # edge_loss = edge_loss.cuda()
        # soft_loss = soft_loss.cuda()

    # initialize weights by Xavizer
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)

    # Save the original model
    torch.save(net.state_dict(), r'./checkpoints/base_L%d_no256_no_dega.pth'%opt.noise_level)

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    psnr_best = 0
    net.train()  # 必须设置为训练模式才会打印

    f = open("./log/0311.txt", mode="a",encoding="utf-8")
    for epoch in range(opt.max_epoch):
        for i, (data, label) in enumerate(train_loader):
            start_time = time.time()
            data = data.cuda(0)
            label = label.cuda(0)

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, label)

            # ditillation_loss = soft_loss(
            #     F.softmax(output / temp, dim=1),
            #     F.softmax(label / temp, dim=1)
            # ) - edge_loss(output, label)
            # loss = alpha * loss + (1 - alpha) * ditillation_loss
            loss.backward()
            optimizer.step()


            if i % 20 == 0:  # save parameters every 20 batches
                es = time.time()- start_time
                mse_loss, psnr_now, ssim_now = test(net,epoch, i)
                print('[%d, %5d] loss:%.10f PSNR:%.3f SSIM:%.3f time:%.10f' % (
                    epoch + 1, (i + 1)*opt.batch_size, mse_loss, psnr_now, ssim_now, es))
                print('[%d, %5d] loss:%.10f PSNR:%.3f SSIM:%.3f time:%.10f' % (
                    epoch + 1, (i + 1) * opt.batch_size, mse_loss, psnr_now, ssim_now, es),file=f)

                if psnr_best < psnr_now:
                    psnr_best = psnr_now
                    torch.save(net.state_dict(), opt.save_model_path)

        # learning rate decay
        if (epoch+1) % 4 == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * opt.lr_decay
            print('learning rate: ', optimizer.param_groups[0]['lr'])
            print('learning rate: ', optimizer.param_groups[0]['lr'],file=f)
    print('Finished Training')
    print('Finished Training',file=f)
    f.close()

# This function is for checking the training effect, not the test code
def test(net1,epoch, i):
    # net1 = Network()
    # net1 = net1.cuda()
    # net1 = nn.DataParallel(net1)
    # net1.load_state_dict(torch.load(opt.save_model_path))
    n = 12
    tmse=0
    tpsnr = 0
    tssim = 0
    for i in range(1, n + 1):
        label_img = './Set12/%.2d.png'%i
        # label_img = r'/home/hainandx/mx/learn/SAR_denoise/test_data/%.2d.png' % i

        transform1 = T.ToTensor()
        transform2 = T.ToPILImage()

        with torch.no_grad():

            img = Image.open(label_img).resize((512, 512))
            # img.show()
            label = np.array(img).astype(np.float32)  # label:0~255
            img_H = img.size[0]
            img_W = img.size[1]
            img = transform1(img)


            img_noise = add_noise(img, opt.noise_level).resize_(1, 1, img_H, img_W)

            output = net1(img_noise)
            output = output.cpu()
            output = output.resize_(img_H, img_W)
            output = torch.clamp(output, min=0, max=1)
            output = transform2(output)

            output = np.array(output)  # output:0~255

            mse, psnr = PSNR(output, label)
            ssim = compare_ssim(output, label, data_range=255)
            # print(ssim)
            # Because of the randomness of Gaussian noise, the output results are different each time.
            # print(i, 'MSE loss:%f, PSNR:%f, SSIM:%.3f' % (mse, psnr, ssim))
            tmse=tmse+mse
            tpsnr = tpsnr + psnr
            tssim = tssim + ssim



    return tmse/n, tpsnr/n, tssim/n

if __name__ == '__main__':
    train()





