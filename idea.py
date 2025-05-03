


import torch
import torch.nn as nn
from einops import rearrange
import typing as t
from configchange import opt
import torch.autograd
from dyconv import DEConv_2, FCAttention, LRCED, CED
import torch.nn.functional as F
# import numpy as np

class DynamicWeightFusion(nn.Module):
    def __init__(self, channels):
        super(DynamicWeightFusion, self).__init__()
        self.channels = channels
        self.weight_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)  # Concatenate feature maps
        weights = self.weight_conv(x)   # Predict blending weights
        x1_weighted = weights[:, 0:1, :, :] * x1
        x2_weighted = weights[:, 1:2, :, :] * x2
        return x1_weighted + x2_weighted

class skip_fe_model(nn.Module):
    def __init__(self, inc, outc):
        super(skip_fe_model, self).__init__()

        self.conv1 = nn.Conv2d(inc, outc, 3, padding=1)
        self.conv2 = nn.Conv2d(outc, outc, 3, padding=1)
        self.conv3 = nn.Conv2d(outc, outc, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)





    def forward(self, x):
        # 通过卷积层 -> 激活函数 -> 池化层
        conv1 = self.conv1(x)
        relu1 = self.relu1(conv1)
        # mp1 = self.avgpool1(relu1)
        # x1 = mp1
        x1 = relu1 + x

        conv2 = self.conv2(x1) + x + x1
        relu2 = self.relu2(conv2)
        # mp2 = self.avgpool2(relu2)
        # x2 = mp2
        x2 = relu2 + x + x1
        conv3 = self.conv3(x2) + x + x1 + x2
        relu3 = self.relu3(conv3)
        # mp3 = self.avgpool3(relu3)
        # x3 = mp3
        x3 = relu3 + x + x1 + x2

        return x3

class single_conv_res(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv_res, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False)
        )

        if in_ch != out_ch:
            self.sc = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            )
        else:
            self.sc = nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        out = out3 + self.sc(x)

        return self.relu(out)


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class skip_connect(nn.Module):
    def __init__(self):
        super(skip_connect, self).__init__()

    def forward(self, x1, x2):

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = x2 + x1
        return x



class Up(nn.Module):
    def __init__(self, in_ch):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        # self.up1 = nn.PixelShuffle(2)
    def forward(self, x):
        x = self.up(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()



    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=7,  # 修改卷积核大小为7x7
            stride=1,
            padding=3,  # 调整padding以保持输出尺寸不变
            groups=in_ch  # 使用分组卷积，每个输入通道独立卷积
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,  # 逐点卷积使用1x1卷积核
            stride=1,
            padding=0,
            groups=1  # 不使用分组
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class CEBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(CEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,  out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels,  out_channels, 1)
        self.conv3 = nn.Conv2d(in_channels,  4*out_channels, 1)
        self.conv4 = nn.Conv2d(4*out_channels, out_channels, 1)
        self.body1 = nn.Sequential(self.conv1,nn.GELU(), CSDN_Tem(in_channels,  out_channels), nn.GELU(), self.conv2
                                   ,nn.GELU(),  self.conv3, nn.GELU(),  self.conv4, nn.GELU())

    def forward(self, x):
        out = self.body1(x)
        return out

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope))

        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.block(x)
        sc = self.shortcut(x)
        out = out + sc
        return out





class SCSA(nn.Module):
    def __init__(
            self,
            dim: int,
            head_num: int,
            window_size: int = 7,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            norm_cfg: t.Dict = dict(type='BN'),
            act_cfg: t.Dict = dict(type='ReLU'),
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super(SCSA, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode

        assert self.dim // 4, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = group_chans = self.dim // 4

        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans
                # dimensionality reduction
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The dim of x is (B, C, H, W)
        """
        # Spatial attention priority calculation
        b, c, h_, w_ = x.size()
        # (B, C, H)
        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        # (B, C, W)
        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        x = x * x_h_attn * x_w_attn

        # Channel attention based on self attention
        # reduce calculations
        y = self.down_func(x)
        y = self.conv_d(y)
        _, _, h_, w_ = y.size()

        # normalization first, then reshape -> (B, H, W, C) -> (B, C, H * W) and generate q, k and v
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        # (B, head_num, head_dim, head_dim)
        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        # (B, head_num, head_dim, N)
        attn = attn @ v
        # (B, C, H_, W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)
        return attn * x



class Deconv_im(nn.Module):
    def __init__(self, dim):
        super(Deconv_im, self).__init__()
        self.conv = DEConv_2(dim)
        self.act = nn.ReLU(inplace=True)
        self.att1 = SCSA(dim,8)
        self.att2 = FCAttention(dim)
        self.sigconv = single_conv(dim,dim)
        self.sigmod = nn.Sigmoid()
        self.a = 0
    def forward(self, x):
        res = self.conv(x)
        res = self.act(res)
        res =  self.sigconv(res + x)
        att1 = self.att1(res)
        att2 = self.att2(res)
        self.a = self.sigmod(att1 + att2)
        out = self.a*att1 + (1-self.a)*att2 + res
        out = out + x
        return out



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.inc = nn.Sequential(

            single_conv(1, 32),
            single_conv(32, 32),
            # Deconv_im(32),

        )

        self.down1 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(inplace=True))

        self.pool1 = nn.AvgPool2d(1)
        self.conv1 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64),
            # Deconv_im(64),

        )

        self.down2 = nn.Sequential(nn.Conv2d(64,128, 3, padding=1, stride=2), nn.ReLU(inplace=True))


        self.conv2 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),

            # Deconv_im(128),

        )

        self.down3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1, stride=2), nn.ReLU(inplace=True))



        self.up1 = Up(256)
        self.conv4 = nn.Sequential(
            single_conv(128,128),
            single_conv(128,128),
            # Deconv_im(128),

        )

        self.up2 = Up(128)
        self.conv5 = nn.Sequential(
            single_conv(64,64),
            single_conv(64,64),
            # Deconv_im(64)
        )

        self.up3 = Up(64)
        self.conv6 = nn.Sequential(
            single_conv(32,32),
            single_conv(32,32),
            # Deconv_im(32),

        )

        self.outc = outconv(32, 1)



        self.skip1 = skip_connect()
        self.skip2 = skip_connect()
        self.skip3 = skip_connect()


        self.mu_1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.gamma_1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.mu_1.data = torch.tensor(0.5)
        # self.gamma_1.data = torch.tensor(0.2)


    def forward(self, input):
        x1 = input
        y = input
        for i in range(2):
            x1 = self.CG_optim(x1, y)
            inx = self.inc(x1)

            down1 = self.down1(inx)
        # down1 = self.hlfd0(down1) + down1
            conv1 = self.conv1(down1)
        # conv1 = self.multi2(conv1) + conv1
        # at0 = self.ca_block1(conv1) + conv1
        # fe1 = self.fe1(conv1)
        # conv1 = self.pool1(conv1)


            down2 = self.down2(conv1)
        # down2 = self.hlfd1(down2) + down2
            conv2 = self.conv2(down2)


            down3 = self.down3(conv2)
        # down3 = self.hlfd2(down3) + down3
        #     conv3 = self.conv3(down3)
        # conv3 = self.multi4(conv3) +  conv3
        # at2 = self.ca_block3(conv3) + conv3
        # conv3 = self.ca_block3(conv3) + conv3
        # conv3 = self.pool3(conv3)


            up1 = self.up1(down3)
            up1 = self.skip1(up1, conv2)
        # up1 = self.multi5(up1) + up1
        # up1 = self.multi2([inx, conv1, conv2], 2) + up1

            conv4 = self.conv4(up1)
        # at3 = self.ca_block4(conv4) + conv4

        # conv4 = self.ehance_block1(conv4)
        # fe4 = self.fe4(conv4)
        # at4 = self.ca_block4(conv4)

            up2 = self.up2(conv4)
            up2 = self.skip2(up2, conv1)
        # up2 = self.multi6(up2) + up2
        # up2 = self.multi1([inx, conv1, conv2], 1) + up2

            conv5 = self.conv5(up2)
        # at4 = self.ca_block5(conv5) + conv5
        # conv5 = self.ehance_block2(conv5)
        # fe5 = self.fe5(conv5)
        # at5 = self.ca_block5(conv5)

            up3 = self.up3(conv5)
            up3 = self.skip3(up3, inx)
        # up3 = self.multi7(up3) + up3
        # up3 = self.multi0([inx, conv1, conv2], 0) + up3
            conv6 = self.conv6(up3)
        # conv6 = self.ehance_block3(conv6)
        # fe6 = self.fe6(conv6)
        # at6 = self.ca_block6(conv6)

        # out = self.ehance_block(conv6)
            res = self.outc(conv6)
            x1 = res + x1
            # out = self.reconnect(v, y)

        return x1

    # def CG_optim(self, u_k, y):
    #     mu = self.mu_1
    #     u = torch.zeros_like(y)
    #     H = 1 + mu  # 系数矩阵 H，因为 K 是单位矩阵
    #     b = y + mu * u_k  # 计算 b
    #     r = b - H * u  # 初始残差 r0
    #     p = r.clone()  # 初始搜索方向 p0
    #
    #     # 单次迭代
    #     for _ in range(2):
    #         # 计算 alpha_k
    #         r_norm_sq = torch.sum(r * r, dim=list(range(1, r.dim())), keepdim=True)
    #         Hp = H * p
    #         pHp = torch.sum(p * Hp, dim=list(range(1, p.dim())), keepdim=True)
    #         alpha = r_norm_sq / (pHp + 1e-10)
    #
    #         # 更新解向量 u
    #         u = u + alpha * p
    #
    #         # 更新残差 r
    #         r_new = r - alpha * Hp
    #
    #         # 计算 beta_k
    #         r_new_norm_sq = torch.sum(r_new * r_new, dim=list(range(1, r_new.dim())), keepdim=True)
    #         beta = r_new_norm_sq / (r_norm_sq + 1e-10)
    #
    #         # 更新搜索方向 p
    #         p = r_new + beta * p
    #
    #         # 更新残差
    #         r = r_new
    #
    #     return u

    def CG_optim(self, u_k, y):
        mu = self.mu_1
        # gamma = self.gamma_1
        u = torch.zeros_like(y)
        H = 1 + mu  # 系数矩阵 H
        # 初始化残差和搜索方向
        r = y + mu * u_k - H * u
        p = r.clone()
        for _ in range(15):  # 您可以根据需要调整迭代次数
            # 计算 H * p
            Hp = H * p

            # 计算步长 alpha_k
            numerator = (r * r).sum(dim=(1, 2, 3), keepdim=True)  # 分子：r^T * r
            denominator = (p * Hp).sum(dim=(1, 2, 3), keepdim=True) + 1e-9  # 分母：p^T * H * p，防止除零
            alpha = numerator / denominator  # alpha_k = (r^T * r) / (p^T * H * p)

            # 更新解向量 u_{k+1}
            u = u + alpha * p # u_{k+1} = u_k + alpha_k * p_k

            # 更新残差 r_{k+1}
            r_new = r - alpha * Hp# r_{k+1} = r_k - alpha_k * H * p_k

            # 计算 beta_k
            numerator_new = (r_new * r_new).sum(dim=(1, 2, 3), keepdim=True)  # 分子：r_{k+1}^T * r_{k+1}
            beta = numerator_new / (numerator + 1e-9)  # beta_k = (r_{k+1}^T * r_{k+1}) / (r_k^T * r_k)

            # 更新搜索方向 p_{k+1}
            p = r_new + beta * p  # p_{k+1} = r_{k+1} + beta_k * p_k

            # 为下一次迭代更新变量
            r = r_new.clone()


        # # 共轭梯度法更新步骤
        # for _ in range(1):  # 单次迭代
        #         # 计算步长 alpha(k)
        #     alpha = torch.mul(r.transpose(2,3) , r) / torch.mul(p.transpose(2,3), (H * p))
        #         # r1 = r - gamma*torch.mul(alpha , p)
        #         # p1 = r1.clone()
        #         # alpha1 = torch.mul(r1.transpose(2,3) , r1) / torch.mul(p1.transpose(2,3) , (H * p1))
        #         # u1 = u + torch.mul(alpha1 , p)
        #     u1 = u + torch.mul(alpha, p)

                # # 更新残差 r(k+1)
            # r_new = r - (alpha * H * p + gamma*torch.mul(alpha, p)) + 1e-8
            #     #
            #     # # 计算 beta(k)
            # beta = torch.mul(r_new.transpose(2,3) , r_new) / torch.mul(r.transpose(2,3) , r)
            #
            #     # # 更新搜索方向 p(k+1)
            # p = r_new + beta * p
            #     #
            #     # # 更新残差为新的残差
            # r = r_new

        return u





class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.unet = UNet()
    def forward(self, x):
        out = self.unet(x)
        return out






if __name__ == '__main__':

    input1 = torch.rand(1, 1, 128, 128).cuda()
    label = torch.rand(1, 1, 128, 128).cuda()

    net = Network().cuda()

    out= net(input1)
    # print(net)
    print(out.size())
    # print(out2.size())