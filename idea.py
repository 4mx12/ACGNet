import torch
import torch.nn as nn
from einops import rearrange
import typing as t
import torch.autograd
from dyconv import DEConv_2, FCAttention
import torch.nn.functional as F


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
            Deconv_im(32),

        )

        self.down1 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64),
            Deconv_im(64),

        )

        self.down2 = nn.Sequential(nn.Conv2d(64,128, 3, padding=1, stride=2), nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            Deconv_im(128),

        )

        self.down3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1, stride=2), nn.ReLU(inplace=True))

        self.up1 = Up(256)
        self.conv4 = nn.Sequential(
            single_conv(128,128),
            single_conv(128,128),
            Deconv_im(128),

        )

        self.up2 = Up(128)
        self.conv5 = nn.Sequential(
            single_conv(64,64),
            single_conv(64,64),
            Deconv_im(64)
        )

        self.up3 = Up(64)
        self.conv6 = nn.Sequential(
            single_conv(32,32),
            single_conv(32,32),
            Deconv_im(32),

        )

        self.outc = outconv(32, 1)

        self.skip1 = skip_connect()
        self.skip2 = skip_connect()
        self.skip3 = skip_connect()

        self.mu_1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.mu_1.data = torch.tensor(0.5)



    def forward(self, input):
        x1 = input
        y = input
        for i in range(2):
            x1 = self.CG_optim(x1, y)

            inx = self.inc(x1)
            down1 = self.down1(inx)

            conv1 = self.conv1(down1)
            down2 = self.down2(conv1)

            conv2 = self.conv2(down2)
            down3 = self.down3(conv2)

            up1 = self.up1(down3)
            up1 = self.skip1(up1, conv2)
            conv4 = self.conv4(up1)

            up2 = self.up2(conv4)
            up2 = self.skip2(up2, conv1)
            conv5 = self.conv5(up2)

            up3 = self.up3(conv5)
            up3 = self.skip3(up3, inx)
            conv6 = self.conv6(up3)

            res = self.outc(conv6)
            x1 = res + x1


        return x1


    def CG_optim(self, u_k, y):
        mu = self.mu_1
        u = torch.zeros_like(y)
        H = 1 + mu  #  H

        r = y + mu * u_k - H * u
        p = r.clone()
        for _ in range(15):  #  adjust the number of iterations as needed
            # compute H * p
            Hp = H * p

            # Compute alpha_k
            numerator = (r * r).sum(dim=(1, 2, 3), keepdim=True)
            denominator = (p * Hp).sum(dim=(1, 2, 3), keepdim=True) + 1e-9  
            alpha = numerator / denominator  # alpha_k = (r^T * r) / (p^T * H * p)

            # Update u_k
            u = u + alpha * p # u_k+1 = u_k + alpha_k * p_k

            # Update r_k+1
            r_new = r - alpha * Hp # r_{k+1} = r_k - alpha_k * H * p_k

            # Compute beta_k
            numerator_new = (r_new * r_new).sum(dim=(1, 2, 3), keepdim=True) 
            beta = numerator_new / (numerator + 1e-9)  # beta_k = (r_{k+1}^T * r_{k+1}) / (r_k^T * r_k)

            # Update p_k+1
            p = r_new + beta * p  # p_k+1 = r_k+1 + beta_k * p_k

            r = r_new.clone()


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
    print(out.size())
